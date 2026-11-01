"""Consistency-DPO: fine-tune a Latent Consistency Model on human preference pairs.

Implements Sec. 3.2 of Croitoru et al., "Curriculum Direct Preference Optimization for
Diffusion and Consistency Models" (CVPR 2025, https://arxiv.org/abs/2405.13637), without the
curriculum. A LoRA on the LCM is the only thing trained. The LCM conventions follow
diffusers/examples/consistency_distillation (Apache-2.0).
"""
import argparse
import io
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    LatentConsistencyModelPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "conv1",
    "conv2",
    "conv_shortcut",
    "downsamplers.0.conv",
    "upsamplers.0.conv",
    "time_emb_proj",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--teacher_model", default="lykon/dreamshaper-7")
    parser.add_argument("--teacher_variant", default="fp16")
    # yuvalkirstain/pickapic_v2 no longer answers anonymous requests; this is a mirror of it.
    parser.add_argument("--dataset", default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--eval_split", default="validation_unique")
    parser.add_argument("--num_pairs", type=int, default=20000)
    parser.add_argument("--pairs_path", default=None, help="defaults to <output_dir>/pairs.pt")
    parser.add_argument("--vae_batch_size", type=int, default=8)
    # The paper trains at 768 on 64 GB of VRAM.
    parser.add_argument("--resolution", type=int, default=512)
    # Values from the paper: beta, rank (alpha = rank) and 10k iterations.
    parser.add_argument("--beta", type=float, default=200)
    parser.add_argument("--lora_rank", type=int, default=64)
    # The paper's 3e-4 diverges on Pick-a-Pic pairs, see the README.
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    # 32 pairs per update, as in the paper's 16 x 2 but within 24 GB.
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # The paper leaves these out; they are the defaults of the authors' code.
    parser.add_argument("--huber_c", type=float, default=0.001)
    parser.add_argument("--num_ddim_timesteps", type=int, default=50)
    parser.add_argument("--w_min", type=float, default=5.0)
    parser.add_argument("--w_max", type=float, default=15.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--resume_from", default=None, help="a checkpoint-N folder to continue from")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def is_usable(entry, eval_prompts):
    """Keeps pairs with a clear human winner whose prompt is not held out for the benchmark."""
    return entry["are_different"] and entry["label_0"] != entry["label_1"] and entry["caption"] not in eval_prompts


def winner_and_loser(entry):
    images = [Image.open(io.BytesIO(entry[key])).convert("RGB") for key in ("jpg_0", "jpg_1")]
    return images if entry["label_0"] > entry["label_1"] else images[::-1]


@torch.no_grad()
def build_pairs(args, path, device):
    """Encodes the pairs to latents once, so a long run never depends on the network."""
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def encode(images):
        pixel_values = torch.stack([transform(image) for image in images]).to(device)
        return (vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor).to(torch.float16).cpu()

    eval_prompts = {entry["caption"] for entry in load_dataset(args.dataset, split=args.eval_split, streaming=True)}
    stream = load_dataset(args.dataset, split="train", streaming=True).shuffle(seed=args.seed, buffer_size=1000)
    entries = itertools.islice((entry for entry in stream if is_usable(entry, eval_prompts)), args.num_pairs)

    latents_w, latents_l, captions = [], [], []
    with tqdm(total=args.num_pairs, desc="Encoding pairs") as progress:
        while batch := list(itertools.islice(entries, args.vae_batch_size)):
            images_w, images_l = zip(*[winner_and_loser(entry) for entry in batch])
            latents_w.append(encode(images_w))
            latents_l.append(encode(images_l))
            captions.extend(entry["caption"] for entry in batch)
            progress.update(len(batch))

    torch.save({"latents_w": torch.cat(latents_w), "latents_l": torch.cat(latents_l), "captions": captions}, path)
    del vae
    torch.cuda.empty_cache()


class PreferencePairs(torch.utils.data.Dataset):
    def __init__(self, path):
        self.pairs = torch.load(path)

    def __len__(self):
        return len(self.pairs["captions"])

    def __getitem__(self, i):
        return self.pairs["latents_w"][i], self.pairs["latents_l"][i], self.pairs["captions"][i]


def pseudo_huber(a, b, c):
    return (torch.sqrt((a.float() - b.float()) ** 2 + c**2) - c).mean(dim=(1, 2, 3))


def consistency_dpo_loss(pred, ref, target, beta, huber_c):
    """Eq. 6-8. Each input holds the winners followed by the losers.

    d* stands in for the DPO log-likelihood ratio: how much further the student sits from the
    reference's target than the reference itself does. It is 0 until the student moves.
    """
    d_star = pseudo_huber(pred, target, huber_c) - pseudo_huber(ref, target, huber_c)
    d_w, d_l = d_star.chunk(2)
    return -F.logsigmoid(-beta * (d_w - d_l)).mean(), d_w, d_l


def consistency_output(model, sample, timesteps, w_embedding, prompt_embeds, scheduler):
    """f(x_t, t, w, c) of an LCM, computed the way LCMScheduler.step does."""
    noise_pred = model(sample, timesteps, timestep_cond=w_embedding, encoder_hidden_states=prompt_embeds).sample.float()
    alpha_prod = scheduler.alphas_cumprod.to(sample.device)[timesteps][:, None, None, None]
    pred_x0 = (sample - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
    c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timesteps)
    return c_skip[:, None, None, None] * sample + c_out[:, None, None, None] * pred_x0


def ddim_step(sample, noise_pred, start_timesteps, timesteps, scheduler):
    """One DDIM step along the PF-ODE, from start_timesteps down to timesteps."""
    alphas_cumprod = scheduler.alphas_cumprod.to(sample.device)
    alpha_prod = alphas_cumprod[start_timesteps][:, None, None, None]
    alpha_prod_prev = alphas_cumprod[timesteps][:, None, None, None]
    pred_x0 = (sample - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
    return alpha_prod_prev.sqrt() * pred_x0 + (1 - alpha_prod_prev).sqrt() * noise_pred


def predict_pair(unet, teacher, latents_w, latents_l, prompt_embeds, uncond_embeds, scheduler, args):
    """Returns f_phi(x_tn+1), f_ref(x_tn+1) and f_ref(x_hat_tn), winners first.

    The winner and loser of a pair share their noise, timestep and guidance scale. f_ref is the
    student with its LoRA switched off, so it is frozen by construction.
    """
    bsz, device = latents_w.shape[0], latents_w.device
    step = scheduler.config.num_train_timesteps // args.num_ddim_timesteps

    def both(x):
        return torch.cat([x, x])

    start_timesteps = both(step * torch.randint(1, args.num_ddim_timesteps + 1, (bsz,), device=device) - 1)
    timesteps = (start_timesteps - step).clamp(min=0)
    noise = both(torch.randn_like(latents_w))
    w = both((args.w_max - args.w_min) * torch.rand(bsz) + args.w_min)
    # The embedding the pipeline uses at inference, where guidance_scale = w + 1.
    w_embedding = LatentConsistencyModelPipeline.get_guidance_scale_embedding(
        None, w, embedding_dim=unet.config.time_cond_proj_dim
    ).to(device)
    w = w.to(device)
    prompt_embeds, uncond_embeds = both(prompt_embeds), both(uncond_embeds)

    sample = scheduler.add_noise(torch.cat([latents_w, latents_l]), noise, start_timesteps)
    pred = consistency_output(unet, sample, start_timesteps, w_embedding, prompt_embeds, scheduler)

    with torch.no_grad():
        teacher_sample = sample.to(teacher.dtype)
        cond = teacher(teacher_sample, start_timesteps, encoder_hidden_states=prompt_embeds).sample.float()
        uncond = teacher(teacher_sample, start_timesteps, encoder_hidden_states=uncond_embeds).sample.float()
        noise_pred = cond + w[:, None, None, None] * (cond - uncond)
        sample_prev = ddim_step(sample, noise_pred, start_timesteps, timesteps, scheduler)

        with unet.disable_adapter():
            ref = consistency_output(unet, sample, start_timesteps, w_embedding, prompt_embeds, scheduler)
            target = consistency_output(unet, sample_prev, timesteps, w_embedding, prompt_embeds, scheduler)

    return pred, ref, target


def save_lora(unet, path):
    # No convert_state_dict_to_diffusers: it renames only the attention keys, and load_lora_weights
    # then silently drops them because the conv keys are left in the PEFT format.
    state_dict = {key.removeprefix("base_model.model."): value for key, value in get_peft_model_state_dict(unet).items()}
    StableDiffusionPipeline.save_lora_weights(path, unet_lora_layers=state_dict)


def load_lora(unet, path):
    """Restores what save_lora wrote, refusing a partial match that would otherwise train on silently."""
    state_dict = load_file(os.path.join(path, "pytorch_lora_weights.safetensors"))
    state_dict = {f"base_model.model.{key.removeprefix('unet.')}": value for key, value in state_dict.items()}
    result = set_peft_model_state_dict(unet, state_dict)
    missing = [key for key in result.missing_keys if "lora_" in key]
    if missing or result.unexpected_keys:
        raise ValueError(f"{path} does not match the LoRA: {len(missing)} missing, {len(result.unexpected_keys)} unexpected")


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision
    )
    device = accelerator.device
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    start_step = int(args.resume_from.rstrip("/").rsplit("-", 1)[1]) if args.resume_from else 0
    # Offset so a resumed run does not replay the batches and noise it already trained on.
    set_seed(args.seed + start_step)

    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = args.pairs_path or os.path.join(args.output_dir, "pairs.pt")
    if not os.path.exists(pairs_path):
        build_pairs(args, pairs_path, device)

    scheduler = LCMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(f"Only epsilon prediction is supported, got {scheduler.config.prediction_type}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    text_encoder.to(device, dtype=weight_dtype).requires_grad_(False)
    teacher = UNet2DConditionModel.from_pretrained(
        args.teacher_model, subfolder="unet", variant=args.teacher_variant or None
    )
    teacher.to(device, dtype=weight_dtype).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    unet.to(device, dtype=weight_dtype).requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # lora_B starts at zero, so f_phi equals f_ref until the first update.
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=LORA_TARGET_MODULES)
    unet = get_peft_model(unet, lora_config)
    cast_training_params(unet, dtype=torch.float32)
    if args.resume_from:
        # Checkpoints hold only the LoRA, so Adam's moments start over.
        load_lora(unet, args.resume_from)
    lora_params = [param for param in unet.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_dataloader = torch.utils.data.DataLoader(
        PreferencePairs(pairs_path), batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    def encode_prompt(captions):
        input_ids = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids
        return text_encoder(input_ids.to(device))[0]

    with torch.no_grad():
        uncond_embeds = encode_prompt([""] * args.train_batch_size)

    global_step = start_step
    window = []
    start = time.time()
    progress = tqdm(initial=start_step, total=args.max_train_steps, desc="Steps")
    with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as metrics:
        while global_step < args.max_train_steps:
            for latents_w, latents_l, captions in train_dataloader:
                with accelerator.accumulate(unet):
                    with torch.no_grad():
                        prompt_embeds = encode_prompt(list(captions))
                    pred, ref, target = predict_pair(
                        unet, teacher, latents_w.float(), latents_l.float(), prompt_embeds, uncond_embeds, scheduler, args
                    )
                    loss, d_w, d_l = consistency_dpo_loss(pred, ref, target, args.beta, args.huber_c)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss became {loss.item()} at step {global_step}")
                window.append((loss.item(), d_w.mean().item(), d_l.mean().item(), (d_w < d_l).float().mean().item()))

                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    if global_step % args.log_steps == 0 or global_step == 1:
                        row = dict(zip(("loss", "d_w", "d_l", "pair_acc"), np.mean(window, axis=0).tolist()))
                        row.update(step=global_step, seconds=round(time.time() - start, 1))
                        row["max_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        metrics.write(json.dumps(row) + "\n")
                        metrics.flush()
                        progress.set_postfix(loss=f"{row['loss']:.4f}", pair_acc=f"{row['pair_acc']:.2f}")
                        window = []
                    if global_step % args.save_steps == 0:
                        save_lora(unet, os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    if global_step >= args.max_train_steps:
                        break

    save_lora(unet, os.path.join(args.output_dir, "unet_lora"))
    return unet


if __name__ == "__main__":
    main(parse_args())
