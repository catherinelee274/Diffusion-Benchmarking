"""Benchmark Consistency-DPO against the plain LCM, following LCM_DPO.ipynb.

Each arm generates one image per Pick-a-Pic validation prompt from the same seed, so the win
rates compare the same starting noise. Images are kept, so a rerun only scores what is missing.
"""
import argparse
import json
import os
from time import time

import numpy as np
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--lora", action="append", default=[], metavar="NAME=PATH", help="an arm to compare with the baseline, repeatable"
    )
    parser.add_argument("--pretrained_model", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--dataset", default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--split", default="validation_unique")
    parser.add_argument("--num_prompts", type=int, default=500)
    # 8 steps at guidance 8 is the paper's LCM setup.
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--processor", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--pickscore_model", default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--clip_model", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_prompts(args):
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    return [entry["caption"] for entry in ds.take(args.num_prompts)]


def load_pipe(args, lora_path=None, device="cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model, torch_dtype=dtype, safety_checker=None)
    if lora_path is not None:
        # Not fused: left separate, the outputs match the trained student exactly.
        pipe.load_lora_weights(lora_path)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def generate(args, name, lora_path, prompts, device="cuda"):
    """Fills <output_dir>/<name>, returning the seconds per image of what it generated."""
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    missing = [i for i in range(len(prompts)) if not os.path.exists(f"{out_dir}/{name}_out_{i:03d}.png")]
    if not missing:
        return None

    pipe = load_pipe(args, lora_path, device)
    cur = time()
    for i in tqdm(missing, desc=name):
        image = pipe(
            prompts[i],
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.resolution,
            width=args.resolution,
            generator=torch.Generator(device).manual_seed(args.seed + i),
        ).images[0]
        image.save(f"{out_dir}/{name}_out_{i:03d}.png")
    seconds = (time() - cur) / len(missing)
    del pipe
    torch.cuda.empty_cache()
    return seconds


def calc_probs(model, processor, prompts, images, device="cuda"):
    image_inputs = processor(images=images, return_tensors="pt").to(device)
    text_inputs = processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    with torch.inference_mode():
        # transformers 5 returns an output object here, older versions the tensor itself.
        image_embs = model.get_image_features(**image_inputs)
        image_embs = getattr(image_embs, "pooler_output", image_embs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = getattr(text_embs, "pooler_output", text_embs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = model.logit_scale.exp() * (text_embs * image_embs).sum(-1)

    return scores.cpu().tolist()


def score_arm(args, name, prompts, model, processor, device="cuda"):
    scores = []
    for start in range(0, len(prompts), 32):
        batch = range(start, min(start + 32, len(prompts)))
        images = [Image.open(f"{args.output_dir}/{name}/{name}_out_{i:03d}.png").convert("RGB") for i in batch]
        scores.extend(calc_probs(model, processor, prompts[start : start + 32], images, device))
    return scores


def summarize(scores, baseline="lcm"):
    results = {}
    for name, arm in scores.items():
        row = {f"median_{metric}": float(np.median(values)) for metric, values in arm.items()}
        if name != baseline:
            for metric, values in arm.items():
                wins = int(np.sum(np.array(values) > np.array(scores[baseline][metric])))
                row[f"{metric}_win_rate"] = wins / len(values)
                row[f"{metric}_sign_test_p"] = binomtest(wins, len(values)).pvalue
        results[name] = row
    return results


def main(args, device="cuda"):
    prompts = load_prompts(args)
    arms = {"lcm": None, **dict(spec.split("=", 1) for spec in args.lora)}
    seconds = {name: generate(args, name, lora_path, prompts, device) for name, lora_path in arms.items()}

    scores = {name: {} for name in arms}
    processor = AutoProcessor.from_pretrained(args.processor)
    for metric, model_name in (("pickscore", args.pickscore_model), ("clip_score", args.clip_model)):
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        for name in arms:
            scores[name][metric] = score_arm(args, name, prompts, model, processor, device)
        del model
        torch.cuda.empty_cache()

    results = summarize(scores)
    for name, row in results.items():
        if seconds[name] is not None:
            row["seconds_per_image"] = seconds[name]
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"results": results, "scores": scores}, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(parse_args())
