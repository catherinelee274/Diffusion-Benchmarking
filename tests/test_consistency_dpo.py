import contextlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState
from diffusers import AutoencoderKL, LCMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, CLIPTextModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_consistency_dpo as evaluate
import train_consistency_dpo as train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY_PIPE = "hf-internal-testing/tiny-stable-diffusion-pipe"
TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


@pytest.fixture(autouse=True)
def fresh_accelerator():
    """Accelerate keeps its mixed precision in a process-wide singleton."""
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    """A tiny LCM directory, and the tiny SD pipeline it pretends to be distilled from."""
    teacher = snapshot_download(TINY_PIPE, ignore_patterns=["*.ckpt", "*.bin.index.json"])
    lcm = tmp_path_factory.mktemp("lcm")

    torch.manual_seed(0)
    unet_config = dict(UNet2DConditionModel.load_config(teacher, subfolder="unet"))
    UNet2DConditionModel.from_config({**unet_config, "time_cond_proj_dim": 16}).save_pretrained(lcm / "unet")
    LCMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear").save_pretrained(lcm / "scheduler")
    AutoencoderKL.from_pretrained(teacher, subfolder="vae").save_pretrained(lcm / "vae")
    CLIPTextModel.from_pretrained(teacher, subfolder="text_encoder").save_pretrained(lcm / "text_encoder")
    AutoTokenizer.from_pretrained(teacher, subfolder="tokenizer").save_pretrained(lcm / "tokenizer")
    return SimpleNamespace(lcm=str(lcm), teacher=teacher)


def write_pairs(path, n=16, gap=0.5):
    latents = torch.randn(n, 4, 16, 16)
    pairs = {
        "latents_w": (latents + gap).to(torch.float16),
        "latents_l": (latents - gap).to(torch.float16),
        "captions": [f"a photo of thing {i}" for i in range(n)],
    }
    torch.save(pairs, path)
    return str(path)


def train_args(tiny_models, tmp_path, **overrides):
    args = train.parse_args(
        [
            "--output_dir", str(tmp_path / "run"),
            "--pretrained_model", tiny_models.lcm,
            "--teacher_model", tiny_models.teacher,
            "--teacher_variant", "",
            "--pairs_path", write_pairs(tmp_path / "pairs.pt"),
            "--lora_rank", "4",
            "--train_batch_size", "2",
            "--gradient_accumulation_steps", "2",
            "--max_train_steps", "3",
            "--log_steps", "1",
            "--save_steps", "2",
            "--mixed_precision", "fp16" if DEVICE == "cuda" else "no",
        ]
    )  # fmt: skip
    return SimpleNamespace(**{**vars(args), **overrides})


def read_metrics(args):
    return [json.loads(line) for line in Path(args.output_dir, "metrics.jsonl").read_text().splitlines()]


def scheduler():
    return LCMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")


def test_loss_is_log2_while_student_matches_reference():
    ref, target = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    loss, d_w, d_l = train.consistency_dpo_loss(ref.clone(), ref, target, beta=200, huber_c=0.001)
    assert loss.item() == pytest.approx(math.log(2))
    assert d_w.abs().max() == 0 and d_l.abs().max() == 0


def test_loss_rewards_consistency_on_winners_and_penalises_it_on_losers():
    ref, target = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    closer = ref + 0.1 * (target - ref)
    further = ref - 0.1 * (target - ref)

    def loss(pred_w, pred_l):
        return train.consistency_dpo_loss(torch.cat([pred_w, pred_l]), torch.cat([ref, ref]), torch.cat([target, target]), 200, 0.001)[0]

    baseline = math.log(2)
    assert loss(closer, further) < baseline
    assert loss(closer, ref) < baseline
    assert loss(ref, further) < baseline
    assert loss(further, closer) > baseline


def test_loss_stays_finite_when_the_sigmoid_saturates():
    ref = torch.zeros(2, 4, 8, 8)
    target = torch.zeros(2, 4, 8, 8)
    pred = torch.cat([torch.full((1, 4, 8, 8), 5.0), torch.zeros(1, 4, 8, 8)]).requires_grad_()
    loss = train.consistency_dpo_loss(pred, ref, target, beta=200, huber_c=0.001)[0]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_ddim_step_stays_on_the_trajectory_given_the_true_noise():
    sched = scheduler()
    x0, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    start, end = torch.tensor([19, 519, 999]), torch.tensor([0, 499, 979])
    sample = sched.add_noise(x0, noise, start)
    stepped = train.ddim_step(sample, noise, start, end, sched)
    assert torch.allclose(stepped, sched.add_noise(x0, noise, end), atol=1e-4)


def test_consistency_output_matches_the_inference_scheduler():
    sched = scheduler()
    sched.set_timesteps(4, original_inference_steps=50)
    timestep = sched.timesteps[1]
    sample, noise_pred = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    model = lambda *args, **kwargs: SimpleNamespace(sample=noise_pred)

    ours = train.consistency_output(model, sample, timestep[None], None, None, sched)
    sched._step_index = 1
    theirs = sched.step(noise_pred, timestep, sample).denoised
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_consistency_output_is_the_identity_at_t0():
    sched = scheduler()
    sample = torch.randn(2, 4, 8, 8)
    model = lambda *args, **kwargs: SimpleNamespace(sample=torch.randn(2, 4, 8, 8))
    output = train.consistency_output(model, sample, torch.zeros(2, dtype=torch.long), None, None, sched)
    assert torch.allclose(output, sample)


class RecordingUNet:
    """Records every forward pass, noting whether the LoRA was on."""

    def __init__(self):
        self.config = SimpleNamespace(time_cond_proj_dim=16)
        self.dtype = torch.float32
        self.lora_enabled = True
        self.calls = []

    @contextlib.contextmanager
    def disable_adapter(self):
        self.lora_enabled = False
        yield
        self.lora_enabled = True

    def __call__(self, sample, timesteps, timestep_cond=None, encoder_hidden_states=None):
        self.calls.append(
            SimpleNamespace(
                sample=sample, timesteps=timesteps, w_embedding=timestep_cond,
                prompt_embeds=encoder_hidden_states, lora_enabled=self.lora_enabled,
            )
        )  # fmt: skip
        # The teacher answers 1 for the prompt and 0 for the empty prompt.
        return SimpleNamespace(sample=encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None] * torch.ones_like(sample))


def test_predict_pair_makes_the_forward_passes_of_the_method():
    sched = scheduler()
    student, teacher = RecordingUNet(), RecordingUNet()
    latents = torch.randn(3, 4, 8, 8)
    prompt_embeds, uncond_embeds = torch.ones(3, 77, 8), torch.zeros(3, 77, 8)
    args = SimpleNamespace(num_ddim_timesteps=50, w_min=7.0, w_max=7.0)

    train.predict_pair(student, teacher, latents, latents.clone(), prompt_embeds, uncond_embeds, sched, args)

    online, reference, target = student.calls
    assert online.lora_enabled and not reference.lora_enabled and not target.lora_enabled

    # f_phi and f_ref see the same point, at a timestep of the 50-step DDIM grid.
    assert torch.equal(online.sample, reference.sample)
    assert torch.equal(online.timesteps, reference.timesteps)
    assert all((t + 1) % 20 == 0 for t in online.timesteps.tolist())

    # The target sits one DDIM step earlier, reached with the teacher's CFG output.
    assert torch.equal(target.timesteps, (online.timesteps - 20).clamp(min=0))
    noise_pred = (1 + 7.0) * torch.ones_like(online.sample)
    expected = train.ddim_step(online.sample, noise_pred, online.timesteps, target.timesteps, sched)
    assert torch.allclose(target.sample, expected)

    # Winner and loser share noise, timestep and guidance, so identical latents match exactly.
    assert torch.equal(*online.sample.chunk(2))
    assert torch.equal(*online.timesteps.chunk(2))
    assert torch.equal(*online.w_embedding.chunk(2))
    assert not torch.equal(online.sample[0], online.sample[1])

    conditional, unconditional = teacher.calls
    assert torch.equal(conditional.timesteps, online.timesteps)
    assert conditional.prompt_embeds.mean() == 1 and unconditional.prompt_embeds.mean() == 0
    assert all(call.lora_enabled for call in teacher.calls)


def test_only_clear_preferences_on_training_prompts_are_kept():
    entry = {"are_different": True, "label_0": 1.0, "label_1": 0.0, "caption": "a dog"}
    assert train.is_usable(entry, set())
    assert not train.is_usable({**entry, "label_0": 0.5, "label_1": 0.5}, set())
    assert not train.is_usable({**entry, "are_different": False}, set())
    assert not train.is_usable(entry, {"a dog"})


def test_build_pairs_encodes_only_usable_pairs(tiny_models, tmp_path, monkeypatch):
    class FakeStream(list):
        def shuffle(self, seed, buffer_size):
            return self

    def jpg():
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(buffer, "JPEG")
        return buffer.getvalue()

    def entry(caption, label_0=1.0, label_1=0.0, are_different=True):
        return dict(
            caption=caption, label_0=label_0, label_1=label_1, are_different=are_different, jpg_0=jpg(), jpg_1=jpg()
        )

    train_rows = [entry("a tie", 0.5, 0.5), entry("held out"), entry("same image", are_different=False)]
    train_rows += [entry(f"keep {i}", 1.0 - i % 2, float(i % 2)) for i in range(7)]
    monkeypatch.setattr(
        train,
        "load_dataset",
        lambda name, split, streaming: FakeStream(train_rows if split == "train" else [entry("held out")]),
    )

    args = train_args(tiny_models, tmp_path, resolution=64, num_pairs=5, vae_batch_size=2)
    path = tmp_path / "built.pt"
    train.build_pairs(args, path, DEVICE)

    pairs = torch.load(path)
    assert pairs["captions"] == [f"keep {i}" for i in range(5)]
    assert pairs["latents_w"].shape == pairs["latents_l"].shape == (5, 4, 32, 32)
    assert pairs["latents_w"].dtype == torch.float16


@pytest.mark.parametrize("label_0, label_1, winner_color", [(1.0, 0.0, "white"), (0.0, 1.0, "black")])
def test_winner_is_the_image_people_picked(label_0, label_1, winner_color):
    def jpg(color):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, "JPEG")
        return buffer.getvalue()

    entry = {"jpg_0": jpg("white"), "jpg_1": jpg("black"), "label_0": label_0, "label_1": label_1}
    winner, loser = train.winner_and_loser(entry)
    assert winner.getpixel((0, 0)) > loser.getpixel((0, 0)) if winner_color == "white" else winner.getpixel((0, 0)) < loser.getpixel((0, 0))


def test_student_starts_exactly_at_the_reference(tiny_models, tmp_path):
    args = train_args(tiny_models, tmp_path, max_train_steps=1, learning_rate=0.0)
    train.main(args)
    assert read_metrics(args)[0]["loss"] == pytest.approx(math.log(2), abs=1e-4)


def test_training_moves_only_the_lora_and_ranks_pairs(tiny_models, tmp_path):
    args = train_args(tiny_models, tmp_path, max_train_steps=40, learning_rate=1e-3, log_steps=10)
    base_before = UNet2DConditionModel.from_pretrained(tiny_models.lcm, subfolder="unet").state_dict()
    unet = train.main(args)

    lora = {name: param for name, param in unet.named_parameters() if "lora_" in name}
    assert any(param.abs().sum() > 0 for name, param in lora.items() if "lora_B" in name)
    assert all(param.requires_grad == ("lora_" in name) for name, param in unet.named_parameters())
    for name, param in unet.named_parameters():
        if "lora_" not in name:
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            assert torch.equal(param.detach().cpu().float(), base_before[key].float().to(param.dtype).float()), name

    metrics = read_metrics(args)
    assert metrics[-1]["loss"] < math.log(2)
    assert metrics[-1]["pair_acc"] > 0.5
    assert metrics[-1]["d_w"] < metrics[-1]["d_l"]


def unet_inputs():
    return dict(
        sample=torch.randn(1, 4, 16, 16, device=DEVICE),
        timestep=torch.tensor([500], device=DEVICE),
        timestep_cond=torch.randn(1, 16, device=DEVICE),
        encoder_hidden_states=torch.randn(1, 77, 32, device=DEVICE),
    )


def test_disabling_the_lora_recovers_the_pretrained_model(tiny_models, tmp_path):
    unet = train.main(train_args(tiny_models, tmp_path, learning_rate=1e-3, mixed_precision="no"))
    base = UNet2DConditionModel.from_pretrained(tiny_models.lcm, subfolder="unet").to(DEVICE)
    inputs = unet_inputs()
    with torch.no_grad():
        expected = base(**inputs).sample
        with unet.disable_adapter():
            reference = unet(**inputs).sample
        student = unet(**inputs).sample
    assert torch.equal(reference, expected)
    assert not torch.allclose(student, expected, atol=1e-4)


def test_every_trained_tensor_reaches_the_benchmark_pipeline(tiny_models, tmp_path):
    args = train_args(tiny_models, tmp_path, mixed_precision="no")
    unet = train.main(args)
    pipe_args = SimpleNamespace(pretrained_model=_as_pipeline(tiny_models, tmp_path))
    pipe = evaluate.load_pipe(pipe_args, f"{args.output_dir}/unet_lora", "cpu")

    def lora_tensors(model, adapter):
        return {name.replace(f".{adapter}.", "."): param for name, param in model.named_parameters() if "lora_" in name}

    trained = {key.removeprefix("base_model.model."): value for key, value in lora_tensors(unet, "default").items()}
    loaded = lora_tensors(pipe.unet, "default_0")
    assert any("attn1.to_q" in key for key in trained)
    assert loaded.keys() == trained.keys()


def test_saved_lora_reproduces_the_student_in_the_benchmark_pipeline(tiny_models, tmp_path):
    args = train_args(tiny_models, tmp_path, learning_rate=1e-3, mixed_precision="no")
    unet = train.main(args)
    pipe_args = SimpleNamespace(pretrained_model=_as_pipeline(tiny_models, tmp_path))
    pipe = evaluate.load_pipe(pipe_args, f"{args.output_dir}/unet_lora", "cpu").to(DEVICE)

    inputs = unet_inputs()
    with torch.no_grad():
        assert torch.equal(pipe.unet(**inputs).sample, unet(**inputs).sample)


def train_then_checkpoint(tiny_models, tmp_path):
    args = train_args(tiny_models, tmp_path, max_train_steps=2, learning_rate=1e-3)
    train.main(args)
    AcceleratorState._reset_state(reset_partial_state=True)
    return args, f"{args.output_dir}/checkpoint-2"


def test_resume_restores_the_checkpoint_exactly(tiny_models, tmp_path):
    _, checkpoint = train_then_checkpoint(tiny_models, tmp_path)
    args = train_args(
        tiny_models, tmp_path, output_dir=str(tmp_path / "resumed"), resume_from=checkpoint, max_train_steps=2
    )
    train.main(args)

    # With no updates left, the LoRA that comes out is the one that went in.
    saved = load_file(f"{checkpoint}/pytorch_lora_weights.safetensors")
    restored = load_file(f"{args.output_dir}/unet_lora/pytorch_lora_weights.safetensors")
    assert saved.keys() == restored.keys()
    assert any(value.abs().sum() > 0 for key, value in saved.items() if "lora_B" in key)
    assert all(torch.equal(saved[key], restored[key]) for key in saved)


def test_resume_continues_the_count_with_new_randomness(tiny_models, tmp_path, monkeypatch):
    first, checkpoint = train_then_checkpoint(tiny_models, tmp_path)
    seeds = []
    monkeypatch.setattr(train, "set_seed", seeds.append)
    args = train_args(tiny_models, tmp_path, output_dir=first.output_dir, resume_from=checkpoint, max_train_steps=4)
    train.main(args)

    assert [row["step"] for row in read_metrics(args)] == [1, 2, 3, 4]
    assert Path(args.output_dir, "checkpoint-4").exists()
    assert seeds == [args.seed + 2]


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_resume_refuses_a_checkpoint_that_only_partly_matches(tiny_models, tmp_path, change):
    first, checkpoint = train_then_checkpoint(tiny_models, tmp_path)
    path = f"{checkpoint}/pytorch_lora_weights.safetensors"
    weights = load_file(path)
    key = next(iter(weights))
    if change == "missing":
        del weights[key]
    else:
        weights["unet.not_a_layer.lora_A.weight"] = weights[key].clone()
    save_file(weights, path)

    args = train_args(tiny_models, tmp_path, output_dir=first.output_dir, resume_from=checkpoint, max_train_steps=4)
    with pytest.raises(ValueError, match=f"1 {change}"):
        train.main(args)


def _as_pipeline(tiny_models, tmp_path):
    """The tiny LCM as a loadable LatentConsistencyModelPipeline."""
    from diffusers import LatentConsistencyModelPipeline

    path = tmp_path / "pipeline"
    LatentConsistencyModelPipeline(
        vae=AutoencoderKL.from_pretrained(tiny_models.lcm, subfolder="vae"),
        text_encoder=CLIPTextModel.from_pretrained(tiny_models.lcm, subfolder="text_encoder"),
        tokenizer=AutoTokenizer.from_pretrained(tiny_models.lcm, subfolder="tokenizer"),
        unet=UNet2DConditionModel.from_pretrained(tiny_models.lcm, subfolder="unet"),
        scheduler=LCMScheduler.from_pretrained(tiny_models.lcm, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).save_pretrained(path)
    return str(path)


def test_benchmark_runs_end_to_end_and_pairs_seeds(tiny_models, tmp_path, monkeypatch):
    args_train = train_args(tiny_models, tmp_path, learning_rate=1e-3)
    train.main(args_train)
    args = evaluate.parse_args(
        [
            "--output_dir", str(tmp_path / "eval"),
            "--lora", f"cdpo={args_train.output_dir}/unet_lora",
            "--pretrained_model", _as_pipeline(tiny_models, tmp_path),
            "--num_inference_steps", "2",
            "--resolution", "32",
            "--processor", TINY_CLIP,
            "--pickscore_model", TINY_CLIP,
            "--clip_model", TINY_CLIP,
        ]
    )  # fmt: skip
    monkeypatch.setattr(evaluate, "load_prompts", lambda args: ["a cat", "a dog", "a bird"])

    results = evaluate.main(args, DEVICE)
    assert set(results) == {"lcm", "cdpo"}
    assert 0 <= results["cdpo"]["pickscore_win_rate"] <= 1
    assert 0 < results["cdpo"]["clip_score_sign_test_p"] <= 1

    # Seeded per prompt, so a rerun of an arm reproduces its images.
    first = np.array(Image.open(tmp_path / "eval/lcm/lcm_out_000.png"))
    (tmp_path / "eval/lcm/lcm_out_000.png").unlink()
    evaluate.generate(args, "lcm", None, ["a cat", "a dog", "a bird"], DEVICE)
    assert np.array_equal(first, np.array(Image.open(tmp_path / "eval/lcm/lcm_out_000.png")))


def test_summary_counts_wins_against_the_baseline():
    scores = {"lcm": {"pickscore": [1.0, 1.0, 1.0, 1.0]}, "cdpo": {"pickscore": [2.0, 2.0, 2.0, 0.0]}}
    results = evaluate.summarize(scores)
    assert results["lcm"] == {"median_pickscore": 1.0}
    assert results["cdpo"]["median_pickscore"] == 2.0
    assert results["cdpo"]["pickscore_win_rate"] == 0.75
    assert results["cdpo"]["pickscore_sign_test_p"] == pytest.approx(0.625)
