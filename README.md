# LDM/LCM Evaluation Results

**TODO**


# lcm_dpo
<img width="629" alt="Screenshot 2024-01-13 at 9 45 23 PM" src="https://github.com/catherinelee274/lcm_dpo/assets/14174625/d10d164c-eb91-478f-8783-2d1f2d42a08e">

## About 

Neurips Applied Ai Research Hackathon 2024

Latent Consistency Models trained using Direct Preference Optimizatoin
- https://arxiv.org/abs/2112.10752
- https://latent-consistency-models.github.io/

## Dataset 
on direct preference data 

## How to Initialize LCM - Lora with DPO
```
def load_pipe(use_dpo: bool = False) -> DiffusionPipeline:
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    unet_params = {}
    if use_dpo:
        unet_params = {"unet": UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1", subfolder="unet", torch_dtype=torch.float16
        )}
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        **unet_params
    )
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    pipe.set_adapters(["lcm"], adapter_weights=[1.0])
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_pipe(False)
```

## Consistency-DPO

[Curriculum DPO for Diffusion and Consistency Models](https://arxiv.org/abs/2405.13637) (Croitoru et al., CVPR 2025) adapts DPO to consistency models, which have no noise-prediction error to stand in for the likelihood. It uses the consistency distance instead, for a frozen reference `f_ref` and a trainable `f_φ` (Eq. 6-8):

```
d*(x) = d(f_φ(x_tn+1), f_ref(x̂_tn)) - d(f_ref(x_tn+1), f_ref(x̂_tn))
L     = -E[log σ(-β (d*(x_w) - d*(x_l)))]
```

`x_tn+1` is a noised image, `x̂_tn` is one DDIM step down the PF-ODE using the teacher with CFG, and `d` is the Pseudo-Huber distance (from the authors' code, the paper leaves it open). There is no EMA target, and only a LoRA on the LCM is trained, so `f_ref` is simply the same UNet with the LoRA switched off.

**Status: no gain at the paper's 10,000 updates.** This run used the defaults below, resuming once at 2,500 updates with Adam restarted, and was benchmarked on 500 prompts with 8 steps at guidance 8. The main test, fixed before the run finished, was 10,000 updates against `lcm` on PickScore win rate, and it did not pass (p = 0.098). The other rows show the trend:

| Arm | Median PickScore | PickScore wins vs `lcm` | p | Median CLIP score | CLIP wins vs `lcm` | p |
|---|---|---|---|---|---|---|
| `lcm` | 0.2161 | | | 32.27 | | |
| 1,000 updates | 0.2161 | 54.0% | 0.081 | 32.01 | 43.4% | 0.004 |
| 2,000 updates | 0.2166 | 55.2% | 0.022 | 32.18 | 48.6% | 0.56 |
| 2,500 updates | 0.2165 | 55.2% | 0.022 | 32.01 | 46.0% | 0.081 |
| 4,000 updates | 0.2161 | 53.0% | 0.19 | 32.17 | 47.4% | 0.26 |
| 6,000 updates | 0.2170 | 56.8% | 0.003 | 31.70 | 47.4% | 0.26 |
| 8,000 updates | 0.2167 | 56.0% | 0.008 | 31.93 | 46.8% | 0.17 |
| 10,000 updates | 0.2156 | 53.8% | 0.098 | 32.07 | 44.4% | 0.014 |

PickScore peaks at 6,000 updates and then falls back, and CLIP score never beats `lcm`. Only 6,000 updates holds up after correcting for the seven checkpoints (p = 0.019), but it was picked after seeing the results and has not been confirmed on new prompts or seeds. Pair accuracy on training pairs kept rising to 0.72 while the benchmark fell, which points to overfitting the 20,000 pairs, each seen about 16 times.

```bash
pip install 'diffusers==0.38.0' 'transformers==5.20.2' 'peft==0.19.1' 'accelerate==1.11.0' 'datasets==3.6.0' scipy pytest
pytest tests/

python train_consistency_dpo.py --output_dir runs/consistency-dpo
python eval_consistency_dpo.py --output_dir runs/consistency-dpo/eval \
    --lora consistency_dpo=runs/consistency-dpo/unet_lora \
    --lora step_1000=runs/consistency-dpo/checkpoint-1000
```

`eval_consistency_dpo.py` writes `results.json` with median PickScore and CLIP score, win rates against the plain LCM and sign test p-values. PickScores are raw, so divide by 100 to match the figures above. Every arm starts from the same seed for each prompt. LoRAs are left unfused so the benchmark runs exactly what was trained, which makes their `seconds_per_image` about 25% worse than a fused LoRA, whose speed equals the baseline.

Compare results with the `lcm` arm of the same run, not with the SDXL figures: this is a different base model at a different step count.

Differences from the paper:

| | Paper | Here | Why |
|---|---|---|---|
| Curriculum | Yes, in its best results | No | It needs a reward model to score every image, and is a separate method |
| Preference pairs (LCM) | Reward-ranked LCM samples | Human labels from Pick-a-Pic | The data this repo already benchmarks on, with no reward model. The paper used Pick-a-Pic for SD only |
| Resolution | 768 | 512 | Out of memory at batch 4 on 24 GB, and about 88 h at batch 2 |
| Pairs per update | 16 × 2 accumulation | 4 × 8 accumulation | Same 32 pairs, within 24 GB |
| Learning rate | 3e-4 | 3e-5 | 3e-4 diverged on these pairs: within 300 updates `d*` reached 0.59 and loss 2.2, with pair accuracy near chance. 3e-5 stayed stable over 10,000 updates, with `d*` rising slowly to 0.019 |

On an RTX 3090 the defaults peak at 17.4 GB and take about 10 s per update, so the paper's 10,000 updates take about 27 hours. `--gradient_checkpointing` lowers that to 7.9 GB at about 20% more time. A LoRA is saved every 1,000 updates, and `--resume_from runs/consistency-dpo/checkpoint-N` continues from one, restarting Adam since only the LoRA is saved. Training streams about 10 GB of Pick-a-Pic to encode its 20,000 pairs, once, and needs about 6 GB of weights. Benchmarking needs about another 8 GB.

# Slides 
https://docs.google.com/presentation/d/11yQJeaHxmU58AsHpO23_oSvRH-aJyc3j5Myx4_Lnlu0/edit#slide=id.p

# Notebooks 
- ldm_eval
- lcm_eval
- ldm_eval_accelerated

# todo
eval server
