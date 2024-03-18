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

# Slides 
https://docs.google.com/presentation/d/11yQJeaHxmU58AsHpO23_oSvRH-aJyc3j5Myx4_Lnlu0/edit#slide=id.p

# Evaluation 

