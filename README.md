# lcm_dpo

## About 

Latent Consistency Models trained using Direct Preference Optimizatoin
- https://arxiv.org/abs/2112.10752
- https://latent-consistency-models.github.io/

## Dataset 
on direct preference data 

## How to Initialize LCM with DPO
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
