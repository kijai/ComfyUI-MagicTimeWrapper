import os
from contextlib import nullcontext
import torch

try:
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        AutoencoderKL, 
        LCMScheduler,
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler
)
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_text_encoder_from_ldm_clip_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config
    )
except:
    print("Diffusers version too old. Please update to 0.26.0 minimum.")

from omegaconf import OmegaConf

from transformers import CLIPTokenizer
from .animatediff.models.unet import UNet3DConditionModel
from .utils.pipeline_magictime import MagicTimePipeline
from .utils.util import  load_diffusers_lora_unet

import comfy.model_management as mm
import comfy.utils
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
    
class magictime_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "motion_model":("MOTION_MODEL_ADE",),
            },
        }

    RETURN_TYPES = ("MAGICTIME",)
    RETURN_NAMES = ("magictime_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "MagicTimeWrapper"

    def loadmodel(self, model, clip, vae, motion_model):
        mm.soft_empty_cache()
        custom_config = {
            'model': model,
            'vae': vae,
            'clip': clip,
            'motion_model': motion_model
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(7)
            self.current_config = custom_config
            # config paths
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            ad_unet_config = OmegaConf.load(os.path.join(script_directory, f"configs/ad_unet_config.yaml"))

            # load models
            
            checkpoint_path = os.path.join(folder_paths.models_dir,'magictime')
            magic_adapter_s_path = os.path.join(checkpoint_path, 'Magic_Weights', 'magic_adapter_s', 'magic_adapter_s.ckpt')
            magic_adapter_t_path = os.path.join(checkpoint_path, 'Magic_Weights', 'magic_adapter_t')
            magic_text_encoder_path = os.path.join(checkpoint_path, 'Magic_Weights', 'magic_text_encoder')

            if not os.path.exists(checkpoint_path):
                print(f"Downloading magictime from https://huggingface.co/BestWishYsh/MagicTime to {checkpoint_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="BestWishYsh/MagicTime", local_dir=checkpoint_path, local_dir_use_symlinks=False)
            
            pbar.update(1)
            
            # get state dict from comfy models
            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)

            pbar.update(1)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)
            pbar.update(1)

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            pbar.update(1)

            # motion module
            motion_module_state_dict = motion_model.model.state_dict()
            if motion_model.model.mm_info.mm_format == "AnimateLCM":
                motion_module_state_dict = {k: v for k, v in motion_module_state_dict.items() if "pos_encoder" not in k}
            converted_unet.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
            converted_unet.pop("animatediff_config", "")
            pbar.update(1)

            unet = UNet3DConditionModel(**ad_unet_config)
            unet.load_state_dict(converted_unet, strict=False)

            pbar.update(1)
            # 3. text_encoder
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            
            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            # 5. scheduler
            scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "linear",
                'steps_offset': 1
            }
            scheduler=DPMSolverMultistepScheduler(**scheduler_config)

            #6. magictime
            from swift import Swift
            magic_adapter_s_state_dict = torch.load(magic_adapter_s_path, map_location="cpu")
            unet = load_diffusers_lora_unet(unet, magic_adapter_s_state_dict, alpha=1.0)
            unet = Swift.from_pretrained(unet, magic_adapter_t_path)
            text_encoder = Swift.from_pretrained(text_encoder, magic_text_encoder_path)
            del sd

            pbar.update(1)
            
            self.pipe = MagicTimePipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer, 
                unet=unet,
                scheduler=scheduler
            )
            
            magictime_model = {
                'pipe': self.pipe,
            }
   
        return (magictime_model,)
    
class magictime_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "magictime_model": ("MAGICTIME",),
            "prompt": ("STRING", {"multiline": True, "default": "positive",}),
            "n_prompt": ("STRING", {"multiline": True, "default": "negative",}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'DDIMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler'
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
            },    
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "MagicTimeWrapper"

    def process(self, magictime_model, prompt, n_prompt, frames, width, height, steps, guidance_scale, seed, scheduler):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        pipe=magictime_model['pipe']
        pipe.to(device, dtype=dtype)

        scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "linear",
                'steps_offset': 1,
            }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        pipe.scheduler = noise_scheduler

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        
            sample = pipe(
            prompt,
            negative_prompt     = n_prompt,
            num_inference_steps = steps,
            guidance_scale      = guidance_scale,
            width               = width,
            height              = height,
            video_length        = frames,
            generator           = generator,
            ).videos
            pipe.to(offload_device)
            image_out = sample.squeeze(0).permute(1, 2, 3, 0).cpu().float()
            return (image_out,)


NODE_CLASS_MAPPINGS = {
    "magictime_model_loader": magictime_model_loader,
    "magictime_sampler": magictime_sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "magic_time_model_loader": "MagicTime Model Loader",
    "magictime_sampler": "MagicTime Sampler",
}
