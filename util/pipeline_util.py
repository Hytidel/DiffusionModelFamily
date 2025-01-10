from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
import torchvision

from PIL import Image

import numpy as np

from util.basic_util import is_none, get_true_value


def load_pipeline(
    pipeline_type: str, 
    pipeline_path: str, 
    torch_dtype: str = None, 
    variant: str = None, 
    device: str = "cpu", 
):
    pipeline = None

    if pipeline_type == "FluxPipeline":
        from diffusers import FluxPipeline

        pipeline = FluxPipeline.from_pretrained(
            pipeline_path, 
            torch_dtype = getattr(torch, torch_dtype), 
            variant = variant
        )
    elif pipeline_type == "StableDiffusionPipeline":
        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline.from_pretrained(
            pipeline_path, 
            torch_dtype = getattr(torch, torch_dtype), 
            variant = variant
        )
    elif pipeline_type == "StableDiffusionXLPipeline":
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pipeline_path, 
            torch_dtype = getattr(torch, torch_dtype), 
            variant = variant
        )

    else:
        raise ValueError(
            f"Unsupported `pipeline_type`, got `{pipeline_type}`. "
        )

    pipeline = pipeline.to(device)

    return pipeline

def load_noise_scheduler(
    scheduler, 
    num_inference_step, 
    pipeline = None, 
    num_train_timesteps = 1000, 
    device = "cpu", 
):
    if scheduler == "DDIM":
        from diffusers import DDIMScheduler

        if pipeline is not None:
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline.scheduler.set_timesteps(
                num_inference_step, 
                device = device
            )

            noise_scheduler = pipeline.scheduler
        else:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps = num_train_timesteps   
            )
            noise_scheduler.set_timesteps(
                num_inference_step, 
                device = device
            )
    else:
        raise NotImplementedError(
            f"Unsupported scheduler: `{scheduler}`. "
        )
    
    return noise_scheduler

def load_unet(
    unet_type: str, 
    unet_path: str, 
    torch_dtype: str = None, 
    device: str = "cpu"
):
    if unet_type == "UNet2DConditionModel":
        from diffusers.models.unet_2d_condition import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            unet_path, 
            torch_dtype = getattr(torch, torch_dtype)
        ).to(device)
    else:
        raise NotImplementedError(
            f"Unsupported U-Net type: `{unet_type}`. "
        )

    return unet

@torch.no_grad()
def img_pil_to_latent(
    img_pil, 
    pipeline
) -> torch.Tensor:
    latent = pipeline.vae.encode(
        # value [0, 255] -> [0, 1] -> [-1, 1]
        torchvision.transforms.functional.to_tensor(img_pil) \
            .unsqueeze(0).to(pipeline.device) * 2 - 1
    )

    latent = 0.18215 * latent.latent_dist.sample()

    return latent

@torch.no_grad()
def img_latent_to_pil(
    img_latent, 
    pipeline
) -> Image.Image:
    if hasattr(pipeline, "decode_latents"):
        img_numpy = pipeline.decode_latents(img_latent)

        return pipeline.numpy_to_pil(img_numpy)
    else:
        # make sure the VAE is in `float32` mode, as it overflows in `float16`
        if (pipeline.vae.dtype == torch.float16) and pipeline.vae.config.force_upcast:
            pipeline.upcast_vae()
            img_latent = img_latent.to(
                next(
                    iter(
                        pipeline.vae.post_quant_conv.parameters()
                    )
                ).dtype
            )

        img_tensor = pipeline.vae.decode(
            img_latent / pipeline.vae.config.scaling_factor, 
            return_dict = False
        )[0]
        
        img_pil = pipeline.image_processor.postprocess(
            img_tensor, 
            output_type = "pil"
        )

        return img_pil

def process_prompt_list(
    prompt: Union[str, List[str]], 
    batch_size: Optional[int] = None, 
    negative_prompt: Union[str, List[str]] = None, 
):
    if isinstance(prompt, list):
        if is_none(batch_size):
            batch_size = len(prompt)
        elif batch_size != len(prompt):
            raise ValueError(
                f"The length of the `prompt` list doesn't match `batch_size`, "
                f"got {len(prompt)} and {batch_size}. "
            )
    elif is_none(batch_size):
        batch_size = 1
        prompt = [prompt]
    else:
        prompt = [prompt] * batch_size

    if negative_prompt is not None:
        if isinstance(negative_prompt, list):
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"The length of the `negative_prompt` list doesn't match `batch_size`, "
                    f"got {len(negative_prompt)} and {batch_size}. "
                    )
        else:
            negative_prompt = [negative_prompt] * batch_size

    return prompt, batch_size, negative_prompt

