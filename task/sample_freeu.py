from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

import torch

from PIL import Image

import os
import shutil

from util.basic_util import pause, get_global_variable, is_none, get_true_value
from util.torch_util import get_generator
from util.image_util import save_pil_as_png
from util.pipeline_util import load_pipeline
from freeu.freeu_util import register_free_upblock2d, register_free_crossattn_upblock2d


@torch.no_grad()
def sample_freeu_implement(
    cfg: DictConfig
):
    # ---------= [Global Variables] =---------
    logger(f"[Global Variables] Loading started. ")

    exp_name = get_global_variable("exp_name")
    device = get_global_variable("device")
    seed = get_global_variable("seed")

    logger(f"[Global Variables] Loading finished. ")

    # ---------= [Task] =---------
    logger(f"[Task] Loading started. ")

    prompt = get_true_value(cfg["task"]["task"]["prompt"])
    height = get_true_value(cfg["task"]["task"]["height"])
    width = get_true_value(cfg["task"]["task"]["width"])
    num_inference_step = get_true_value(cfg["task"]["task"]["num_inference_step"])
    guidance_scale = get_true_value(cfg["task"]["task"]["guidance_scale"])
    negative_prompt = get_true_value(cfg["task"]["task"]["negative_prompt"])
    # num_images_per_prompt = get_true_value(cfg["task"]["task"]["num_images_per_prompt"])
    # eta = get_true_value(cfg["task"]["task"]["eta"])
    # latents = get_true_value(cfg["task"]["task"]["latents"])
    # prompt_embeds = get_true_value(cfg["task"]["task"]["prompt_embeds"])
    # negative_prompt_embeds = get_true_value(cfg["task"]["task"]["negative_prompt_embeds"])
    # output_type = get_true_value(cfg["task"]["task"]["output_type"])
    # cross_attention_kwargs = get_true_value(cfg["task"]["task"]["cross_attention_kwargs"])
    # guidance_rescale = get_true_value(cfg["task"]["task"]["guidance_rescale"])

    logger(f"    prompt: {prompt}")
    logger(f"    (height, width): ({height}, {width})")
    logger(f"    num_inference_step: {num_inference_step}")
    logger(f"    guidance_scale: {guidance_scale}")
    logger(f"    negative_prompt: {negative_prompt}")
    # logger(f"    num_images_per_prompt: {num_images_per_prompt}")
    # logger(f"    eta: {eta}")
    # logger(f"    latents: {latents}")
    # logger(f"    prompt_embeds: {prompt_embeds}")
    # logger(f"    negative_prompt_embeds: {negative_prompt_embeds}")
    # logger(f"    output_type: {output_type}")
    # logger(f"    cross_attention_kwargs: {cross_attention_kwargs}")
    # logger(f"    guidance_rescale: {guidance_rescale}")
    
    logger(f"[Task] Loading finished. ")

    # ---------= [Pipeline & Model] =---------
    logger(f"[Pipeline & Model] Loading started. ")

    # load pipeline
    if cfg["pipeline"]["load_pipeline"]:
        pipeline_type = get_true_value(cfg["pipeline"]["pipeline_type"])
        pipeline_path = get_true_value(cfg["pipeline"]["pipeline_path"])
        torch_dtype = get_true_value(cfg["pipeline"]["torch_dtype"])
        variant = get_true_value(cfg["pipeline"]["variant"])

        pipeline = load_pipeline(
            pipeline_type = pipeline_type, 
            pipeline_path = pipeline_path, 
            torch_dtype = torch_dtype, 
            variant = variant, 
            device = device
        )

        # logger(f"    pipeline: {type(pipeline)}")
        logger(f"    pipeline: {pipeline}")

    logger(
        f"[Pipeline & Model] Loading finished. "
        "\n"
    )

    # ---------= [FreeU] =---------
    logger(f"[FreeU] Loading started. ")

    b1 = get_true_value(cfg["task"]["freeu"]["b1"])
    s1 = get_true_value(cfg["task"]["freeu"]["s1"])
    b2 = get_true_value(cfg["task"]["freeu"]["b2"])
    s2 = get_true_value(cfg["task"]["freeu"]["s2"])

    logger(f"    b1: {b1}, s1: {s1}")
    logger(f"    b2: {b2}, s2: {s2}")

    logger(
        f"[FreeU] Loading finished. "
        "\n"
    )

    # ---------= [Sample] =---------
    logger(f"[Sample] Loading started. ")

    sample_standard = get_true_value(cfg["task"]["sample"]["sample_standard"])

    logger(f"    sample_standard: {sample_standard}")

    save_sample = get_true_value(cfg["task"]["sample"]["save_sample"])
    sample_root_path = get_true_value(cfg["task"]["sample"]["sample_root_path"])
    sample_root_path = os.path.join(sample_root_path, exp_name)

    logger(f"    save_sample: {save_sample}")
    logger(f"    sample_root_path: {sample_root_path}")

    logger(
        f"[Sample] Loading finished. "
        "\n"
    )

    # ---------= [Main] =---------
    def sample(
        b1: float, s1: float, 
        b2: float, s2: float
    ) -> Image.Image:
        register_free_upblock2d(
            pipeline, 
            b1 = b1, s1 = s1, 
            b2 = b2, s2 = s2
        )
        register_free_crossattn_upblock2d(
            pipeline, 
            b1 = b1, s1 = s1, 
            b2 = b2, s2 = s2
        )

        # refresh generator
        generator = get_generator(
            seed = seed, 
            device = device
        )

        img_pil = pipeline(
            prompt = prompt, 
            negative_prompt = negative_prompt, 
            height = height, width = width, 
            guidance_scale = guidance_scale, 
            num_inference_steps = num_inference_step, 
            generator = generator
        ).images[0]

        return img_pil

    if sample_standard:
        img_pil = sample(
            b1 = 1.0, s1 = 1.0, 
            b2 = 1.0, s2 = 1.0
        )

        save_pil_as_png(
            pil = img_pil, 
            png_root_path = sample_root_path, 
            png_name = "standard.png"
        )
    
    img_pil = sample(
        b1 = b1, s1 = s1, 
        b2 = b2, s2 = s2
    )

    save_pil_as_png(
        pil = img_pil, 
        png_root_path = sample_root_path, 
        png_name = "freeu.png"
    )

    # `sample_freeu_implement()` done
    pass

def sample_freeu(
    cfg: DictConfig
):
    sample_freeu_implement(cfg)

    pass