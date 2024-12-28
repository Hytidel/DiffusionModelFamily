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


@torch.no_grad()
def sample_sdxl_implement(
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

    logger(f"    prompt: {prompt}")
    logger(f"    (height, width): ({height}, {width})")
    logger(f"    num_inference_step: {num_inference_step}")
    logger(f"    guidance_scale: {guidance_scale}")
    
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

        # save VRAM by offloading the model to CPU
        pipeline.enable_model_cpu_offload()

        # logger(f"    pipeline: {type(pipeline)}")
        logger(f"    pipeline: {pipeline}")

    logger(
        f"[Pipeline & Model] Loading finished. "
        "\n"
    )

    # ---------= [Sample] =---------
    logger(f"[Sample] Loading started. ")

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
    # get generator
    generator = get_generator(
        seed = seed, 
        device = device
    )

    img_pil = pipeline(
        prompt = prompt, 
        height = height, width = width, 
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_step, 
        generator = generator
    ).images[0]

    save_pil_as_png(
        pil = img_pil, 
        png_root_path = sample_root_path, 
        png_name = "sample.png"
    )

    # `sample_sdxl_implement()` done
    pass

def sample_sdxl(
    cfg: DictConfig
):
    sample_sdxl_implement(cfg)

    pass
