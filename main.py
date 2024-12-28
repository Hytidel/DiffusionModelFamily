from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

from util.basic_util import pause, set_global_variable_dict, get_global_variable, set_global_variable


def sample_flux_1_dev(
    cfg: DictConfig
):
    from task.sample_flux_1_dev import sample_flux_1_dev
    sample_flux_1_dev(cfg)

    # `sample_flux_1_dev()` done
    pass

def sample_sd_2_1(
    cfg: DictConfig
):
    from task.sample_sd_2_1 import sample_sd_2_1
    sample_sd_2_1(cfg)

    # `sample_sd_2_1()` done
    pass

def sample_sd_turbo(
    cfg: DictConfig
):
    from task.sample_sd_turbo import sample_sd_turbo
    sample_sd_turbo(cfg)

    # `sample_sd_turbo()` done
    pass

def sample_sdxl(
    cfg: DictConfig
):
    from task.sample_sdxl import sample_sdxl
    sample_sdxl(cfg)

    # `sample_sdxl()` done
    pass

def sample_sdxl_turbo(
    cfg: DictConfig
):
    from task.sample_sdxl_turbo import sample_sdxl_turbo
    sample_sdxl_turbo(cfg)

    # `sample_sdxl_turbo()` done
    pass

def sample_freeu(
    cfg: DictConfig
):
    from task.sample_freeu import sample_freeu
    sample_freeu(cfg)

    # `sample_freeu()` done
    pass

def run_task(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name.startswith("sample_flux.1-dev"):
        sample_flux_1_dev(cfg)
    elif task_name.startswith("sample_sd_2.1"):
        sample_sd_2_1(cfg)
    elif task_name.startswith("sample_sd-turbo"):
        sample_sd_turbo(cfg)
    elif task_name.startswith("sample_sdxl"):
        sample_sdxl(cfg)
    elif task_name.startswith("sample_sdxl_turbo"):
        sample_sdxl_turbo(cfg)
    elif task_name.startswith("sample_freeu"):
        sample_freeu(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

@hydra.main(version_base = None, config_path = "config", config_name = "cfg")
def main(
    cfg: DictConfig
):
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(
        cfg, 
        resolve = True
    )

    set_global_variable_dict(cfg)

    exp_name = get_global_variable("exp_name")
    logger(f"Start experiment `{exp_name}`. ")

    run_task(cfg)

    logger(f"Experiment `{exp_name}` finished. ")

    # `main()` done
    pass

if __name__ == "__main__":
    main()
    