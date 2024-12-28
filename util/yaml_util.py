from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

import numpy as np

import os
import shutil


def load_yaml(
    yaml_path: str, 
    ret_type: Optional[str] = "DictConfig"  # ["DictConfig", "dict"]
) -> DictConfig:
    if not os.path.isfile(yaml_path):
        raise ValueError(
            f"File `{yaml_path}` not exists. "
        )

    cfg = OmegaConf.load(yaml_path)

    if ret_type == "DictConfig":
        pass
    elif ret_type == "dict":
        cfg = OmegaConf.to_container(
            cfg, 
            resolve = True
        )
    
    return cfg

def save_yaml(
    cfg: Union[Dict, DictConfig], 
    yaml_root_path, 
    yaml_name
):
    if not os.path.exists(yaml_root_path):
        os.makedirs(yaml_root_path)
    
    yaml_path = os.path.join(yaml_root_path, yaml_name)

    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    
    OmegaConf.save(
        cfg, 
        yaml_path
    )

def convert_numpy_type_to_native_type(
    var
):
    res = var

    if isinstance(var, np.integer):
        res = int(var)
    elif isinstance(var, np.float64):
        res = float(res)
    elif isinstance(var, np.ndarray):
        res = var.tolist()
    elif isinstance(var, list):
        res = [convert_numpy_type_to_native_type(val) for val in var]
    elif isinstance(var, tuple):
        res = tuple([convert_numpy_type_to_native_type(val) for val in var])
    elif isinstance(var, dict):
        res = {
            key: convert_numpy_type_to_native_type(val) \
                for key, val in var.items()
        }

    return res
    