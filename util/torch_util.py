from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

import numpy as np

import os
import shutil


def get_device():
    device = "cuda" if torch.cuda.is_available() \
        else "cpu"

    return device

def get_optim(
    optim_name, 
    model, 
    lr
):
    if optim_name == "Adam":
        optim = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()), 
            lr = lr
        )
    else:
        raise NotImplementedError(
            f"Unsupported optimizer:` {optim_name}`. "
        )

    return optim

def get_lr_scheduler(
    lr_scheduler_name, 
    optim, 
    mode = "min", 
    factor = None, 
    patience = None, 
    verbose = True,
):
    """
    Args:
        verbose (`bool`, *optional*, defaults to True):
            Set `verbose = True` for `lr_scheduler` to print prompt messages 
            when the learning rate changes. 
    """

    if lr_scheduler_name is None:
        lr_scheduler = None
    elif lr_scheduler_name == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode = mode, 
            factor = factor, 
            patience = patience, 
            verbose = verbose
        )
    else:
        raise NotImplementedError(
            f"Unsupported learning rate scheduler:` {lr_scheduler_name}`. "
        )
    
    return lr_scheduler

def get_criterion(
    criterion_name
):
    if criterion_name == "L1":
        criterion = F.l1_loss
    elif criterion_name in ["L2", "MSE"]:
        criterion = F.mse_loss
    elif criterion_name == "Huber":
        criterion = F.smooth_l1_loss
    else:
        raise NotImplementedError(
            f"Unsupported criterion:` {criterion_name}`. "
        )

    return criterion

def save_model_state_dict(
    state_dict: Dict, 
    ckpt_root_path: str, 
    ckpt_name: str
):
    if not os.path.exists(ckpt_root_path):
        os.makedirs(ckpt_root_path)
    
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)

    torch.save(
        state_dict, 
        ckpt_path
    )

def save_model_ckpt(
    model, 
    ckpt_root_path: str, 
    ckpt_name: str
):
    state_dict = model.state_dict()

    save_model_state_dict(
        state_dict, 
        ckpt_root_path, 
        ckpt_name
    )

def load_model_state_dict(
    state_dict_path: str, 
    device: str
) -> Dict:
    if (state_dict_path is None) or (state_dict_path == "None") \
        or (not os.path.exists(state_dict_path)):
        logger(
            f"State dict `{state_dict_path}` not exists, continue with initial model parameters. ", 
            log_type = "warning"
        )

        return None
    
    state_dict = torch.load(
        state_dict_path, 
        map_location = device
    )

    state_dict_name = os.path.split(state_dict_path)[-1]

    logger(
        f"Loaded model state_dict `{state_dict_name}`.", 
        log_type = "info"
    )

    return state_dict

def load_model_ckpt(
    model, 
    ckpt_path: str, 
    device: str
):
    if (ckpt_path is None) or (ckpt_path == "None") \
        or (not os.path.exists(ckpt_path)):
            logger(
                f"Model checkpoint `{ckpt_path}` not exists, continue with initial model parameters. ", 
                log_type = "info"
            )

            return

    state_dict = torch.load(
        ckpt_path, 
        map_location = device
    )

    model.load_state_dict(state_dict)

    ckpt_name = os.path.split(ckpt_path)[-1]
    logger(
        f"Loaded model checkpoint `{ckpt_name}`.", 
        log_type = "info"
    )

def get_model_num_params(
    model
):
    model_num_param = sum(
        [
            param.numel() \
                for param in model.parameters()
        ]
    )

    return model_num_param

def get_current_lr_list(
    optim
):
    cur_lr_list = [
        param_group["lr"] \
            for param_group in optim.param_groups
    ]

    return cur_lr_list

def get_generator(
    seed, 
    device
):
    generator = torch.Generator(device).manual_seed(seed)
    
    return generator

def get_selected_state_dict(
    model, 
    selected_param_name_list: List[str]
) -> Dict:
    state_dict = model.state_dict()

    selected_state_dict = {
        name: state_dict[name] \
            for name in selected_param_name_list
    }

    return selected_state_dict
