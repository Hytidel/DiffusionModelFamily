from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import pickle

import numpy as np
import torch

import os
import shutil


def load_pkl(
    pkl_path: str
) -> Union[List, np.ndarray, torch.Tensor]:
    if not os.path.isfile(pkl_path):
        raise ValueError(
            f"File `{pkl_path}` not exists. "
        )

    res = None
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    
    return res

def save_pkl(
    var: Union[List, np.ndarray, torch.Tensor], 
    pkl_root_path: str, 
    pkl_name: str
): 
    if not os.path.exists(pkl_root_path):
        os.makedirs(pkl_root_path)
    
    pkl_path = os.path.join(pkl_root_path, pkl_name)

    with open(pkl_path, "wb") as f:
        pickle.dump(var, f)
    