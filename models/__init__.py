import importlib
from .Unet import Unet
import random
import os
import torch
import numpy as np

def get_option_setter():
    """Return the static method <modify_commandline_options> of the model class."""
    return Unet.modify_commandline_options

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
