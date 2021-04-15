import numpy as np
import random
import warnings

import torch

"""
乱数を固定する関数
"""

def seed_everything(seed):
    warnings.filterwarnings('ignore')
    random.seed(71)
    torch.manual_seed(71)
    np.random.seed(71)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False