"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from trains.multiTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # multi-task
            'v1': V1,
            'v2': V1,
            'v3': V1,
            'v4': V1,
            'v5': V2,
            'v6': V2,
            'v7': V2,
            'uw': V2,
            'dwa': V4,
        }
    
    def getTrain(self, args):
        # 根据加权方法不同选择训练器
        return self.TRAIN_MAP[args.weighting_method](args)
