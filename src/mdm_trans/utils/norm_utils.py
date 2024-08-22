import numpy as np
import torch


def normalize(data, axis):
    return (data                               - data.min(axis=axis, keepdims=True)) \
         / (data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True) + 1e-7)


class Normalizer():
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x, feature_idx=-1):
        mean, std = self.adjust_dim(x, feature_idx)
        x = (x - mean) / std
        return x

    def backward(self, x, feature_idx=-1):
        mean, std = self.adjust_dim(x, feature_idx)
        x = x * std + mean
        return x

    def adjust_dim(self, x, feature_idx=-1):
        mean = self.motion_mean
        std = self.motion_std

        if feature_idx == -1:
            return mean, std
        
        for _ in range(feature_idx):  
            mean = mean.unsqueeze_(0)
            std = std.unsqueeze_(0)

        for _ in range(feature_idx+1, x.ndim):  
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)

        return mean, std



