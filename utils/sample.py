import torch
import torch.nn.functional as F
import numpy as np

def sample1d(x, L):
    B, C, _ = x.shape
    x4d = x.reshape(B*C, 1, -1, 1)
    x = F.interpolate(x4d, size=(L, 1), mode='bicubic', align_corners=False)
    x = x[..., 0].reshape(B, C, -1)
    return x

def sample2d(x, H, W):
    x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    return x


