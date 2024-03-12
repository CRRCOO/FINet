import torch
import numpy as np
from math import sqrt


def get_model_complexity(model, inputs, round=3):
    """
    Modified according to https://github.com/GewelsJI/SINet-V2/blob/main/utils/utils.py
    """
    from thop import profile, clever_format
    flops, params = profile(model, inputs=inputs)
    if round is not None:
        flops, params = clever_format([flops, params], f"%.{round}f")
        return flops, params
    return int(flops), int(params)


def normalize(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img
