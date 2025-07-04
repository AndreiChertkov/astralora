# code taken from https://github.com/JeremieMelo/pytorch-onn/blob/main/torchonn/layers/mzi_linear.py

import numpy as np
import torch
from torch import Tensor


def create_mzi_linear(d_inp, d_out):

    v_max = 10.8
    v_pi = 4.36
    gamma = np.pi / v_pi**2
    w_bit = 32
    in_bit = 32
    photodetect = True
    decompose_alg = 'clements'

    def build_parameters() -> None:
        ...
    
    def bb(x, w):
        weight = build_weight_from_phase(w).view(d_inp, d_out)
        return x @ weight

    w = build_parameters()
    
    return bb, w



