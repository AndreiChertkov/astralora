"""bb_layer_mzi.

Code taken from https://github.com/JeremieMelo/pytorch-onn/blob/main/torchonn/layers/mzi_linear.py

"""
import torch
from torch import Tensor


def create_bb_layer_mzi(d_inp, d_out):
    raise NotImplementedError

    v_max = 10.8
    v_pi = 4.36
    gamma = torch.pi / v_pi**2
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