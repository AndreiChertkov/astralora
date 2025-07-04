"""bb_layer_mrr.

Code taken from https://github.com/JeremieMelo/pytorch-onn/blob/main/torchonn/layers/mrr_linear.py

"""
import math
import numpy as np
import torch
from torch import Tensor


def create_bb_layer_mrr(d_inp, d_out):
    v_max = 10.8
    v_pi = 4.36
    gamma = np.pi / v_pi**2

    mrr_tr_to_weight = lambda x: 2 * x - 1

    def build_parameters() -> None:
        phase = torch.empty((d_out, d_inp))
        torch.nn.init.kaiming_uniform_(phase, a=math.sqrt(5))
        phase = phase.reshape(-1)
        return phase
    
    def build_weight_from_phase(phases: Tensor) -> Tensor:
        return mrr_tr_to_weight(mrr_roundtrip_phase_to_tr(phases))
    
    def build_weight(phases: Tensor) -> Tensor:
        return build_weight_from_phase(phases)
    
    def bb(x, w):
        weight = build_weight_from_phase(w).view(d_inp, d_out)
        return x @ weight

    w = build_parameters()

    dw = 0.01
    
    return bb, w, dw


def mrr_roundtrip_phase_to_tr(
    rt_phi, a: float = 0.8, r: float = 0.9, intensity: bool = False
):

    # use slow but accurate mode from theoretical equation
    # create e^(-j phi) first
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # ephi = torch.view_as_complex(polar_to_complex(mag=None, angle=-rt_phi)) ## this sign is from the negativity of phase lag
    # ### Jiaqi: Since PyTorch 1.7 rsub is not supported for autograd of complex, so have to use negate and add
    # a_ephi = -a * ephi
    # t = torch.view_as_real((r + a_ephi)/(1 + r * a_ephi))

    # if(intensity):
    #     t = get_complex_energy(t)
    # else:
    #     t = get_complex_magnitude(t)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_time', row_limit=5))
    ra_cosphi_by_n2 = -2 * r * a * rt_phi.cos()
    t = (a * a + r * r + ra_cosphi_by_n2) / (1 + r * r * a * a + ra_cosphi_by_n2)
    if not intensity:
        # as long as a is not equal to r, t cannot be 0.
        t = t.sqrt()
    return t