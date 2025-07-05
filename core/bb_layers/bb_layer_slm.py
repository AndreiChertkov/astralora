"""bb_layer_slm.

Simple SLM layer.

"""
import math
import torch


def create_bb_layer_slm(d_inp, d_out):
    def bb(x, w):
        x = x.to(torch.cfloat)
        w = w.view(d_out, d_inp)
        w = torch.exp(1j * w)
        y = torch.einsum('ij, ...j -> ...i', w, x)
        y = y / torch.sqrt(torch.tensor(d_inp, dtype=torch.cfloat))
        return torch.real(y)

    w = torch.empty((d_out, d_inp))
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    w = w.reshape(-1)

    dw = 0.01
    
    return bb, w, dw