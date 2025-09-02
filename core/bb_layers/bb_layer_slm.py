"""bb_layer_slm.

Simple SLM layer.

"""
import math
import torch


def create_bb_layer_slm(d_inp, d_out):
    def bb(x, w):
        w = w.view(d_out, d_inp)
        y = torch.einsum('ij, ...j -> ...i', torch.cos(w), x)
        y = y / math.sqrt(d_inp)
        return y

        # w = torch.nn.functional.sigmoid(w) - 0.5
        # w = w * 2 * math.pi
        # y = torch.einsum('ij, ...j -> ...i', w, x)
        # return y

    w = torch.empty((d_out, d_inp))
    # torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    torch.nn.init.uniform_(w, math.pi/2 - 0.1, math.pi/2 + 0.1)
    w = w.reshape(-1)

    dw = 1.E-4
    
    return bb, w, dw