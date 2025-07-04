"""bb_layer_matvec.

Simple matvec (linear) layer.

"""
import math
import torch


def create_bb_layer_matvec(d_inp, d_out):
    def bb(x, w):
        A = w.reshape(d_out, d_inp)
        return x @ A.T

    w = torch.empty((d_out, d_inp))
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    w = w.reshape(-1)

    dw = 1.
    
    return bb, w, dw