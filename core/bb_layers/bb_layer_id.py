"""bb_layer_id.

Simple ID layer (i.e., no layer).

"""
import torch


def create_bb_layer_id(d_inp, d_out):
    def bb(x, w):
        return x

    w0 = torch.Tensor([])
    
    dw = 1.
    
    return bb, w0, dw