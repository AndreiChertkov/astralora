"""bb_layer_lowrank.

Low-rank layer implementation following the same pattern as other bb layers.

"""
import math
import torch
import torch.nn as nn
import numpy as np
import warnings


def create_bb_layer_lowrank(d_inp, d_out, rank=None, w_get_matrix=False):
    # If rank is not specified, use a default rank (minimum of input and output dimensions)
    if rank is None:
        rank = min(d_inp, d_out)
        warnings.warn(f"Rank not specified, using default rank: {rank}")
    
    # Ensure rank is valid
    rank = min(rank, min(d_inp, d_out))
    
    # Total number of parameters: rank * (d_inp + d_out)
    d = rank * (d_inp + d_out)

    def bb(x, w):
        # Split weights into left and right factors
        left_weights = w[:rank * d_inp].view(rank, d_inp)
        right_weights = w[rank * d_inp:].view(d_out, rank)

        # Low-rank matrix multiplication: (d_out, rank) @ (rank, d_inp) @ (d_inp, ...) -> (d_out, ...)
        output = torch.einsum("or,ri,...i->...o", right_weights, left_weights, x)

        return output
    
    def get_matrix(w):
        x = torch.eye(d_inp, dtype=torch.float, device=w.device)
        return bb(x, w)

    w = torch.empty(d)
    # Initialize weights with small random values for better convergence
    torch.nn.init.normal_(w, mean=0.0, std=0.1)

    dw = 1.E-4
    
    if w_get_matrix:
        return bb, w, dw, get_matrix
    else:
        return bb, w, dw



