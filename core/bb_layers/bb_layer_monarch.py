"""bb_layer_monarch.

Monarch layer implementation following the same pattern as other bb layers.

"""
import math
import torch
import torch.nn as nn
import numpy as np


def create_bb_layer_monarch(d_inp, d_out, digital_mode=False, output_encoding="real", w_get_matrix=False):
    # decompose the dimensions into the product of their two closest integer factors.
    num_inp_blks, inp_bsize = get_closest_factors(d_inp)
    num_out_blks, out_bsize = get_closest_factors(d_out)
    d = num_inp_blks * num_out_blks * (inp_bsize + out_bsize)


    def bb(x, w):
        if not digital_mode:
            x = x.to(torch.cfloat)

        # Pad the last dimension of x to num_inp_blks * inp_bsize if necessary
        last_dim = x.shape[-1]
        target_dim = num_inp_blks * inp_bsize
        if last_dim < target_dim:
            pad_size = target_dim - last_dim
            pad_shape = list(x.shape[:-1]) + [pad_size]
            pad_tensor = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad_tensor], dim=-1)
        elif last_dim > target_dim:
            raise ValueError(f"Input dimension {last_dim} is greater than the target dimension {target_dim}. Input shape: {x.shape}")

        x = x.unflatten(-1, (num_inp_blks, inp_bsize))
        w1 = w[0 : num_inp_blks * num_out_blks * inp_bsize].view(num_out_blks, num_inp_blks, inp_bsize)
        w2 = w[num_inp_blks * num_out_blks * inp_bsize :].view(out_bsize, num_out_blks, num_inp_blks)

        if not digital_mode:
            w1 = _get_slm_stack(w1)
            w2 = _get_slm_stack(w2)

        output = torch.einsum("qki, kij, ...ij -> ...qk", w2, w1, x)
        output = output.flatten(start_dim=-2)

        # Crop the last dimension of output to d_out
        if output.shape[-1] >= d_out:
            output = output[..., :d_out]
        else:
            raise ValueError(f"Output dimension {output.shape[-1]} is less than the target dimension {d_out}. Output shape: {output.shape}")


        if digital_mode:
            return output

        if output_encoding == "real":
            return torch.real(output)
        elif output_encoding == "intensity":
            return torch.abs(output)
    
    def get_matrix(w):
        x = torch.eye(d_inp, dtype=torch.cfloat if not digital_mode else torch.float, device=w.device)
        return bb(x, w)


    w = torch.empty(d)
    # torch.nn.init.uniform_(w, a=0, b=torch.pi)
    torch.nn.init.uniform_(w, math.pi/2 - 0.1, math.pi/2 + 0.1)

    dw = 1.E-4
    
    if w_get_matrix:
        return bb, w, dw, get_matrix
    else:
        return bb, w, dw


def get_closest_factors(N: int):
    """
    Decomposes a number N, which is a power of 2, into the product
    of its two closest integer factors.

    Args:
        N (int): The number to decompose. Must be a power of 2.

    Returns:
        tuple[int, int]: A tuple containing the two closest integer factors,
                         sorted in ascending order.

    Raises:
        ValueError: If N is not a power of 2 or not a positive integer.
    """

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Input N must be a positive integer.")

    # if (N & (N - 1)) != 0:
    #     raise ValueError(f"Input {N} is not a power of 2.")
    k = np.log2(N)

    exp1 = np.ceil(k / 2)
    exp2 = np.ceil(k / 2)

    factor1 = np.power(2, exp1)
    factor2 = np.power(2, exp2)

    return int(factor1), int(factor2)


def _get_slm_stack(tps):
    """
    Get tensor of multiple slms.

    Args:
        tps (Tensor[num_slm, N_out, N_inp]): phase-shifts of num_slm SLMs with given N_inp and N_out each.

    Returns:
        Tensor[num_slm, N_out, N_inp]: Tensor torch.exp(1j * tps) / torch.sqrt(torch.tensor(tps.size(2), dtype=torch.cfloat))
    """

    w = torch.exp(1j * tps) / torch.sqrt(torch.tensor(tps.size(2), dtype=torch.cfloat))
    return w


