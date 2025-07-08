import torch
import torch.nn as nn
from torch import cos, exp, sin, tensor

import numpy as np

from optinet.layers.layer import Layer, LayerConfig


class MonarchSlmLayer(Layer):
    def __init__(self, config: LayerConfig):
        if config.d_inp != config.d_out:
            raise ValueError(f"Expected d_inp = d_out; got d_inp = {config.d_inp}, d_out = {config.d_out}.")

        monarch_dims = get_closest_factors(config.d_inp)
        config.d = monarch_dims[0] * monarch_dims[1] * (monarch_dims[0] + monarch_dims[1])

        super().__init__(config=config)
        self.weight = nn.Parameter(torch.Tensor(config.d), requires_grad=True)  # real parameters
        nn.init.uniform_(self.weight, a=0, b=torch.pi)
        self.bb_set_func(build_monarchslm_func(monarch_dims))


def build_monarchslm_func(dims):

    def func(x, w):
        x = x.to(torch.cfloat)
        x = x.unflatten(-1, (dims[0], dims[1]))
        ps1 = w[0 : dims[0] * dims[1] ** 2].view(dims[0], dims[1], dims[1])
        ps2 = w[dims[0] * dims[1] ** 2 :].view(dims[1], dims[0], dims[0])

        # tensors of size [dim[0], dim[1], dim[1]] and [dim[1], dim[0], dim[0]]
        w1 = _get_slm_stack(ps1)
        w2 = _get_slm_stack(ps2)

        output = torch.einsum("...qi, qij -> ...qj", x, w1)
        output = torch.einsum("...qi, iqj -> ...qj", output, w2)

        return torch.real(output)

    return func


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

    if (N & (N - 1)) != 0:
        raise ValueError(f"Input {N} is not a power of 2.")
    k = np.log2(N)

    exp1 = np.floor(k / 2)
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
