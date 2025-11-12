import numpy as np
import torch


class ClampSteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min=0, max=1):
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class RoundSteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantization_fixed(q, a=-np.pi, b=+np.pi, n_bins=256):
    # We expect q to be (array of) float from [a, b]

    # Convert to float [0, 1]:
    q = (q - a) / (b - a)

    # Convert to float [0, n_bins - 1]:
    q = ClampSteFn.apply(q * (n_bins - 1), 0, n_bins - 1)      
    
    # Convert to int [0, n_bins - 1]:
    q = RoundSteFn.apply(q)   
    
    # Convert to float [0, 1]:
    q = q / (n_bins - 1)
    
    # Convert to float [a, b]:
    q = q * (b - a) + a
    
    return q