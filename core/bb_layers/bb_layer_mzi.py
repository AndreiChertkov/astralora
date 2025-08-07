"""bb_layer_mzi.

"""
import torch
import torch.nn as nn
from torch import Tensor
from torch import tensor, sqrt, cos, sin, exp
import warnings
import numpy as np


def create_bb_layer_mzi(d_inp, d_out):
    """Create MZI (Mach-Zehnder Interferometer) BB layer using MZI3ULayer.
    
    This function creates a BB layer that uses the MZI3U interferometer
    implementation for optical computing.
    
    Args:
        d_inp: Input dimension
        d_out: Output dimension
        
    Returns:
        tuple: (bb_function, initial_weights, learning_rate_adjustment)
    """
    # Find minimal N that satisfies MZI3U requirements
    # N must be even and N*N >= max(d_inp, d_out)
    # This allows the N-dimensional MZI space to accommodate both input and output
    
    # Find minimal N such that N*N can accommodate both dimensions
    min_N = max(d_inp, d_out)
    
    # Make sure N is even (required for MZI3U)
    if min_N % 2 == 1:
        N = min_N + 1
    else:
        N = min_N
    
    # N must also be at least 2 for MZI3U to work
    N = max(N, 2)
    
    # Calculate number of parameters needed for N x N MZI3U network
    # Each MZI needs 2 parameters (theta and phi)
    num_mzis_0 = (N // 2) * (N // 2)  # MZIs in even layers
    num_mzis_1 = (N // 2) * max(0, N // 2 - 1)  # MZIs in odd layers  
    total_mzis = num_mzis_0 + num_mzis_1
    d = total_mzis * 2
    
    def build_parameters():
        """Initialize parameters for MZI3U network."""
        w = torch.zeros(d)
        # Initialize with uniform distribution as in original MZI3ULayer
        nn.init.uniform_(w, a=0, b=1.0)
        return w
    
    # Get the BB function from MZI3U implementation
    bb_func = build_3mzi_func(N)
    
    def bb(x, w):
        """
        BB function that applies MZI3U transformation.
        
        Args:
            x: Input tensor of shape [..., d_inp]
            w: Weight parameters of shape [d]
            
        Returns:
            Output tensor of shape [..., d_out]
        """
        # Step 1: Pad input from d_inp to N if necessary
        needed_size = N
        if x.shape[-1] < needed_size:
            # Pad input to size N
            padding = torch.zeros(*x.shape[:-1], needed_size - x.shape[-1], 
                                dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > needed_size:
            # This shouldn't happen since we chose N >= d_inp, but handle it
            x_padded = x[..., :needed_size]
            warnings.warn(f"Input dimension {x.shape[-1]} is greater than needed size {needed_size}. Truncating input.")
        else:
            x_padded = x
            
        # Step 2: Apply MZI3U transformation (N -> N)
        output_mzi = bb_func(x_padded, w)
        
        # Step 3: Handle output mapping from N to d_out
        # Since N*N >= d_out, we can always fit d_out in the N-dimensional output
        if d_out <= output_mzi.shape[-1]:
            # Simply take the first d_out elements
            output = output_mzi[..., :d_out]
        else:
            raise ValueError(f"Output dimension {d_out} is greater than the output dimension of the MZI3U layer {output_mzi.shape[-1]}.")
            
        return output
    
    # Initialize parameters
    w0 = build_parameters()
    
    # Learning rate adjustment factor (from original code)
    dw = 0.01
    
    return bb, w0, dw


# Note: MZI3ULayer and MZI3UConfig classes have been moved to use the 
# create_bb_layer_mzi function approach, which is compatible with the
# existing astralora framework. The core MZI3U functionality is preserved
# in the build_3mzi_func and related functions below.


def build_3mzi_func(N):

    def func(x, w):
        x = x.to(torch.cfloat)

        interferometer = MZI3_Interferometer(N, w)
        U = interferometer.U

        output = torch.einsum("ij, ...j -> ...i", U, x)

        return torch.real(output)

    return func


class MZI3_Interferometer:
    def __init__(self, N, params):
        self.device = params.device
        self.params = params
        self.N = N
        self.n_mzi = N * (N - 1) // 2
        self.U = self.get_matrix()

    def get_matrix(self):
        params0 = self.params[0 : self.N**2 // 2]  # for layers 0, 2, 4, ...
        params1 = self.params[self.N**2 // 2 : self.N**2 - self.N]  # for layers 1, 3, 5, ...
        blocks0 = _get_3mzi_blocks(params0).view((self.N // 2, self.N // 2, 2, 2))  # layers 0, 2, 4, ...
        blocks1 = _get_3mzi_blocks(params1).view((self.N // 2, self.N // 2 - 1, 2, 2))  # layers 1, 3, 5, ...

        U = torch.eye(self.N, dtype=torch.cfloat, device=self.device)
        for i in range(self.N // 2):
            # First einsum - avoid view/reshape which can break autograd
            U_reshaped = U.reshape(self.N // 2, 2, self.N) 
            U = torch.einsum("bij,bjk->bik", blocks0[i], U_reshaped)
            U = U.reshape(self.N, self.N)
            
            # Second einsum - avoid in-place operations and views
            U_middle = U[1:-1, :]
            U_middle_reshaped = U_middle.reshape(self.N // 2 - 1, 2, self.N)
            U_new_middle = torch.einsum("bij,bjk->bik", blocks1[i], U_middle_reshaped)
            U_new_middle = U_new_middle.reshape(self.N - 2, self.N)
            
            # Combine results without in-place operations
            U = torch.cat([U[:1], U_new_middle, U[-1:]], dim=0)

            # old version, left for reference
            # U = torch.einsum("bij, bjk -> bik", blocks0[i], U.view((self.N // 2, 2, self.N))).view((self.N, self.N))
            # U[1:-1, :] = torch.einsum("bij, bjk -> bik", blocks1[i], U[1:-1, :].view((self.N // 2 - 1, 2, self.N))).view(
            #     (self.N - 2, self.N)
            # )

        return U


def _get_3mzi_blocks(params):
    """
    Args:
    pshifts: phase-shift tensors of size [num_mzis, 2]
    pshifts[:,0] corresponds to thetas
    pshifts[:,1] corresponds to phis

    Return:
    tmzi: tensor of size [num_mzis, 2, 2]
    """
    device = params.device
    pshifts = params.view(-1, 2)

    n_mzis = pshifts.size(0)
    tmzi = torch.zeros((n_mzis, 2, 2), device=device).to(torch.cfloat)

    ps1 = (pshifts[:, 0] - pshifts[:, 1]) / 2
    ps2 = (pshifts[:, 0] + pshifts[:, 1]) / 2

    tmzi[:, 0, 0] = exp(1j * ps2) * (sin(ps1) + 1j * sin(ps2)) / sqrt(tensor(2))
    tmzi[:, 1, 1] = exp(1j * ps2) * (sin(ps1) - 1j * sin(ps2)) / sqrt(tensor(2))
    tmzi[:, 0, 1] = exp(1j * ps2) * (-cos(ps1) + 1j * cos(ps2)) / sqrt(tensor(2))
    tmzi[:, 1, 0] = exp(1j * ps2) * (cos(ps1) + 1j * cos(ps2)) / sqrt(tensor(2))

    return tmzi
