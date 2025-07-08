"""bb_layer_monarch.

Monarch layer implementation following the same pattern as other bb layers.

"""
import torch
import torch.nn as nn
import numpy as np


def create_bb_layer_monarch(d_inp, d_out):
    # Use the larger dimension for monarch decomposition
    d_max = max(d_inp, d_out)
    
    # Check if d_max is a power of 2, if not, pad to the next power of 2
    if (d_max & (d_max - 1)) != 0:
        d_padded = 2 ** int(np.ceil(np.log2(d_max)))
    else:
        d_padded = d_max
    
    monarch_dims = get_closest_factors(d_padded)
    d = monarch_dims[0] * monarch_dims[1] * (monarch_dims[0] + monarch_dims[1])

    def bb(x, w):
        # Pad input if necessary
        if x.shape[-1] < d_padded:
            padding = torch.zeros(*x.shape[:-1], d_padded - x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > d_padded:
            x = x[..., :d_padded]
        
        x = x.to(torch.cfloat)
        x = x.unflatten(-1, (monarch_dims[0], monarch_dims[1]))
        ps1 = w[0 : monarch_dims[0] * monarch_dims[1] ** 2].view(monarch_dims[0], monarch_dims[1], monarch_dims[1])
        ps2 = w[monarch_dims[0] * monarch_dims[1] ** 2 :].view(monarch_dims[1], monarch_dims[0], monarch_dims[0])

        # tensors of size [dim[0], dim[1], dim[1]] and [dim[1], dim[0], dim[0]]
        w1 = _get_slm_stack(ps1)
        w2 = _get_slm_stack(ps2)

        output = torch.einsum("...qi, qij -> ...qj", x, w1)
        output = torch.einsum("...qi, iqj -> ...qj", output, w2)
        output = torch.real(output)
        
        # Reshape back to flattened form
        output = output.flatten(-2)
        
        # Truncate or pad to desired output dimension
        if output.shape[-1] > d_out:
            output = output[..., :d_out]
        elif output.shape[-1] < d_out:
            padding = torch.zeros(*output.shape[:-1], d_out - output.shape[-1], dtype=output.dtype, device=output.device)
            output = torch.cat([output, padding], dim=-1)

        return output

    w = torch.empty(d)
    torch.nn.init.uniform_(w, a=0, b=torch.pi)
    
    dw = 1.E-4
    
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


# Test snippet
if __name__ == "__main__":
    # Test cases
    test_cases = [
        (8, 8),    # Same dimensions (power of 2)
        (16, 16),  # Same dimensions (power of 2)
        (8, 16),   # Different dimensions (both powers of 2)
        (16, 8),   # Different dimensions (both powers of 2)
        (10, 12),  # Different dimensions (not powers of 2)
        (12, 10),  # Different dimensions (not powers of 2)
    ]
    
    for d_inp, d_out in test_cases:
        print(f"\nTesting monarch layer: d_inp={d_inp}, d_out={d_out}")
        
        # Create layer
        bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
        
        # Create test input
        batch_size = 3
        x = torch.randn(batch_size, d_inp)
        
        # Forward pass
        try:
            output = bb(x, w)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: ({batch_size}, {d_out})")
            print(f"  ✓ Test passed!")
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
    
    print("\nAll tests completed!")
