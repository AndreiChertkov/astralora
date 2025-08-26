import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from core.bb_layers.bb_layer_slm import create_bb_layer_slm
from core.bb_layers.bb_layer_mrr import create_bb_layer_mrr
from core.bb_layers.bb_layer_monarch import create_bb_layer_monarch


def _estimate_grad_w_original(bb_func, x, w, grad_output, shift, u_samples):
    """Original stochastic estimator used in _backprop_stochastic for_w."""
    x = torch.clone(x.detach())
    w = torch.clone(w.detach())

    device = x.device
    value = w
    grad = torch.zeros(value.shape, device=device)

    y0 = bb_func(x, w)
    p0 = torch.einsum("ij,ij->i", y0, grad_output)

    for u in u_samples:
        y_new = bb_func(x, w + shift * u)

        p_new = torch.einsum("ij,ij->i", y_new, grad_output)
        
        ein = "j,i->j"
        grad_sampled = torch.einsum(ein, u, (p_new - p0) / shift)
        
        grad = grad + grad_sampled

    grad = grad / len(u_samples)

    return grad


def _estimate_grad_w_compressed(bb_func, x, w, grad_output, shift, u_samples):
    """Compressed-batch variant: replace batch B with J = d_out using x_M = G^T @ x."""
    device = x.device
    dtype = w.dtype

    B = x.shape[0]
    x2d = x.view(B, -1)
    G = grad_output  # [B, J]
    J = G.shape[1]

    # Build compressed batch of size J
    M = (G.transpose(0, 1) @ x2d).contiguous()  # [J, D_flat]
    M = M.view(J, *x.shape[1:])  # [J, ...]

    # Baseline scalar is the trace of bb(M, w)
    Y0 = bb_func(M, w)  # [J, J]
    s0 = torch.einsum("ii->", Y0)  # trace

    grad = torch.zeros_like(w, device=device, dtype=dtype)
    for u in u_samples:
        Y_new = bb_func(M, w + shift * u)  # [J, J]
        s_new = torch.einsum("ii->", Y_new)
        s = (s_new - s0) / shift
        grad = grad + u * s

    grad = grad / len(u_samples)
    return grad


def plot_grad_comparison():
    """Generate comparison plots for different bb layers."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    
    # Test configurations
    configs = [
        ("SLM", create_bb_layer_slm, 16, 8, 256, 10, 1.0),
        ("MRR", create_bb_layer_mrr, 16, 8, 256, 10, 1.0),
        ("Monarch", create_bb_layer_monarch, 16, 16, 128, 10, 1.0),
    ]
    
    os.makedirs("_tmp", exist_ok=True)
    
    for layer_name, factory, d_inp, d_out, B, samples, shift in configs:
        print(f"Processing {layer_name}...")
        
        bb, w0, _ = factory(d_inp, d_out)
        x = torch.randn(B, d_inp, device=device)
        
        # Upstream gradient must match bb output shape
        with torch.no_grad():
            y0 = bb(x, w0)
        grad_output = torch.randn_like(y0)
        
        # Pre-sample identical perturbations for both estimators
        u_samples = [torch.normal(
            mean=torch.zeros_like(w0),
            std=torch.ones_like(w0),
            generator=torch.Generator(device=device).manual_seed(1000 + i)
        ) for i in range(samples)]
        
        grad_orig = _estimate_grad_w_original(bb, x, w0, grad_output, shift, u_samples)
        grad_comp = _estimate_grad_w_compressed(bb, x, w0, grad_output, shift, u_samples)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Gradient Comparison: {layer_name} Layer', fontsize=16)
        
        # Plot 1: Scatter plot of original vs compressed
        axes[0, 0].scatter(grad_orig.detach().numpy(), grad_comp.detach().numpy(), alpha=0.6)
        axes[0, 0].plot([grad_orig.min(), grad_orig.max()], [grad_orig.min(), grad_orig.max()], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Original Gradient')
        axes[0, 0].set_ylabel('Compressed Gradient')
        axes[0, 0].set_title('Scatter Plot: Original vs Compressed')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Difference histogram
        diff = (grad_orig - grad_comp).detach().numpy()
        axes[0, 1].hist(diff, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Gradient Difference (Original - Compressed)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Difference Distribution\nMax diff: {diff.max():.2e}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient magnitude comparison
        mag_orig = torch.norm(grad_orig).item()
        mag_comp = torch.norm(grad_comp).item()
        axes[1, 0].bar(['Original', 'Compressed'], [mag_orig, mag_comp], alpha=0.7)
        axes[1, 0].set_ylabel('Gradient Magnitude')
        axes[1, 0].set_title(f'Gradient Magnitudes\nRel diff: {abs(mag_orig-mag_comp)/mag_orig:.2e}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Relative error by parameter index
        rel_error = torch.abs(grad_orig - grad_comp) / (torch.abs(grad_orig) + 1e-8)
        axes[1, 1].plot(rel_error.detach().numpy(), alpha=0.7)
        axes[1, 1].set_xlabel('Parameter Index')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_title(f'Relative Error by Parameter\nMax rel error: {rel_error.max():.2e}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'_tmp/grad_comparison_{layer_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"  {layer_name}: max abs diff = {diff.max():.2e}, max rel error = {rel_error.max():.2e}")
    
    print(f"\nPlots saved in _tmp/ folder:")
    for layer_name, _, _, _, _, _, _ in configs:
        print(f"  - grad_comparison_{layer_name.lower()}.png")


if __name__ == "__main__":
    plot_grad_comparison()
