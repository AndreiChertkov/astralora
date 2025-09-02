import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoModelForCausalLM
from tqdm import trange
from core.bb_layers.bb_layer_monarch import create_bb_layer_monarch


def initialize_low_rank_from_svd(target_matrix_t: torch.Tensor, rank: int):
    """
    Initialize low-rank factors L and R from SVD of the target matrix.

    Args:
        target_matrix_t: Target matrix with shape [d_inp, d_out]
        rank: Desired low-rank

    Returns:
        L: [d_inp, r]
        R: [r, d_out]
        effective_rank: Possibly clamped rank actually used
    """
    d_inp, d_out = target_matrix_t.shape
    max_rank = min(d_inp, d_out)
    effective_rank = min(rank, max_rank)

    # Compute economical/full-m=False SVD: target_matrix_t = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(target_matrix_t, full_matrices=False)
    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]

    # Factor as L @ R with balanced scaling via sqrt of singular values
    sqrt_S = torch.sqrt(S_r)
    L = U_r * sqrt_S.unsqueeze(0)
    R = (sqrt_S.unsqueeze(1) * Vh_r)

    return L, R, effective_rank


def compute_parameter_ratio(d_inp: int, d_out: int, rank: int) -> float:
    """Return (parameters in low-rank) / (parameters in full matrix)."""
    low_rank_params = rank * (d_inp + d_out)
    full_params = d_inp * d_out
    return low_rank_params / full_params


# Create _tmp folder if it doesn't exist
os.makedirs('_tmp', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Low-rank + Monarch approximation parameters
rank = 1020  # low-rank	r
mode = 'dynamic'  # 'static' or 'dynamic'
batch_size = 256

print("Low-rank + Monarch approximation parameters:")
print(f"  Rank: {rank}")
print(f"  Mode: {mode}")


# Download the model from Hugging Face
print("Loading Qwen3 model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = model.to(device)

# Inspect first transformer block
first_block = model.model.layers[0]
print("First transformer block structure:")
print(first_block)

if hasattr(first_block, "mlp"):
    mlp = first_block.mlp
    if hasattr(mlp, "up_proj"):
        linear_layer = mlp.up_proj
        print("Feedforward linear layer (up_proj) weight shape:", linear_layer.weight.shape)

        # target_matrix: [d_out, d_inp]
        target_matrix = linear_layer.weight.data.clone()
        d_inp, d_out = target_matrix.shape[1], target_matrix.shape[0]

        print(f"Target up_proj matrix shape: {target_matrix.shape}")
        print(f"d_inp: {d_inp}, d_out: {d_out}")

        # Prepare target in [d_inp, d_out]
        target_matrix_t = target_matrix.T  # [d_inp, d_out]

        # Initialize low-rank via SVD
        L, R, effective_rank = initialize_low_rank_from_svd(target_matrix_t.to(device), rank)
        L = L.to(device)
        R = R.to(device)
        param_ratio = compute_parameter_ratio(d_inp, d_out, effective_rank)
        print(f"Effective rank: {effective_rank}")
        print(f"Parameter ratio (low-rank/original): {param_ratio:.4f}")

        # Create BB monarch layer
        print("Creating BB monarch layer...")
        bb_func, w, dw, get_matrix = create_bb_layer_monarch(d_inp, d_out, digital_mode=True, w_get_matrix=True)
        w = w.to(device)
        torch.nn.init.uniform_(w, -0.1, 0.1)
        w = w.requires_grad_(True)

        # Low-rank trainability based on mode
        if mode == 'dynamic':
            L = L.requires_grad_(True)
            R = R.requires_grad_(True)
        else:
            L = L.detach()
            R = R.detach()

        # Optimizer
        params = [w]
        if mode == 'dynamic':
            params += [L, R]
        optimizer = optim.Adam(params, lr=0.001)

        # Training params
        num_epochs = 5_000
        losses = []
        print("Starting optimization (vector-action loss)...")

        for epoch in trange(num_epochs, desc="Optimizing"):
            optimizer.zero_grad()

            # Sample random input vectors X: [B, d_inp]
            X = torch.randn(batch_size, d_inp, device=device)

            # Current matrices
            monarch_matrix = get_matrix(w)  # [d_inp, d_out]
            if mode == 'dynamic':
                low_rank_matrix = L @ R  # [d_inp, d_out]
            else:
                # For static mode, recompute from detached L,R (constant)
                low_rank_matrix = L @ R

            full_matrix = low_rank_matrix + monarch_matrix

            # Vector-action outputs
            y_pred = X @ full_matrix
            y_target = X @ target_matrix_t.to(device)

            # Loss on vector outputs
            loss = torch.mean((y_pred - y_target) ** 2)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        print("Optimization completed!")

        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'Low-Rank+Monarch Approximation Loss (mode={mode}, ratio={param_ratio:.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (vector outputs)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_lowrank_{mode}_loss.png', dpi=300, bbox_inches='tight')
        plt.show()

        np.save(f'_tmp/monarch_lowrank_{mode}_losses.npy', np.array(losses))

        # Matrix comparison and error analysis
        print("\nMatrix comparison:")
        print(f"Target matrix norm: {torch.norm(target_matrix):.6f}")

        approx_monarch = get_matrix(w).detach().cpu()
        if mode == 'dynamic':
            low_rank_final = (L @ R).detach().cpu()
        else:
            low_rank_final = (L @ R).cpu()
        final_matrix = low_rank_final + approx_monarch  # [d_inp, d_out]
        target_matrix_cpu = target_matrix_t.detach().cpu()

        print(f"Low-rank matrix norm: {torch.norm(low_rank_final):.6f}")
        print(f"Monarch matrix norm: {torch.norm(approx_monarch):.6f}")
        print(f"Final matrix norm: {torch.norm(final_matrix):.6f}")

        # Distributions of elements
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(target_matrix_cpu.flatten().numpy(), bins=100, alpha=0.6, label='Target', color='blue')
        plt.hist(final_matrix.flatten().numpy(), bins=100, alpha=0.6, label='LowRank+Monarch', color='orange')
        plt.title('Element Distribution: Target vs Final')
        plt.xlabel('Element Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Elementwise errors
        abs_err_matrix = torch.abs(final_matrix - target_matrix_cpu)
        rel_err_matrix = abs_err_matrix / (torch.abs(target_matrix_cpu) + 1e-8)

        plt.subplot(1, 3, 2)
        rel_err_data = rel_err_matrix.flatten().numpy()
        print(rel_err_data.mean(), f'mean relative error of low-rank + monarch at rank {effective_rank} {mode} mode and ratio {param_ratio:.4f}')
        abs_err_data = abs_err_matrix.flatten().numpy()
        print(abs_err_data.mean(), f'mean absolute error of low-rank + monarch at rank {effective_rank} {mode} mode and ratio {param_ratio:.4f}')
        min_val = np.percentile(rel_err_data, 1)
        max_val = np.percentile(rel_err_data, 99)
        plt.hist(rel_err_data, bins=100, alpha=0.7, label='Relative Error', color='green', range=(min_val, max_val))
        plt.title('Elementwise Relative Error: Final Matrix')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Show parameter ratio
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.title('Parameter Ratio Info')
        txt = f'Rank: {effective_rank}\nLow-rank params: {effective_rank * (d_inp + d_out):,}\nFull params: {d_inp * d_out:,}\nRatio: {param_ratio:.4f}'
        print(txt)
        plt.text(0.1, 0.5, txt, fontsize=12)

        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_lowrank_{mode}_matrix_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Matrix analysis saved to _tmp/monarch_lowrank_{mode}_matrix_analysis.png")

        # Random vector tests
        num_test_vecs = 1000
        test_inputs = torch.randn(num_test_vecs, d_inp)
        approx_outputs_monarch = test_inputs @ approx_monarch
        approx_outputs_final = test_inputs @ final_matrix
        target_outputs = test_inputs @ target_matrix_cpu

        abs_err_vec_monarch = torch.abs(approx_outputs_monarch - target_outputs)
        abs_err_vec_final = torch.abs(approx_outputs_final - target_outputs)
        rel_err_vec_monarch = abs_err_vec_monarch / (torch.abs(target_outputs) + 1e-8)
        rel_err_vec_final = abs_err_vec_final / (torch.abs(target_outputs) + 1e-8)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(abs_err_vec_monarch.flatten().numpy(), bins=100, alpha=0.7, color='red', label='Monarch only')
        plt.hist(abs_err_vec_final.flatten().numpy(), bins=100, alpha=0.7, color='blue', label='LowRank+Monarch')
        plt.title('Elementwise Absolute Error: Random Vectors')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        rel_err_data_monarch = rel_err_vec_monarch.flatten().numpy()
        rel_err_data_final = rel_err_vec_final.flatten().numpy()
        min_val = min(np.percentile(rel_err_data_monarch, 1), np.percentile(rel_err_data_final, 1))
        max_val = max(np.percentile(rel_err_data_monarch, 99), np.percentile(rel_err_data_final, 99))
        plt.hist(rel_err_data_monarch, bins=100, alpha=0.7, color='red', label='Monarch only', range=(min_val, max_val))
        plt.hist(rel_err_data_final, bins=100, alpha=0.7, color='blue', label='LowRank+Monarch', range=(min_val, max_val))
        plt.title('Elementwise Relative Error: Random Vectors')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_lowrank_{mode}_vector_error.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Vector error analysis saved to _tmp/monarch_lowrank_{mode}_vector_error.png")

        # Summary statistics
        print("\nSummary Statistics:")
        print(f"Monarch only - Vector absolute error: mean={abs_err_vec_monarch.mean():.6f}, std={abs_err_vec_monarch.std():.6f}")
        print(f"Monarch only - Vector relative error: mean={rel_err_vec_monarch.mean():.6f}, std={rel_err_vec_monarch.std():.6f}")
        print(f"LowRank+Monarch - Vector absolute error: mean={abs_err_vec_final.mean():.6f}, std={abs_err_vec_final.std():.6f}")
        print(f"LowRank+Monarch - Vector relative error: mean={rel_err_vec_final.mean():.6f}, std={rel_err_vec_final.std():.6f}")

        # Save low-rank factors for later analysis
        np.save(f'_tmp/monarch_lowrank_{mode}_L.npy', low_rank_final.numpy())
        np.save(f'_tmp/monarch_lowrank_{mode}_final_matrix.npy', final_matrix.numpy())
        print(f"Saved low-rank and final matrices to _tmp/")

    else:
        print("Could not find 'up_proj' in mlp.")
else:
    print("Could not find 'mlp' in the first transformer block.")


