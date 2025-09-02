import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoModelForCausalLM
from tqdm import trange
from core.bb_layers.bb_layer_monarch import create_bb_layer_monarch


def create_sparse_mask(target_matrix, sparse_rate, mode='static'):
    """
    Create a sparse mask based on the target matrix.

    Args:
        target_matrix: Target matrix to approximate
        sparse_rate: Fraction of elements to keep sparse (0 to 1)
        mode: 'static' or 'dynamic'

    Returns:
        sparse_mask: Boolean mask where True indicates elements to keep sparse
    """

    abs_values = torch.abs(target_matrix)
    num_elements = target_matrix.numel()
    num_sparse = int(sparse_rate * num_elements)

    # Get indices of top-k elements by absolute value
    _, indices = torch.topk(abs_values.flatten(), k=num_sparse, largest=True)
    print(f"Number of indices: {len(indices)}")
    # Create mask
    sparse_mask = torch.zeros_like(target_matrix, dtype=torch.bool).flatten()
    sparse_mask[indices] = True
    sparse_mask = sparse_mask.reshape(target_matrix.shape)

    return sparse_mask


def get_matrix_with_sparse(approx_matrix, target_matrix, sparse_mask):
    """
    Combine approximated matrix with sparse elements from target matrix.

    Args:
        approx_matrix: Matrix from BB monarch approximation [d_inp, d_out]
        target_matrix: Original target matrix [d_inp, d_out]
        sparse_mask: Boolean mask indicating sparse elements [d_inp, d_out]

    Returns:
        combined_matrix: Matrix with sparse elements from target and approximated elements elsewhere
    """
    combined_matrix = approx_matrix.clone()
    combined_matrix[sparse_mask] = target_matrix[sparse_mask]
    return combined_matrix


def compute_sparse_loss(approx_matrix, target_matrix, sparse_mask):
    """
    Compute loss only on non-sparse elements.

    Args:
        approx_matrix: Matrix from BB monarch approximation [d_inp, d_out]
        target_matrix: Original target matrix [d_inp, d_out]
        sparse_mask: Boolean mask indicating sparse elements [d_inp, d_out]

    Returns:
        loss: MSE loss computed only on non-sparse elements
    """
    # Create mask for non-sparse elements
    non_sparse_mask = ~sparse_mask

    # Compute loss only on non-sparse elements
    if non_sparse_mask.sum() > 0:
        loss = torch.mean((approx_matrix[non_sparse_mask] - target_matrix[non_sparse_mask]) ** 2)
    else:
        # If all elements are sparse, loss is zero
        loss = torch.tensor(0.0, device=approx_matrix.device, requires_grad=True)

    return loss


def update_dynamic_sparse_mask(approx_matrix, target_matrix, sparse_rate):
    """
    Update sparse mask for dynamic mode based on current loss values.

    Args:
        approx_matrix: Current approximated matrix [d_inp, d_out]
        target_matrix: Original target matrix [d_inp, d_out]
        sparse_rate: Fraction of elements to keep sparse

    Returns:
        sparse_mask: Updated boolean mask [d_inp, d_out]
    """
    # Compute elementwise loss
    assert approx_matrix.shape == target_matrix.shape
    elementwise_loss = (approx_matrix - target_matrix) ** 2

    # Get indices of elements with highest loss
    num_elements = target_matrix.numel()
    num_sparse = int(sparse_rate * num_elements)

    _, indices = torch.topk(elementwise_loss.flatten(), k=num_sparse, largest=True)

    # Create mask
    sparse_mask = torch.zeros_like(target_matrix, dtype=torch.bool).flatten()
    sparse_mask[indices] = True
    sparse_mask = sparse_mask.reshape(*approx_matrix.shape)
    return sparse_mask


# Create _tmp folder if it doesn't exist
os.makedirs('_tmp', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Sparse approximation parameters
sparse_rate = 0.9
sparse_mode = 'static'  # 'static' or 'dynamic'

print(f"Sparse approximation parameters:")
print(f"  Sparse rate: {sparse_rate:.1%}")
print(f"  Sparse mode: {sparse_mode}")

# Download the model from Hugging Face
print("Loading Qwen3 model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = model.to(device)

# Try to find the feedforward linear layer
# The exact module name may depend on the model architecture.
# Let's try to find a linear layer in the feedforward block.

# Qwen3-8B uses a transformer architecture. Let's inspect the first transformer block.
first_block = model.model.layers[0]

# The feedforward block is often called 'mlp' or similar.
# Let's print the structure to find the linear layer.
print("First transformer block structure:")
print(first_block)

# Try to access the feedforward linear layer
if hasattr(first_block, "mlp"):
    mlp = first_block.mlp
    # Print the size of the first linear layer
    if hasattr(mlp, "up_proj"):
        linear_layer = mlp.up_proj
        print("Feedforward linear layer (up_proj) weight shape:", linear_layer.weight.shape)

        # Get the target matrix for approximation
        target_matrix = linear_layer.weight.data.clone()  # Shape: [d_out, d_inp]
        d_inp, d_out = target_matrix.shape[1], target_matrix.shape[0]

        print(f"Target up_proj matrix shape: {target_matrix.shape}")
        print(f"d_inp: {d_inp}, d_out: {d_out}")

        # Create sparse mask (note: target_matrix is [d_out, d_inp], but we need to work with [d_inp, d_out])
        # We'll create the mask for the transposed target matrix to match the approximated matrix shape
        target_matrix_t = target_matrix.T  # [d_inp, d_out]
        sparse_mask = create_sparse_mask(target_matrix_t, sparse_rate, sparse_mode)
        num_sparse_elements = sparse_mask.sum().item()
        total_elements = target_matrix_t.numel()
        print(f"Sparse elements: {num_sparse_elements} / {total_elements} ({num_sparse_elements/total_elements:.1%})")

        # Create BB monarch layer
        print("Creating BB monarch layer...")
        bb_func, w, dw, get_matrix = create_bb_layer_monarch(d_inp, d_out, digital_mode=True)
        w = w.to(device)

        # Initialize w with better starting values
        torch.nn.init.uniform_(w, -0.1, 0.1)

        # Make w trainable
        w = w.requires_grad_(True)

        # Create optimizer
        optimizer = optim.Adam([w], lr=0.001)

        # Training parameters
        num_epochs = 5_000  # Reduced for testing
        losses = []

        print("Starting optimization...")

        # Training loop
        for epoch in trange(num_epochs, desc="Optimizing"):
            optimizer.zero_grad()

            # Use get_matrix to generate the BB monarch layer's matrix approximation
            y_pred = get_matrix(w)

            # Compute the target outputs using the actual up_proj matrix
            # (target_matrix is [d_out, d_inp], so target_matrix.T is [d_inp, d_out])
            # get_matrix(w) returns [d_inp, d_out]
            y_target = target_matrix.T  # [d_inp, d_out]

            # Update sparse mask for dynamic mode
            if sparse_mode == 'dynamic':
                sparse_mask = update_dynamic_sparse_mask(y_pred, y_target, sparse_rate)

            # Compute sparse loss
            loss = compute_sparse_loss(y_pred, y_target, sparse_mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        print("Optimization completed!")

        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'BB Monarch + Sparse Layer Approximation Loss ({sparse_mode} mode, {sparse_rate:.1%} sparse)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error Loss (Non-sparse elements only)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_sparse_{sparse_mode}_loss.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save the loss data
        np.save(f'_tmp/monarch_sparse_{sparse_mode}_losses.npy', np.array(losses))

        # Print final statistics
        final_loss = losses[-1]
        print(f"Final loss: {final_loss:.6f}")
        print(f"Loss curve saved to _tmp/monarch_sparse_{sparse_mode}_loss.png")
        print(f"Loss data saved to _tmp/monarch_sparse_{sparse_mode}_losses.npy")

        # Matrix comparison and error analysis
        print("\nMatrix comparison:")
        print(f"Target matrix norm: {torch.norm(target_matrix):.6f}")

        # Get the approximated matrix from the BB Monarch layer
        approx_matrix = get_matrix(w).detach().cpu()
        target_matrix_cpu = target_matrix.detach().cpu().T
        if sparse_mode == 'dynamic':
            sparse_mask = update_dynamic_sparse_mask(approx_matrix, target_matrix_cpu, sparse_rate)
        sparse_mask_cpu = sparse_mask.detach().cpu()
        print(f'amount of non-zeroes in final mask:', sparse_mask_cpu.flatten().sum())

        # Create final matrix with sparse elements
        final_matrix = get_matrix_with_sparse(approx_matrix, target_matrix_cpu, sparse_mask_cpu)

        print(f"Approximated matrix norm: {torch.norm(approx_matrix):.6f}")
        print(f"Final matrix (with sparse) norm: {torch.norm(final_matrix):.6f}")

        # Plot distributions of elements of target and final matrices
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(target_matrix_cpu.flatten().numpy(), bins=100, alpha=0.6, label='Target', color='blue')
        plt.hist(final_matrix.flatten().numpy(), bins=100, alpha=0.6, label='Final (Monarch+Sparse)', color='orange')
        plt.title('Element Distribution: Target vs Final Matrix')
        plt.xlabel('Element Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot sparse mask visualization
        plt.subplot(1, 3, 2)
        plt.imshow(sparse_mask_cpu.numpy(), cmap='Reds', aspect='auto')
        plt.title(f'Sparse Mask ({sparse_rate:.1%} sparse elements)')
        plt.xlabel('Input dimension')
        plt.ylabel('Output dimension')
        plt.colorbar(label='Sparse element')

        # Compute elementwise absolute and relative error
        abs_err_matrix = torch.abs(final_matrix - target_matrix_cpu)
        rel_err_matrix = abs_err_matrix / (torch.abs(target_matrix_cpu) + 1e-8)

        plt.subplot(1, 3, 3)
        rel_err_data = rel_err_matrix.flatten().numpy()
        print(rel_err_data.mean(), f'mean relative error of sparse + monarch at {sparse_rate} {sparse_mode} sparsity')
        # print(rel_err_data,f'mean_relative error for non-sparse elements of sparse + monarch at {sparse_rate} {sparse_mode} sparsity')
        # print(, f'mean_relative error for sparse elements of sparse + monarch at {sparse_rate} {sparse_mode} sparsity')
        min_val = np.percentile(rel_err_data, 1)
        max_val = np.percentile(rel_err_data, 99)
        plt.hist(rel_err_data, bins=100, alpha=0.7, label='Relative Error', color='green', range=(min_val, max_val))
        plt.title('Elementwise Error: Final Matrix')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_sparse_{sparse_mode}_matrix_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Matrix analysis saved to _tmp/monarch_sparse_{sparse_mode}_matrix_analysis.png")

        # Now, test on random vectors
        num_test_vecs = 1000
        d_inp = target_matrix.shape[1]
        d_out = target_matrix.shape[0]
        test_inputs = torch.randn(num_test_vecs, d_inp)
        approx_outputs = test_inputs @ approx_matrix
        final_outputs = test_inputs @ final_matrix
        target_outputs = test_inputs @ target_matrix_cpu

        abs_err_vec_approx = torch.abs(approx_outputs - target_outputs)
        abs_err_vec_final = torch.abs(final_outputs - target_outputs)
        rel_err_vec_approx = abs_err_vec_approx / (torch.abs(target_outputs) + 1e-8)
        rel_err_vec_final = abs_err_vec_final / (torch.abs(target_outputs) + 1e-8)

        # Plot distributions of absolute and relative error for random vectors
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(abs_err_vec_approx.flatten().numpy(), bins=100, alpha=0.7, color='red', label='Monarch only')
        plt.hist(abs_err_vec_final.flatten().numpy(), bins=100, alpha=0.7, color='blue', label='Monarch+Sparse')
        plt.title('Elementwise Absolute Error: Random Vectors')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        rel_err_data_approx = rel_err_vec_approx.flatten().numpy()
        rel_err_data_final = rel_err_vec_final.flatten().numpy()
        min_val = min(np.percentile(rel_err_data_approx, 1), np.percentile(rel_err_data_final, 1))
        max_val = max(np.percentile(rel_err_data_approx, 99), np.percentile(rel_err_data_final, 99))
        plt.hist(rel_err_data_approx, bins=100, alpha=0.7, color='red', label='Monarch only', range=(min_val, max_val))
        plt.hist(rel_err_data_final, bins=100, alpha=0.7, color='blue', label='Monarch+Sparse', range=(min_val, max_val))
        plt.title('Elementwise Relative Error: Random Vectors')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'_tmp/monarch_sparse_{sparse_mode}_vector_error.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Vector error analysis saved to _tmp/monarch_sparse_{sparse_mode}_vector_error.png")

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Monarch only - Matrix absolute error: mean={abs_err_matrix.mean():.6f}, std={abs_err_matrix.std():.6f}")
        print(f"Monarch only - Matrix relative error: mean={rel_err_matrix.mean():.6f}, std={rel_err_matrix.std():.6f}")
        print(f"Monarch only - Vector absolute error: mean={abs_err_vec_approx.mean():.6f}, std={abs_err_vec_approx.std():.6f}")
        print(f"Monarch only - Vector relative error: mean={rel_err_vec_approx.mean():.6f}, std={rel_err_vec_approx.std():.6f}")
        print(f"Monarch+Sparse - Vector absolute error: mean={abs_err_vec_final.mean():.6f}, std={abs_err_vec_final.std():.6f}")
        print(f"Monarch+Sparse - Vector relative error: mean={rel_err_vec_final.mean():.6f}, std={rel_err_vec_final.std():.6f}")

        # Save sparse mask for later analysis
        np.save(f'_tmp/monarch_sparse_{sparse_mode}_mask.npy', sparse_mask_cpu.numpy())
        print(f"Sparse mask saved to _tmp/monarch_sparse_{sparse_mode}_mask.npy")

    else:
        print("Could not find 'up_proj' in mlp.")
else:
    print("Could not find 'mlp' in the first transformer block.")
