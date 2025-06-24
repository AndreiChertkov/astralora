import torch
import numpy as np
import matplotlib.pyplot as plt
from layers.helpers.astralora_bb import bb_appr, bb_build


def test_svd_convergence():
    """Compare normal SVD with learning-based SVD approximation"""
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    d_inp = 1000
    d_out = 1000
    rank = 200
    n_samples = 10000
    max_iter = 50
    lr = 0.001
    
    print(f"Testing SVD convergence with:")
    print(f"  Input dimension: {d_inp}")
    print(f"  Output dimension: {d_out}")
    print(f"  Rank: {rank}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Learning rate: {lr}")
    print()
    
    # Create a test matrix
    A = torch.randn(d_out, d_inp, device=device)
    
    # Normal SVD
    print("Computing normal SVD...")
    U_true, s_true, V_true = torch.linalg.svd(A, full_matrices=False)
    U_true = U_true[:, :rank]
    S_true = torch.diag(s_true[:rank])
    V_true = V_true[:rank, :]
    
    # Create black-box function
    bb, w = bb_build(d_inp, d_out, d_inp * d_out, kind='matvec')
    
    # Move w to device
    w = w.to(device)
    
    # Learning-based SVD
    print("Computing learning-based SVD...")
    losses = []
    
    def log_loss(loss_val):
        losses.append(loss_val)
        if len(losses) % 100 == 0:
            print(f"  Iteration {len(losses)}, Loss: {loss_val:.6f}")
    
    U_learned, S_learned, V_learned = bb_appr(
        bb, d_inp, d_out, w, rank=rank, 
        log=log_loss, n_samples=n_samples, 
        lr=lr, max_iter=max_iter
    )

    
    
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    # 1. Matrix reconstruction error
    A_reconstructed_true = U_true @ S_true @ V_true
    A_reconstructed_learned = U_learned @ S_learned @ V_learned
    
    error_true = torch.norm(A - A_reconstructed_true, 'fro')
    error_learned = torch.norm(A - A_reconstructed_learned, 'fro')
    
    print(f"Matrix reconstruction error:")
    print(f"  True SVD: {error_true:.6f}")
    print(f"  Learned SVD: {error_learned:.6f}")
    print(f"  Relative difference: {abs(error_true - error_learned) / error_true * 100:.2f}%")
    
    # 2. Singular values comparison
    print(f"\nSingular values comparison:")
    print(f"  True SVD singular values: {s_true[:rank]}")
    print(f"  Learned SVD singular values: {torch.diag(S_learned).sort(descending=True)[0]}")
    
    # 3. Test on random inputs
    print(f"\nTesting on random inputs...")
    X_test = torch.randn(100, d_inp, device=device)
    Y_true = X_test @ A.T
    Y_approx_true = X_test @ V_true.T @ S_true @ U_true.T
    Y_approx_learned = X_test @ V_learned.T @ S_learned @ U_learned.T
    
    error_true_test = torch.mean((Y_true - Y_approx_true) ** 2)
    error_learned_test = torch.mean((Y_true - Y_approx_learned) ** 2)
    
    print(f"Test error (MSE):")
    print(f"  True SVD: {error_true_test:.6f}")
    print(f"  Learned SVD: {error_learned_test:.6f}")
    print(f"  Relative difference: {abs(error_true_test - error_learned_test) / error_true_test * 100:.2f}%")
    
    # 4. Plot convergence
    if losses:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Learning-based SVD Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.yscale('log')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Move tensors to CPU for plotting
        s_true_cpu = s_true[:rank].cpu()
        S_learned_cpu = torch.diag(S_learned).cpu()
        plt.plot(s_true_cpu, 'o-', label='True SVD', markersize=8)
        plt.plot(S_learned_cpu, 's-', label='Learned SVD', markersize=8)
        plt.title('Singular Values Comparison')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('svd_convergence_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nConvergence plot saved as 'svd_convergence_comparison.png'")
    
    # 5. Orthogonality check
    print(f"\nOrthogonality check:")
    U_ortho_error = torch.norm(U_learned.T @ U_learned - torch.eye(rank, device=device), 'fro')
    V_ortho_error = torch.norm(V_learned @ V_learned.T - torch.eye(rank, device=device), 'fro')
    print(f"  U orthogonality error: {U_ortho_error:.6f}")
    print(f"  V orthogonality error: {V_ortho_error:.6f}")
    
    return {
        'U_true': U_true,
        'S_true': S_true, 
        'V_true': V_true,
        'U_learned': U_learned,
        'S_learned': S_learned,
        'V_learned': V_learned,
        'losses': losses,
        'reconstruction_error_true': error_true,
        'reconstruction_error_learned': error_learned,
        'test_error_true': error_true_test,
        'test_error_learned': error_learned_test
    }


if __name__ == "__main__":
    results = test_svd_convergence() 