# run like
# export CUDA_VISIBLE_DEVICES=0 && TEST_D_INP=1000 TEST_D_OUT=1000 TEST_BATCH=256 TEST_STEPS=3000 python -m pytest tests/visualize/test_fit_mzi_to_unitary.py -q -s > _tmp/mzi_fit_unitary_1000x1000_cuda.log 2>&1
# or simply
# CUDA_VISIBLE_DEVICES=7 TEST_D_INP=1000 TEST_D_OUT=1000 TEST_BATCH=256 TEST_STEPS=3000 python tests/visualize/test_fit_mzi_to_unitary.py

import os
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
from core.bb_layers.bb_layer_mzi import create_bb_layer_mzi


def _compute_effective_matrix(bb_func, w, d_inp, d_out):
    """Return the effective real matrix A (shape d_out x d_inp) realized by bb with params w.

    We evaluate the layer on the identity inputs to recover columns.
    """
    device = w.device
    x_eye = torch.eye(d_inp, device=device, dtype=torch.float32)
    y = bb_func(x_eye, w)  # shape: (d_inp, d_out); row k equals column k of A
    a_mat = y.T.contiguous()  # (d_out, d_inp)
    return a_mat


def _vector_to_square_grid(vec: torch.Tensor) -> torch.Tensor:
    """Pad a vector to the next square length with zeros and reshape to (k, k)."""
    numel = vec.numel()
    k = int(math.ceil(math.sqrt(numel)))
    total = k * k
    if total == numel:
        return vec.reshape(k, k)
    out = torch.zeros(total, dtype=vec.dtype)
    out[:numel] = vec.reshape(-1).cpu()
    return out.reshape(k, k)


def _random_real_orthogonal(n: int, device: torch.device) -> torch.Tensor:
    """Generate a random n x n real orthogonal matrix via QR decomposition."""
    a = torch.randn(n, n, device=device)
    q, r = torch.linalg.qr(a)
    # Make Q unique by fixing the sign (ensure positive diagonal of R)
    d = torch.sign(torch.diagonal(r))
    d[d == 0] = 1.0
    q = q @ torch.diag(d)
    return q  # orthogonal (real unitary)


def test_visualize_fit_mzi_to_unitary():
    """
    Fit an MZI layer to the real part of a random unitary and save visualizations.

    Environment overrides (optional):
    - TEST_D_INP, TEST_D_OUT: integers to override default dims
    - TEST_STEPS: number of training steps (default 300)
    - TEST_BATCH: minibatch size of random input vectors per step (default 256)
    - TEST_LR: learning rate (default 0.05)
    - TEST_SEED: random seed (default 0)
    """
    # Configuration
    d_inp = int(os.environ.get('TEST_D_INP', 8))
    d_out = int(os.environ.get('TEST_D_OUT', 8))
    steps = int(os.environ.get('TEST_STEPS', 300))
    batch = int(os.environ.get('TEST_BATCH', 256))
    lr = float(os.environ.get('TEST_LR', 0.001))
    seed = int(os.environ.get('TEST_SEED', 0))

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build MZI layer (bb function) and parameters
    bb, w0, _ = create_bb_layer_mzi(d_inp, d_out)
    w = w0.clone().to(device).detach().requires_grad_(True)

    # Determine internal N used by the layer from parameter count (for target unitary size)
    # Parameter formula: total_mzis = (N//2)^2 + (N//2)*(N//2 - 1); d = 2 * total_mzis
    d_params = w.numel()

    def _infer_n_from_params(d_params_val: int) -> int:
        # Try even N starting from max(d_inp, d_out) up to a reasonable bound
        start = max(d_inp, d_out)
        if start % 2 == 1:
            start += 1
        for n_try in range(start, start + 512, 2):
            m0 = (n_try // 2) * (n_try // 2)
            m1 = (n_try // 2) * max(0, n_try // 2 - 1)
            total = (m0 + m1) * 2
            if total == d_params_val:
                return n_try
        raise ValueError('Failed to infer internal N from parameter count')

    n_internal = _infer_n_from_params(d_params)

    # Build target matrix: block of a random real orthogonal matrix of size N
    u = _random_real_orthogonal(n_internal, device=device)
    target_full = u  # (N, N), real and orthogonal
    target = target_full[:d_out, :d_inp].to(device=device, dtype=torch.float32)

    # Optimizer
    optimizer = torch.optim.Adam([w], lr=lr)

    # Trackers
    losses = []
    w_init = w.detach().cpu().clone()

    # Initial matrix and initial loss (on a random minibatch)
    with torch.no_grad():
        a_init = _compute_effective_matrix(bb, w, d_inp, d_out).detach().cpu()
        x0 = torch.randn(batch, d_inp, device=device, dtype=torch.float32)
        y0_t = torch.matmul(target, x0.T).T
        y0_p = bb(x0, w)
        loss_init = torch.mean((y0_p - y0_t).pow(2)).item()

    # Train on random vector minibatches
    for _ in trange(steps, desc="Training"):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(batch, d_inp, device=device, dtype=torch.float32)
        y_t = torch.matmul(target, x.T).T
        y_p = bb(x, w)
        loss = torch.mean((y_p - y_t).pow(2))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if _ % 100 == 0:
            print(f"Loss: {loss.detach().item()}")

    with torch.no_grad():
        a_final = _compute_effective_matrix(bb, w, d_inp, d_out).detach().cpu()
        w_final = w.detach().cpu().clone()
        loss_final = torch.mean((a_final.to(device) - target).pow(2)).item()

    # Prepare output dir
    out_dir = '_tmp'
    os.makedirs(out_dir, exist_ok=True)
    fname = f'mzi_fit_unitary_{d_inp}x{d_out}.png'
    fpath = os.path.join(out_dir, fname)

    # Plot all visuals in one PNG
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1) Target matrix
    ax = axes[0, 0]
    im = ax.imshow(target.cpu(), aspect='auto', cmap='RdBu_r')
    ax.set_title(f'Target (Re[U]) {d_out}x{d_inp}')
    ax.set_xlabel('inp index')
    ax.set_ylabel('out index')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2) Final MZI matrix
    ax = axes[0, 1]
    im = ax.imshow(a_final, aspect='auto', cmap='RdBu_r')
    ax.set_title('Final MZI matrix')
    ax.set_xlabel('inp index')
    ax.set_ylabel('out index')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3) Loss evolution
    ax = axes[0, 2]
    ax.plot(losses, color='tab:blue', linewidth=1.5)
    ax.set_title(f'Loss evolution (init {loss_init:.3e} → final {loss_final:.3e})')
    ax.set_xlabel('step')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)

    # 4) Initial parameters (heatmap) and its distribution (next subplot)
    ax = axes[1, 0]
    w_init_grid = _vector_to_square_grid(w_init)
    im = ax.imshow(w_init_grid, aspect='equal', cmap='viridis')
    ax.set_title('Initial parameters (heatmap)')
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_hist = axes[1, 1]
    ax_hist.hist(w_init.numpy(), bins=50, color='gray', alpha=0.9)
    ax_hist.set_title('Initial parameters distribution')
    ax_hist.set_xlabel('value')
    ax_hist.set_ylabel('count')

    # 5) Final parameters distribution (hist only)
    ax = axes[1, 2]
    ax.hist(w_final.numpy(), bins=50, color='tab:orange', alpha=0.9)
    ax.set_title('Final parameters distribution')
    ax.set_xlabel('value')
    ax.set_ylabel('count')

    # Save figure
    fig.suptitle(f'MZI fit to unitary (d_inp={d_inp}, d_out={d_out}, steps={steps}, lr={lr})', fontsize=12)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

    # Basic sanity assertion: loss should decrease substantially
    assert loss_final < loss_init, 'Training did not reduce loss'


if __name__ == '__main__':
    test_visualize_fit_mzi_to_unitary()

