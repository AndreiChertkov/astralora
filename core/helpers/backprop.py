import torch


def backprop_wrap(bb_func, generator, samples_w=1, shift_w=1., skip_sm=False,
                  samples_batch_frac=-1):
    class FuncCustom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, U, S, V):
            ctx.save_for_backward(x, w, U, S, V)
            return bb_func(x, w)

        @staticmethod
        def backward(ctx, grad_output):
            x, w, U, S, V = ctx.saved_tensors

            if skip_sm:
                # We use GD for x-grad:
                with torch.enable_grad():
                    x_grad = x.detach().requires_grad_(True)
                    y_x = bb_func(x_grad, w.detach())
                    grad_x, = torch.autograd.grad(
                        outputs=y_x,
                        inputs=x_grad,
                        grad_outputs=grad_output)
            else:
                # We use SM for x-grad:
                grad_x = grad_output @ U @ S @ V

            if samples_w == -1:
                # We use GD for w-grad:
                # (for linear layer it would be:
                # grad_w = grad_output.t() @ x
                # grad_w = grad_w.reshape(-1)  )
                with torch.enable_grad():
                    w_grad = w.detach().requires_grad_(True)
                    y_w = bb_func(x.detach(), w_grad)
                    grad_w, = torch.autograd.grad(
                        outputs=y_w,
                        inputs=w_grad,
                        grad_outputs=grad_output)
            else:
                # We use stochastic formula for w-grad:
                if samples_w < 1:
                    raise ValueError('Invalid number of samples to update w')
                grad_w = _backprop_stochastic(bb_func, x, w, grad_output, 
                    generator, samples_w, shift_w, 
                    samples_batch_frac=samples_batch_frac)

            return grad_x, grad_w, None, None, None

    return FuncCustom.apply




def _backprop_stochastic(bb_func, x, w, grad_output, generator,
                         samples=1, shift=1., for_x=False, 
                         samples_batch_frac=-1):
    device = grad_output.device

    if samples_batch_frac > 0:
        num = int(x.shape[0] * samples_batch_frac)
        ind = torch.randperm(x.size(0))[:num]
        x = x[ind]
        grad_output = grad_output[ind]

    x = torch.clone(x.detach())
    w = torch.clone(w.detach())

    value = x if for_x else w
    grad = torch.zeros(value.shape, device=device)

    # For w-gradients, use compressed-batch optimization when B > J
    if not for_x and x.shape[0] > grad_output.shape[1]:
        # Compressed-batch optimization: replace batch B with J = d_out
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
        
        for _ in range(samples):
            u_m = torch.zeros(value.shape, device=device)
            u_s = torch.tensor(1., device=device)
            u = torch.normal(u_m, std=u_s, generator=generator)
            
            Y_new = bb_func(M, w + shift * u)  # [J, J]
            s_new = torch.einsum("ii->", Y_new)
            s = (s_new - s0) / shift
            grad = grad + u * s
        
        grad = grad / samples
        return grad
    
    # Original implementation for x-gradients or when B <= J
    y0 = bb_func(x, w)
    p0 = torch.einsum("ij,ij->i", y0, grad_output)

    for _ in range(samples):
        u_m = torch.zeros(value.shape, device=device)
        u_s = torch.tensor(1., device=device)
        u = torch.normal(u_m, std=u_s, generator=generator)

        if for_x:
            y_new = bb_func(x + shift * u, w)
        else:
            y_new = bb_func(x, w + shift * u)

        p_new = torch.einsum("ij,ij->i", y_new, grad_output)
        
        ein = "ij,i->ij" if for_x else "j,i->j"
        grad_sampled = torch.einsum(ein, u, (p_new - p0) / shift)
        
        grad = grad + grad_sampled

    grad = grad / samples

    return grad


def _backprop_stochastic_x_megabatch(bb_func, x, w, grad_output, generator,
                                     samples=1, shift=1.):
    device = grad_output.device

    x = torch.clone(x.detach())
    batch_size = x.shape[0]

    # Generate all samples at once
    u_m = torch.zeros((samples, *x.shape), device=device)
    u_s = torch.tensor(1., device=device)
    u = torch.normal(u_m, std=u_s, generator=generator)

    # Reshape for batch processing
    x_expanded = x.unsqueeze(0).expand(samples, -1, -1)
    x_perturbed = x_expanded + shift * u
    
    # Process all samples in one forward pass
    y0 = bb_func(x, w)
    p0 = torch.einsum("ij,ij->i", y0, grad_output)
    
    y_new = bb_func(x_perturbed.reshape(-1, *x.shape[1:]), w)
    y_new = y_new.reshape(samples, batch_size, -1)
    
    p_new = torch.einsum("sij,ij->si", y_new, grad_output)
    sampled_g = torch.einsum("sij,si->sij",
        u, (p_new - p0) / shift)
    
    x_grad = sampled_g.mean(dim=0)

    return x_grad


def _backprop_stochastic_subset(
    bb_func,
    x,
    w,
    grad_output,
    generator,
    samples=1,
    shift=1.0,
    subset_size=None,
    subset_fraction=None,
):
    """Stochastic zero-order gradient approximation over a random subset of parameters.

    Computes a finite-difference gradient estimate for ``w`` using random Gaussian
    perturbations applied only on a randomly sampled subset of parameter indices
    on each sample. Parameters outside the sampled subset receive zero updates
    for that sample. The final result is averaged over all samples.

    Arguments
    - bb_func: callable taking (x, w) and returning output tensor shaped like y.
    - x: input tensor (batch, in_features...) used for evaluating bb_func.
    - w: parameter tensor for which the gradient is approximated.
    - grad_output: upstream gradient w.r.t. bb_func(x, w), same shape as output.
    - generator: torch.Generator used for reproducible randomness.
    - samples: number of stochastic samples to average over (int >= 1).
    - shift: finite-difference step size for perturbations.
    - subset_size: number of parameter elements to perturb per sample. If None,
      it is inferred from subset_fraction.
    - subset_fraction: fraction (0, 1] of parameters to perturb per sample if
      subset_size is not provided. Defaults to 0.1 if both are None.

    Returns
    - grad: tensor of the same shape as ``w`` with the approximated gradient.
    """
    device = grad_output.device

    if samples < 1:
        raise ValueError('samples must be >= 1')
    if shift <= 0:
        raise ValueError('shift must be > 0')

    # Work on detached clones to avoid autograd interactions
    x = torch.clone(x.detach())
    w = torch.clone(w.detach())

    total_params = w.numel()
    if subset_size is None:
        if subset_fraction is None:
            subset_fraction = 0.1
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError('subset_fraction must be in (0, 1]')
        subset_size = max(1, int(round(total_params * subset_fraction)))
    else:
        if subset_size < 1 or subset_size > total_params:
            raise ValueError('subset_size must be in [1, numel(w)]')

    grad = torch.zeros_like(w, device=device)

    # Baseline projection with current parameters
    y0 = bb_func(x, w)
    p0 = torch.einsum('ij,ij->i', y0, grad_output)

    for _ in range(samples):
        # Sample subset indices and Gaussian perturbations for those indices
        indices = torch.randperm(total_params, device=device, generator=generator)[:subset_size]

        u_flat = torch.zeros(total_params, device=device)
        u_subset = torch.normal(
            mean=torch.zeros(subset_size, device=device),
            std=torch.ones(subset_size, device=device),
            generator=generator,
        )
        u_flat.index_copy_(0, indices, u_subset)
        u = u_flat.reshape_as(w)

        # Forward with perturbed parameters only on the chosen subset
        y_new = bb_func(x, w + shift * u)
        p_new = torch.einsum('ij,ij->i', y_new, grad_output)

        # Directional derivative times perturbation gives estimator of grad
        grad_sampled_flat = torch.einsum('j,i->j', u_flat, (p_new - p0) / shift)
        grad = grad + grad_sampled_flat.reshape_as(w)

    grad = grad / samples
    return grad