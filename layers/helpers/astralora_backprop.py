import torch


def bb_backprop_wrap(bb_func, generator, samples_x=1, samples_w=1,
                     use_sm=True):
    class FuncCustom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, U, S, V):
            ctx.save_for_backward(x, w, U, S, V)
            return bb_func(x, w)

        @staticmethod
        def backward(ctx, grad_output):
            x, w, U, S, V = ctx.saved_tensors
            
            if use_sm:
                grad_x = grad_output @ U @ S @ V
            else:
                grad_x = _backprop_stochastic(bb_func, x, w, grad_output, 
                    generator, samples_x, for_x=True)

            if True:
                # grad_w = grad_output.t() @ x
                grad_w = _backprop_stochastic(bb_func, x, w, grad_output, 
                    generator, samples_w, for_x=False)

            return grad_x, grad_w, None, None, None

    func = FuncCustom.apply
    return func


def _backprop_stochastic(bb_func, x, w, grad_output, generator,
                         samples=1, shift=1., for_x=True):
    device = grad_output.device

    x = torch.clone(x.detach())
    w = torch.clone(w.detach())

    value = x if for_x else w
    grad = torch.zeros(value.shape, device=device)

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
        sampled_grad = torch.einsum(ein, u, (p_new - p0) / shift)
        grad = grad + sampled_grad

    if for_x:
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