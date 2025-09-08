import torch
import torch.nn as nn

from .layer import AstraloraLayer


class SparseBBLayer(nn.Module):
    def __init__(self, base_layer, d_inp, d_out,
                 bb_kind, rank, samples_bb, samples_sm,
                 samples_bb_batch_frac, skip_sm, use_residual,
                 sparse_top_p=0.1, log=print, nepman=None):
        super().__init__()

        self.d_inp = d_inp
        self.d_out = d_out
        self.log = log
        self.nepman = nepman

        # Initialize sparse mask and weights from the provided base layer (if available)
        if hasattr(base_layer, 'ыweight') and base_layer.weight is not None:
            with torch.no_grad():
                full_weight = base_layer.weight.data.detach().clone()
        else:
            full_weight = torch.zeros(d_out, d_inp)

        p = float(sparse_top_p) if sparse_top_p is not None else 0.0
        p = max(0.0, min(1.0, p))

        if p <= 0.0:
            mask = torch.zeros_like(full_weight, dtype=torch.bool)
        elif p >= 1.0:
            mask = torch.ones_like(full_weight, dtype=torch.bool)
        else:
            k = max(1, int(round(p * full_weight.numel())))
            flat = full_weight.abs().flatten()
            if k >= flat.numel():
                threshold = flat.min()
            else:
                threshold = torch.topk(flat, k, largest=True).values.min()
            mask = (full_weight.abs() >= threshold)

        self.register_buffer('sparse_mask', mask.to(dtype=torch.float32))

        # Trainable sparse weight; gradients flow only where mask==1 due to multiplication
        self.sparse_weight = nn.Parameter(torch.zeros_like(full_weight))
        with torch.no_grad():
            self.sparse_weight.data.copy_(full_weight * self.sparse_mask)

        # Optional bias retained as part of sparse/digital parameters (main optimizer)
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.detach().clone())
        else:
            self.bias = None

        # Underlying BB sub-layer that learns the complement
        self.bb_layer = AstraloraLayer(
            d_inp=d_inp,
            d_out=d_out,
            kind=bb_kind,
            rank=rank,
            samples_bb=samples_bb,
            samples_sm=samples_sm,
            samples_bb_batch_frac=samples_bb_batch_frac,
            skip_sm=skip_sm,
            use_residual=use_residual,
            log=log,
            nepman=nepman)

        # If using matvec BB, initialize its weight to the complement (non-sparse) part
        if bb_kind == 'matvec':
            with torch.no_grad():
                remainder = full_weight * (1.0 - self.sparse_mask)
                self.bb_layer.w.data.copy_(remainder.reshape(-1))

    def forward(self, x):
        # x: [..., d_inp]
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])

        # Sparse/digital path
        w_sparse = self.sparse_weight * self.sparse_mask
        y_sparse = x2 @ w_sparse.t()

        # BB path
        y_bb = self.bb_layer(x2)

        y = y_sparse + y_bb
        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*shape[:-1], y.shape[-1])

    def rebuild(self):
        # Delegate to inner BB layer
        self.bb_layer.rebuild()


