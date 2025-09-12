import torch
import torch.nn as nn

from .layer import AstraloraLayer


class SparseBBLayer(nn.Module):
    def __init__(self, base_layer, d_inp, d_out,
                 bb_kind, rank, samples_bb, samples_sm,
                 samples_bb_batch_frac, skip_sm, use_residual,
                 sparse_top_p=0.1, log=print, nepman=None,
                 bb_pretrain_steps=10000, bb_pretrain_batch=1024,
                 bb_pretrain_lr=1e-3, bb_pretrain_seed=42,
                 bb_pretrain_weight_decay=0.0,
                 bb_pretrain_clip_grad_norm=None,
                 bb_pretrain_inner_batches=1,
                 bb_pretrain_eval_batches=32,
                 bb_pretrain_use_cosine=True,
                 bb_pretrain_min_lr=0.0,
                 bb_pretrain_warmup_frac=0.1):
        super().__init__()

        self.d_inp = d_inp
        self.d_out = d_out
        self.log = log
        self.nepman = nepman

        # Initialize sparse mask and weights from the provided base layer (if available)
        if hasattr(base_layer, 'weight') and base_layer.weight is not None:
            with torch.no_grad():
                full_weight = base_layer.weight.data.detach().clone()
        else:
            full_weight = torch.zeros(d_out, d_inp)

        # Keep a copy of the original full weight as target for BB pretraining
        self.register_buffer('full_weight', full_weight.clone())

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
            b = base_layer.bias.data.detach().clone()
            self.bias = nn.Parameter(b.clone())
            # Keep a fixed copy for target during pretraining
            self.register_buffer('full_bias', b.clone())
        else:
            self.bias = None
            # Fixed zero bias target
            self.register_buffer('full_bias', torch.zeros(d_out, dtype=full_weight.dtype))

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

        # BB pretraining hyperparameters
        self.bb_pretrain_steps = int(bb_pretrain_steps) if bb_pretrain_steps is not None else 0
        self.bb_pretrain_batch = int(bb_pretrain_batch)
        self.bb_pretrain_lr = float(bb_pretrain_lr)
        self.bb_pretrain_seed = int(bb_pretrain_seed)
        self.bb_pretrain_weight_decay = float(bb_pretrain_weight_decay)
        self.bb_pretrain_clip_grad_norm = (None if bb_pretrain_clip_grad_norm is None
                                           else float(bb_pretrain_clip_grad_norm))
        self.bb_pretrain_inner_batches = max(1, int(bb_pretrain_inner_batches))
        self.bb_pretrain_eval_batches = max(1, int(bb_pretrain_eval_batches))
        self.bb_pretrain_use_cosine = bool(bb_pretrain_use_cosine)
        self.bb_pretrain_min_lr = float(bb_pretrain_min_lr)
        self.bb_pretrain_warmup_frac = float(bb_pretrain_warmup_frac)
        self._bb_pretrained = False

        # If using matvec BB, initialize its weight to the complement (non-sparse) part
        # if bb_kind == 'matvec':
        #     with torch.no_grad():
        #         remainder = full_weight * (1.0 - self.sparse_mask)
        #         self.bb_layer.w.data.copy_(remainder.reshape(-1))

        # Decorate the underlying Astralora _build to run BB pretraining
        original_build = self.bb_layer._build

        def _decorated_build(device=None, _orig=original_build):
            # Run pretraining once on the correct device before building wrappers/factors
            self._bb_pretrain(device=device)
            return _orig(device=device)

        self.bb_layer._build = _decorated_build

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

    def _bb_pretrain(self, device=None):
        if self._bb_pretrained or self.bb_pretrain_steps <= 0:
            return

        # Resolve device/dtype
        param_device = self.sparse_weight.device
        dev = device if device is not None else param_device
        dtype = self.sparse_weight.dtype

        # Optimizer over BB weights and sparse parameters (and bias if present)
        opt_params = [self.bb_layer.w, self.sparse_weight]
        if self.bias is not None:
            opt_params.append(self.bias)
        optimizer = torch.optim.AdamW(opt_params, lr=self.bb_pretrain_lr,
                                      weight_decay=self.bb_pretrain_weight_decay)
        mse = torch.nn.MSELoss()

        # Optional cosine LR scheduler with warmup
        scheduler = None
        if self.bb_pretrain_use_cosine and self.bb_pretrain_steps > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.bb_pretrain_steps, eta_min=self.bb_pretrain_min_lr)
        warmup_steps = int(self.bb_pretrain_warmup_frac * self.bb_pretrain_steps)

        # RNG for repeatability on the same device
        generator = torch.Generator(device=dev)
        generator.manual_seed(self.bb_pretrain_seed)

        # Helper to sample a batch
        def sample_inputs(batch):
            return torch.randn((batch, self.d_inp), device=dev, dtype=dtype, generator=generator)

        # Compute loss given inputs
        def compute_loss(X):
            # Sparse path (trainable during pretraining)
            w_sparse = self.sparse_weight * self.sparse_mask
            y_sparse = X @ w_sparse.t()

            # BB forward uses the raw bb mapping without wrapper
            y_bb = self.bb_layer.bb(X, self.bb_layer.w)

            # Prediction combines sparse + bb (+trainable bias)
            y_pred = y_sparse + y_bb
            if self.bias is not None:
                y_pred = y_pred + self.bias

            # Target from original full weight (+fixed original bias)
            y_tgt = X @ self.full_weight.t()
            y_tgt = y_tgt + self.full_bias

            return mse(y_pred, y_tgt)

        # Eval helper over multiple batches (no grad)
        def eval_avg_loss(num_batches):
            total = 0.0
            with torch.no_grad():
                for _ in range(num_batches):
                    X = sample_inputs(self.bb_pretrain_batch)
                    total += compute_loss(X).item()
            return total / float(num_batches)

        # Initial loss (multi-batch)
        init_loss = eval_avg_loss(self.bb_pretrain_eval_batches)
        self.log(f"... [BB pretrain] init MSE: {init_loss:.6e}")

        # Training loop
        for step in range(self.bb_pretrain_steps):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.bb_pretrain_inner_batches):
                X = sample_inputs(self.bb_pretrain_batch)
                loss = compute_loss(X) / float(self.bb_pretrain_inner_batches)
                loss.backward()
                accum_loss += float(loss.detach())

            if self.bb_pretrain_clip_grad_norm is not None and self.bb_pretrain_clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=self.bb_pretrain_clip_grad_norm)

            optimizer.step()

            # Warmup then cosine schedule
            if warmup_steps > 0 and step < warmup_steps:
                scale = float(step + 1) / float(warmup_steps)
                for g in optimizer.param_groups:
                    g['lr'] = self.bb_pretrain_lr * scale
            elif scheduler is not None:
                scheduler.step()

        # Final loss (multi-batch)
        final_loss = eval_avg_loss(self.bb_pretrain_eval_batches)
        self.log(f"... [BB pretrain] final MSE: {final_loss:.6e}")

        self._bb_pretrained = True
