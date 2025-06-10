import sys
import os
import torch
from torch import nn
import torch.nn.functional as F


from layers.astralora_layer import AstraloraLayer
from layers.nograd_layer import NoGradLinear
from layers.truelowrank_layer import TrueLowRankLinear


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([
                Block(config, num) for num in range(config.n_layer)])))

        self._head_init()
        
    def forward(self, idx, targets=None, return_logits=True):
        # Forward the GPT model itself
        # (token embeddings of shape (b, t, n_embd)):
        
        x = self._body_forward(idx)

        return self._head_forward(x, targets, return_logits)
    
    def _body_forward(self, idx):
        # body forward, useful to implement stuff for advanced sampling strategies
        x = self.transformer.wte(idx) 
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generation of the new sequence."""
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it:
            if idx.size(1) <= self.config.n_embd:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.n_embd:]
            
            # Forward the model to get the logits for the index in the sequence:
            logits, _ = self(idx_cond)
            
            # Pluck the logits at the final step and scale by temperature:
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options:
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities:
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution:
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue:
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_head_params(self):
        return self.lm_head.parameters()

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.
        
        For non-embedding count (default), the position embeddings get
        subtracted. The token embeddings would too, except due to the parameter
        sharing these params are actually used as weights in the final layer,
        so we include them.
        
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _head_forward(self, x, targets=None, return_logits=True):
        if targets is not None:
            # if we are given some desired targets also calculate the loss:
            logits = self.lm_head(x)
            logits = logits.float() # Use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the
            # very last position:
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :]) 
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # There are performance reasons why not returning logits is prudent,
        # if not needed:
        if not return_logits:
            logits = None

        return logits, loss
    
    def _head_init(self):
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight


class Block(nn.Module):
    def __init__(self, config, num):
        super().__init__()

        self.attn = CausalSelfAttention(config)

        self.mlp = MLP(config, num)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class MLP(nn.Module):
    def __init__(self, config, num):
        super().__init__()

        # TODO: make it cleaner
        is_last = num == config.n_layer-1
        is_bb = config.mode == 'bb'
        is_bb = is_bb or (config.mode == 'bb_one' and is_last)
        is_nograd = config.mode == 'nograd'
        is_truelowrank = config.mode == 'truelowrank'

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

        if is_nograd:
            self.c_proj = NoGradLinear(4 * config.n_embd, config.n_embd)
        elif is_truelowrank:
            self.c_proj = TrueLowRankLinear(4 * config.n_embd, config.n_embd, rank=config.rank)
        elif not is_bb:
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
            self.c_proj.weight.data.zero_()
        else:
            self.c_proj = AstraloraLayer(4 * config.n_embd, config.n_embd,
                rank=config.rank, log=config.log,
                samples_bb=config.samples_bb, samples_sm=config.samples_sm)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        # Re-assemble all head outputs side by side:
        y = y.transpose(1, 2).contiguous().view_as(x) 
        y = self.c_proj(y)
        return y


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)