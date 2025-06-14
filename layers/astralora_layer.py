import torch
import torch.nn as nn


from .helpers.astralora_backprop import bb_backprop_wrap
from .helpers.astralora_bb import bb_appr
from .helpers.astralora_bb import bb_build
from .helpers.astralora_psi import psi_implicit


class AstraloraLayer(nn.Module):
    def __init__(self, d_inp, d_out, d=None, kind='matvec', rank=1,
                 samples_bb=100, samples_sm=100, use_sm=True,
                 log=print, nepman=None):
        super().__init__()
        
        self.d_inp = d_inp
        self.d_out = d_out
        self.d = d
        if self.d is None or self.d < 0:
            self.d = self.d_inp * self.d_out
        self.kind = kind
        self.rank = rank
        self.samples_bb = samples_bb
        self.samples_sm = samples_sm
        self.log = log
        self.nepman = nepman
        self.use_sm = use_sm

        self.log('... [DEBUG] Init Astralora layer : ' + self.extra_repr())

        self.bb, w = bb_build(self.d_inp, self.d_out, self.d, self.kind)
        self.w = nn.Parameter(w)
        self.w_old = None

        self.device = None
        self.bb_wrapper = None
      
    def extra_repr(self):
        text = ''
        text += f'd_inp={self.d_inp}, '
        text += f'd_out={self.d_out}, '
        text += f'd={self.d}, '
        text += f'kind={self.kind}, '
        text += f'rank={self.rank}, '
        text += f'samples_bb={self.samples_bb}, '
        text += f'samples_sm={self.samples_sm}, '
        text += f'use_sm={self.use_sm}'
        return text

    def forward(self, x):
        if self.bb_wrapper is None:
            self._build(device=x.device)

        shape = x.shape
        x = x.reshape(-1, shape[-1])

        y = self.bb_wrapper(x, self.w, self.U, self.S, self.V)

        if self.training and self.use_sm and self.w_old is not None:
            self._update_factors(x.detach().clone(), y.detach().clone())

        self.w_old = self.w.data.detach().clone()
            
        y = y.reshape(*shape[:-1], y.shape[-1])

        return y

    def _build(self, device=None):
        self.device = device
        
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(42)

        if self.use_sm:
            U, S, V = bb_appr(self.bb, self.d_inp, self.d_out,
                self.w.data.clone(), self.rank, self.log, self.nepman)
            self.register_buffer('U', U)
            self.register_buffer('S', S)
            self.register_buffer('V', V)
            self._debug_err()  
        else:
            self.register_buffer('U', None)
            self.register_buffer('S', None)
            self.register_buffer('V', None)

        self.bb_wrapper = bb_backprop_wrap(self.bb, self.generator,
            self.samples_sm, self.samples_bb, use_sm=self.use_sm)

    def _debug_err(self):
        # TODO: now it use the exact form of bb. We should remove it later
        with torch.no_grad():
            A = self.w.data.detach().clone().reshape(self.d_out, self.d_inp)
            A_appr = self.U @ self.S @ self.V
            err = torch.norm(A_appr - A) / torch.norm(A)
            self.log(f'... [DEBUG] Error A : {err:-12.5e}')
            if self.nepman:
                self.nepman['astralora/A_error'].append(err)

    def _update_factors(self, x, y):
        with torch.no_grad():
            w_old = self.w_old
            w_new = self.w.data.detach().clone()

            delta = torch.norm(w_new - w_old) / torch.norm(w_old)
            self.log(f'... [DEBUG] Delta w : {delta:-12.5e}')
            if self.nepman:
                self.nepman['astralora/w_delta'].append(delta)

            if delta < 1.E-12:
                return

            def f_old(X):
                return self.bb(X, w_old)

            def f_new(X):
                return self.bb(X, w_new)

            self.U, self.S, self.V = psi_implicit(f_old, f_new,
                self.U, self.S, self.V, self.samples_sm)

            self._debug_err()