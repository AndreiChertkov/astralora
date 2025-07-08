import torch
import torch.nn as nn
import torch.nn.functional as F


from .helpers.approximation import approximation
from .helpers.approximation import bb_appr_w_svd
from .helpers.backprop import backprop_wrap
from .helpers.psi import psi_implicit
from .bb_layers.bb_layer_id import create_bb_layer_id
from .bb_layers.bb_layer_matvec import create_bb_layer_matvec
from .bb_layers.bb_layer_monarch import create_bb_layer_monarch
from .bb_layers.bb_layer_mrr import create_bb_layer_mrr
from .bb_layers.bb_layer_mzi import create_bb_layer_mzi
from .bb_layers.bb_layer_slm import create_bb_layer_slm


class AstraloraLayer(nn.Module):
    def __init__(self, d_inp, d_out, kind, rank, samples_bb, samples_sm, 
                 skip_sm, use_residual, log=print, nepman=None):
        super().__init__()
        
        self.d_inp = d_inp
        self.d_out = d_out
        self.kind = kind
        self.rank = rank
        self.samples_bb = samples_bb
        self.samples_sm = samples_sm
        self.skip_sm = skip_sm
        self.use_residual = use_residual
        self.log = log
        self.nepman = nepman

        self.log('... Init Astralora layer : \n     ' + self.extra_repr())

        if self.kind == 'id':
            self.bb, w0, self.dw = create_bb_layer_id(
                self.d_inp, self.d_out)
        elif self.kind == 'matvec':
            self.bb, w0, self.dw = create_bb_layer_matvec(
                self.d_inp, self.d_out)
        elif self.kind == 'monarch':
            self.bb, w0, self.dw = create_bb_layer_monarch(
                self.d_inp, self.d_out)
        elif self.kind == 'mrr':
            self.bb, w0, self.dw = create_bb_layer_mrr(
                self.d_inp, self.d_out)
        elif self.kind == 'mzi':
            self.bb, w0, self.dw = create_bb_layer_mzi(
                self.d_inp, self.d_out)
        elif self.kind == 'slm':
            self.bb, w0, self.dw = create_bb_layer_slm(
                self.d_inp, self.d_out)
        else:
            raise NotImplementedError
        
        self.w = nn.Parameter(w0)
        self.scale = nn.Parameter(torch.ones(1))
        self.w.ast_bb = True
        self.w_old = None

        self.device = None
        self.bb_wrapper = None
      
    def extra_repr(self):
        text = ''
        text += f'd_inp={self.d_inp}, '
        text += f'd_out={self.d_out}, '
        text += f'kind={self.kind}, '
        text += f'rank={self.rank}, '
        text += f'samples_bb={self.samples_bb}, '
        text += f'samples_sm={self.samples_sm}'
        if self.skip_sm:
            text += f', skip_sm={self.skip_sm}'
        if self.use_residual:
            text += f', use_residual={self.use_residual}'
        return text

    def forward(self, x):
        if self.bb_wrapper is None:
            self._build(device=x.device)

        shape = x.shape
        x = x.reshape(-1, shape[-1])

        y = self.bb_wrapper(x, self.w, self.U, self.S, self.V)

        if self.training and self.w_old is not None and not self.skip_sm:
            self._update_factors(x.detach().clone(), y.detach().clone())

        self.w_old = self.w.data.detach().clone()
            
        y = y.reshape(*shape[:-1], y.shape[-1])
        y = y * self.scale
        if self.use_residual:
            y = y + self._add_residual(x, y)

        return y

    def _add_residual(self, x, y):
        x_ch = x.shape[1]
        y_ch = y.shape[1]

        if x_ch == y_ch:
            residual = x
 
        elif x_ch <= y_ch:
            residual = torch.zeros_like(y)
            residual[:, :x_ch] = x
        
        else:
            residual = x[:, :y_ch]

        return residual

    def _build(self, device=None):
        self.device = device
        
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(42)

        U, S, V = None, None, None
        if not self.skip_sm:
            U, S, V = approximation(self.bb, self.d_inp, self.d_out,
                self.w.data.clone(), self.rank, self.log, self.nepman)
        self._set_factors(U, S, V, init=True)

        self.bb_wrapper = backprop_wrap(self.bb, self.generator,
            self.samples_bb, self.dw, self.skip_sm)

    def _debug_err(self):
        # TODO: now it use the exact form of bb. We should remove it later
        # (it works ok only for linear layer)
        with torch.no_grad():
            A = self.w.data.detach().clone().reshape(self.d_out, self.d_inp)
            A_appr = self.U @ self.S @ self.V
            err = torch.norm(A_appr - A) / torch.norm(A)
            self.log(f'... [DEBUG] Error A : {err:-12.5e}')
            if self.nepman:
                self.nepman['astralora/A_error'].append(err)

    def _set_factors(self, U=None, S=None, V=None, init=False):
        if init:
            self.register_buffer('U', U)
            self.register_buffer('S', S)
            self.register_buffer('V', V)
        else:
            self.U = U # .data.copy_(U) TODO: check this
            self.S = S # .data.copy_(S)
            self.V = V # .data.copy_(V)

    def _update_factors(self, x, y, thr=1.E-12):
        with torch.no_grad():
            w_old = self.w_old
            w_new = self.w.data.detach().clone()

            delta = torch.norm(w_new - w_old) / torch.norm(w_old)
            # self.log(f'... [DEBUG] Delta w : {delta:-12.5e}')
            if self.nepman:
                self.nepman['astralora/w_delta'].append(delta)

            if delta < thr:
                return

            if self.samples_sm == -1: # "Exact" update:
                self._set_factors(*bb_appr_w_svd(self.bb,
                    self.d_inp, self.d_out, w_new, self.rank))
                return

            if self.samples_sm < 1:
                raise ValueError('Invalid number of samples to update SM')

            def f_old(X):
                return self.bb(X, w_old)

            def f_new(X):
                return self.bb(X, w_new)

            self._set_factors(*psi_implicit(f_old, f_new,
                self.U, self.S, self.V, self.samples_sm))