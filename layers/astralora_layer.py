import math
import torch
from torch.autograd import Function
import torch.nn as nn


class AstraloraLayer(nn.Module):
    def __init__(self, d_inp, d_out, d=None, kind='matvec', rank=1,
                 samples_bb=100, samples_sm=100, log=print, nepman=None):
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

        self.log('... [DEBUG] Building Astralora layer : ' + self.extra_repr())

        self._init_bb()
        self._init_factors()
        
    def extra_repr(self):
        text = ''
        text += f'd_inp={self.d_inp}, '
        text += f'd_out={self.d_out}, '
        text += f'd={self.d}, '
        text += f'kind={self.kind}, '
        text += f'rank={self.rank}, '
        text += f'samples_bb={self.samples_bb}, '
        text += f'samples_sm={self.samples_sm}'
        return text

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        y = LowRankGradientFunction.apply(x, self.A, self.U, self.S, self.V)

        if self.training:
            self._update_factors(x.detach().clone(), y.detach().clone())

        self.A_old = self.A.detach().clone()
            
        y = y.reshape(*shape[:-1], y.shape[-1])
        
        return y

    def _debug_err(self):
        with torch.no_grad():
            # TODO: fix it:
            A = self.w.data.clone().reshape(self.d_out, self.d_inp)
            A_approx = self.U @ self.S @ self.V
            err = torch.norm(A - A_approx) / torch.norm(A)
            self.log(f'... [DEBUG] Error A : {err:-12.5e}')
            if self.nepman:
                self.nepman['astralora/A_error'].append(err)

    def _init_bb(self):
        self.w = nn.Parameter(torch.Tensor(self.d))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        self.bb = bb_build(self.d_inp, self.d_out, self.d, self.kind)

        #self.A = nn.Parameter(torch.Tensor(self.d_out, self.d_inp))
        #self.A_old = self.A.detach().clone()

    def _init_factors(self):
        w = self.w.data.clone()
        U, S, V = bb_appr(self.bb, self.d_inp, self.d_out, w, self.rank)
        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('V', V)
        self._debug_err()            

    def _update_factors(self, x, y):
        with torch.no_grad():

            A_old = self.A_old.data.clone()
            A_new = self.A.data.clone()

            delta = torch.norm(A_new - A_old) / torch.norm(A_old)
            self.log(f'... [DEBUG] Delta A : {delta:-12.5e}')
            if self.nepman:
                self.nepman['astralora/A_delta'].append(delta)

            if delta < 1.E-12:
                return

            def f_old(x):
                return x @ A_old.T

            def f_new(x):
                return x @ A_new.T

            self.U, self.S, self.V = psi_implicit(f_old, f_new,
                self.U, self.S, self.V.T, self.samples_sm)
            self.V = self.V.T

            self._debug_err()


class LowRankGradientFunction(Function):
    @staticmethod
    def forward(ctx, x, A, U, S, V):
        ctx.save_for_backward(x, A, U, S, V)
        return x @ A.t()

    @staticmethod
    def backward(ctx, grad_output):
        x, A, U, S, V = ctx.saved_tensors
        grad_A = grad_output.t() @ x
        grad_x = grad_output @ U @ S @ V
        return grad_x, grad_A, None, None, None


def bb_appr(bb, d_inp, d_out, w, rank=10):
    # TODO: add real bb-approximation here
    A = w.reshape(d_out, d_inp)
    
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    U = U[:, :rank]
    S = torch.diag(torch.sqrt(s[:rank]))
    V = V[:rank, :]

    return U, S, V


def bb_build(d_inp, d_out, d, kind='matvec'):
    if kind == 'matvec':
        assert d_inp * d_out == d
    
        def bb(x, w):
            A = w.reshape(d_out, d_inp)
            return x @ A.T

    else:
        raise NotImplementedError
    
    return bb


def psi_implicit(f_old, f_new, U0, S0, V0, samples=100):
    """A projector-splitting integrator (PSI) for dynamical low-rank appr."""
    def compute_P1(f_new, f_old, V0): # Compute dA @ V0
        V0_batch = V0.T
        res_old = f_old(V0_batch)
        res_new = f_new(V0_batch)
        return (res_new - res_old).T

    P1 = compute_P1(f_new, f_old, V0)

    K1 = U0 @ S0 + P1
    U1, S0_tld = torch.linalg.qr(K1, mode='reduced')
    S0_hat = S0_tld - U1.T @ P1

    def compute_P2(f_new, f_old, U1, d_inp, samples): # Compute dA^t @ U1
        Z = torch.randn(samples, d_inp, device=U1.device, dtype=U1.dtype)
        AZ = f_new(Z) - f_old(Z)
        AZT_U = AZ @ U1
        return Z.T @ AZT_U / samples

    d_inp = V0.shape[0]
    P2 = compute_P2(f_new, f_old, U1, d_inp, samples)

    L1 = V0 @ S0_hat.T + P2

    V1, S1_T = torch.linalg.qr(L1, mode='reduced')
    S1 = S1_T.T

    return U1, S1, V1