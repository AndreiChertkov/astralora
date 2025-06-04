import math
import torch
from torch.autograd import Function
import torch.nn as nn


class AstraloraLayer(nn.Module):
    def __init__(self, d_inp, d_out, rank=1, lr=0.01, log=print):
        super().__init__()
        
        self.d_inp = d_inp
        self.d_out = d_out
        self.rank = rank
        self.lr = lr
        self.log = log

        self.log('... [DEBUG] Building Astralora layer : ' + self.extra_repr())

        self._init_bb()
        self._init_factors()
        
    def extra_repr(self):
        text = ''
        text += f'd_inp={self.d_inp}, '
        text += f'd_out={self.d_out}, '
        text += f'rank={self.rank}, '
        text += f'lr={self.lr}'
        return text

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        y = LowRankGradientFunction.apply(x, self.A, self.U, self.V)

        if self.training:
            self._update_factors(x.detach().clone(), y.detach().clone())
            
        y = y.reshape(*shape[:-1], y.shape[-1])
        
        return y
    
    def _init_bb(self):
        self.A = nn.Parameter(torch.Tensor(self.d_out, self.d_inp))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def _init_factors(self):
        with torch.no_grad():
            A = self.A.data.clone()
            
            U, s, V = torch.linalg.svd(A, full_matrices=False)
            U = U[:, :self.rank]
            S = torch.diag(torch.sqrt(s[:self.rank]))
            V = V[:self.rank, :]

            self.register_buffer('U', U @ S)
            self.register_buffer('V', S @ V)

            A_approx = self.U @ self.V
            err = torch.norm(A - A_approx) / torch.norm(A)
            self.log(f'... [DEBUG] Error : {err:-12.5e} (init)')

    def _update_factors(self, x, y):
        with torch.no_grad():
            batch_size = x.size(0)

            A_appr = self.U @ self.V
            y_pred = x @ A_appr.t()
            
            e = y_pred - y
            
            grad_A_appr = e.t() @ x
            grad_A_appr /= (batch_size * self.d_out)
            
            grad_U = grad_A_appr @ self.V.t()
            grad_V = self.U.t() @ grad_A_appr
            
            self.U = self.U - self.lr * grad_U
            self.V = self.V - self.lr * grad_V

            A = self.A.data.clone()
            A_appr = self.U @ self.V
            err = torch.norm(A - A_appr) / torch.norm(A)
            n1 = torch.norm(grad_U)
            n2 = torch.norm(grad_V)

            text = f'... [DEBUG] Error : {err:-12.5e}'
            text += f' Grad norms : {n1:-8.1e}, {n2:-8.1e}'
            self.log(text)


class LowRankGradientFunction(Function):
    @staticmethod
    def forward(ctx, x, A, U, V):
        ctx.save_for_backward(x, A, U, V)
        return x @ A.t()

    @staticmethod
    def backward(ctx, grad_output):
        x, A, U, V = ctx.saved_tensors
        
        grad_A = grad_output.t() @ x
        
        A_appr = U @ V
        grad_x = grad_output @ A_appr
        
        return grad_x, grad_A, None, None