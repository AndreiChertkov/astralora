import math
import torch
from torch.autograd import Function
import torch.nn as nn


class AstraloraGDLayer(nn.Module):
    def __init__(self, d_inp, d_out, rank=1, surrogate_lr=0.01, log=print):
        super().__init__()
        
        self.d_inp = d_inp
        self.d_out = d_out
        self.rank = rank
        self.surrogate_lr = surrogate_lr
        self.lr = surrogate_lr
        self.log = log

        self.clip_grad_norm = 1.
        self.reg_lambda = 1.E-4
        self.stable_update_count = 0

        self.log('... [DEBUG] Building AstraloraGD layer : ' + self.extra_repr())

        self._init_bb()
        self._init_factors()
        
    def extra_repr(self):
        text = ''
        text += f'd_inp={self.d_inp}, '
        text += f'd_out={self.d_out}, '
        text += f'rank={self.rank}, '
        text += f'surrogate_lr={self.surrogate_lr}'
        return text

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        y = LowRankGradientFunction.apply(x, self.A, self.U, self.V)

        if self.training:
            self._update_factors(x.detach().clone(), y.detach().clone())
            
        y = y.reshape(*shape[:-1], y.shape[-1])
        
        return y
    
    def reset_learning_rate(self):
        self.lr = self.base_lr
        self.stable_update_count = 0

    def _adjust_learning_rate(self, stable):
        if stable:
            self.stable_update_count += 1
            if self.stable_update_count >= 10:
                self.lr = min(self.base_lr * 1.1, self.base_lr * 5)
        else:
            self.stable_update_count = 0
            self.lr = max(self.lr * 0.5, self.base_lr * 0.01)
            # print(f"LR is reduced to {self.lr:.6f}")

    def _check_stability(self, tensor, name=""):
        if torch.isnan(tensor).any():
            print(f"Found NaN in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"Found Inf in {name}")
            return False
        if tensor.abs().max() > 1e6:
            print(f"Found big value in {name} (max={tensor.abs().max():.2f})")
            return False
        return True

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

            reg_term = self.reg_lambda * A_appr
            
            e = y_pred - y
            
            grad_A_appr = e.t() @ x + reg_term
            grad_A_appr /= (batch_size * self.d_out)
            
            grad_U = grad_A_appr @ self.V.t()
            grad_V = self.U.t() @ grad_A_appr
            
            if self.clip_grad_norm > 0:
                grad_U_norm = torch.norm(grad_U)
                if grad_U_norm > self.clip_grad_norm:
                    grad_U = grad_U * (self.clip_grad_norm / grad_U_norm)

                grad_V_norm = torch.norm(grad_V)
                if grad_V_norm > self.clip_grad_norm:
                    grad_V = grad_V * (self.clip_grad_norm / grad_V_norm)


            stable = True
            stable &= self._check_stability(grad_U, "grad_U")
            stable &= self._check_stability(grad_V, "grad_V")

            self._adjust_learning_rate(stable)
        
            if stable:
                self.U = self.U - self.lr * grad_U
                self.V = self.V - self.lr * grad_V

            self.U.clamp_(-1.E5, 1.E5)
            self.V.clamp_(-1.E5, 1.E5)
            
            A = self.A.data.clone()
            A_appr = self.U @ self.V
            err = torch.norm(A - A_appr) / torch.norm(A)
            n1 = torch.norm(grad_U)
            n2 = torch.norm(grad_V)

            text = f'... [DEBUG] Error : {err:-12.5e}'
            text += f' Grad norms : {n1:-8.1e}, {n2:-8.1e} | stable: {stable}'
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
        grad_x = grad_output @ U @ V
        return grad_x, grad_A, None, None