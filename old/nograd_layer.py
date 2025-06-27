import torch
import torch.nn as nn
from torch.autograd import Function
import math


class NoGradLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Save original shape and reshape for linear operation
        ctx.save_for_backward(x, weight)
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        return torch.matmul(x_2d, weight.t()).reshape(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # Reshape for gradient computation
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        
        # Compute gradient only for weight, not for input
        grad_weight = torch.matmul(grad_output_2d.t(), x_2d)
        
        # Return zero gradient for input to maintain DDP compatibility
        grad_x = torch.zeros_like(x)
        return grad_x, grad_weight


class NoGradLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        return NoGradLinearFunction.apply(x, self.weight)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}' 