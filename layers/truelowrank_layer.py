import torch
import torch.nn as nn
from torch.autograd import Function
import math


class TrueLowRankGradientFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, rank):
        # Save original shape and reshape for linear operation
        ctx.save_for_backward(x, weight)
        ctx.rank = rank
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        return torch.matmul(x_2d, weight.t()).reshape(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        rank = ctx.rank
        
        # Reshape for gradient computation
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        
        # Compute low-rank decomposition on-the-fly
        U, s, V = torch.linalg.svd(weight, full_matrices=False)
        
        # Compute the low-rank approximation
        weight_lowrank = torch.matmul(U[:, :rank] * s[:rank].unsqueeze(0), V[:rank, :])
        
        # Compute gradient through the low-rank approximation
        grad_x = torch.matmul(grad_output_2d, weight_lowrank).reshape(shape)
        
        # Compute gradient for the weight matrix
        grad_weight = torch.matmul(grad_output_2d.t(), x_2d)
        
        return grad_x, grad_weight, None


class TrueLowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Set default rank if not provided
        if rank is None:
            rank = min(in_features, out_features) // 2
        self.rank = rank
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        return TrueLowRankGradientFunction.apply(x, self.weight, self.rank)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}' 