import torch
import numpy as np
from layers.helpers.astralora_backprop import _backprop_stochastic
from tqdm import tqdm

DEVICE_ID = 2

def test_backprop_accuracy(device=None):
    # Set device
    if device is None:
        device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Running test on: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a simple linear function for testing
    def linear_func(x, w):
        # Reshape w back to 2D for the linear operation
        w_2d = w.reshape(input_dim, output_dim)
        return x @ w_2d
    
    # Generate random input data
    batch_size = 16000
    input_dim = 10
    output_dim = 10
    
    x = torch.randn(batch_size, input_dim, device=device, requires_grad=True)
    w = torch.randn(input_dim, output_dim, device=device, requires_grad=True)
    w_flat = w.reshape(-1)  # Flatten weights for stochastic backprop
    
    # Forward pass
    y = linear_func(x, w)
    
    # Generate random gradient
    grad_output = torch.randn_like(y)
    
    # Compute standard backprop gradients
    y.backward(grad_output)
    standard_grad_x = x.grad.clone()
    standard_grad_w = w.grad.clone()
    
    # Reset gradients
    x.grad = None
    w.grad = None
    
    # Test different numbers of samples
    sample_counts = [1, 10, 100, 1000, 10000]
    errors_x = []
    errors_w = []
    
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    for samples in tqdm(sample_counts, desc="Testing sample counts"):
        # Compute stochastic gradients
        stochastic_grad_x = _backprop_stochastic(
            linear_func, x, w_flat, grad_output, generator, samples=samples, for_x=True
        )
        stochastic_grad_w = _backprop_stochastic(
            linear_func, x, w_flat, grad_output, generator, samples=samples, for_x=False
        )

        
        # Reshape weight gradient back to original shape
        stochastic_grad_w = stochastic_grad_w.reshape(input_dim, output_dim)
        
        print(stochastic_grad_x.shape, standard_grad_x.shape, 'shapes of x [stochastic and standard]')
        print(stochastic_grad_w.shape, standard_grad_w.shape, 'shapes of w [stochastic and standard]')
        print(stochastic_grad_x.norm(), standard_grad_x.norm(), 'norms of x [stochastic and standard]')
        print(stochastic_grad_w.norm(), standard_grad_w.norm(), 'norms of w [stochastic and standard]')
        # Calculate relative errors
        error_x = torch.norm(stochastic_grad_x - standard_grad_x) / torch.norm(standard_grad_x)
        error_w = torch.norm(stochastic_grad_w - standard_grad_w) / torch.norm(standard_grad_w)
        
        errors_x.append(error_x.item())
        errors_w.append(error_w.item())
        
        print(f"\nSamples: {samples}")
        print(f"X gradient error: {error_x:.6f}")
        print(f"W gradient error: {error_w:.6f}")
        print("-" * 50)
    
    return sample_counts, errors_x, errors_w

if __name__ == "__main__":
    # You can specify device explicitly if needed
    # device = torch.device('cuda:0')  # for specific GPU
    # device = torch.device('cpu')     # for CPU
    sample_counts, errors_x, errors_w = test_backprop_accuracy()
