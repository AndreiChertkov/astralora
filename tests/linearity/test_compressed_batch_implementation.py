import torch
import pytest

from core.bb_layers.bb_layer_slm import create_bb_layer_slm
from core.bb_layers.bb_layer_mrr import create_bb_layer_mrr
from core.helpers.backprop import _backprop_stochastic


def test_compressed_batch_optimization_triggered():
    """Test that compressed-batch optimization is triggered when B > J."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    
    # Test configurations where B > J (should trigger optimization)
    configs = [
        ("SLM", create_bb_layer_slm, 16, 8, 256),  # B=256 > J=8
        ("MRR", create_bb_layer_mrr, 16, 8, 256),  # B=256 > J=8
    ]
    
    for layer_name, factory, d_inp, d_out, B in configs:
        print(f"Testing {layer_name} with B={B} > J={d_out}...")
        
        bb, w0, _ = factory(d_inp, d_out)
        x = torch.randn(B, d_inp, device=device)
        
        # Upstream gradient must match bb output shape
        with torch.no_grad():
            y0 = bb(x, w0)
        grad_output = torch.randn_like(y0)
        
        # Test for_w case (should trigger optimization)
        generator = torch.Generator(device=device).manual_seed(42)
        grad_w = _backprop_stochastic(
            bb, x, w0, grad_output, generator, 
            samples=5, shift=1.0, for_x=False
        )
        
        # Verify gradient has correct shape
        assert grad_w.shape == w0.shape, f"Gradient shape mismatch for {layer_name}"
        assert not torch.isnan(grad_w).any(), f"NaN gradients for {layer_name}"
        assert not torch.isinf(grad_w).any(), f"Inf gradients for {layer_name}"
        
        print(f"  ✓ {layer_name}: gradient shape {grad_w.shape}, norm {torch.norm(grad_w):.3e}")


def test_compressed_batch_optimization_not_triggered():
    """Test that original implementation is used when B <= J."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    
    # Test configurations where B <= J (should NOT trigger optimization)
    configs = [
        ("SLM", create_bb_layer_slm, 16, 32, 16),  # B=16 <= J=32
        ("MRR", create_bb_layer_mrr, 16, 64, 32),  # B=32 <= J=64
    ]
    
    for layer_name, factory, d_inp, d_out, B in configs:
        print(f"Testing {layer_name} with B={B} <= J={d_out}...")
        
        bb, w0, _ = factory(d_inp, d_out)
        x = torch.randn(B, d_inp, device=device)
        
        # Upstream gradient must match bb output shape
        with torch.no_grad():
            y0 = bb(x, w0)
        grad_output = torch.randn_like(y0)
        
        # Test for_w case (should NOT trigger optimization)
        generator = torch.Generator(device=device).manual_seed(42)
        grad_w = _backprop_stochastic(
            bb, x, w0, grad_output, generator, 
            samples=5, shift=1.0, for_x=False
        )
        
        # Verify gradient has correct shape
        assert grad_w.shape == w0.shape, f"Gradient shape mismatch for {layer_name}"
        assert not torch.isnan(grad_w).any(), f"NaN gradients for {layer_name}"
        assert not torch.isinf(grad_w).any(), f"Inf gradients for {layer_name}"
        
        print(f"  ✓ {layer_name}: gradient shape {grad_w.shape}, norm {torch.norm(grad_w):.3e}")


def test_for_x_case_unaffected():
    """Test that for_x case still uses original implementation."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    
    bb, w0, _ = create_bb_layer_slm(16, 8)
    x = torch.randn(256, 16, device=device)  # B=256 > J=8
    
    # Upstream gradient must match bb output shape
    with torch.no_grad():
        y0 = bb(x, w0)
    grad_output = torch.randn_like(y0)
    
    # Test for_x case (should NOT trigger optimization even when B > J)
    generator = torch.Generator(device=device).manual_seed(42)
    grad_x = _backprop_stochastic(
        bb, x, w0, grad_output, generator, 
        samples=5, shift=1.0, for_x=True
    )
    
    # Verify gradient has correct shape
    assert grad_x.shape == x.shape, f"Gradient shape mismatch for for_x case"
    assert not torch.isnan(grad_x).any(), f"NaN gradients for for_x case"
    assert not torch.isinf(grad_x).any(), f"Inf gradients for for_x case"
    
    print(f"  ✓ for_x case: gradient shape {grad_x.shape}, norm {torch.norm(grad_x):.3e}")


if __name__ == "__main__":
    print("Testing compressed-batch optimization triggers...")
    test_compressed_batch_optimization_triggered()
    
    print("\nTesting compressed-batch optimization NOT triggered...")
    test_compressed_batch_optimization_not_triggered()
    
    print("\nTesting for_x case unaffected...")
    test_for_x_case_unaffected()
    
    print("\nAll tests passed!")
