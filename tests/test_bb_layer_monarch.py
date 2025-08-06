"""
Test suite for bb_layer_monarch module.

This module tests the create_bb_layer_monarch function and related Monarch layer functionality.
"""

import pytest
import torch
import numpy as np
from core.bb_layers.bb_layer_monarch import create_bb_layer_monarch


class TestCreateBBLayerMonarch:
    """Test class for create_bb_layer_monarch function."""
    
    def test_same_dimensions_power_of_2(self):
        """Test monarch layer with same dimensions that are powers of 2."""
        test_cases = [
            (8, 8),
            (16, 16),
            (32, 32),
        ]
        
        for d_inp, d_out in test_cases:
            bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
            
            # Check return types
            assert callable(bb), "bb should be a callable function"
            assert isinstance(w, torch.Tensor), "w should be a torch.Tensor"
            assert isinstance(dw, (int, float)), "dw should be a number"
            
            # Test forward pass
            batch_size = 3
            x = torch.randn(batch_size, d_inp)
            output = bb(x, w)
            
            # Check output shape
            assert output.shape == (batch_size, d_out), f"Expected shape ({batch_size}, {d_out}), got {output.shape}"
    
    def test_different_dimensions_power_of_2(self):
        """Test monarch layer with different dimensions that are powers of 2."""
        test_cases = [
            (8, 16),
            (16, 8),
            (4, 32),
            (32, 4),
        ]
        
        for d_inp, d_out in test_cases:
            bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
            
            # Check return types
            assert callable(bb), "bb should be a callable function"
            assert isinstance(w, torch.Tensor), "w should be a torch.Tensor"
            assert isinstance(dw, (int, float)), "dw should be a number"
            
            # Test forward pass
            batch_size = 3
            x = torch.randn(batch_size, d_inp)
            output = bb(x, w)
            
            # Check output shape
            assert output.shape == (batch_size, d_out), f"Expected shape ({batch_size}, {d_out}), got {output.shape}"
    
    def test_non_power_of_2_dimensions(self):
        """Test monarch layer with dimensions that are not powers of 2."""
        test_cases = [
            (10, 12),
            (12, 10),
            (6, 9),
            (15, 20),
        ]
        
        for d_inp, d_out in test_cases:
            bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
            
            # Check return types
            assert callable(bb), "bb should be a callable function"
            assert isinstance(w, torch.Tensor), "w should be a torch.Tensor"
            assert isinstance(dw, (int, float)), "dw should be a number"
            
            # Test forward pass
            batch_size = 3
            x = torch.randn(batch_size, d_inp)
            output = bb(x, w)
            
            # Check output shape
            assert output.shape == (batch_size, d_out), f"Expected shape ({batch_size}, {d_out}), got {output.shape}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through the monarch layer properly."""
        d_inp, d_out = 8, 16
        bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
        
        # Enable gradients
        w.requires_grad_(True)
        
        # Create input with gradients
        x = torch.randn(2, d_inp, requires_grad=True)
        
        # Forward pass
        output = bb(x, w)
        
        # Compute a simple loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert w.grad is not None, "Parameters should have gradients"
        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(w.grad, torch.zeros_like(w.grad)), "Parameter gradients should be non-zero"
    
    def test_parameter_count(self):
        """Test that the parameter count is reasonable for different dimensions."""
        test_cases = [
            (8, 8),
            (16, 16),
            (8, 16),
            (10, 12),
        ]
        
        for d_inp, d_out in test_cases:
            bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
            
            # Parameter count should be reasonable (not too large)
            max_reasonable_params = max(d_inp, d_out) ** 2
            assert w.numel() <= max_reasonable_params, f"Too many parameters: {w.numel()} > {max_reasonable_params}"
            
            # Should have some parameters
            assert w.numel() > 0, "Should have at least some parameters"
    
    def test_deterministic_output(self):
        """Test that the monarch layer produces deterministic output for the same input and weights."""
        d_inp, d_out = 8, 8
        bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
        
        # Create input
        x = torch.randn(2, d_inp)
        
        # Run forward pass twice with the same weights
        output1 = bb(x, w)
        output2 = bb(x, w)
        
        # Should produce identical outputs
        assert torch.allclose(output1, output2), "Output should be deterministic"
    
    def test_batch_processing(self):
        """Test that the monarch layer handles different batch sizes correctly."""
        d_inp, d_out = 16, 8
        bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
        
        batch_sizes = [1, 3, 5, 10]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, d_inp)
            output = bb(x, w)
            
            assert output.shape == (batch_size, d_out), f"Failed for batch size {batch_size}"


if __name__ == "__main__":
    # Run basic tests when executed directly
    test_monarch = TestCreateBBLayerMonarch()
    
    print("Running basic monarch layer tests...")
    
    # Test cases from the original snippet
    test_cases = [
        (8, 8),    # Same dimensions (power of 2)
        (16, 16),  # Same dimensions (power of 2)
        (8, 16),   # Different dimensions (both powers of 2)
        (16, 8),   # Different dimensions (both powers of 2)
        (10, 12),  # Different dimensions (not powers of 2)
        (12, 10),  # Different dimensions (not powers of 2)
    ]
    
    for d_inp, d_out in test_cases:
        print(f"\nTesting monarch layer: d_inp={d_inp}, d_out={d_out}")
        
        # Create layer
        bb, w, dw = create_bb_layer_monarch(d_inp, d_out)
        
        # Create test input
        batch_size = 3
        x = torch.randn(batch_size, d_inp)
        
        # Forward pass
        try:
            output = bb(x, w)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: ({batch_size}, {d_out})")
            print(f"  ✓ Test passed!")
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
    
    print("\nAll tests completed!")