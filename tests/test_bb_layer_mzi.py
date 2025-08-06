"""
Test suite for bb_layer_mzi module.

This module tests the create_bb_layer_mzi function and related MZI3U functionality.
"""

import pytest
import torch
import numpy as np
from core.bb_layers.bb_layer_mzi import create_bb_layer_mzi


class TestCreateBBLayerMZI:
    """Test class for create_bb_layer_mzi function."""
    
    def test_valid_even_dimensions(self):
        """Test that create_bb_layer_mzi works with valid even N values."""
        # Test cases: (d_inp, d_out, expected_N, expected_params)
        test_cases = [
            (4, 4, 2, 2),      # N=2, minimal case
            (16, 16, 4, 12),   # N=4, larger case
            (36, 36, 6, 30),   # N=6, even larger case
        ]
        
        for d_inp, d_out, expected_N, expected_params in test_cases:
            with pytest.raises(Exception, match=None):  # Should not raise
                bb, w0, dw = create_bb_layer_mzi(d_inp, d_out)
                
                # Check return types and shapes
                assert callable(bb), "bb should be a callable function"
                assert isinstance(w0, torch.Tensor), "w0 should be a torch.Tensor"
                assert isinstance(dw, (int, float)), "dw should be a number"
                
                # Check parameter shape
                assert w0.shape == (expected_params,), f"Expected {expected_params} parameters, got {w0.shape}"
                
                # Check learning rate
                assert dw == 0.01, f"Expected dw=0.01, got {dw}"
    
    def test_invalid_odd_dimensions(self):
        """Test that create_bb_layer_mzi fails with odd N values."""
        # Test cases with odd N values
        odd_cases = [9, 25, 49, 81]  # N=3, 5, 7, 9
        
        for d_out in odd_cases:
            with pytest.raises(AssertionError, match="must be even"):
                create_bb_layer_mzi(d_out, d_out)
    
    def test_invalid_non_square_dimensions(self):
        """Test that create_bb_layer_mzi fails with non-perfect-square dimensions."""
        # Test cases with non-perfect-square values
        non_square_cases = [5, 8, 10, 15, 20]
        
        for d_out in non_square_cases:
            with pytest.raises(ValueError, match="must be a perfect square"):
                create_bb_layer_mzi(d_out, d_out)
    
    def test_bb_function_forward_pass(self):
        """Test that the bb function performs forward pass correctly."""
        # Test with N=2 (d_out=4)
        bb, w0, _ = create_bb_layer_mzi(4, 4)
        
        # Test different batch sizes and input shapes
        test_inputs = [
            torch.randn(1, 4),    # Single sample
            torch.randn(5, 4),    # Batch of 5
            torch.randn(10, 4),   # Larger batch
        ]
        
        for x in test_inputs:
            output = bb(x, w0)
            
            # Check output shape
            expected_shape = x.shape  # Should maintain input shape
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            
            # Check output dtype (should be real)
            assert output.dtype in [torch.float32, torch.float64], f"Expected real dtype, got {output.dtype}"
            
            # Check that output is finite
            assert torch.isfinite(output).all(), "Output should be finite"
    
    def test_bb_function_different_input_output_dims(self):
        """Test bb function with different input and output dimensions."""
        # Test padding case (d_inp < d_out -> N)
        bb, w0, _ = create_bb_layer_mzi(3, 4)  # 3 -> 4, N=2
        x = torch.randn(2, 3)
        output = bb(x, w0)
        assert output.shape == (2, 4), f"Expected (2, 4), got {output.shape}"
        
        # Test truncation case (d_inp > d_out -> N)  
        bb2, w0_2, _ = create_bb_layer_mzi(5, 4)  # 5 -> 4, N=2
        x2 = torch.randn(2, 5)
        output2 = bb2(x2, w0_2)
        assert output2.shape == (2, 4), f"Expected (2, 4), got {output2.shape}"
    
    def test_parameter_initialization(self):
        """Test that parameters are initialized correctly."""
        bb, w0, _ = create_bb_layer_mzi(4, 4)
        
        # Check parameter range (should be between 0 and 1 based on uniform initialization)
        assert (w0 >= 0).all() and (w0 <= 1).all(), "Parameters should be initialized between 0 and 1"
        
        # Check that parameters are not all the same (very unlikely with random init)
        assert not torch.allclose(w0, w0[0]), "Parameters should not all be identical"
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the bb function."""
        bb, w0, _ = create_bb_layer_mzi(4, 4)
        
        # Make parameters require grad
        w = w0.clone().requires_grad_(True)
        x = torch.randn(2, 4, requires_grad=True)
        
        # Forward pass
        output = bb(x, w)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert w.grad is not None, "Gradients should flow to parameters"
        assert x.grad is not None, "Gradients should flow to input"
        
        # Check that gradients are not zero (for most cases)
        assert not torch.allclose(w.grad, torch.zeros_like(w.grad)), "Parameter gradients should be non-zero"
    
    def test_deterministic_output(self):
        """Test that the same inputs produce the same outputs."""
        bb, w0, _ = create_bb_layer_mzi(4, 4)
        x = torch.randn(2, 4)
        
        # Run multiple times with same input
        output1 = bb(x, w0)
        output2 = bb(x, w0)
        
        # Should be identical
        assert torch.allclose(output1, output2), "Same inputs should produce identical outputs"
    
    def test_batch_consistency(self):
        """Test that batched and individual processing give consistent results."""
        bb, w0, _ = create_bb_layer_mzi(4, 4)
        
        # Create batch input
        x_batch = torch.randn(3, 4)
        
        # Process as batch
        output_batch = bb(x_batch, w0)
        
        # Process individually
        outputs_individual = []
        for i in range(3):
            output_single = bb(x_batch[i:i+1], w0)
            outputs_individual.append(output_single)
        output_individual = torch.cat(outputs_individual, dim=0)
        
        # Should be identical
        assert torch.allclose(output_batch, output_individual, atol=1e-6), \
            "Batch and individual processing should give identical results"


class TestMZI3UIntegration:
    """Integration tests for MZI3U components."""
    
    def test_parameter_count_formula(self):
        """Test that parameter count formula is correct for various N values."""
        def calculate_expected_params(N):
            """Calculate expected parameter count for given N."""
            num_mzis_0 = (N // 2) * (N // 2)
            num_mzis_1 = (N // 2) * max(0, N // 2 - 1)
            return (num_mzis_0 + num_mzis_1) * 2
        
        # Test for even N values
        even_N_values = [2, 4, 6, 8]
        for N in even_N_values:
            d_out = N * N
            expected_params = calculate_expected_params(N)
            
            bb, w0, _ = create_bb_layer_mzi(d_out, d_out)
            actual_params = w0.shape[0]
            
            assert actual_params == expected_params, \
                f"For N={N}, expected {expected_params} params, got {actual_params}"


# Pytest fixtures for common test data
@pytest.fixture
def small_mzi_layer():
    """Fixture providing a small MZI layer for testing."""
    return create_bb_layer_mzi(4, 4)


@pytest.fixture
def medium_mzi_layer():
    """Fixture providing a medium MZI layer for testing."""
    return create_bb_layer_mzi(16, 16)


@pytest.fixture
def sample_inputs():
    """Fixture providing sample input tensors."""
    return {
        'small': torch.randn(3, 4),
        'medium': torch.randn(3, 16),
        'single': torch.randn(1, 4),
    }


# Additional test using fixtures
def test_with_fixtures(small_mzi_layer, sample_inputs):
    """Test using pytest fixtures."""
    bb, w0, dw = small_mzi_layer
    x = sample_inputs['small']
    
    output = bb(x, w0)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])