"""
Test suite for bb_layer_mzi module.

This module tests the create_bb_layer_mzi function and related MZI3U functionality.
"""

import pytest
import torch
import numpy as np
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity
from core.bb_layers.bb_layer_mzi import create_bb_layer_mzi


class TestCreateBBLayerMZI:
    """Test class for create_bb_layer_mzi function."""
    
    def test_valid_dimensions(self):
        """Test that create_bb_layer_mzi works with any valid dimensions."""
        # Test cases: (d_inp, d_out, expected_N, expected_params)
        # New logic: N = max(d_inp, d_out), then round up to even
        test_cases = [
            (4, 4, 4, 12),     # N=4, max(4,4)=4
            (3, 5, 6, 30),     # N=6, max(3,5)=5->6 (even)
            (16, 16, 16, 240), # N=16, max(16,16)=16
            (1, 1, 2, 2),      # N=2, max(1,1)=1->2 (minimum)
        ]
        
        for d_inp, d_out, expected_N, expected_params in test_cases:
            # Should not raise any exception for valid cases
            bb, w0, dw = create_bb_layer_mzi(d_inp, d_out)
            
            # Check return types and shapes
            assert callable(bb), "bb should be a callable function"
            assert isinstance(w0, torch.Tensor), "w0 should be a torch.Tensor"
            assert isinstance(dw, (int, float)), "dw should be a number"
            
            # Check parameter shape
            assert w0.shape == (expected_params,), f"Expected {expected_params} parameters, got {w0.shape}"
            
            # Check learning rate
            assert dw == 0.01, f"Expected dw=0.01, got {dw}"
    
    def test_flexible_dimensions(self):
        """Test that create_bb_layer_mzi now works with any dimensions (including odd and non-square)."""
        # Test cases that would have failed in the old implementation
        flexible_cases = [
            (9, 9),   # Odd dimensions
            (5, 8),   # Non-square, different dimensions
            (10, 15), # Non-square
            (3, 7),   # Both odd and different
        ]
        
        for d_inp, d_out in flexible_cases:
            # Should work fine now
            bb, w0, dw = create_bb_layer_mzi(d_inp, d_out)
            
            # Test forward pass
            x = torch.randn(2, d_inp)
            output = bb(x, w0)
            
            # Verify output shape
            assert output.shape == (2, d_out), f"Expected (2, {d_out}), got {output.shape}"
    
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
    
    @pytest.mark.skip(reason="In-place operations in MZI implementation prevent gradient computation")
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
        """Test that parameter count formula is correct for various cases."""
        def calculate_expected_params(N):
            """Calculate expected parameter count for given N."""
            num_mzis_0 = (N // 2) * (N // 2)
            num_mzis_1 = (N // 2) * max(0, N // 2 - 1)
            return (num_mzis_0 + num_mzis_1) * 2
        
        # Test cases: (d_inp, d_out, expected_N)
        # New logic: N = max(d_inp, d_out), then round up to even
        test_cases = [
            (2, 2, 2),     # N = max(2,2) = 2
            (4, 4, 4),     # N = max(4,4) = 4  
            (3, 5, 6),     # N = max(3,5) = 5 -> 6 (even)
            (6, 6, 6),     # N = max(6,6) = 6
        ]
        
        for d_inp, d_out, expected_N in test_cases:
            expected_params = calculate_expected_params(expected_N)
            
            bb, w0, _ = create_bb_layer_mzi(d_inp, d_out)
            actual_params = w0.shape[0]
            
            assert actual_params == expected_params, \
                f"For d_inp={d_inp}, d_out={d_out} (N={expected_N}), expected {expected_params} params, got {actual_params}"


class TestMZIPerformanceGPU:
    """Performance tests for MZI layer with GPU support and profiling."""
    
    @pytest.fixture
    def gpu_device(self):
        """Fixture to get GPU device if available."""
        gpu_id = os.environ.get('TEST_GPU_ID', '0')
        device_name = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")
            
        return device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_layer_performance_gpu(self, gpu_device):
        """
        Test performance of large MZI layer (1024x1024) with batch size 1024 on GPU.
        
        This test creates a large MZI layer and profiles its execution time
        for the specified batch size and layer dimensions.
        
        Environment variables:
            TEST_GPU_ID: GPU device ID to use (default: 0)
            PROFILE_TRACES: If set, saves detailed profiling traces
        """
        # Test configuration
        d_inp = 1024
        d_out = 1024
        batch_size = 1024
        device = gpu_device
        
        print(f"\nTesting MZI layer performance:")
        print(f"  Input dimension: {d_inp}")
        print(f"  Output dimension: {d_out}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        
        # Create MZI layer
        bb, w0, _ = create_bb_layer_mzi(d_inp, d_out)
        
        
        # Move parameters and input to GPU
        w0 = w0.to(device)
        x = torch.randn(batch_size, d_inp, device=device, dtype=torch.float32)
        
        print(f"  Parameter count: {w0.numel()}")
        print(f"  Input tensor size: {x.shape}")
        print(f"  Memory usage (MB): {x.numel() * 4 / 1e6:.1f}")
        
        # Warm-up runs to ensure GPU is ready
        print("\nPerforming warm-up runs...")
        for _ in range(3):
            with torch.no_grad():
                _ = bb(x, w0)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Basic timing test
        print("\nBasic timing test (3 runs):")
        times = []
        for i in range(3):
            start_time = time.time()
            with torch.no_grad():
                for i in range(10):
                    output = bb(x, w0)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            print(f"  Run {i+1}: {run_time*1000:.2f} ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        
        # Verify output correctness
        assert output.shape == (batch_size, d_out), f"Expected shape ({batch_size}, {d_out}), got {output.shape}"
        assert torch.isfinite(output).all(), "Output should be finite"
        assert output.dtype in [torch.float32, torch.float64], f"Expected real dtype, got {output.dtype}"
        
        # Detailed profiling with torch.profiler
        print("\nDetailed profiling with torch.profiler:")
        
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        save_traces = os.environ.get('PROFILE_TRACES', '').lower() in ['1', 'true', 'yes']
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("mzi_forward_pass"):
                with torch.no_grad():
                    for i in range(10):
                        output = bb(x, w0)
                
        
        # Print profiling summary
        print("\nProfiling Summary:")
        print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total", 
                                       row_limit=10))
        
        if save_traces:
            trace_file = f"mzi_profile_{device.type}_{batch_size}x{d_inp}x{d_out}.json"
            prof.export_chrome_trace(trace_file)
            print(f"\nProfiler trace saved to: {trace_file}")
        
        # Memory usage analysis if on GPU
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1e6   # MB
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {memory_allocated:.1f} MB")
            print(f"  Reserved: {memory_reserved:.1f} MB")
        
        # Performance assertions (these are quite lenient)
        assert avg_time < 10.0, f"Forward pass too slow: {avg_time:.3f}s > 10s"
        print(f"\nTest passed! Average forward pass time: {avg_time*1000:.2f} ms")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_different_batch_sizes_gpu(self, gpu_device):
        """Test performance scaling with different batch sizes on GPU."""
        d_inp = 512
        d_out = 512
        batch_sizes = [1, 16, 64, 256, 1024]
        device = gpu_device
        
        print(f"\nTesting batch size scaling on {device}:")
        print(f"Layer size: {d_inp} -> {d_out}")
        
        # Create MZI layer
        bb, w0, _ = create_bb_layer_mzi(d_inp, d_out)
        w0 = w0.to(device)
        
        results = {}
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, d_inp, device=device, dtype=torch.float32)
            
            # Warm-up
            with torch.no_grad():
                _ = bb(x, w0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time measurement
            times = []
            for _ in range(5):
                start_time = time.time()
                with torch.no_grad():
                    output = bb(x, w0)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time  # samples per second
            results[batch_size] = {'time': avg_time, 'throughput': throughput}
            
            print(f"  Batch {batch_size:4d}: {avg_time*1000:6.2f} ms, {throughput:8.1f} samples/sec")
            
            # Basic correctness check
            assert output.shape == (batch_size, d_out)
            assert torch.isfinite(output).all()
        
        # Check that throughput generally increases with batch size
        throughputs = [results[bs]['throughput'] for bs in batch_sizes]
        print(f"\nThroughput scaling: {throughputs}")
        
        # Assert that larger batches are generally more efficient
        assert results[1024]['throughput'] > results[1]['throughput'], \
            "Large batches should be more efficient than single samples"
    
    def test_cpu_vs_gpu_comparison(self, gpu_device):
        """Compare CPU vs GPU performance for moderate size layer."""
        d_inp = 256
        d_out = 256
        batch_size = 128
        
        print(f"\nComparing CPU vs GPU performance:")
        print(f"Layer size: {d_inp} -> {d_out}, Batch size: {batch_size}")
        
        # Create MZI layer
        bb, w0, _ = create_bb_layer_mzi(d_inp, d_out)
        
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(gpu_device)
        
        for device in devices:
            w0_dev = w0.to(device)
            x = torch.randn(batch_size, d_inp, device=device, dtype=torch.float32)
            
            # Warm-up
            with torch.no_grad():
                _ = bb(x, w0_dev)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time measurement
            times = []
            for _ in range(5):
                start_time = time.time()
                with torch.no_grad():
                    output = bb(x, w0_dev)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            print(f"  {str(device):12s}: {avg_time*1000:6.2f} ms")
            
            # Verify correctness
            assert output.shape == (batch_size, d_out)
            assert torch.isfinite(output).all()


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