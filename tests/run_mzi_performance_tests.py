#!/usr/bin/env python3
"""
Script to run MZI performance tests with GPU support and profiling.

This script demonstrates how to run the new performance tests for MZI layers.
It includes examples of different configurations and environment variables.
"""

import os
import subprocess
import sys

def run_test(test_name, gpu_id=None, profile_traces=False, verbose=True):
    """
    Run a specific MZI performance test.
    
    Args:
        test_name: Name of the test to run
        gpu_id: GPU device ID to use (None for default)
        profile_traces: Whether to save profiling traces
        verbose: Whether to use verbose output
    """
    env = os.environ.copy()
    
    if gpu_id is not None:
        env['TEST_GPU_ID'] = str(gpu_id)
    
    if profile_traces:
        env['PROFILE_TRACES'] = '1'
    
    cmd = [
        sys.executable, '-m', 'pytest',
        f'tests/bb_layers/test_bb_layer_mzi.py::{test_name}',
        '-s',  # Don't capture output
    ]
    
    if verbose:
        cmd.append('-v')
    
    print(f"Running command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"Using GPU device: {gpu_id}")
    if profile_traces:
        print("Profiling traces will be saved")
    print("-" * 50)
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

def main():
    """Main function to demonstrate different test configurations."""
    
    print("MZI Performance Test Runner")
    print("=" * 50)
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available - tests will run on CPU only")
    except ImportError:
        print("PyTorch not available")
        return 1
    
    print("\n" + "=" * 50)
    
    if not cuda_available:
        print("Skipping GPU tests - CUDA not available")
        return 0
    
    # Example 1: Run the main performance test with default GPU
    print("\n1. Running main performance test (1024x1024, batch=1024)...")
    return_code = run_test(
        "TestMZIPerformanceGPU::test_large_layer_performance_gpu",
        gpu_id=0,
        profile_traces=True
    )
    
    if return_code != 0:
        print("Test failed!")
        return return_code
    
    # Example 2: Run batch size scaling test
    print("\n2. Running batch size scaling test...")
    return_code = run_test(
        "TestMZIPerformanceGPU::test_different_batch_sizes_gpu",
        gpu_id=0,
        profile_traces=False
    )
    
    if return_code != 0:
        print("Test failed!")
        return return_code
    
    # Example 3: Run CPU vs GPU comparison
    print("\n3. Running CPU vs GPU comparison...")
    return_code = run_test(
        "TestMZIPerformanceGPU::test_cpu_vs_gpu_comparison",
        gpu_id=0,
        profile_traces=False
    )
    
    if return_code != 0:
        print("Test failed!")
        return return_code
    
    print("\nAll tests completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
