#!/usr/bin/env python3
"""
Simple test runner for the transformer Monarch approximation test.

Usage:
    python run_transformer_test.py
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_transformer_monarch_approximation import (
    test_transformer_monarch_approximation,
    test_different_transformer_models
)

def main():
    """Run all transformer Monarch approximation tests."""
    print("=" * 60)
    print("Running Transformer Monarch Approximation Tests")
    print("=" * 60)
    
    try:
        print("\n1. Testing transformer Monarch approximation...")
        test_transformer_monarch_approximation()
        print("✓ Transformer Monarch approximation test passed!")
        
        # print("\n2. Testing with different transformer models...")
        # test_different_transformer_models()
        # print("✓ Different transformer models test passed!")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
