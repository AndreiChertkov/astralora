# Transformer Monarch Approximation Tests

This directory contains tests for approximating transformer feed-forward layers using Monarch black-box gradient-based methods.

## Overview

The tests demonstrate how to:
1. Load transformer models from Hugging Face Hub
2. Extract linear layers from the feed-forward parts of transformers
3. Approximate these layers using Monarch black-box gradient-based methods
4. Compare the performance between original and approximated layers

## Files

- `test_transformer_monarch_approximation.py` - Main test file with comprehensive tests
- `run_transformer_test.py` - Simple test runner script
- `README.md` - This documentation file

## Requirements

The tests require the following dependencies (already included in the project):
- `torch` >= 2.6.0
- `transformers` (for loading Hugging Face models)
- `numpy`
- `pytest` (for running tests)

## Usage

### Running the Tests

1. **Using the test runner script:**
   ```bash
   cd tests/initialization
   python run_transformer_test.py
   ```

2. **Using pytest:**
   ```bash
   cd tests/initialization
   pytest test_transformer_monarch_approximation.py -v
   ```

3. **Running individual test functions:**
   ```bash
   cd tests/initialization
   python -c "from test_transformer_monarch_approximation import test_monarch_layer_properties; test_monarch_layer_properties()"
   ```

### Test Functions

1. **`test_transformer_monarch_approximation()`**
   - Loads a DialoGPT model from Hugging Face Hub
   - Finds feed-forward layers in the model
   - Approximates a linear layer using Monarch black-box methods
   - Compares original vs approximated outputs
   - Tests with different batch sizes

2. **`test_monarch_layer_properties()`**
   - Tests basic properties of the Monarch layer
   - Verifies input/output dimensions
   - Tests forward pass functionality
   - Checks training/eval mode switching

3. **`test_different_transformer_models()`**
   - Tests with multiple transformer models
   - Currently includes DialoGPT and DistilBERT
   - Demonstrates compatibility across different architectures

## Key Components

### TransformerLayerExtractor
Helper class that:
- Loads transformer models from Hugging Face Hub
- Finds feed-forward layers in the model
- Provides layer information and metadata

### MonarchApproximator
Class that handles:
- Creation of Monarch black-box layers
- Training the approximation using gradient-based methods
- Comparison between original and approximated layers
- Performance metrics calculation

## Expected Output

When running the tests successfully, you should see output similar to:

```
============================================================
Running Transformer Monarch Approximation Tests
============================================================

1. Testing Monarch layer properties...
✓ Monarch layer properties test passed!

2. Testing transformer Monarch approximation...
Loading model: microsoft/DialoGPT-small
Model loaded successfully on cuda:0
Found 12 feed-forward layers
Testing layer: {'name': 'transformer.h.0.mlp.c_fc', 'in_features': 768, 'out_features': 3072, ...}
Approximating layer: 768 -> 3072
Epoch 0/50, Loss: 0.123456
Epoch 20/50, Loss: 0.045678
Epoch 40/50, Loss: 0.023456
Test passed!
Final MSE: 0.023456
Final MAE: 0.123456
Relative Error: 0.045678
✓ Transformer Monarch approximation test passed!

3. Testing with different transformer models...
Model microsoft/DialoGPT-small: MSE = 0.023456
Model distilbert-base-uncased: MSE = 0.034567
✓ Different transformer models test passed!

============================================================
All tests completed successfully! 🎉
============================================================
```

## Configuration

The tests use the following default configurations:
- **Model**: `microsoft/DialoGPT-small` (small transformer for quick testing)
- **Rank**: Adaptive based on layer dimensions (min(10, min(in_features, out_features) // 4))
- **Samples**: 50 for both black-box and surrogate model training
- **Training**: 500 samples, 50 epochs, learning rate 0.01

You can modify these parameters in the test functions to experiment with different settings.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce the number of samples or use a smaller model
2. **Model loading fails**: Check internet connection and Hugging Face Hub access
3. **Import errors**: Ensure you're running from the correct directory with proper Python path

### Performance Notes

- The tests are designed to run quickly for development purposes
- For production use, you may want to increase the number of training samples and epochs
- Larger models will require more memory and computation time

## Extending the Tests

To add new transformer models or test different configurations:

1. Add model names to the `model_names` list in `test_different_transformer_models()`
2. Modify the `find_feed_forward_layers()` method to handle different layer naming conventions
3. Adjust hyperparameters in `MonarchApproximator` for different layer sizes
4. Add new test functions for specific use cases

## References

- Monarch black-box implementation: `core/bb_layers/bb_layer_monarch.py`
- Astralora layer wrapper: `core/layer.py`
- Project documentation: `README.md`
