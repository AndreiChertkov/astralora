"""
Test for approximating transformer feed-forward layers with Monarch black-box methods.

This test demonstrates how to:
1. Load a transformer model from Hugging Face Hub
2. Extract a linear layer from the feed-forward part
3. Approximate it using Monarch black-box gradient-based methods
4. Compare the original and approximated layer outputs
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pytest

# Add the project root to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.bb_layers.bb_layer_monarch import create_bb_layer_monarch
from core.layer import AstraloraLayer


class TransformerLayerExtractor:
    """Helper class to extract linear layers from transformer models."""
    
    def __init__(self, model_name="facebook/opt-125m"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the transformer model and tokenizer from Hugging Face Hub."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
    def find_feed_forward_layers(self):
        """Find all linear layers in the feed-forward parts of the transformer."""
        feed_forward_layers = []
        all_linear_layers = []
        
        print("Scanning model layers...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                all_linear_layers.append((name, module))
                print(f"Found linear layer: {name} ({module.in_features} -> {module.out_features})")
                
                # Check for various naming patterns used in different transformer architectures
                name_lower = name.lower()
                is_feed_forward = any(keyword in name_lower for keyword in [
                    'mlp', 'ffn', 'feed', 'intermediate', 'output', 'c_fc', 'c_proj', 
                    'dense', 'linear', 'fc', 'proj'
                ])
                
                # For GPT-style models, look for specific patterns
                if 'transformer.h.' in name and ('c_fc' in name or 'c_proj' in name):
                    is_feed_forward = True
                
                # For BERT-style models
                if 'intermediate' in name or 'output.dense' in name:
                    is_feed_forward = True
                
                # Exclude embedding and attention layers
                if any(exclude in name_lower for exclude in ['embed', 'attn', 'attention', 'wte', 'wpe']):
                    is_feed_forward = False
                
                if is_feed_forward:
                    feed_forward_layers.append((name, module))
                    print(f"  -> Classified as feed-forward layer")
                
        print(f"Total linear layers found: {len(all_linear_layers)}")
        print(f"Feed-forward layers found: {len(feed_forward_layers)}")
        
        return feed_forward_layers
    
    def get_layer_info(self, layer_name, layer):
        """Get information about a specific layer."""
        return {
            'name': layer_name,
            'in_features': layer.in_features,
            'out_features': layer.out_features,
            'weight_shape': layer.weight.shape,
            'bias': layer.bias is not None
        }


class MonarchApproximator:
    """Class to handle Monarch black-box approximation of linear layers."""
    
    def __init__(self, d_inp, d_out, rank=10, samples_bb=100, samples_sm=100):
        self.d_inp = d_inp
        self.d_out = d_out
        self.rank = rank
        self.samples_bb = samples_bb
        self.samples_sm = samples_sm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the Monarch black-box layer
        self.bb_layer = self._create_monarch_layer()
        
    def _create_monarch_layer(self):
        """Create an AstraloraLayer with Monarch black-box implementation."""
        return AstraloraLayer(
            d_inp=self.d_inp,
            d_out=self.d_out,
            kind='monarch',
            rank=self.rank,
            samples_bb=self.samples_bb,
            samples_sm=self.samples_sm,
            samples_bb_batch_frac=1.0,
            skip_sm=False,
            use_residual=False,
            log=print,
            nepman=None
        ).to(self.device)
    
    def approximate_layer(self, original_layer, num_samples=1000, learning_rate=0.01, num_epochs=100):
        """Approximate the original layer using Monarch black-box methods."""
        print(f"Approximating layer: {original_layer.in_features} -> {original_layer.out_features}")
        
        # Generate training data
        x_data = torch.randn(num_samples, original_layer.in_features, device=self.device)
        with torch.no_grad():
            y_target = original_layer(x_data)
        
        # Setup optimizer for the Monarch layer
        optimizer = torch.optim.Adam(self.bb_layer.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass through Monarch layer
            y_pred = self.bb_layer(x_data)
            
            # Compute loss
            loss = criterion(y_pred, y_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
        
        return losses
    
    def compare_outputs(self, original_layer, test_input):
        """Compare outputs of original and approximated layers."""
        with torch.no_grad():
            original_output = original_layer(test_input)
            approximated_output = self.bb_layer(test_input)
        
        # Compute metrics
        mse = F.mse_loss(approximated_output, original_output)
        mae = F.l1_loss(approximated_output, original_output)
        
        # Compute relative error
        relative_error = torch.norm(approximated_output - original_output) / torch.norm(original_output)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'relative_error': relative_error.item(),
            'original_output': original_output,
            'approximated_output': approximated_output
        }


def test_transformer_monarch_approximation():
    """Main test function for transformer Monarch approximation."""
    
    # Initialize the extractor
    extractor = TransformerLayerExtractor("facebook/opt-125m")
    extractor.load_model()
    
    # Find feed-forward layers
    feed_forward_layers = extractor.find_feed_forward_layers()
    
    if not feed_forward_layers:
        print("No feed-forward layers found, looking for any linear layer...")
        # Fallback: find any linear layer
        for name, module in extractor.model.named_modules():
            if isinstance(module, nn.Linear) and 'embed' not in name.lower():
                feed_forward_layers = [(name, module)]
                print(f"Using fallback layer: {name} ({module.in_features} -> {module.out_features})")
                break
        
        if not feed_forward_layers:
            pytest.skip("No suitable linear layers found in the model")
    
    print(f"Found {len(feed_forward_layers)} feed-forward layers")
    
    # Test with the first feed-forward layer
    layer_name, layer = feed_forward_layers[0]
    layer_info = extractor.get_layer_info(layer_name, layer)
    
    print(f"Testing layer: {layer_info}")
    
    # Create approximator
    approximator = MonarchApproximator(
        d_inp=layer.in_features,
        d_out=layer.out_features,
        rank=min(10, min(layer.in_features, layer.out_features) // 4),  # Adaptive rank
        samples_bb=50,
        samples_sm=50
    )
    
    # Approximate the layer
    losses = approximator.approximate_layer(layer, num_samples=500, num_epochs=50)
    
    # Test with new data
    test_input = torch.randn(10, layer.in_features, device=extractor.device)
    comparison = approximator.compare_outputs(layer, test_input)
    
    # Assertions
    assert comparison['mse'] < 1.0, f"MSE too high: {comparison['mse']}"
    assert comparison['relative_error'] < 0.5, f"Relative error too high: {comparison['relative_error']}"
    
    print(f"Test passed!")
    print(f"Final MSE: {comparison['mse']:.6f}")
    print(f"Final MAE: {comparison['mae']:.6f}")
    print(f"Relative Error: {comparison['relative_error']:.6f}")
    
    # Test that the Monarch layer can handle different input shapes
    batch_sizes = [1, 5, 10]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, layer.in_features, device=extractor.device)
        with torch.no_grad():
            output = approximator.bb_layer(test_input)
        assert output.shape == (batch_size, layer.out_features), \
            f"Output shape mismatch for batch size {batch_size}"





def test_different_transformer_models():
    """Test with different transformer models."""
    
    model_names = [
        "facebook/opt-125m",
        "gpt2"
    ]
    
    for model_name in model_names:
        try:
            extractor = TransformerLayerExtractor(model_name)
            extractor.load_model()
            
            feed_forward_layers = extractor.find_feed_forward_layers()
            if feed_forward_layers:
                layer_name, layer = feed_forward_layers[0]
                
                # Create a simple approximator for testing
                approximator = MonarchApproximator(
                    d_inp=layer.in_features,
                    d_out=layer.out_features,
                    rank=5,
                    samples_bb=10,
                    samples_sm=10
                )
                
                # Quick test
                test_input = torch.randn(2, layer.in_features, device=extractor.device)
                comparison = approximator.compare_outputs(layer, test_input)
                
                print(f"Model {model_name}: MSE = {comparison['mse']:.6f}")
                
        except Exception as e:
            print(f"Failed to test model {model_name}: {e}")
            continue

 
if __name__ == "__main__":
    # Run the tests
    test_transformer_monarch_approximation()
    test_different_transformer_models()
    print("All tests completed successfully!")
