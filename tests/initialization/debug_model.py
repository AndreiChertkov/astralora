#!/usr/bin/env python3
"""
Debug script to inspect the DialoGPT model structure.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def debug_model():
    """Debug the DialoGPT model structure."""
    
    model_name = "microsoft/DialoGPT-small"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    print(f"Model type: {type(model)}")
    print(f"Model loaded successfully on {device}")
    
    # Print model structure
    print("\nModel structure:")
    print(model)
    
    # Count different types of layers
    linear_count = 0
    embedding_count = 0
    attention_count = 0
    
    print("\nScanning all modules:")
    for name, module in model.named_modules():
        print(f"Module: {name} - Type: {type(module)}")
        
        if isinstance(module, nn.Linear):
            linear_count += 1
            print(f"  -> Linear layer: {module.in_features} -> {module.out_features}")
        elif isinstance(module, nn.Embedding):
            embedding_count += 1
            print(f"  -> Embedding layer: {module.num_embeddings} x {module.embedding_dim}")
        elif 'attention' in name.lower() or 'attn' in name.lower():
            attention_count += 1
            print(f"  -> Attention-related module")
    
    print(f"\nSummary:")
    print(f"Linear layers: {linear_count}")
    print(f"Embedding layers: {embedding_count}")
    print(f"Attention modules: {attention_count}")
    
    # Try to find any linear layers in the model
    print("\nLooking for linear layers specifically:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Found linear layer: {name}")
            print(f"  Input features: {module.in_features}")
            print(f"  Output features: {module.out_features}")
            print(f"  Has bias: {module.bias is not None}")
            print(f"  Weight shape: {module.weight.shape}")

if __name__ == "__main__":
    debug_model()
