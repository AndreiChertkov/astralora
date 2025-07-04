# Dataset Support for ViT-B/32 Training

This script supports both Tiny ImageNet and ImageNet-1K datasets with automatic download capabilities.

## Datasets

### Tiny ImageNet (Default)
- **Classes**: 200
- **Image size**: 64x64 (automatically resized to 224x224)
- **Download**: Automatic from Stanford CS231n
- **Storage**: `vgg19_tiny/data/tiny-imagenet/`

```bash
# Train on Tiny ImageNet (default)
python run.py --dataset tiny-imagenet
```

### ImageNet-1K
- **Classes**: 1000
- **Image size**: Variable (automatically resized to 224x224)
- **Download**: Automatic from HuggingFace (`ILSVRC/imagenet-1k`)
- **Storage**: `vgg19_tiny/data/imagenet-1k/`

```bash
# Train on ImageNet-1K
python run.py --dataset imagenet-1k
```

## Requirements

### For Tiny ImageNet
No additional requirements - uses standard libraries.

### For ImageNet-1K
```bash
pip install datasets pillow
```

**Note**: ImageNet-1K download may require HuggingFace authentication:
```bash
pip install huggingface-hub
huggingface-cli login
```

## Data Augmentation

The script automatically applies dataset-specific augmentation:

**Tiny ImageNet:**
- Initial resize from 64x64 to 256x256
- RandomResizedCrop to 224x224 (scale: 0.8-1.0)
- Moderate color jitter and rotation

**ImageNet-1K:**
- RandomResizedCrop to 224x224 (scale: 0.2-1.0)
- Stronger color jitter for better generalization
- Less rotation due to larger dataset diversity

## Model Configuration

The model automatically configures the number of output classes:
- **Tiny ImageNet**: 200 classes
- **ImageNet-1K**: 1000 classes

## Example Commands

```bash
# Train ViT-B/32 on Tiny ImageNet with LoRA on all layers
python run.py --dataset tiny-imagenet --arch vit_b_32 --epochs 100

# Train ViT-B/32 on ImageNet-1K with LoRA on specific layers  
python run.py --dataset imagenet-1k --arch vit_b_32 --replace-layers 0,1,2,3 --epochs 300

# Use different learning rates for different datasets
python run.py --dataset tiny-imagenet --lr 1e-3   # Good for Tiny ImageNet
python run.py --dataset imagenet-1k --lr 3e-3     # Good for ImageNet-1K
```

## Storage Structure

```
vgg19_tiny/data/
├── tiny-imagenet/
│   ├── train/
│   │   ├── n01443537/
│   │   └── ...
│   └── val/
│       ├── n01443537/
│       └── ...
└── imagenet-1k/
    ├── train/
    │   ├── 0000/
    │   └── ...
    └── val/
        ├── 0000/
        └── ...
```

Both datasets can coexist in the same data directory without conflicts. 