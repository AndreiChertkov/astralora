"""
ImageNet-1K dataset downloader using HuggingFace datasets.

This script downloads and prepares the ImageNet-1K dataset from HuggingFace.
The dataset is organized into train/val directories with class subdirectories.

Requirements:
  pip install datasets pillow

Note: ImageNet-1K download from HuggingFace (ILSVRC/imagenet-1k) may require:
- HuggingFace account with access to the dataset
- Authentication via huggingface-hub login

Usage:
  python download_1k.py [data_dir]
"""

import os
import argparse
from tqdm import tqdm


def download_imagenet_1k(data_dir='imagenet-1k'):
    """
    Download and prepare ImageNet-1K dataset using HuggingFace datasets.
    
    Args:
        data_dir (str): Directory to save the dataset
        
    Returns:
        str: Path to the prepared dataset directory
    """
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        raise ImportError("Please install datasets and PIL: pip install datasets pillow")
    
    # if os.path.exists(os.path.join(data_dir, 'train')) and os.path.exists(os.path.join(data_dir, 'val')):
    #     print(f"ImageNet-1K dataset already exists in {data_dir}")
    #     return data_dir
    
    print("Downloading ImageNet-1K dataset from HuggingFace...")
    print("This may take a while as ImageNet-1K is a large dataset...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        print("Loading dataset from ILSVRC/imagenet-1k...")
        dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=os.path.join(data_dir, 'cache'), trust_remote_code=True)
        
        # Create train and val directories
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        def save_split(split_name, split_data, output_dir):
            print(f"Processing {split_name} split...")
            class_dirs = {}
            
            for idx, example in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                label = example['label']
                image = example['image']
                
                # Create class directory if it doesn't exist
                if label not in class_dirs:
                    class_dir = os.path.join(output_dir, str(label).zfill(4))  # Zero-pad to 4 digits
                    os.makedirs(class_dir, exist_ok=True)
                    class_dirs[label] = class_dir
                
                # Save image
                image_path = os.path.join(class_dirs[label], f"{idx:08d}.JPEG")
                if isinstance(image, Image.Image):
                    image.save(image_path, "JPEG")
                else:
                    # Handle case where image might be in different format
                    Image.fromarray(image).save(image_path, "JPEG")
        
        # Save training split
        save_split("train", dataset["train"], train_dir)
        
        # Save validation split  
        save_split("validation", dataset["validation"], val_dir)
        
        print(f"ImageNet-1K dataset prepared in {data_dir}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        
    except Exception as e:
        raise e
    
    return data_dir


def main():
    parser = argparse.ArgumentParser(description='Download ImageNet-1K dataset')
    parser.add_argument('--data_dir', default='imagenet-1k', 
                       help='Directory to save the dataset (default: imagenet-1k)')
    
    args = parser.parse_args()
    
    try:
        download_imagenet_1k(args.data_dir)
        print(f"ImageNet-1K dataset successfully downloaded to {args.data_dir}")
    except Exception as e:
        print(f"Failed to download ImageNet-1K: {e}")
        exit(1)


if __name__ == '__main__':
    main()
