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
  python download_1k.py [data_dir] [--workers N] [--batch_size N]
"""

import os
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time


def process_batch(batch_data, output_dir, split_name):
    """
    Process a batch of images and save them to disk.
    
    Args:
        batch_data (list): List of (idx, example) tuples
        output_dir (str): Output directory for this split
        split_name (str): Name of the split (train/validation)
    
    Returns:
        int: Number of images processed
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install PIL: pip install pillow")
    
    processed_count = 0
    
    for idx, example in batch_data:
        try:
            label = example['label']
            image = example['image']
            
            # Create class directory if it doesn't exist
            class_dir = os.path.join(output_dir, str(label).zfill(4))  # Zero-pad to 4 digits
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate filename
            image_path = os.path.join(class_dir, f"{idx:08d}.JPEG")
            
            # Skip if already exists
            if os.path.exists(image_path):
                processed_count += 1
                continue
            
            # Save image
            if isinstance(image, Image.Image):
                image.save(image_path, "JPEG", quality=95)
            else:
                # Handle case where image might be in different format
                Image.fromarray(image).save(image_path, "JPEG", quality=95)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {idx} in {split_name}: {e}")
            continue
    
    return processed_count


def save_split_parallel(split_name, dataset_split, output_dir, num_workers=4, batch_size=100):
    """
    Save a dataset split using parallel processing with streaming.
    
    Args:
        split_name (str): Name of the split
        dataset_split: HuggingFace dataset split (streaming)
        output_dir (str): Output directory
        num_workers (int): Number of worker processes
        batch_size (int): Batch size for processing
    """
    print(f"Processing {split_name} split with {num_workers} workers...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total items for progress bar (this is approximate for streaming datasets)
    try:
        total_items = len(dataset_split)
    except:
        # For streaming datasets, we might not know the exact length
        total_items = None
    
    # Initialize progress bar
    pbar = tqdm(desc=f"Saving {split_name}", total=total_items)
    
    # Prepare batches for parallel processing
    batch = []
    batch_idx = 0
    
    # Create process pool
    with mp.Pool(processes=num_workers) as pool:
        active_tasks = []
        
        for idx, example in enumerate(dataset_split):
            batch.append((idx, example))
            
            # Process batch when it's full
            if len(batch) >= batch_size:
                # Submit batch for processing
                task = pool.apply_async(
                    process_batch, 
                    (batch.copy(), output_dir, split_name)
                )
                active_tasks.append(task)
                batch = []
                batch_idx += 1
                
                # Check for completed tasks and update progress
                completed_tasks = []
                for task in active_tasks:
                    if task.ready():
                        try:
                            processed_count = task.get()
                            pbar.update(processed_count)
                            completed_tasks.append(task)
                        except Exception as e:
                            print(f"Batch processing error: {e}")
                            completed_tasks.append(task)
                
                # Remove completed tasks
                for task in completed_tasks:
                    active_tasks.remove(task)
                
                # Limit number of active tasks to control memory usage
                while len(active_tasks) >= num_workers * 2:
                    time.sleep(0.1)
                    completed_tasks = []
                    for task in active_tasks:
                        if task.ready():
                            try:
                                processed_count = task.get()
                                pbar.update(processed_count)
                                completed_tasks.append(task)
                            except Exception as e:
                                print(f"Batch processing error: {e}")
                                completed_tasks.append(task)
                    
                    for task in completed_tasks:
                        active_tasks.remove(task)
        
        # Process remaining batch
        if batch:
            task = pool.apply_async(
                process_batch, 
                (batch, output_dir, split_name)
            )
            active_tasks.append(task)
        
        # Wait for all remaining tasks to complete
        for task in active_tasks:
            try:
                processed_count = task.get()
                pbar.update(processed_count)
            except Exception as e:
                print(f"Final batch processing error: {e}")
    
    pbar.close()


def download_imagenet_1k(data_dir='imagenet-1k', num_workers=None, batch_size=100):
    """
    Download and prepare ImageNet-1K dataset using HuggingFace datasets.
    
    Args:
        data_dir (str): Directory to save the dataset
        num_workers (int): Number of worker processes (default: CPU count)
        batch_size (int): Batch size for processing
        
    Returns:
        str: Path to the prepared dataset directory
    """
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        raise ImportError("Please install datasets and PIL: pip install datasets pillow")
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    
    print(f"Using {num_workers} worker processes with batch size {batch_size}")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace with streaming to avoid loading everything into memory
        print("Loading dataset from ILSVRC/imagenet-1k in streaming mode...")
        dataset = load_dataset(
            "ILSVRC/imagenet-1k", 
            cache_dir=os.path.join(data_dir, 'cache'), 
            trust_remote_code=True,
            streaming=True  # This is key to avoid memory issues
        )
        
        # Create train and val directories
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Process training split
        print("Starting training split processing...")
        save_split_parallel("train", dataset["train"], train_dir, num_workers, batch_size)
        
        # Process validation split  
        print("Starting validation split processing...")
        save_split_parallel("validation", dataset["validation"], val_dir, num_workers, batch_size)
        
        print(f"ImageNet-1K dataset prepared in {data_dir}")
        
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        raise e
    
    return data_dir


def main():
    parser = argparse.ArgumentParser(description='Download ImageNet-1K dataset')
    parser.add_argument('--data_dir', default='imagenet-1k', 
                       help='Directory to save the dataset (default: imagenet-1k)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count, max 8)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    
    args = parser.parse_args()
    
    try:
        download_imagenet_1k(args.data_dir, args.workers, args.batch_size)
        print(f"ImageNet-1K dataset successfully downloaded to {args.data_dir}")
    except Exception as e:
        print(f"Failed to download ImageNet-1K: {e}")
        exit(1)


if __name__ == '__main__':
    main()
