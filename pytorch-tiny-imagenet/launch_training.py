import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from test_model import test_model
from train_model import train_model


def setup_device():
    """Setup and return the appropriate device (CUDA/CPU)"""
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Using CUDA")
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def setup_data_loaders(data_dir, batch_size=100):
    """Setup data transforms and data loaders"""
    num_workers = {"train": 4, "val": 0, "test": 0}
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    }
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
        for x in ["train", "val", "test"]
    }
    
    dataloaders = {
        x: data.DataLoader(
            image_datasets[x], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers[x]
        )
        for x in ["train", "val", "test"]
    }
    
    return dataloaders


def setup_model(model_type, device):
    """Setup and return the model"""
    torch.manual_seed(42)
    
    if model_type == "ResNet18":
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        # Finetune Final few layers to adjust for tiny imagenet input
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, 200)
        
    elif model_type == "AlexNet":
        model_ft = models.alexnet(weights="IMAGENET1K_V1")
        num_features = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_features, 200)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_ft = model_ft.to(device)
    return model_ft


def main():
    parser = argparse.ArgumentParser(description='Train models on Tiny ImageNet')
    parser.add_argument('--model', type=str, default='ResNet18', 
                       choices=['ResNet18', 'AlexNet'],
                       help='Model to train (default: ResNet18)')
    parser.add_argument('--input-size', type=int, default=224,
                       choices=[64, 224],
                       help='Input image size (default: 224)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Training batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input Size: {args.input_size}x{args.input_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print()
    
    # Setup device
    device = setup_device()
    
    # Setup data directory
    data_dir = f"tiny-{args.input_size}/"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Setup data loaders
    print("Setting up data loaders...")
    dataloaders = setup_data_loaders(data_dir, args.batch_size)
    
    # Setup model
    print(f"Setting up {args.model} model...")
    model_ft = setup_model(args.model, device)
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    output_path = f"{args.model}_{args.input_size}"
    
    best_epoch = train_model(
        output_path=output_path,
        model=model_ft,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=args.epochs,
    )
    
    print(f"Training complete. Best epoch: {best_epoch}")
    
    # Test the model
    print("Testing the model...")
    model_ft.load_state_dict(torch.load(f"models/{output_path}/model_{best_epoch}_epoch.pt"))
    test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)
    
    print("Training and testing complete!")


if __name__ == "__main__":
    main() 