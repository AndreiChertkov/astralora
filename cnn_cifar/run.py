import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


from core.astralora import Astralora


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.35),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run():
    ast = Astralora('cnn_cifar', with_neptune=False)

    loader_trn, loader_tst = _build_data(
        ast.args.root_data, ast.args.batch_size)

    model = Model()
    model.classifier[1] = ast.build(model.classifier[1])
    model = model.to(ast.device) # Do it after ast.build!

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ast.args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=0.5, patience=5)

    losses_trn = []
    losses_tst = []

    accs_trn = []
    accs_tst = []

    for epoch in range(ast.args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader_trn:
            inputs, labels = inputs.to(ast.device), labels.to(ast.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        loss_trn = running_loss / len(loader_trn)
        acc_trn = 100. * correct / total
        losses_trn.append(loss_trn)
        accs_trn.append(acc_trn)
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader_tst:
                inputs, labels = inputs.to(ast.device), labels.to(ast.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        loss_tst = running_loss / len(loader_tst)
        acc_tst = 100. * correct / total
        losses_tst.append(loss_tst)
        accs_tst.append(acc_tst)
        
        scheduler.step(loss_tst)

        ast.step(epoch, loss_trn, loss_tst, acc_trn, acc_tst)

    ast.save_model(model)

    _plot(losses_trn, losses_tst, accs_trn, accs_tst, ast.path('result.png'))


def _build_data(fpath, batch_size):
    transform_trn = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_tst = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_trn = torchvision.datasets.CIFAR10(transform=transform_trn,
        root=fpath, download=True, train=True)
    dataset_tst = torchvision.datasets.CIFAR10(transform=transform_tst,
        root=fpath, download=True, train=False)

    loader_trn = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size,
        shuffle=True, num_workers=2)
    loader_tst = torch.utils.data.DataLoader(dataset_tst, batch_size=batch_size,
        shuffle=False, num_workers=2)
    
    return loader_trn, loader_tst


def _plot(losses_trn, losses_tst, accs_trn, accs_tst, fpath):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses_trn, label='Train Loss')
    plt.plot(losses_tst, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Evolution')

    plt.subplot(1, 2, 2)
    plt.plot(accs_trn, label='Train Accuracy')
    plt.plot(accs_tst, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Evolution')
    plt.tight_layout()

    plt.savefig(fpath)


if __name__ == '__main__':
    run()