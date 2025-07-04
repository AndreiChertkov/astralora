"""
ViT-B/32 training script for ImageNet-1K and Tiny ImageNet.

Key features:
- Uses torchvision's Vision Transformer implementation
- ViT-B/32 model with configurable classes (1000 for ImageNet-1K, 200 for Tiny ImageNet)
- Applied LoRA to attention and MLP layers
- AdamW optimizer with cosine annealing scheduler
- Dataset-specific data augmentation optimized for each dataset type
- Automatic dataset switching with --dataset argument
- Automatic dataset download for Tiny ImageNet from web
- Supports both ImageNet-1K and Tiny ImageNet datasets

Requirements for ImageNet-1K:
  pip install datasets pillow
  
Note: ImageNet-1K must be downloaded separately using download_1k.py:
- Requires HuggingFace account with access to the dataset
- Authentication via huggingface-hub login

Usage:
  python run.py --dataset tiny-imagenet    # Train on Tiny ImageNet (default)
  python run.py --dataset imagenet-1k     # Train on ImageNet-1K (requires pre-downloaded dataset)
  
To download ImageNet-1K:
  python download_1k.py --data_dir imagenet-1k
"""

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from core.astralora import Astralora


def check_imagenet_1k_exists(data_dir):
    """
    Check if ImageNet-1K dataset exists in the specified directory.
    
    Args:
        data_dir (str): Directory to check for ImageNet-1K dataset
        
    Raises:
        FileNotFoundError: If the dataset doesn't exist
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"ImageNet-1K dataset not found in {data_dir}. "
            f"Please download it first using: python download_1k.py --data_dir {data_dir}"
        )
    
    print(f"ImageNet-1K dataset found at {data_dir}")
    return data_dir


def download_tiny_imagenet(data_dir='tiny-imagenet'):
    """
    Download and prepare Tiny ImageNet dataset if not already present.
    """
    if os.path.exists(os.path.join(data_dir, 'train')) and os.path.exists(os.path.join(data_dir, 'val')):
        print(f"Tiny ImageNet dataset already exists in {data_dir}")
        return data_dir
    
    print("Downloading Tiny ImageNet dataset...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URL for Tiny ImageNet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    
    # Download the dataset
    if not os.path.exists(zip_path):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download completed!")
    else:
        print("Zip file already exists, skipping download.")
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move files to the expected structure
    extracted_dir = os.path.join(data_dir, "tiny-imagenet-200")
    
    # Create train and val directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Move training data
    train_source = os.path.join(extracted_dir, "train")
    if os.path.exists(train_source):
        for class_folder in os.listdir(train_source):
            class_path = os.path.join(train_source, class_folder)
            if os.path.isdir(class_path):
                # Move images from images/ subfolder to class folder
                images_path = os.path.join(class_path, "images")
                if os.path.exists(images_path):
                    target_path = os.path.join(train_dir, class_folder)
                    shutil.move(images_path, target_path)
    
    # Move validation data
    val_source = os.path.join(extracted_dir, "val")
    if os.path.exists(val_source):
        # Read validation annotations
        val_annotations = os.path.join(extracted_dir, "val", "val_annotations.txt")
        if os.path.exists(val_annotations):
            # Create class directories for validation
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_name, class_name = parts[0], parts[1]
                        class_dir = os.path.join(val_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        # Move image to appropriate class directory
                        src_image = os.path.join(val_source, "images", image_name)
                        dst_image = os.path.join(class_dir, image_name)
                        if os.path.exists(src_image):
                            shutil.move(src_image, dst_image)
    
    # Clean up extracted directory and zip file
    shutil.rmtree(extracted_dir)
    os.remove(zip_path)
    
    print(f"Tiny ImageNet dataset prepared in {data_dir}")
    return data_dir






best_acc1 = 0


def main():
    
    ast = Astralora('vgg19_tiny', with_neptune=False)
    args = ast.args
    
    # Determine dataset type based on the dataset argument
    dataset_type = args.dataset
    
    # Handle dataset preparation based on type
    if not args.dummy:
        if dataset_type == 'tiny-imagenet':
            print("Using Tiny ImageNet dataset")
            tiny_data_dir = os.path.join(args.data, 'tiny-imagenet')
            args.data = download_tiny_imagenet(tiny_data_dir)
        elif dataset_type == 'imagenet-1k':
            print("Using ImageNet-1K dataset")
            imagenet_data_dir = os.path.join(args.data, 'imagenet-1k')
            # Check if dataset exists, raise error if not
            args.data = check_imagenet_1k_exists(imagenet_data_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Store dataset type for later use
    args.dataset_type = dataset_type

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, ast))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, ast)


def main_worker(gpu, ngpus_per_node, args, ast):
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.node_rank == -1:
            args.node_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.node_rank = args.node_rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.node_rank)

    # create model
    print("=> creating ViT-B/32 model from scratch")
    
    # Use torchvision's ViT implementation
    from torchvision.models import vit_b_32
    
    model = vit_b_32(weights=None)  # Train from scratch
    
    # Determine number of classes based on dataset
    if args.dataset_type == 'tiny-imagenet':
        num_classes = 200
        print(f"=> Configuring model for Tiny ImageNet ({num_classes} classes)")
    elif args.dataset_type == 'imagenet-1k':
        num_classes = 1000
        print(f"=> Configuring model for ImageNet-1K ({num_classes} classes)")
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # Modify the classifier head
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Apply LoRA to the transformer blocks
    if args.replace_layers:
        replace_layers = [int(i) for i in args.replace_layers.split(',')]
    else:
        replace_layers = []

    for i, block in enumerate([block for block in model.encoder.layers if hasattr(block, 'mlp')]):        
        # Apply LoRA to MLP layers
        if i in replace_layers or replace_layers == []:  # First linear layer in MLP
            print(f"  Replacing MLP layer {i} with LoRA")
            block.mlp[0] = ast.build(block.mlp[0])
            block.mlp[3] = ast.build(block.mlp[3])

    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # For ViT models, use standard DataParallel
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.DataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    # Use AdamW optimizer for ViT with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
    
    # Use Cosine Annealing scheduler for ViT
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        # Use appropriate number of classes for dummy data
        dummy_classes = 200 if args.dataset_type == 'tiny-imagenet' else 1000
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), dummy_classes, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), dummy_classes, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # Configure data augmentation based on dataset type
        if args.dataset_type == 'tiny-imagenet':
            print("=> Using Tiny ImageNet data transforms")
            # Enhanced data augmentation for ViT training on Tiny ImageNet
            train_transform = transforms.Compose([
                transforms.Resize(256),  # Resize first since Tiny ImageNet is 64x64
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                normalize,
            ])
        elif args.dataset_type == 'imagenet-1k':
            print("=> Using ImageNet-1K data transforms")
            # Enhanced data augmentation for ViT training on ImageNet-1K
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        # Validation transforms are the same for both datasets
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
        
        ast.step(epoch, train_loss, val_loss, train_acc1, val_acc1)
        
        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.node_rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, ast)

    
    ast.done(model)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)

    # switch to train mode
    model.train()

    end = time.time()
    
    # Create tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', 
                unit='batch', disable=args.distributed and args.node_rank != 0)
    
    for i, (images, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected in epoch {epoch+1}, batch {i}. Skipping batch.")
            continue

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN gradients and clip them
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"Warning: NaN gradient detected in epoch {epoch+1}, batch {i}")
                    param.grad.data.zero_()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar description
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    pbar.close()
    
    # Print final epoch summary
    if not args.distributed or args.node_rank == 0:
        print(f'Epoch {epoch+1} Summary: Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            
            # Create tqdm progress bar for validation
            pbar = tqdm(loader, desc='Validation', unit='batch', 
                       disable=args.distributed and args.node_rank != 0)
            
            for i, (images, target) in enumerate(pbar):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type=='cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Update progress bar description
                if i % args.print_freq == 0 or i == len(loader) - 1:
                    pbar.set_postfix({
                        'Loss': f'{losses.avg:.4f}',
                        'Acc@1': f'{top1.avg:.2f}%',
                        'Acc@5': f'{top5.avg:.2f}%'
                    })
            
            pbar.close()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    # Print validation summary
    if not args.distributed or args.node_rank == 0:
        print(f'Validation Summary: Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, ast):
    checkpoint_path = ast.path('checkpoint.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = ast.path('model_best.pth')
        shutil.copyfile(checkpoint_path, best_model_path)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()