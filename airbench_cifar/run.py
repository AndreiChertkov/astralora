"""Train the CNN model on CIFAR10 datacet.

The code is based on the repo: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96_faster.py#L357

"""
from math import ceil
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import uuid


from core.astralora import Astralora


hyp = {
    'opt': {
        'train_epochs': 100.0,
        'batch_size': 1024,
        'batch_size_masked': 512,
        'lr': 18., # 15.0,               # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.012,   # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,     # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3, # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 12,
    },
    'proxy': {
        'widths': {
            'block1': 32,
            'block2': 64,
            'block3': 64,
        },
        'depth': 2,
        'scaling_factor': 1/9,
    },
    'net': {
        'widths': {
            'block1': 128,
            'block2': 384,
            'block3': 512,
        },
        'depth': 3,
        'scaling_factor': 1/9,
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    }}


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out


def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask


def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


def set_random_state(seed, state):
    if seed is None:
        # If we don't get a data seed, then make sure to randomize the state using independent generator, since
        # it might have already been set by the model seed.
        import random
        torch.manual_seed(random.randint(0, 2**63))
    else:
        seed1 = 1000 * seed + state # just don't do more than 1000 epochs or else there will be overlap
        torch.manual_seed(seed1)


class InfiniteCifarLoader:
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    The purpose is to support cross-epoch batches (in case the batch size does not divide the number of train examples),
    and support stochastic iteration counts in order to preserve perfect linearity/independence.
    """

    def __init__(self, path, device, train=True, batch_size=500, aug=None, altflip=True, subset_mask=None, aug_seed=None, order_seed=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=device) #'cuda')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        # self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.images = (self.images / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.altflip = altflip
        self.subset_mask = subset_mask if subset_mask is not None else torch.tensor([True]*len(self.images)).to(device) #.cuda()
        self.train = train
        self.aug_seed = aug_seed
        self.order_seed = order_seed

    def __iter__(self):

        # Preprocess
        images0 = self.normalize(self.images)
        # Pre-randomly flip images in order to do alternating flip later.
        if self.aug.get('flip', False) and self.altflip:
            set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        # Pre-pad images to save time when doing random translation
        pad = self.aug.get('translate', 0)
        if pad > 0:
            images0 = F.pad(images0, (pad,)*4, 'reflect')
        labels0 = self.labels

        # Iterate forever
        epoch = 0
        batch_size = self.batch_size

        # In the below while-loop, we will repeatedly build a batch and then yield it.
        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
        batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:

            # Assume we need to generate more data to add to the batch.
            assert len(batch_images) < batch_size

            # If we have already exhausted the current epoch, then begin a new one.
            if current_pointer >= num_examples:
                # If we already reached the end of the last epoch then we need to generate
                # a new augmented epoch of data (using random crop and alternating flip).
                epoch += 1

                set_random_state(self.aug_seed, epoch)
                if pad > 0:
                    images1 = batch_crop(images0, 32)
                if self.aug.get('flip', False):
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)
                if self.aug.get('cutout', 0) > 0:
                    images1 = batch_cutout(images1, self.aug['cutout'])

                set_random_state(self.order_seed, epoch)
                indices = (torch.randperm if self.train else torch.arange)(len(self.images), device=images0.device)

                # The effect of doing subsetting in this manner is as follows. If the permutation wants to show us
                # our four examples in order [3, 2, 0, 1], and the subset mask is [True, False, True, False],
                # then we will be shown the examples [2, 0]. It is the subset of the ordering.
                # The purpose is to minimize the interaction between the subset mask and the randomness.
                # So that the subset causes not only a subset of the total examples to be shown, but also a subset of
                # the actual sequence of examples which is shown during training.
                indices_subset = indices[self.subset_mask[indices]]
                current_pointer = 0

            # Now we are sure to have more data in this epoch remaining.
            # This epoch's remaining data is given by (images1[current_pointer:], labels0[current_pointer:])
            # We add more data to the batch, up to whatever is needed to make a full batch (but it might not be enough).
            remaining_size = batch_size - len(batch_images)

            # Given that we want `remaining_size` more training examples, we construct them here, using
            # the remaining available examples in the epoch.

            extra_indices = indices_subset[current_pointer:current_pointer+remaining_size]
            extra_images = images1[extra_indices]
            extra_labels = labels0[extra_indices]
            current_pointer += remaining_size
            batch_indices = torch.cat([batch_indices, extra_indices])
            batch_images = torch.cat([batch_images, extra_images])
            batch_labels = torch.cat([batch_labels, extra_labels])

            # If we have a full batch ready then yield it and reset.
            if len(batch_images) == batch_size:
                assert len(batch_images) == len(batch_labels)
                yield (batch_indices, batch_images, batch_labels)
                batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
                batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
                batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)


def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0
            x = self.activ(x)
        return x


def make_net(hyp):
    widths = hyp['widths']
    scaling_factor = hyp['scaling_factor']
    depth = hyp['depth']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], depth),
        ConvGroup(widths['block1'], widths['block2'], depth),
        ConvGroup(widths['block2'], widths['block3'], depth),
        nn.MaxPool2d(3),
        Flatten(),
        # This layer will be replaced by BB:
        nn.Linear(widths['block3'], 1024, bias=False),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 10, bias=False),
        # nn.Linear(widths['block3'], 10, bias=False),
        Mul(scaling_factor),
    )
    net[0].weight.requires_grad = False
    # net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net


def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()


def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()


def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)


def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))


class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)


def main(ast, hyp, model):
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']

    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    set_random_state(None, 0)
    import random
    data_seed = random.randint(0, 2**50)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = InfiniteCifarLoader(ast.args.root_data, ast.device, 
        train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader(ast.args.root_data, ast.device,
        train=True, batch_size=batch_size, aug=hyp['aug'], aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    # reinit_net(model)
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_trainbias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_freezebias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, get_lr)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer_freezebias, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    #lookahead_state = LookaheadState(model)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    # init_whitening_conv(model._orig_mod[0], train_images) TODO
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for indices, inputs, labels in train_loader:
        optimizer = optimizer_trainbias
        scheduler = scheduler_trainbias

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            starter.record()
            model.train()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_steps += 1
        #if current_steps % 5 == 0:
        #    lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        #if current_steps == total_train_steps:
        #    if lookahead_state is not None:
        #        lookahead_state.update(model, decay=1.0)

        if (current_steps % steps_per_epoch == 0) or (current_steps == total_train_steps):
            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)

            # Save the accuracy and loss from the last training batch of the epoch
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)

            ast.step(epoch, train_loss, None, train_acc, val_acc)

        if current_steps == total_train_steps:
            break

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'

    return tta_val_acc


def run():
    ast = Astralora('airbench_cifar', with_neptune=False)

    model = make_net(hyp['net'])
    model[7] = ast.build(model[7])
    model = model.to(ast.device, memory_format=torch.channels_last)
    # model = torch.compile(model, mode='max-autotune')
    acc = main(ast, hyp, model)
    ast.done(model)


if __name__ == "__main__":
    run()