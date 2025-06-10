from datetime import datetime
import os
import shutil
import torch


def init_log(msg='', fpath='log.txt', enable=True):
    def log(text, kind='', only_file=False):
        if not enable:
            return

        pref = ''

        # if kind != 'ini' and kind != 'log':
        #    pref += '[' + datetime.now().strftime('%H-%M-%S') + '] > '

        if kind == 'prc':
            pref = '... ' + pref
        if kind == 'res':
            pref = '+++ ' + pref
        if kind == 'wrn':
            pref = 'WRN ' + pref
        if kind == 'err':
            pref = '!!! ' + pref

        text = pref + text
        with open(fpath, 'w' if kind == 'ini' else 'a+', encoding='utf-8') as f:
            f.write(text + '\n')
        if not only_file:
            print(text)

    dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    content = 'Start computations'
    text = f'[{dt}] >> {content}'
    text += '\n' + '=' * 24 + ' ' + '-' * len(content) + '\n'
    if msg:
        text += msg
        text += '\n' + '=' * (25 + len(content)) + '\n'
    log(text, 'ini')

    return log


def init_path(name, root='result', rewrite=False):
    os.makedirs(root, exist_ok=True)
    folder = f'{root}/{name}'
    if os.path.isdir(folder):
        if rewrite:
            act = 'y'
        else:
            msg = f'Path "{folder}" already exists. Remove? [y/n] '
            act = input(msg)
        if act == 'y':
            shutil.rmtree(folder)
        else:
            raise ValueError('Folder with results is already exists')
    os.makedirs(folder)


def modify_gpu_args_for_cryri(args):
    # Get the number of requested GPUs from args.gpus:
    requested_gpus = [int(g.strip()) for g in args.gpus.split(',')]
    num_requested = len(requested_gpus)
    
    # Get the number of available GPUs
    num_available = torch.cuda.device_count()
    
    # Check if we have enough GPUs
    if num_requested > num_available:
        raise ValueError(f'Requested {num_requested} GPUs but only {num_available} are available')
    elif num_requested != num_available:
        print('Executing on Hopper, no GPU number swapping')
    else:
        print('Executing with jobs, GPU number swapping')
    
        available_gpus = list(range(num_available))
        args.gpus = ','.join(map(str, available_gpus))
    
    return args