from collections import defaultdict
from datetime import datetime
from io import StringIO
import neptune
import numpy as np
import os
import random
import re
import shutil
import sys
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


def init_neptune(name, config_path, args):
    with open(config_path, 'r') as f:
        text = f.read()
        project = text.split('NEPTUNE_PROJECT="')[1].split('"')[0]
        token = text.split('NEPTUNE_API_TOKEN="')[1].split('"')[0]

    original_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    nepman = neptune.init_run(project=project, api_token=token, name=name)
    sys.stdout = original_stdout

    output = captured_output.getvalue()
    url_match = re.search(r'https://app\.neptune\.ai/[^\s]+', output)
    if url_match:
        url = url_match.group()
    
    nepman['parameters'] = vars(args)
    nepman['system/gpu'] = torch.cuda.get_device_name(0)
    nepman['system/cuda_version'] = torch.version.cuda
    nepman['system/pytorch_version'] = torch.version.__version__

    return nepman, url


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


def init_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True


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
        pass # print('Executing on Hopper, no GPU number swapping')
    else:
        print('Executing with jobs, GPU number swapping')
    
        available_gpus = list(range(num_available))
        args.gpus = ','.join(map(str, available_gpus))
    
    return args


def save_args_to_markdown(args, parser, filename):
    group_actions = defaultdict(list)
    group_titles = {}

    for group in parser._action_groups:
        if not group._group_actions or group.title == 'options':
            continue
            
        title = group.title
        group_titles[title] = title
        
        for action in group._group_actions:
            group_actions[title].append(action)

    content = [f'# Arguments\n\n\n']
    
    for group_title in group_actions:
        content.append(f'## {group_title}:\n\n')
        
        for action in group_actions[group_title]:
            name = ', '.join(action.option_strings) if action.option_strings else action.dest
            
            value = getattr(args, action.dest, None)
            if value is None:
                value = '⛔ not set'
            elif action.const and isinstance(value, bool):
                value = '✅' if value else '❌'
                
            content.append(f'- **{name}**: `{value}`\n')
        
        content.append('\n\n')

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(content)