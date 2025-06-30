import os
import random
import string
import subprocess
import sys
import time


from core.config import config


ARGS_OWN = {
    'device_num': {
        'type': int,
        'help': 'Number of required GPU devices to use [it is argument only for "script.py", not for "run.py"]',
        'default': 1},
    'task': {
        'type': str,
        'choices': ['airbench_cifar', 'cnn_cifar', 'ecapa_urbansound8k', 'nanogpt_fineweb', 'vgg19_tiny'],
        'help': 'Name of the task (model / data) to solve',
        'default': 'cnn_cifar'}
}


def parse_args_with_separator():
    """Parse arguments, separating cryri args from custom command using '--' separator"""
    if '--' in sys.argv:
        separator_index = sys.argv.index('--')
        cryri_args = sys.argv[1:separator_index]
        custom_command_parts = sys.argv[separator_index + 1:]
    else:
        cryri_args = sys.argv[1:]
        custom_command_parts = []
    
    # Temporarily modify sys.argv for the config function
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + cryri_args
    
    try:
        args, _ = config('TASK_PLACEHOLDER', ARGS_OWN)
    finally:
        sys.argv = original_argv
    
    return args, custom_command_parts


def args_to_str(args):
    args_str = ''
    for arg_name in vars(args):
        if arg_name in ARGS_OWN:
            continue
        arg_val = getattr(args, arg_name)
        if arg_name == 'suffix' and not arg_val:
            continue # TODO: fix it
        args_str += f'--{arg_name} {arg_val} '
    return args_str


def get_text_ssh():
    return """
        #!/bin/bash
        set -e
        conda create --name astralora python=3.10 -y
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate astralora
        conda install -c conda-forge conda-ecosystem-user-package-isolation -y
        pip install -e .
        conda install gcc_linux-64 -y && conda install gxx_linux-64 -y
        SCRIPT_COMMAND_PLACEHOLDER
        echo ">>> WORK IS DONE <<<"
    """


def get_text_yaml():
    return """
        container:
          image: "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py39:0.0.36"
          command: ???                     
          work_dir: '.'
          run_from_copy: False
          cry_copy_dir: "/home/jovyan/<LOCAL FOLDER>/.cryri"
        cloud:
          region: "SR006"   # Cloud region to deploy the job
          instance_type: "GPU_PLACEHOLDER"
          n_workers: 1  # Number of worker instances, 1 is only option
          priority: "medium" # ['high', 'medium', 'low']
          description: "astralora job"
    """


def script():
    args, custom_command_parts = parse_args_with_separator()

    # Always handle task-based placeholder replacements
    args.root_data = args.root_data.replace('TASK_PLACEHOLDER', args.task)
    args.root = args.root.replace('TASK_PLACEHOLDER', args.task)

    # Determine what command to run
    if custom_command_parts:
        # Custom command mode - use whatever comes after --
        script_command = f'torchrun --standalone --nproc_per_node=1 {args.task}/run.py ' + ' '.join(custom_command_parts)
        description = f'{args.task} custom: {script_command}'
    else:
        # Original torchrun mode
        args_str = args_to_str(args)
        script_command = f'torchrun --standalone --nproc_per_node=1 {args.task}/run.py {args_str}'.strip()
        description = f'astralora {args.task}'

    os.makedirs('_tmp', exist_ok=True)
    name_ssh = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_ssh = f'_tmp/{name_ssh}.sh'
    text_ssh = get_text_ssh().replace('    ', '')
    text_ssh = text_ssh.replace('SCRIPT_COMMAND_PLACEHOLDER', script_command)

    print('-' * 50)
    print(text_ssh)
    print('-' * 50)

    with open(fpath_ssh, 'w') as f:
        f.write(text_ssh)

    name_yaml = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_yaml = f'_tmp/{name_yaml}.yaml'
    text_yaml = get_text_yaml()
    
    # GPU configuration
    if args.device_num == 1:
        gpu = 'a100plus.1gpu.80vG.12C.96G'
    elif args.device_num == 4:
        gpu = 'a100plus.4gpu.80vG.48C.728G'
    elif args.device_num == 8:
        gpu = 'a100plus.8gpu.80vG.96C.768G'
    else:
        raise NotImplementedError(f"Unsupported device_num: {args.device_num}")
    
    text_yaml = text_yaml.replace('GPU_PLACEHOLDER', gpu)

    with open(fpath_yaml, 'w') as f:
        f.write(text_yaml)

    comment = f'run {description}'

    cmd = f'cryri {fpath_yaml} --script {fpath_ssh} --comment "{comment}"'

    out = subprocess.getoutput(cmd)
    time.sleep(5)

    if not 'Job submitted' in out:
        raise ValueError(out)
    
    uid = out[-8:]

    print(f'\n\n' + '-' * 50)
    print(f'Job is started. UID is  : {uid}')
    print(f'See logs as             : cryri --logs {uid}')
    print(f'Kill job as             : cryri --kill {uid}')
    print(f'See all jobs as         : cryri --jobs_c')
    print('-' * 50 + '\n\n')

    # os.remove(fpath_ssh)
    os.remove(fpath_yaml)


if __name__ == '__main__':
    script()