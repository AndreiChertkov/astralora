"""Script to perform computation on our cluster.

Run it as:

- "python script.py --task TASK_NAME ...ARGS_TO_RUN_THE_SCRIPT..."

You can provide the number of required devices (1, 2, 4, or 8) by "device_num" argument, and you could set "torchrun 1" to run the python script for the task inside "script.py" by "torchrun" command.

"""
import os
import random
import string
import subprocess
import sys
import time


from core.config import config


TASKS = ['airbench_cifar', 'cnn_cifar', 'ecapa_urbansound8k',  
    'nanogpt_fineweb', 'vgg19_tiny']


ARGS_OWN = {
    'device_num': {
        'type': int,
        'help': 'Number of required GPU devices to use [it is argument only for "script.py", not for "run.py"]',
        'default': 1},
    'task': {
        'type': str,
        'choices': TASKS,
        'help': 'Name of the task (model / data) to solve [it is argument only for "script.py", not for "run.py"]'},
    'torchrun': {
        'type': int,
        'help': 'Do we use "torchrun" to run the script (if =1) [it is argument only for "script.py", not for "run.py"]',
        'default': 0}}


def args_to_str(args):
    keys_to_remove = ['--' + key for key in ARGS_OWN.keys()]

    new_args = []
    i = 0
    n = len(args)
    while i < n:
        if args[i] in keys_to_remove:
            i += 1
            if i < n and not args[i].startswith('--'):
                i += 1
        else:
            new_args.append(args[i])
            i += 1

    return (' '.join(new_args)).strip()


def get_text_ssh():
    return """
        #!/bin/bash
        set -e
        conda create --name astralora python=3.10 -y
        source $(conda info --base)/etc/profile.d/conda.sh
        source set_neptune_env.sh
        conda activate astralora
        conda install -c conda-forge conda-ecosystem-user-package-isolation -y
        pip install -e . --cache-dir cache
        conda install gcc_linux-64 -y && conda install gxx_linux-64 -y
        echo -e "\\n\\n\\n\\n\\n>>>>>>>>> WORK_START <<<<<<<<<\\n\\n\\n\\n\\n"
        SCRIPT_COMMAND_PLACEHOLDER
        echo -e "\\n\\n\\n\\n\\n>>>>>>>>> WORK_DONE <<<<<<<<<\\n\\n\\n\\n\\n"
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
          description: "astralora"
    """


def script():
    task = sys.argv[sys.argv.index("--task")+1]
    
    args, _ = config(task, ARGS_OWN)
    args_str = args_to_str(sys.argv[1:])

    if args.torchrun:
        runner = f'torchrun --standalone --nproc_per_node={args.device_num}'
    else:
        runner = 'python'

    script_command = f'{runner} {args.task}/run.py {args_str}'

    os.makedirs('_tmp', exist_ok=True)
    name_ssh = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_ssh = f'_tmp/{name_ssh}.sh'
    text_ssh = get_text_ssh().replace('    ', '')
    text_ssh = text_ssh.replace('SCRIPT_COMMAND_PLACEHOLDER', script_command)

    if False: # For debug:
        print(fpath_ssh)
        print(text_ssh)
        raise ValueError('stop')

    with open(fpath_ssh, 'w') as f:
        f.write(text_ssh)

    name_yaml = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_yaml = f'_tmp/{name_yaml}.yaml'
    text_yaml = get_text_yaml()
    
    # GPU configuration
    if args.device_num == 1:
        # gpu = 'a100plus.1gpu.80vG.12C.96G'
        # gpu = 'a100.1gpu.40vG.12C.230G'
        gpu = 'a100.1gpu.8C.243G'
    elif args.device_num == 2:
        gpu = 'a100plus.2gpu.80vG.24C.192G'
    elif args.device_num == 4:
        gpu = 'a100plus.4gpu.80vG.48C.728G'
    elif args.device_num == 8:
        gpu = 'a100plus.8gpu.80vG.96C.768G'
    else:
        raise NotImplementedError(f"Unsupported device_num: {args.device_num}")
    
    text_yaml = text_yaml.replace('GPU_PLACEHOLDER', gpu)

    with open(fpath_yaml, 'w') as f:
        f.write(text_yaml)

    comment = f'{args.task} (args: {args_str})'

    cmd = f'cryri {fpath_yaml} --script {fpath_ssh} --comment "{comment}"'

    out = subprocess.getoutput(cmd)
    time.sleep(5)

    if not 'Job submitted' in out:
        raise ValueError(out)
    
    uid = out[-8:]

    print(f'\n\n' + '-' * 50)
    print(f'Job is started. UID is  : {uid}.')
    print(f'See logs as             : cryri --logs {uid}')
    print(f'Kill job as             : cryri --kill {uid}')
    print(f'See all jobs as         : cryri --jobs_c')
    print('-' * 50 + '\n\n')

    # os.remove(fpath_ssh)
    os.remove(fpath_yaml)


if __name__ == '__main__':
    script()