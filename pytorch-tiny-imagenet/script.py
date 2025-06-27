import os
import random
import string
import subprocess
import time
import argparse


ARGS_OWN = {
    'device_num': {
        'type': int,
        'help': 'Number of required GPU devices to use [it is argument only for "script.py", not for "launch_training.py"]',
        'default': 1}
}


def create_parser():
    """Create argument parser for pytorch-tiny-imagenet training"""
    parser = argparse.ArgumentParser(description='Launch Pytorch Tiny ImageNet training on cloud')
    
    # Add pytorch-tiny-imagenet specific arguments
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
    
    # Add script-specific arguments
    for arg_name, arg_config in ARGS_OWN.items():
        parser.add_argument(f'--{arg_name}', **arg_config)
    
    return parser


def args_to_str(args):
    """Convert arguments to string for launch_training.py"""
    args_str = ''
    for arg_name in vars(args):
        if arg_name in ARGS_OWN:
            continue
        arg_val = getattr(args, arg_name)
        args_str += f'--{arg_name.replace("_", "-")} {arg_val} '
    return args_str


def get_text_ssh():
    """Generate SSH script for cloud execution"""
    return """
        #!/bin/bash
        set -e
        conda create --name pytorch-tiny-imagenet python=3.9 -y
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate pytorch-tiny-imagenet
        
        # Install dependencies
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install opencv-python tqdm livelossplot
        
        # Prepare dataset
        python prepare_dataset.py
        
        # Run training
        python launch_training.py ARGS_PLACEHOLDER
        
        echo ">>> PYTORCH TINY IMAGENET TRAINING IS DONE <<<"
    """


def get_text_yaml():
    """Generate YAML configuration for cloud deployment"""
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
          description: "pytorch tiny imagenet training"
    """


def script():
    parser = create_parser()
    args = parser.parse_args()

    args_str = args_to_str(args)

    os.makedirs('_tmp', exist_ok=True)
    name_ssh = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_ssh = f'_tmp/{name_ssh}.sh'
    text_ssh = get_text_ssh().replace('    ', '')
    text_ssh = text_ssh.replace('ARGS_PLACEHOLDER', args_str)

    with open(fpath_ssh, 'w') as f:
        f.write(text_ssh)

    name_yaml = ''.join(random.choices(string.ascii_uppercase, k=25))
    fpath_yaml = f'_tmp/{name_yaml}.yaml'
    text_yaml = get_text_yaml()
    
    # Set GPU configuration based on device_num
    if args.device_num == 1:
        gpu = 'a100plus.1gpu.80vG.12C.96G'
    elif args.device_num == 4:
        gpu = 'a100plus.4gpu.80vG.48C.728G'
    elif args.device_num == 8:
        gpu = 'a100plus.8gpu.80vG.96C.768G'
    else:
        raise NotImplementedError(f"device_num={args.device_num} is not supported")
    
    text_yaml = text_yaml.replace('GPU_PLACEHOLDER', gpu)

    with open(fpath_yaml, 'w') as f:
        f.write(text_yaml)

    comment = f'pytorch tiny imagenet training: {args.model} {args.input_size}x{args.input_size} {args.epochs}epochs'

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

    # Clean up temporary files
    os.remove(fpath_ssh)
    os.remove(fpath_yaml)


if __name__ == '__main__':
    script() 