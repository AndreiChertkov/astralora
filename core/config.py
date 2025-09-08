import argparse
from distutils.util import strtobool
import os


def config(task, args_add={}):
    parser = argparse.ArgumentParser(
        prog='astralora',
        description='Adaptive Surrogate TRAining with LOw RAnk',
        epilog = 'Andrei Chertkov, Artem Basharin ©')

    for name, opts in args_add.items():
        parser.add_argument(f'--{name}', **opts)

    _config_base(task,
        parser.add_argument_group("Parameters for general options"))

    _config_astralora(task,
        parser.add_argument_group("Parameters for custom layer training"))

    if task == 'airbench_cifar':
        _config_airbench_cifar(task,
            parser.add_argument_group("Parameters for task airbench/cifar"))

    if task == 'cnn_cifar':
        _config_cnn_cifar(task,
            parser.add_argument_group("Parameters for task cnn/cifar"))

    if task == 'ecapa_urbansound8k':
        _config_ecapa_urbansound8k(task,
            parser.add_argument_group("Parameters for task ecapa/urbansound8k"))

    if task == 'nanogpt_fineweb':
        _config_nanogpt_fineweb(task,
            parser.add_argument_group("Parameters for task nanogpt/fineweb"))

    if task == 'vgg19_tiny':
        _config_vgg19_tiny_imagenet(task,
            parser.add_argument_group("Parameters for task vgg19/tiny"))

    if 'JPY_PARENT_PID' in os.environ:
        # Jupyter can not use the console arguments, hence empty list:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args, parser


def _config_airbench_cifar(task, parser):
    parser.add_argument('--epochs',
        type=int,
        help='Number of training epochs',
        default=100)


def _config_astralora(task, parser):
    parser.add_argument('--rank',
        type=int,
        help='Rank for the low-rank model.',
        default=10)

    parser.add_argument('--samples_bb',
        type=int,
        help='Number of samples to train bb (its parameters). If =-1, then "exact" (gradient-based) computation is performed (only for debug)',
        default=0)
    
    parser.add_argument('--samples_sm',
        type=int,
        help='Number of samples to update surrogate model. If =-1, then "exact" (svd-based) computation is performed (only for debug)',
        default=0)

    parser.add_argument('--step_sm_rebuild',
        type=int,
        help='Number of training steps after which we rebuild the surrogate model. If =-1, then rebuild is skipped',
        default=-1)

    parser.add_argument('--samples_bb_batch_frac',
        type=float,
        help='The fraction of the x-batch to use within stochastic formula. If < 0, then the full x-batch is used',
        default=-1)

    parser.add_argument('--skip_sm',
        type=lambda x: bool(strtobool(x)),
        help='If True, then we use the autograd to pass a gradient through a bb-layer instead of the surrogate model (only for debug)',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--use_residual',
        type=lambda x: bool(strtobool(x)),
        help='If True, then we add the residual connections for bb-layer',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--lr_bb',
        type=float,
        help='Learning rate for the BB layer',
        default=0.01)

    parser.add_argument('--sparse_top_p',
        type=float,
        help='Top-p fraction (0..1) of absolute weight values to keep as learnable sparse part when using mode=sparse_bb',
        default=0.1)


def _config_base(task, parser):
    parser.add_argument('--name',
        type=str,
        help='Name of the computation (it relates to the subfolder name in the folder with results)',
        default='test')
    
    parser.add_argument('--root',
        type=str,
        help='Path to the root folder, where results are saved',
        default=f'{task}/result')
    
    parser.add_argument('--root_data',
        type=str,
        help='Path to save the train / test data',
        default=f'{task}/_data')

    parser.add_argument('--mode',
        type=str,
        help='Kind of the model',
        choices=[
            'digital',
            'bb',
            'sparse_bb',
        ],
        default='digital')

    parser.add_argument('--seed',
        type=int,
        help='Initial global random seed value (use "None" for no seed)',
        default=1)
    
    parser.add_argument('--device',
        type=int,
        help='GPU device id to use. If it is not set (= -1), then auto selection will be performed',
        default=0)

    parser.add_argument("--bb_kind",
        type=str, 
        help="Kind of the used black box",
        choices=[
            "id",                    # Identity layer
            "matvec",                # Simple linear-like layer
            "monarch",               # Monarch layer
            "mrr",                   # MRR layer from torch-onn
            "mzi",                   # 3-MZI layer from torch-onn
            "mzi_classical",         # MZI layer from torch-onn
            "slm",                   # SLM layer
            "lowrank",               # Low-rank layer
        ],
        default="matvec")

    parser.add_argument("--load_digital",
        type=str, 
        help="Optional path to load the initial (digital) model",
        default=None)

    parser.add_argument('--save_model',
        type=lambda x: bool(strtobool(x)),
        help='Do we save model into file after train',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--rewrite',
        type=lambda x: bool(strtobool(x)),
        help='Do we auto delete the computation with the same name',
        nargs='?',
        const=True,
        default=False)
    

def _config_cnn_cifar(task, parser):
    parser.add_argument('--epochs',
        type=int,
        help='Number of training epochs',
        default=50)

    parser.add_argument('--batch_size',
        type=int,
        help='Batch size',
        default=512)

    parser.add_argument('--lr',
        type=float,
        help='Learning rate',
        default=0.001)


def _config_ecapa_urbansound8k(task, parser):
    parser.add_argument('--epochs',
        type=int,
        help='Number of training epochs',
        default=50)


def _config_nanogpt_fineweb(task, parser):
    parser.add_argument('--epochs',
        type=int,
        help='Number of iterations ("epochs") to run',
        default=5000)

    parser.add_argument('--bb_num',
        type=int,
        help='Number of BB-layers (it should be <= num_blocks)',
        default=1)

    parser.add_argument('--replace_feedforward',
        type=lambda x: bool(strtobool(x)),
        help='Do we replace the whole feedforward layer by BB (if True) or only linear layer (if False)',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--device_total',
        type=int,
        help='Number of required GPU devices to use',
        default=1)

    parser.add_argument('--batch_size',
        type=int,
        help='Batch size (per device)',
        default=16)

    parser.add_argument('--warmup_iters',
        type=int,
        default=0)

    parser.add_argument('--warmdown_iters',
        type=int,
        help='Number of iterations of linear warmup/warmdown',
        default=1450)

    parser.add_argument('--weight_decay',
        type=float,
        default=0)

    parser.add_argument('--tokens_vld',
        type=int,
        help='How many tokens of validation data?',
        default=10485760)

    parser.add_argument('--data_chunks',
        type=int,
        help='Number of downloaded Fineweb data chunks (note that the maximum possible value is 103)',
        default=103)

    parser.add_argument('--accumulation_steps',
        type=int,
        help='Number of accumulation steps while train',
        default=20) # TODO: note this argument (it should be 20 if one device)

    parser.add_argument('--sequence_length',
        type=int,
        help='Sequence length in tokens',
        default=1024)
    
    parser.add_argument('--num_blocks',
        type=int,
        help='Number of blocks inside the base model',
        default=12)

    parser.add_argument('--block_size',
        type=int,
        help='Size of the block',
        default=1024)

    parser.add_argument('--num_head',
        type=int,
        help='Number of attention heads inside the base model',
        default=6)

    parser.add_argument('--vld_every',
        type=int,
        help='Every how many steps to evaluate validation loss?',
        default=500)

    parser.add_argument('--lr_embed',
        type=float,
        help='Learning rate for head parameters',
        default=0.0036)

    parser.add_argument('--lr_muon',
        type=float,
        help='Learning rate for transformer.h parameters',
        default=0.02)


def _config_vgg19_tiny_imagenet(task, parser):
    parser.add_argument('--epochs',
        type=int,
        help='Number of training epochs',
        metavar='N',
        default=200)

    parser.add_argument('data', metavar='DIR', nargs='?', default='vgg19_tiny/data',
                    help='path to dataset (default: vgg19_tiny/data)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19',
                        choices=['vgg19', "vit_b_32"], help='vgg19')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--node-rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

    parser.add_argument('--replace-layers', type=str, default='',
                        help='comma separated list of layer indices to replace with LoRA')
    
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        choices=['tiny-imagenet', 'imagenet-1k'],
                        help='Dataset to use: tiny-imagenet or imagenet-1k (default: tiny-imagenet)')