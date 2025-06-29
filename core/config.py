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

    if task == 'cnn_cifar':
        _config_cnn_cifar(task,
            parser.add_argument_group("Parameters for cnn / cifar"))

    if task == 'nanogpt_fineweb':
        _config_nanogpt_fineweb(task,
            parser.add_argument_group("Parameters for nanogpt / fineweb"))

    if 'JPY_PARENT_PID' in os.environ:
        # Jupyter can not use the console arguments, hence empty list:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args, parser


def _config_astralora(task, parser):
    parser.add_argument('--rank',
        type=int,
        help='Rank for the low-rank model. Used in bb, bb_one, and truelowrank modes',
        default=10)

    parser.add_argument('--samples_bb',
        type=int,
        help='Number of samples to train bb (its parameters)',
        default=100)
    
    parser.add_argument('--samples_sm',
        type=int,
        help='Number of samples to update surrogate model',
        default=100)

    parser.add_argument('--gd_update_iters',
        type=int,
        default=1)
    
    parser.add_argument('--use_gd_update',
        type=lambda x: bool(strtobool(x)),
        help='It true, we use gd-based naive update instead of PSI',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--use_stochastic_w',
        type=lambda x: bool(strtobool(x)),
        help='Do we use stochastic formula to update w; if not, a surrogate model will be used',
        nargs='?',
        const=True,
        default=False)

    parser.add_argument('--bb_do_baseline',
        type=lambda x: bool(strtobool(x)),
        help='Do we perform exact computations of gradients and SVD (fto otain a baseline)',
        nargs='?',
        const=True,
        default=False)


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
            'bb'
        ],
        default='digital')

    parser.add_argument('--seed',
        type=int,
        help='Initial global random seed value',
        default=1)
    
    parser.add_argument('--device',
        type=int,
        help='GPU device id to use. If it is not set (= -1), then auto selection will be performed',
        default=0)

    parser.add_argument('--bb_d',
        type=int,
        help='Number of parameters in the bb-layer. If = -1, then auto set',
        default=-1)

    parser.add_argument("--bb_kind",
        type=str, 
        help="Kind of the used black box",
        choices=[
            "matvec",                # Simple linear-like layer
            "slm",                   # SLM layer
        ],
        default="matvec")

    parser.add_argument('--gpus',
        type=str,
        help='Numbers of used GPUs (comma-separated). Note that "nproc_per_node" argument for torchrun should be chosen accordingly (! TODO ! remove nproc_per_node)',
        default='0')

    parser.add_argument('--save_model',
        type=lambda x: bool(strtobool(x)),
        help='Do we auto-save model after train',
        nargs='?',
        const=True,
        default=True)

    parser.add_argument('--rewrite',
        type=lambda x: bool(strtobool(x)),
        help='Do we auto delete the computation with the same name',
        nargs='?',
        const=True,
        default=False)
    

def _config_cnn_cifar(task, parser):
    parser.add_argument('--batch_size',
        type=int,
        help='Batch size',
        default=512)

    parser.add_argument('--epochs',
        type=int,
        help='Number of training epochs',
        default=50)

    parser.add_argument('--lr',
        type=float,
        help='Learning rate',
        default=0.001)


def _config_nanogpt_fineweb(task, parser):
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
        default=1)

    parser.add_argument('--sequence_length',
        type=int,
        help='Sequence length in tokens',
        default=1024)
    
    parser.add_argument('--num_blocks',
        type=int,
        help='Number of blocks inside the base model',
        default=12)

    parser.add_argument('--num_head',
        type=int,
        help='Number of attention heads inside the base model',
        default=6)

    parser.add_argument('--num_iterations',
        type=int,
        help='Number of iterations to run (note: it was 5100 in base repo)',
        default=5000)

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
    
    parser.add_argument('--lr_bb',
        type=float,
        help='Learning rate for bb-layers',
        default=0.0036)
    
    parser.add_argument('--surrogate_lr',
        type=float,
        default=0.01)