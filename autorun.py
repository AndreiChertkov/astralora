"""Script to perform multiple computations in automatic mode.

Run it as:

- "python autorun.py airbench_cifar matvec"

"""
import os
import subprocess
import sys
from types import SimpleNamespace


BB_KINDS = ['matvec', 'monarch', 'mrr', 'mzi', 'slm']


def autorun(task, kind_only=None):
    for kind in BB_KINDS:
        print(f'>>> Run task "{task}" for the layer "{kind}".')
        
        if kind_only is not None and kind != kind_only:
            print('... skip.')
            continue

        elif task == 'airbench_cifar':
            autorun_airbench_cifar(task, kind)
    
        elif task == 'cnn_cifar':
            autorun_cnn_cifar(task, kind)

        elif task == 'ecapa_urbansound8k':
            autorun_ecapa_urbansound8k(task, kind)

        elif task == 'nanogpt_fineweb_layers12':
            autorun_nanogpt_fineweb_layers12('nanogpt_fineweb', kind)

        elif task == 'nanogpt_fineweb_layers1':
            autorun_nanogpt_fineweb_layers1('nanogpt_fineweb', kind)

        elif task == 'finetune_prepare':
            autorun_finetune_prepare(task, kind)

        elif task == 'finetune':
            autorun_finetune(task, kind)

        else:
            raise NotImplementedError
    

def autorun_airbench_cifar(task, kind):
    root = f'{task}/result_{kind}'
    seeds = [1, 2, 3, 4, 5]
    ranks = [1, 3, 5, 7, 10, 50, 100]
    samples = [1, 10, 100, 1000]

    for seed in seeds:
        if kind == 'matvec':
            args = SimpleNamespace(**{'task': task, 'root': root})
            args.name = f'digital_seed{seed}'
            args.mode = 'digital'
            args.seed = seed
            _run(args)

        if False:
            args = SimpleNamespace(**{'task': task, 'root': root})
            args.name = f'bb_{kind}_baseline_gd-gd_seed{seed}'
            args.mode = 'bb'
            args.seed = seed
            args.bb_kind = kind
            args.samples_bb = -1
            args.samples_sm = 0
            args.skip_sm = True
            _run(args)

        for rank in ranks:
            args = SimpleNamespace(**{'task': task, 'root': root})
            args.name = f'bb_{kind}_rank{rank}_baseline_gd-svd_seed{seed}'
            args.mode = 'bb'
            args.seed = seed
            args.rank = rank
            args.bb_kind = kind
            args.samples_bb = -1
            args.samples_sm = -1
            _run(args)

            for s in samples:
                args = SimpleNamespace(**{'task': task, 'root': root})
                args.name = f'bb_{kind}_rank{rank}_samples{s}_seed{seed}'
                args.mode = 'bb'
                args.seed = seed
                args.rank = rank
                args.bb_kind = kind
                args.samples_bb = s
                args.samples_sm = s
                if kind == 'mzi': # TODO: note this
                    args.step_sm_rebuild = 100
                _run(args)


def autorun_cnn_cifar(task, kind):
    raise NotImplementedError


def autorun_ecapa_urbansound8k(task, kind, samples_bb=1000, samples_sm=1000):
    seeds = [1, 2, 3, 4, 5]
    ranks = [1, 5, 10, 50, 100]

    for seed in seeds:
        if False:
            args = SimpleNamespace(**{
                'task': task, 'seed': seed, 'root': f'{task}/result_{kind}'})
            args.name = f'digital_seed{seed}'
            args.mode = 'digital'
            _run(args)

        args = SimpleNamespace(**{
            'task': task, 'seed': seed, 'root': f'{task}/result_{kind}'})
        args.name = f'bb_{kind}_baseline_gd-gd_seed{seed}'
        args.mode = 'bb'
        args.bb_kind = kind
        args.samples_bb = -1
        args.skip_sm = True
        _run(args)

        for rank in ranks:
            args = SimpleNamespace(**{
                'task': task, 'seed': seed, 'root': f'{task}/result_{kind}'})
            args.name = f'bb_{kind}_rank{rank}_seed{seed}'
            args.mode = 'bb'
            args.rank = rank
            args.bb_kind = kind
            args.samples_bb = samples_bb
            args.samples_sm = samples_sm
            _run(args)


def autorun_finetune_prepare(task, kind):
    if kind != 'matvec':
        return

    if True:
        task = 'airbench_cifar'
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune/digital',
            'name': 'digital',
            'mode': 'digital',
            'task': task,
            'epochs': 1,
            'save_model': True}))

    if True:
        task = 'cnn_cifar'
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune/digital',
            'name': 'digital',
            'mode': 'digital',
            'task': task,
            'epochs': 1,
            'save_model': True}))

    if True:
        task = 'ecapa_urbansound8k'
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune/digital',
            'name': 'digital',
            'mode': 'digital',
            'task': task,
            'epochs': 1,
            'save_model': True}))
        
    if True:
        task = 'nanogpt_fineweb'
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune/digital',
            'name': 'digital',
            'mode': 'digital',
            'task': task,
            'torchrun': 1,
            'epochs': 1,
            'save_model': True}))


def autorun_finetune(task, kind):
    if kind != 'matvec':
        return

    tasks = [
        'airbench_cifar',
        'cnn_cifar',
        'ecapa_urbansound8k',
        'nanogpt_fineweb',
    ]

    for task in tasks:
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune/digital_test',
            'name': 'digital',
            'mode': 'digital',
            'task': task,
            'load_digital': f'{task}/result_finetune/digital',
            'torchrun': 1 if task == 'nanogpt_fineweb' else 0,
            'epochs': 1}))


def autorun_nanogpt_fineweb_layers1(task, kind, rank=100):
    if kind == 'matvec':
        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-1/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'digital'
        args.mode = 'digital'
        _run(args)

    if True:
        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-1/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'bb_{kind}_baseline_bb-gd_sm-gd'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.bb_num = 1
        args.samples_bb = -1
        args.skip_sm = True
        _run(args)

    for s in [1, 10, 100, 1000]:
        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-1/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'bb_{kind}_baseline_bb-gd_s{s}'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.bb_num = 1
        args.samples_bb = -1
        args.samples_sm = s
        _run(args)

        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-1/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'bb_{kind}_s{s}'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.bb_num = 1
        args.samples_bb = s
        args.samples_sm = s
        _run(args)


def autorun_nanogpt_fineweb_layers12(task, kind, samples_sm=1000):
    ranks = [1, 5, 10, 50, 100, 500]

    if False:
        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-12/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'digital'
        args.mode = 'digital'
        _run(args)

    for rank in ranks:
        args = SimpleNamespace(**{
            'root': f'{task}/result_layers-12/result_{kind}',
            'task': task,
            'torchrun': 1})
        args.name = f'bb_{kind}_rank{rank}_baseline_gd'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.bb_num = 12
        args.samples_bb = -1
        args.samples_sm = samples_sm
        _run(args)


def autorun_spec(task, kind):
    raise NotImplementedError('Outdated code')

    print(f'>>> Run task "{task}" for the layer "{kind}".')

    root = f'{task}/result_{kind}'
    ranks = [1, 10, 50, 100]

    args = SimpleNamespace(**{'task': task, 'root': root})
    args.name = f'digital'
    args.mode = 'digital'
    _run(args)

    args = SimpleNamespace(**{'task': task, 'root': root})
    args.name = f'baseline_full_{kind}'
    args.mode = 'bb'
    args.bb_kind = kind
    args.samples_bb = -1
    args.skip_sm = True
    _run(args)

    for rank in ranks:
        args = SimpleNamespace(**{'task': task, 'root': root})
        args.name = f'baseline_{kind}_rank{rank}'
        args.mode = 'bb'
        args.bb_kind = kind
        args.samples_bb = -1
        args.samples_sm = -1
        args.rank = rank
        _run(args)

    for rank in ranks:
        args = SimpleNamespace(**{'task': task, 'root': root})
        args.name = f'bb-gd_{kind}_rank{rank}'
        args.mode = 'bb'
        args.bb_kind = kind
        args.samples_bb = -1
        args.samples_sm = 1000
        args.rank = rank
        _run(args)

    for rank in ranks:
        args = SimpleNamespace(**{'task': task, 'root': root})
        args.name = f'bb_{kind}_rank{rank}'
        args.mode = 'bb'
        args.bb_kind = kind
        args.samples_bb = 100 if task == 'nanogpt_fineweb' else 1000
        args.samples_sm = 1000
        args.rank = rank
        _run(args)


def _args_to_command(args):
    command = ['python', 'script.py']
    for name, value in vars(args).items():
        command.append('--' + name)
        command.append(str(value))
    return command


def _check(args):
    fpath = f'{args.root}/{args.name}/result.npz'
    # fpath = f'{args.root}/{args.name}/log.txt'
    return not os.path.isfile(fpath)


def _run(args):
    if not _check(args):
        return

    command = _args_to_command(args)
    
    try:
        result = subprocess.run(command, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert 'Job is started. UID is' in result.stdout
        
        uid = result.stdout.split('UID is  : ')[1].split('.')[0]
        print(f'>>> DONE [uid={uid}]:\n        ', ' '.join(command))

    except subprocess.CalledProcessError as e:
        print(f'>>> ERROR for', ' '.join(command))
        print(f'... Details (code {e.returncode}):')
        print(e.stderr)
        raise ValueError


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'airbench_cifar'
    kind_only = sys.argv[2] if len(sys.argv) > 2 else None
    
    autorun(task, kind_only)