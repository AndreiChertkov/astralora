"""Script to perform multiple computations in automatic mode.

Run it as:

- "python autorun.py airbench_cifar matvec"

"""
import os
import subprocess
import sys
from types import SimpleNamespace


BB_KINDS = ['matvec', 'monarch', 'mrr', 'slm']
BB_KINDS_ADD = ['mzi', 'mzi_classical']


def autorun(task, kind_only=None):
    if task == 'airbench_cifar':
        kinds = BB_KINDS + BB_KINDS_ADD
    else:
        kinds = BB_KINDS 

    for kind in kinds:
        print(f'>>> Run task "{task}" for the layer "{kind}".')
        
        if kind_only is not None and kind != kind_only:
            print('... skip.')
            continue

        elif task == 'airbench_cifar':
            autorun_airbench_cifar(task, kind)
    
        elif task == 'airbench_cifar_check':
            autorun_airbench_cifar_check('airbench_cifar', kind)
    
        elif task == 'cnn_cifar':
            autorun_cnn_cifar(task, kind)

        elif task == 'ecapa_urbansound8k':
            autorun_ecapa_urbansound8k(task, kind)

        elif task == 'ecapa_urbansound8k_check':
            autorun_ecapa_urbansound8k_check('ecapa_urbansound8k', kind)

        elif task == 'nanogpt_fineweb':
            autorun_nanogpt_fineweb('nanogpt_fineweb', kind)

        elif task == 'nanogpt_fineweb_baseline':
            autorun_nanogpt_fineweb_baseline('nanogpt_fineweb', kind)

        elif task == 'finetune_prepare':
            autorun_finetune_prepare(task, kind)

        elif task == 'finetune':
            autorun_finetune(task, kind)

        elif task == 'finetune_gd':
            autorun_finetune_gd(task, kind)

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
            if False:
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
                args.rewrite = True # TODO: note this
                if kind in ['mzi', 'mzi_classical']: # TODO: note this
                    args.step_sm_rebuild = 100
                _run(args)


def autorun_airbench_cifar_check(task, kind, rank=3):
    root = f'{task}/result_{kind}'
    seeds = [1, 2, 3, 4, 5]
    samples = [1, 10, 100, 1000]

    for seed in seeds:
        for s in samples:
            args = SimpleNamespace(**{'task': task, 'root': root})
            args.name = f'bb_{kind}_bs_samples{s}_sm-skip_seed{seed}'
            args.mode = 'bb'
            args.seed = seed
            args.bb_kind = kind
            args.samples_bb = s
            args.samples_sm = 0
            args.skip_sm = True
            _run(args)

        for s in samples:
            args = SimpleNamespace(**{'task': task, 'root': root})
            args.name = f'bb_{kind}_bs_rank{rank}_samples{s}_bb-skip_seed{seed}'
            args.mode = 'bb'
            args.seed = seed
            args.rank = rank
            args.bb_kind = kind
            args.samples_bb = -1
            args.samples_sm = s
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


def autorun_ecapa_urbansound8k_check(task, kind, rank=10, samples=1000):
    seeds = [1, 2, 3, 4, 5]

    for seed in seeds:
        args = SimpleNamespace(**{
            'task': task, 'seed': seed, 'root': f'{task}/result_{kind}'})
        args.name = f'bb_{kind}_rank{rank}_sm-skip_seed{seed}'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.samples_bb = samples
        args.samples_sm = 0
        args.skip_sm = True
        _run(args)

        args = SimpleNamespace(**{
            'task': task, 'seed': seed, 'root': f'{task}/result_{kind}'})
        args.name = f'bb_{kind}_rank{rank}_bb-skip_seed{seed}'
        args.mode = 'bb'
        args.rank = rank
        args.bb_kind = kind
        args.samples_bb = -1
        args.samples_sm = samples
        _run(args)


def autorun_finetune_prepare(task, kind):
    if kind != 'matvec':
        return

    task = 'airbench_cifar'
    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': 'digital',
        'mode': 'digital',
        'task': task,
        'save_model': True}))

    task = 'cnn_cifar'
    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': 'digital',
        'mode': 'digital',
        'task': task,
        'save_model': True}))

    task = 'ecapa_urbansound8k'
    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': 'digital',
        'mode': 'digital',
        'task': task,
        'save_model': True}))
    
    task = 'nanogpt_fineweb'
    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': 'digital',
        'mode': 'digital',
        'task': task,
        'torchrun': 1,
        'save_model': True}))


def autorun_finetune(task, kind, rank=100,
                     bb_num=4, samples_bb=100, samples_sm=1000):
    task = 'nanogpt_fineweb'

    if False:
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune',
            'name': f'digital_check',
            'mode': 'digital',
            'task': task,
            'load_digital': 'nanogpt_fineweb/result_finetune/digital',
            'rewrite': True,
            'torchrun': 1}))

    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': f'bb_{kind}_l{bb_num}_base',
        'mode': 'bb',
        'rank': rank,
        'task': task,
        'bb_kind': kind,
        'bb_num': bb_num,
        'samples_bb': samples_bb,
        'samples_sm': samples_sm,
        'load_digital': 'nanogpt_fineweb/result_finetune/digital',
        'rewrite': True,
        'torchrun': 1}))

    for p in [1, 2, 3, 4, 5]:
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune',
            'name': f'bb_{kind}_l{bb_num}_sparse-{p}',
            'mode': 'sparse_bb',
            'rank': rank,
            'task': task,
            'bb_kind': kind,
            'bb_num': bb_num,
            'samples_bb': samples_bb,
            'samples_sm': samples_sm,
            'load_digital': 'nanogpt_fineweb/result_finetune/digital',
            'sparse_top_p': p * 1./10,
            'rewrite': True,
            'torchrun': 1}))

        if True:
            _run(SimpleNamespace(**{
                'root': f'{task}/result_finetune',
                'name': f'id_l{bb_num}_sparse-{p}',
                'mode': 'sparse_bb',
                'rank': rank,
                'task': task,
                'bb_kind': 'id',
                'bb_num': bb_num,
                'samples_bb': -1,
                'skip_sm': True,
                'load_digital': 'nanogpt_fineweb/result_finetune/digital',
                'sparse_top_p': p * 1./10,
                'rewrite': True,
                'torchrun': 1}))


def autorun_finetune_gd(task, kind, rank=100,
                        bb_num=4, samples_sm=1000):    
    task = 'nanogpt_fineweb'

    _run(SimpleNamespace(**{
        'root': f'{task}/result_finetune',
        'name': f'bb_{kind}_l{bb_num}_bb-gd_base',
        'mode': 'bb',
        'rank': rank,
        'task': task,
        'bb_kind': kind,
        'bb_num': bb_num,
        'samples_bb': -1,
        'samples_sm': samples_sm,
        'load_digital': 'nanogpt_fineweb/result_finetune/digital',
        'rewrite': True,
        'torchrun': 1}))

    for p in [1, 2, 3, 4, 5]:
        _run(SimpleNamespace(**{
            'root': f'{task}/result_finetune',
            'name': f'bb_{kind}_l{bb_num}_bb-gd_sparse-{p}',
            'mode': 'sparse_bb',
            'rank': rank,
            'task': task,
            'bb_kind': kind,
            'bb_num': bb_num,
            'samples_bb': -1,
            'samples_sm': samples_sm,
            'load_digital': 'nanogpt_fineweb/result_finetune/digital',
            'sparse_top_p': p * 1./10,
            'rewrite': True,
            'torchrun': 1}))


def autorun_nanogpt_fineweb(task, kind,
                            samples_bb=100, samples_sm=1000, rank=100):
    if False: #kind == 'matvec':
        _run(SimpleNamespace(**{
            'root': f'{task}/result',
            'name': f'digital',
            'mode': 'digital',
            'task': task,
            'torchrun': 1}))

    for l in [1, 4, 8, 12]:
        _run(SimpleNamespace(**{
            'root': f'{task}/result_feedforward', # _feedforward
            'name': f'bb_{kind}_l{l}', # f'bb_gd_{kind}_l{l}',
            'mode': 'bb',
            'rank': rank,
            'task': task,
            'bb_kind': kind,
            'bb_num': l,
            'samples_bb': samples_bb,  # -1,
            'samples_sm': samples_sm,
            'replace_feedforward': True, # True
            'torchrun': 1}))
    

def autorun_nanogpt_fineweb_baseline(task, kind, rank=50):
    if kind != 'matvec':
        return

    i = 1 # for i in range(2):
    for l in [1, 4, 8, 12]:
        root = f'{task}/result' if i == 0 else f'{task}/result_feedforward'
        _run(SimpleNamespace(**{
            'root': root,
            'name': f'bb_gd_sm_gd_id_l{l}',
            'mode': 'bb',
            'rank': rank,
            'task': task,
            'bb_kind': 'id',
            'bb_num': l,
            'samples_bb': -1,
            'skip_sm': True,
            'replace_feedforward': i == 1,
            'torchrun': 1}))
        _run(SimpleNamespace(**{
            'root': root,
            'name': f'bb_gd_sm_gd_lowrank_l{l}',
            'mode': 'bb',
            'rank': rank,
            'task': task,
            'bb_kind': 'lowrank',
            'bb_num': l,
            'samples_bb': -1,
            'skip_sm': True,
            'replace_feedforward': i == 1,
            'torchrun': 1}))


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
    # TODO: note this:
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