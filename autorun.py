import os
import subprocess
import sys
from types import SimpleNamespace


def autorun(task, kind):
    print(f'>>> Run task "{task}" for the layer "{kind}".')
    
    if task == 'airbench_cifar':
        return autorun_airbench_cifar(task, kind)
    
    raise NotImplementedError
    

def autorun_airbench_cifar(task, kind):
    seeds = [1] #, 2, 3, 4, 5]
    ranks = [1] #, 3, 5, 7, 10, 50, 100]
    samples = [1] #, 10, 100, 1000]

    args = SimpleNamespace(**{
        'task': task,
        'root': f'{task}/result_{kind}',
        'mode': None,
        'seed': None,
        'name': None,
        'bb_kind': kind})

    for seed in seeds:
        args.seed = seed

        args.mode = 'digital'
        args.name = f'digital_seed{seed}'
        _run(args)

        args.mode = 'bb'

        for rank in ranks:
            args.rank = rank
            
            args.name = f'bb_{kind}_rank{rank}_baseline_seed{seed}'
            args.samples_bb = -1
            args.samples_sm = -1
            _run(args)

            for s in samples:
                args.name = f'bb_{kind}_rank{rank}_samples{s}_seed{seed}'
                args.samples_bb = s
                args.samples_sm = s
                _run(args)


def _args_to_command(args):
    command = ['python', 'script.py']
    for name, value in vars(args).items():
        command.append('--' + name)
        command.append(str(value))
    return command


def _check(args):
    fpath = f'{args.root}/{args.name}/result.npz'
    if not os.path.isfile(fpath):
        return True

    return False


def _run(args):
    if _check(args):
        command = _args_to_command(args)
        try:
            result = subprocess.run(command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
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
    kind = sys.argv[2] if len(sys.argv) > 2 else 'matvec'
    autorun(task, kind)