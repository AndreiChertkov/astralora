import os
import subprocess


def autorun(task):
    seeds = [1, 2, 3, 4, 5]
    ranks = [1, 3, 5, 7, 10, 50, 100]
    samples = [1, 10, 100, 1000]

    for seed in seeds:
        _run(task, [
            "python", "script.py",
            "--task", task,
            "--mode", "digital",
            "--name", f"digital_seed{seed}",
            "--save_model", "False",
            '--seed', str(seed)])

        for r in ranks:
            _run(task, [
                "python", "script.py",
                "--task", task,
                "--mode", "bb",
                "--name", f"bb_matvec_rank{r}_baseline_seed{seed}",
                "--rank", str(r),
                "--bb_do_baseline",
                "--save_model", "False",
                '--seed', str(seed)])

            for s in samples:
                _run(task, [
                    "python", "script.py",
                    "--task", task,
                    "--mode", "bb",
                    "--name", f"bb_matvec_rank{r}_samples{s}_seed{seed}",
                    "--rank", str(r),
                    "--samples_bb", str(s),
                    "--samples_sm", str(s),
                    "--save_model", "False",
                    '--seed', str(seed)])


def _check(task, name):
    fpath = f'{task}/result/{name}/result.npz'
    if not os.path.isfile(fpath):
        return True

    return False


def _run(task, command):
    if _check(task, command[7]):
        result = subprocess.run(command, check=True)
        print("DONE:", " ".join(command))


if __name__ == '__main__':
    autorun('airbench_cifar')
    # autorun('cnn_cifar')