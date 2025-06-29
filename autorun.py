import os
import subprocess


def autorun_cnn_cifar():
    seeds = [1, 2, 3, 4, 5]
    ranks = [1, 3, 5, 7, 10, 50, 100]
    samples = [1, 10, 100, 1000]

    for seed in seeds:
        _run([
            "python", "script.py",
            "--task", "cnn_cifar",
            "--mode", "digital",
            "--name", f"digital_seed{seed}",
            "--save_model", "False",
            '--seed', str(seed)])

        for r in ranks:
            _run([
                "python", "script.py",
                "--task", "cnn_cifar",
                "--mode", "bb",
                "--name", f"bb_matvec_rank{r}_baseline_seed{seed}",
                "--rank", str(r),
                "--bb_do_baseline",
                "--save_model", "False",
                '--seed', str(seed)])

            for s in samples:
                _run([
                    "python", "script.py",
                    "--task", "cnn_cifar",
                    "--mode", "bb",
                    "--name", f"bb_matvec_rank{r}_samples{s}_seed{seed}",
                    "--rank", str(r),
                    "--samples_bb", str(s),
                    "--samples_sm", str(s),
                    "--save_model", "False",
                    '--seed', str(seed)])


def _check(name):
    fpath = f'cnn_cifar/result/{name}/result.npz'
    if not os.path.isfile(fpath):
        return True

    return False


def _run(command):
    if _check(command[7]):
        result = subprocess.run(command, check=True)
        print("DONE:", " ".join(command))


if __name__ == '__main__':
    autorun_cnn_cifar()