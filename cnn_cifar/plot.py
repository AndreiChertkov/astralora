import matplotlib as mpl
import numpy as np
import os

mpl.rcParams.update({
    'font.size': 30,
    'text.usetex': False,
})


import matplotlib.pyplot as plt


seeds = [1, 2, 3, 4, 5]
ranks = [1, 3, 5, 7, 10, 50, 100]
samples_list = [1, 10, 100, 1000][::-1]
base_dir = "result" 


def load():
    results = {s: [] for s in samples_list}
    results['digital'] = []
    results['baseline'] = []

    exp_name = f"digital"
    acc = load_one(exp_name)
    for rank in [0]+ranks:
        results['digital'].append(acc)

    for rank in ranks:
        for samples_val in samples_list:
            exp_name = f"bb_matvec_rank{rank}_samples{samples_val}"
            results[samples_val].append(load_one(exp_name))
        
        exp_name = f"bb_matvec_rank{rank}_baseline"
        results['baseline'].append(load_one(exp_name))

    return results


def load_one(name):
    accs = []
    for seed in seeds:
        fpath = os.path.join(base_dir, name + f'_seed{seed}', "result.npz")
        data = np.load(fpath, allow_pickle=True)
        res = data['res'].item()
        accs.append(res['accs_tst'][-1])
    return np.mean(accs)


def plot():
    results = load()

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.)

    plt.xlabel("Rank")
    plt.ylabel("Accuracy, %")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.plot([0]+ranks, results['digital'], '--', linewidth=6, label=f"Digital")

    valid_ranks = []
    valid_accs = []

    for i, rank in enumerate(ranks):
        acc = results['baseline'][i]
        if acc is not None:
            valid_ranks.append(rank)
            valid_accs.append(acc)

    plt.plot(valid_ranks, valid_accs, '--', linewidth=6, marker='*', markersize=12,
        label=f"Baseline")

    for samples_val in samples_list:
        valid_ranks = []
        valid_accs = []
        
        for i, rank in enumerate(ranks):
            acc = results[samples_val][i]
            if acc is not None:
                valid_ranks.append(rank)
                valid_accs.append(acc)
        
        label = f"{samples_val} samples"
        if samples_val == 1:
            label = label[:-1]
        plt.plot(valid_ranks, valid_accs, 'o-', linewidth=3, markersize=10,
            label=label)

    plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(ranks, labels=[str(r) for r in ranks])

    plt.legend(loc="best")
    plt.tight_layout()

    ax.grid(ls=":")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    fpath = base_dir + "/result_cnn_cifar.png"
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved plot in:', fpath)


if __name__ == '__main__':
    plot()