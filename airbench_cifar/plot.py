import matplotlib as mpl
import numpy as np
import os


mpl.rcParams.update({
    'font.size': 30,
    'text.usetex': False})


import matplotlib.pyplot as plt


BB_KINDS = ['monarch', 'mrr', 'slm', 'matvec']
seeds = [1, 2, 3, 4, 5]
ranks = [1, 3, 5, 7, 10, 50, 100]
samples_list = [1, 10, 100, 1000][::-1]
Y_TICKS = [
    [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    ['', 50, '', 60, '', 70, '', 80, '', 90, '', 100]]


def load(kind, base_dir, bs='gd-svd'):
    results = {s: [] for s in samples_list}
    results['digital'] = []
    results['baseline'] = []

    exp_name = "digital"
    digital_accs = load_one(base_dir, exp_name)
    results['digital'] = digital_accs

    for rank in ranks:
        exp_name = f"bb_{kind}_rank{rank}_baseline_{bs}"
        accs = load_one(base_dir, exp_name)
        results['baseline'].append(accs)

        for samples_val in samples_list:
            exp_name = f"bb_{kind}_rank{rank}_samples{samples_val}"
            accs = load_one(base_dir, exp_name)
            results[samples_val].append(accs)

    return results


def load_one(base_dir, name):
    accs = []
    for seed in seeds:
        fpath = os.path.join(base_dir, name + f'_seed{seed}', "result.npz")
        data = np.load(fpath, allow_pickle=True)
        res = data['res'].item()
        accs.append(res['accs_tst'][-1] * 100)
    return accs


def plot_line(x, y, label, linestyle='-', linewidth=4,
              marker='*', markersize=12, alpha=0.3):
    x = np.array(x)
    y = np.array(y)
    y_avg = np.mean(y, axis=1)
    dy = np.std(y, axis=1)

    plt.plot(x, y_avg, linestyle, linewidth=linewidth, label=label,
        marker=marker, markersize=markersize)
    plt.fill_between(x, y_avg - dy, y_avg + dy, alpha=alpha)


def plot(kind, base_dir="result", fpath_plot="result.png"):
    results = load(kind, base_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.)
    
    plt.xlabel("Rank")
    plt.ylabel("Accuracy, %")
    
    plot_line([0, max(ranks)], [results['digital'], results['digital']],
        "Digital", linestyle='--', linewidth=7, marker=None)

    plot_line(ranks, results['baseline'],
        "GD+SVD", linestyle='--', linewidth=7, marker='o', markersize=16)

    for samples_val in samples_list:
        label = f"{samples_val} sample{'s' if samples_val > 1 else ''}"
        plot_line(ranks, results[samples_val], label)

    plt.xscale('log')
    plt.xticks(ranks, labels=[str(r) for r in ranks])
    plt.yticks(*Y_TICKS)
    plt.ylim(Y_TICKS[0][0], Y_TICKS[0][-1])
    plt.legend(loc="lower right")
    plt.tight_layout()

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.savefig(fpath_plot, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('Saved plot in:', fpath_plot)


if __name__ == '__main__':
    os.makedirs("_plot", exist_ok=True)
    for kind in BB_KINDS:
        plot(kind, base_dir=f"result_{kind}",
            fpath_plot=f"_plot/airbench_cifar_result_{kind}.png")