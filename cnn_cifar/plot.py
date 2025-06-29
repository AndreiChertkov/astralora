import matplotlib as mpl
import numpy as np


mpl.rcParams.update({
    'font.size': 30,
    'text.usetex': False,
})


import matplotlib.pyplot as plt
import os


ranks = [1, 3, 5, 7, 10, 25, 50, 75, 100]
samples_list = [1, 10, 50, 100, 500, 1000][::-1]
base_dir = "result" 

results = {s: [] for s in samples_list}
results['digital'] = []
results['baseline'] = []

for rank in ranks:
    for samples_val in samples_list:
        exp_name = f"bb_matvec_rank{rank}_samples{samples_val}"
        file_path = os.path.join(base_dir, exp_name, "result.npz")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            res = data['res'].item()
            accs_tst = res['accs_tst']
            
            final_acc = accs_tst[-1]
            results[samples_val].append(final_acc)
            
            print(f"Loaded: rank={rank}, samples={samples_val}, accuracy={final_acc:.2f}%")
            
        except Exception as e:
            results[samples_val].append(None)
            print(f"Error for {exp_name}: {str(e)}")
    
    exp_name = f"bb_matvec_rank{rank}_baseline"
    file_path = os.path.join(base_dir, exp_name, "result.npz")
    data = np.load(file_path, allow_pickle=True)
    res = data['res'].item()
    accs_tst = res['accs_tst']
    final_acc = accs_tst[-1]
    results['baseline'].append(final_acc)
    print(f"Loaded: rank={rank}, baseline, accuracy={final_acc:.2f}%")

exp_name = f"digital"
file_path = os.path.join(base_dir, exp_name, "result.npz")
data = np.load(file_path, allow_pickle=True)
res = data['res'].item()
accs_tst = res['accs_tst']
final_acc = accs_tst[-1]
for rank in [0]+ranks:
    results['digital'].append(final_acc)
print(f"Loaded: digital, accuracy={final_acc:.2f}%")

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

plt.savefig(base_dir + "/result_cnn_cifar.png", dpi=300, bbox_inches='tight')
plt.close(fig)