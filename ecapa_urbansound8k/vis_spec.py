from collections import defaultdict
import numpy as np
import os


FPATH = 'vis_spec_log.txt'
KINDS = ['matvec', 'mrr', 'slm', 'monarch']
RANK = 10
SEEDS = [1, 2, 3, 4, 5]


mode_types = {
    'BS': 'base',
    'ZO': 'bb_sm-skip',
    'SM': 'sm_bb-skip',
    'GD': 'gd',
}


display_mode_names = {
    'BS': 'ZO+SM',
    'ZO': 'ZO-only',
    'SM': 'SM-only',
    'GD': 'GD-only',
}


all_results = defaultdict(dict)
for kind in KINDS:
    current_kind_results = defaultdict(list)
    for mt_key, mt_folder_suffix in mode_types.items():
        for seed in SEEDS:
            if mt_key == 'BS':
                folder = f'bb_{kind}_rank{RANK}_seed{seed}'
            elif mt_key == 'ZO':
                folder = f'bb_{kind}_rank{RANK}_sm-skip_seed{seed}'
            elif mt_key == 'SM':
                folder = f'bb_{kind}_rank{RANK}_bb-skip_seed{seed}'
            elif mt_key == 'GD':
                folder = f'bb_{kind}_baseline_gd-gd_seed{seed}'
            else:
                raise NotImplementedError(f"Unknown mode type: {mt_key}")

            npz_path = os.path.join(f'./result_{kind}', folder, 'result.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                res = data['res'].item()
                accs_tst = res['accs_tst']
                if len(accs_tst) > 0:
                    last_acc = accs_tst[-1]
                    current_kind_results[mt_key].append(last_acc)
                data.close()
            else:
                print(f"Warning: {npz_path} not found.")

    for mt_key in mode_types.keys():
        accs = current_kind_results[mt_key]
        if accs:
            all_results[kind][mt_key] = np.mean(accs)
        else:
            all_results[kind][mt_key] = np.nan
            print(f"Warning: No data for kind={kind}, mode={display_mode_names[mt_key]}")


num_columns = 1 + len(KINDS)
header_columns = ["Calculation Type"] + [k.capitalize() for k in KINDS]
header_line = '\n' + " & ".join(header_columns) + '\n'
with open(FPATH, 'w') as f:
    f.write(r"""
        \begin{table}[t!]
        \centering
        \caption{Mean test accuracy (last epoch) averaged over seeds 1-5 for different sampling regimes (transposed).}
        \begin{tabular}{""".replace('        ', '').strip() + "c" * num_columns + r"""}
        \hline

        """.replace('        ', '').strip() + header_line + r"""
        \\ \hline
    """.replace('        ', '').strip() + '\n\n')

    for mt_key in list(mode_types.keys()):
        row_data = [display_mode_names[mt_key]]
        for kind in KINDS:
            mean_val = all_results[kind].get(mt_key, np.nan)
            if np.isnan(mean_val):
                formatted_mean = "N/A"
            else:
                formatted_mean = f"{mean_val*100:.2f}"
            row_data.append(formatted_mean)
        row_text = " & ".join(row_data)
        f.write(row_text + "\n\\\\ \\hline\n")

    f.write(r"""
        \end{tabular}
        \end{table}
        """.strip().replace('        ', '') + '\n')