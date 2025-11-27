from collections import defaultdict
import numpy as np
import os
import re
from sklearn.metrics import classification_report
import sys


TASK = 'ecapa_urbansound8k'
LOG_DIR = '.'
LOG_FILE = 'vis_log.txt'


def calc_score(confusion_matrix, info=False):
    if confusion_matrix is None:
        return None

    num_classes = confusion_matrix.shape[0]

    support_per_class = np.sum(confusion_matrix, axis=1)

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    if info:
        print("Метрики по классам:")
        print("{:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format("Class", "Precision", "Recall", "F1-Score", "Support"))
        print("-" * 65)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        
        if info:
            print("{:<10} | {:<10.4f} | {:<10.4f} | {:<10.4f} | {:<10}".format(
                f"Class {i}", precision, recall, f1, support_per_class[i]
            ))

    if info:
        print("-" * 65)

    # Общие метрики
    total_samples = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix) # Сумма диагональных элементов (TP для всех классов)
    overall_accuracy = correct_predictions / total_samples

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    weighted_precision = np.sum([p * s for p, s in zip(precision_per_class, support_per_class)]) / total_samples
    weighted_recall = np.sum([r * s for r, s in zip(recall_per_class, support_per_class)]) / total_samples
    weighted_f1 = np.sum([f * s for f, s in zip(f1_per_class, support_per_class)]) / total_samples

    if info:
        print("\nСводные метрики:")
        print(f"Overall Accuracy (Micro F1): {overall_accuracy:.4f}")
        print(f"Macro-averaged Precision:    {macro_precision:.4f}")
        print(f"Macro-averaged Recall:       {macro_recall:.4f}")
        print(f"Macro-averaged F1-Score:     {macro_f1:.4f}")
        print(f"Weighted-averaged Precision: {weighted_precision:.4f}")
        print(f"Weighted-averaged Recall:    {weighted_recall:.4f}")
        print(f"Weighted-averaged F1-Score:  {weighted_f1:.4f}")

    return weighted_f1


def load_confusion_matrix(root, name):
    try:
        fpath = f'{root}/{name}/speechbrain_output/log.txt'
        with open(fpath, 'r') as f:
            text = f.read()
    except Exception as e:
        log(f'WARNING: can not find {root}/{name}')
        return None
    if ' Confusion Matrix: ' not in text:
        log(f'WARNING: invalid file {root}/{name}')
        return None

    C = text.split(' Confusion Matrix: ')[1].split(' - test loss')[0]
    C = C.strip().strip('[]')
    lines = [line.strip() for line in C.split('\n') if line.strip()]
    matrix = []
    for line in lines:
        clean_line = re.sub(r'[\[\]]', '', line)
        numbers = [int(x) for x in clean_line.split() if x]
        if numbers:
            matrix.append(numbers)

    return np.array(matrix)


def log(text, init=False):
    fpath = os.path.join(LOG_DIR, LOG_FILE)
    with open(fpath, 'w' if init else 'a') as f:
        f.write(text + '\n')


def vis_base():
    log('>>> Start vis for "base" task\n')

    BB_KINDS = KINDS = ['matvec', 'mrr', 'slm', 'monarch']
    RANKS = [1, 5, 10, 50] #, 100]
    SEEDS = [1, 2, 3, 4, 5]

    res = []
    for seed in SEEDS:
        root = f'result_digital'
        name = f'digital_seed{seed}'
        s = calc_score(load_confusion_matrix(root, name))
        if s is not None:
            res.append(s)
    result_ref = np.mean(res)

    results = {rank: {kind: None for kind in KINDS} for rank in RANKS}
    for kind in KINDS:
        for rank in RANKS:
            res = []
            for seed in SEEDS:
                root = f'result_{kind}'
                name = f'bb_{kind}_rank{rank}_seed{seed}'
                s = calc_score(load_confusion_matrix(root, name))
                if s is not None:
                    res.append(s)
                results[rank][kind] = np.mean(res)

    # Convert accuracy values to percentages and format
    formatted_results = {}
    for rank in RANKS:
        formatted_results[rank] = {}
        for kind in BB_KINDS:
            val = results[rank][kind]
            if val is not None:
                formatted_results[rank][kind] = f"{val * 100:.2f}"
            else:
                formatted_results[rank][kind] = "--"

    # Build LaTeX table
    latex = [
        "\\begin{table}[t!]",
        "\\centering",
        "\\caption{Accuracy results (\%) for the speaker recognition problem using the UrbanSound8K dataset with the ECAPA-TDNN architecture. We consider different BB layer types and ranks of the SM. Reference accuracy of the digital model is " + f'{result_ref * 100:.2f}' + '\%.}',
        "\\label{tbl:results_ecapa_urbansound8k}",
        "\\begin{tabular}{c" + " c" * len(BB_KINDS) + "}",
        "\\toprule",
        "Rank & " + " & ".join(f"\\texttt{{{kind}}}" for kind in BB_KINDS) + " \\\\",
        "\\midrule"
    ]
    
    # Add table rows
    for rank in RANKS:
        row = [str(rank)]
        for kind in BB_KINDS:
            row.append(formatted_results[rank][kind])
        latex.append(" & ".join(row) + " \\\\")
    
    # Finalize table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    log('\n' + '-'*75 + '\n')
    log("\n".join(latex))


def vis_contrib():
    log('>>> Start vis for "contrib" task\n')

    KINDS = ['matvec', 'mrr', 'slm', 'monarch']
    RANK = 10
    SEEDS = [1, 2, 3, 4, 5]

    mode_types = {
        'BS': 'base',
        'ZO': 'bb_sm-skip',
        'SM': 'sm_bb-skip',
        'GD': 'baseline_gd-gd',
    }

    display_mode_names = {
        'BS': 'ZO+SM',
        'ZO': 'ZO-only',
        'SM': 'SM-only',
        'GD': 'GD',
    }

    all_results = defaultdict(dict)

    for kind in KINDS:
        current_kind_results = defaultdict(list)
        for mt_key, mt_folder_suffix in mode_types.items():
            for seed in SEEDS:
                if mt_key == 'BS':
                    root = f'result_{kind}'
                    name = f'bb_{kind}_rank{RANK}_seed{seed}'
                elif mt_key == 'ZO':
                    root = f'result_{kind}_contrib'
                    name = f'bb_{kind}_rank{RANK}_sm-skip_seed{seed}'
                elif mt_key == 'SM':
                    root = f'result_{kind}_contrib'
                    name = f'bb_{kind}_rank{RANK}_bb-skip_seed{seed}'
                elif mt_key == 'GD':
                    root = f'result_{kind}_contrib'
                    name = f'bb_{kind}_baseline_gd-gd_seed{seed}'
                else:
                    raise NotImplementedError(f"Unknown mode type: {mt_key}")

                s = calc_score(load_confusion_matrix(root, name))
                if s is not None:
                    current_kind_results[mt_key].append(s)

        for mt_key in mode_types.keys():
            accs = current_kind_results[mt_key]
            if accs:
                all_results[kind][mt_key] = np.mean(accs)
            else:
                all_results[kind][mt_key] = np.nan
                log(f'WARNING: no data for kind={kind}, mode={display_mode_names[mt_key]}')

    num_columns = 1 + len(KINDS)
    header_columns = ["Calculation Type"] + [k.capitalize() for k in KINDS]
    header_line = '\n' + " & ".join(header_columns) + '\n'

    text = r"""
        \begin{table}[t!]
        \centering
        \caption{Mean test accuracy (last epoch) averaged over seeds 1-5 for different sampling regimes (transposed).}
        \begin{tabular}{""".replace('        ', '').strip() + "c" * num_columns + r"""}
        \hline

        """.replace('        ', '').strip() + header_line + r"""
        \\ \hline
    """.replace('        ', '').strip() + '\n\n'

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
        text += row_text + "\n\\\\ \\hline\n"

    text += r"""
        \end{tabular}
        \end{table}
        """.strip().replace('        ', '') + '\n'
    
    log('\n' + '-'*75 + '\n')
    log(text)


def vis_noise():
    log('>>> Start vis for "noise" task\n')

    KINDS = ['matvec', 'mrr', 'slm', 'monarch']
    RANK = 10
    SAMPLES = 1000
    SEEDS = [1, 2, 3, 4, 5]
    NOISES = [10, 100, -1]

    all_results = defaultdict(dict)
    
    for kind in KINDS:
        current_kind_results = defaultdict(list)
        for noise in NOISES:
            for seed in SEEDS:
                if noise == -1:
                    root = f'result_{kind}'
                    name = f'bb_{kind}_rank{RANK}_seed{seed}'
                else:
                    root = f'result_{kind}_noise'
                    name = f'bb_{kind}_rank{RANK}_noise{noise}_seed{seed}'
                s = calc_score(load_confusion_matrix(root, name))
                if s is not None:
                    current_kind_results[noise].append(s)

        for noise in NOISES:
            accs = current_kind_results[noise]
            if accs:
                all_results[kind][noise] = np.mean(accs)
            else:
                all_results[kind][noise] = np.nan
                log(f'WARNING: no data for kind={kind}, noise={noise}')

    num_columns = 1 + len(KINDS)
    header_columns = ["Noise Level"] + [k.capitalize() for k in KINDS]
    header_line = '\n' + " & ".join(header_columns) + '\n'

    text = r"""
        \begin{table}[t!]
        \centering
        \caption{Mean test accuracy (last epoch) averaged over seeds 1-5 for different noise levels.}
        \label{tab:ecapa_urbansound8k_noise}
        \begin{tabular}{""".replace('        ', '').strip() + "c" * num_columns + r"""}
        \toprule

        """.replace('        ', '').strip() + header_line + r"""
        \\ \midrule
    """.replace('        ', '').strip() + '\n\n'

    for noise in NOISES:
        row_data = [str(noise)]
        for kind in KINDS:
            mean_val = all_results[kind].get(noise, np.nan)
            if np.isnan(mean_val):
                formatted_mean = "N/A"
            else:
                formatted_mean = f"{mean_val*100:.2f}"
            row_data.append(formatted_mean)
        row_text = " & ".join(row_data)
        text += row_text + " \\\\ \n"

    text += '\n' + r"""
        \end{tabular}
        \end{table}
        """.strip().replace('        ', '') + '\n'
    
    log('\n' + '-'*75 + '\n')
    log(text)


def vis_quan():
    log('>>> Start vis for "quan" task\n')

    KIND = 'slm'
    QUANS = [4, 8, 16, 256]
    RANKS = [1, 5, 10, 50]
    SEEDS = [1, 2, 3, 4, 5]

    quan_levels_for_columns = QUANS + ['none']
    display_quan_names = {
        4: '2-bit',
        8: '3-bit',
        16: '4-bit',
        256: '8-bit',
        'none': '32-bit',
    }

    all_results = defaultdict(lambda: defaultdict(list))
    
    for rank in RANKS:
        for seed in SEEDS:
            root = f'result_{KIND}'
            name = f'bb_{KIND}_rank{rank}_seed{seed}'
            s = calc_score(load_confusion_matrix(root, name))
            if s is not None:
                all_results[rank]['none'].append(s)

        for quan in QUANS:
            for seed in SEEDS:
                root = f'result_{KIND}_quan'
                name = f'bb_{KIND}_rank{rank}_quan-{quan}_seed{seed}'
                s = calc_score(load_confusion_matrix(root, name))
                if s is not None:
                    all_results[rank][quan].append(s)

    mean_results = defaultdict(dict)
    for rank in RANKS:
        for quan_level in quan_levels_for_columns:
            accs = all_results[rank][quan_level]
            if accs:
                mean_results[rank][quan_level] = np.mean(accs)
            else:
                mean_results[rank][quan_level] = np.nan
                log(f'WARNING: no data for r={rank}, q={display_quan_names[quan_level]}')

    num_columns = 1 + len(quan_levels_for_columns)
    header_columns = ["Rank"] + [display_quan_names[q] for q in quan_levels_for_columns]
    header_line = " & ".join(header_columns)

    text = r"""
    \begin{table}[t!]
    \centering
    \caption{Mean test accuracy (last epoch, \%) for {slm} layer on {ecapa_urbansound8k} dataset, averaged over seeds 1-5, for various ranks and quantization levels.}
    \label{tab:ecapa_urbansound8k_quan}
    \begin{tabular}{""".replace('    ', '').strip() + "c" * num_columns + r"""}
        \toprule
        """.replace('    ', '') + header_line.replace('    ', '').strip() + '\n' + r""" \\ \midrule
    """.replace('    ', '').strip() + '\n\n'

    for rank in RANKS:
        row_data = [str(rank)]
        for quan_level in quan_levels_for_columns:
            mean_val = mean_results[rank].get(quan_level, np.nan)
            if np.isnan(mean_val):
                formatted_mean = "N/A"
            else:
                formatted_mean = f"{mean_val*100:.2f}"
            row_data.append(formatted_mean)
        row_text = " & ".join(row_data)
        text += row_text + r" \\" + "\n"

    text += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """.replace('    ', '')

    log('\n' + '-'*75 + '\n')
    log(text)


if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)
    log('', init=True)
    
    log('-'*75)
    vis_base()
    log('\n' + '='*75 + '\n\n')

    log('-'*75)
    vis_contrib()
    log('\n' + '='*75 + '\n\n')

    log('-'*75)
    vis_noise()
    log('\n' + '='*75 + '\n\n')

    log('-'*75)
    vis_quan()
    log('\n' + '='*75 + '\n\n')
