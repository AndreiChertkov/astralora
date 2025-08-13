import glob
import numpy as np
import os


BB_KINDS = ['matvec', 'slm', 'mrr', 'monarch']
RANKS = [1, 5, 10, 50, 100]
SEEDS = [1, 2, 3, 4, 5]


def load_results():
    results = {rank: {kind: None for kind in BB_KINDS} for rank in RANKS}
    
    kind_ref = 'slm' # TODO: note this
    pattern = os.path.join(f'result_{kind_ref}', f'digital_seed*', 'result.npz')
    seed_files = glob.glob(pattern)

    seed_values = []
    for seed_path in seed_files:
        try:
            with np.load(seed_path, allow_pickle=True) as data:
                seed_values.append(parse_result(data['res']))
        except:
            continue

    if len(seed_values) != 5:
        print('WARNING for ref')
    
    result_ref = np.mean(seed_values)

    for kind in BB_KINDS: 
        for rank in RANKS:
            pattern = os.path.join(f'result_{kind}', f'bb_{kind}_rank{rank}_seed*', 'result.npz')
            seed_files = glob.glob(pattern)

            seed_values = []
            for seed_path in seed_files:
                try:
                    with np.load(seed_path, allow_pickle=True) as data:
                        seed_values.append(parse_result(data['res']))
                except:
                    continue

            if len(seed_values) != 5:
                print(f'WARNING for {kind}')
            
            if len(seed_values) > 0:
                results[rank][kind] = np.mean(seed_values)
    
    return result_ref, results


def parse_result(res):
    return res[()]['accs_tst'][-1]
    # return np.max(res[()]['accs_tst'])


def vis(result_ref, results):
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
    
    return "\n".join(latex)


if __name__ == "__main__":
    result_ref, results = load_results()
    print(vis(result_ref, results))