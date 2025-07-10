import numpy as np
import os


BB_KINDS = ['matvec', 'slm', 'mrr', 'monarch']
RANKS = [1, 5, 10, 50, 100]


def load_results():
    ref_values = {}
    results = {rank: {kind: None for kind in BB_KINDS} for rank in RANKS}
    
    for kind in BB_KINDS:
        # Load digital reference
        ref_path = os.path.join(f'result_{kind}', 'digital', 'result.npz')

        if os.path.exists(ref_path):
            with np.load(ref_path, allow_pickle=True) as data:
                ref_values[kind] = parse_result(data['res'])
        else:
            ref_values[kind] = None

        for rank in RANKS:
            res_path = os.path.join(f'result_{kind}', f'bb_{kind}_rank{rank}', 'result.npz')
            if os.path.exists(res_path):
                with np.load(res_path, allow_pickle=True) as data:
                    results[rank][kind] = parse_result(data['res'])
    
    return ref_values, results


def parse_result(res):
    return res[()]['accs_tst'][-1]
    # return np.max(res[()]['accs_tst'])


def vis(ref_values, results):
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
    
    # Format reference values
    ref_strs = []
    for kind in BB_KINDS:
        if ref_values[kind] is not None:
            ref_strs.append(f"{ref_values[kind]*100:.2f}\%")
    
    # Build LaTeX table
    refs = ', '.join(ref_strs)
    latex = [
        "\\begin{table}[t!]",
        "\\centering",
        "\\caption{Accuracy results (\%) for ecapa\\_urbansound8k for different layer types and ranks of surrogate model. Reference digital accuracies: " + refs + '}'
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
    ref_values, results = load_results()
    print(vis(ref_values, results))
