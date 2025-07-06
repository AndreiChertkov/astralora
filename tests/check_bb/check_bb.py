import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch


sys.path.append('../..')
from core.layer import AstraloraLayer
from core.helpers.backprop import _backprop_stochastic


DEVICE = 'cuda:4'
FOLDER = '_plot_check_bb'
KINDS = ['matvec', 'mrr', 'slm']
SHIFTS = [1e-7, 1e-5, 1e-3, 1e-1, 1e+1, 1e+3]
SAMPLES = [1, 10, 100, 1000, 10000, 100000, 1000000]


def check_bb(kind, d_inp=1000, d_out=2000, batch_size=1000):
    _log(f'\n\n>>> Check layer "{kind}"')

    device = DEVICE

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    os.makedirs(FOLDER, exist_ok=True)

    layer = AstraloraLayer(d_inp, d_out, kind,
        rank=1, samples_bb=0, samples_sm=0, 
        skip_sm=True, use_residual=False, log=_log)
    layer = layer.to(device)

    x = _random((batch_size, d_inp), generator, device)
    w = _random(d_inp * d_out, generator, device)
    g = _random((batch_size, d_out), generator, device)

    y = layer.bb(x, w)

    _plot_distributions(x, y, kind)

    with torch.enable_grad():
        w_grad = w.detach().requires_grad_(True)
        y_w = layer.bb(x.detach(), w_grad)
        grad_w_real, = torch.autograd.grad(
            outputs=y_w,
            inputs=w_grad,
            grad_outputs=g)

    _log()

    errors_dict_an = {s: [] for s in SHIFTS}
    errors_dict_sc = {s: [] for s in SHIFTS}

    for shift in SHIFTS:
        for samples in SAMPLES:
            grad_w_pred = _backprop_stochastic(
                layer.bb, x, w, g, generator, samples, shift)
            
            gr = grad_w_real.flatten()
            gp = grad_w_pred.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(gr, gp, dim=0)
            e_an = np.degrees(torch.acos(torch.clamp(cos_sim, -1., 1.)).item())
            errors_dict_an[shift].append(e_an.item())
            
            e_sc = torch.norm(grad_w_pred) / torch.norm(grad_w_real)
            errors_dict_sc[shift].append(e_sc.item())

            text = ''
            text += f'samples = {samples:-7d} | '
            text += f'shift = {shift:-8.2e} | '
            text += f'e_an = {e_an:-8.2e} | '
            text += f'e_sc = {e_sc:-8.2e}'
            _log(text)
        
        _log()

    _plot_errors(errors_dict_an, kind, is_angle=True)
    _plot_errors(errors_dict_sc, kind, is_angle=False)


def _log(text='', init=False):
    with open('_log.txt', 'w' if init else 'a+') as f:
        f.write(text+'\n')
    print(text)


def _plot_distributions(x, y, kind):
    plt.figure(figsize=(12, 7))
    
    x_np = x.cpu().detach().numpy().flatten()
    y_np = y.cpu().detach().numpy().flatten()

    bins = 100
    alpha = 0.6
    log_scale = False

    plt.hist(x_np, bins=bins, alpha=alpha, 
             label='Input (x)', density=True, log=log_scale)
    plt.hist(y_np, bins=bins, alpha=alpha, 
             label='Output (y)', density=True, log=log_scale)

    plt.title(f'Input/Output Distribution: {kind}', fontsize=16)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density' if not log_scale else 'Log Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{FOLDER}/distribution_{kind}.png',
        dpi=150, bbox_inches='tight')
    plt.close()


def _plot_errors(errors_dict, kind, is_angle=True):
    plt.figure(figsize=(10, 7))

    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'd', 'v', '<', '>']
    colors = plt.cm.viridis(np.linspace(0, 1, len(SHIFTS)))
    
    for i, shift_val in enumerate(SHIFTS):
        style_idx = i % len(line_styles)
        marker_idx = i % len(marker_styles)
        
        plt.plot(
            SAMPLES, errors_dict[shift_val],
            label=f'{shift_val:.0e}',
            linestyle=line_styles[style_idx],
            marker=marker_styles[marker_idx],
            markersize=4,
            color=colors[i]
        )
    
    plt.semilogx()
    plt.semilogy()
    plt.title(f'Gradient evaluation error for layer "{kind}"', fontsize=18)
    plt.xlabel('Number of Samples', fontsize=16)
    if is_angle:
        plt.ylabel('Angle', fontsize=16)
    else:
        plt.ylabel('Scale', fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='upper right', fontsize=14, title='shift:')
    
    fkind = '_angle' if is_angle else '_scale'
    fpath = f'{FOLDER}/grad_error_{kind}_{fkind}.png'
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()


def _random(shape, generator, device):
    m = torch.zeros(shape, device=device)
    s = torch.tensor(1., device=device)
    v = torch.normal(m, std=s, generator=generator)
    return v


if __name__ == "__main__":
    _log('', init=True)
    for kind in KINDS:
        check_bb(kind=kind)