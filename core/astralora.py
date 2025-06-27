import matplotlib.pyplot as plt
import numpy as np
import os
import torch


from .config import config
from .layer import AstraloraLayer
from .utils import init_log
from .utils import init_neptune
from .utils import init_path
from .utils import init_seed
from .utils import modify_gpu_args_for_cryri
from .utils import save_args_to_markdown


class Astralora:
    def __init__(self, task, with_neptune=True):
        self.args, args_parser = config(task)
        self.args = modify_gpu_args_for_cryri(self.args)

        init_seed(self.args.seed)
        init_path(self.args.name, self.args.root, self.args.rewrite)
        self.args.folder = os.path.join(self.args.root, self.args.name)

        fpath = os.path.join(self.args.folder, 'log.txt')
        self.log = init_log(fpath=fpath)

        self.device = torch.device(f'cuda:{self.args.device}')

        if with_neptune:
            self.nepman, self.nepman_url = init_neptune(
                self.args.name, 'set_neptune_env.sh', self.args)
            self.log('Use neptune. See: ' + self.url, 'res')
        else:
            self.nepman = None
            self.nepman_url = ''

        save_args_to_markdown(self.args, args_parser, self.path('args.md'))

        self.losses_trn = []
        self.losses_tst = []

        self.accs_trn = []
        self.accs_tst = []

    def build(self, layer):
        d_inp = layer.in_features
        d_out = layer.out_features

        if self.args.mode == 'digital':
            return layer

        if self.args.mode == 'bb':
            return AstraloraLayer(d_inp, d_out,
                self.args.bb_d, self.args.bb_kind,
                self.args.rank, self.args.samples_bb, self.args.samples_sm,
                not self.args.use_stochastic_w,
                self.args.use_gd_update, self.args.gd_update_iters,
                self.log, self.nepman)
        
        raise NotImplementedError

    def done(self, model):
        torch.save(model.state_dict(), self.path('model.pth'))

        np.savez_compressed(self.path('result.npz'), res={
            'args': self._args_to_dict(),
            'losses_trn': self.losses_trn,
            'losses_tst': self.losses_tst,
            'accs_trn': self.accs_trn,
            'accs_tst': self.accs_tst,
        })
        # res = np.load(fpath, allow_pickle=True).get('res').item()

        self.plot()

    def path(self, fpath):
        return os.path.join(self.args.folder, fpath)

    def plot(self):
        if len(self.losses_trn) == 0 or len(self.accs_trn) == 0:
            return
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_trn, label='Train Loss')
        plt.plot(self.losses_tst, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Evolution')

        plt.subplot(1, 2, 2)
        plt.plot(self.accs_trn, label='Train Accuracy')
        plt.plot(self.accs_tst, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Evolution')
        plt.tight_layout()

        plt.savefig(self.path('plot.png'))

    def step(self, epoch, loss_trn, loss_tst, acc_trn=None, acc_tst=None):
        if loss_trn is not None:
            self.losses_trn.append(loss_trn)
        if loss_tst is not None:
            self.losses_tst.append(loss_tst)
        if acc_trn is not None:
            self.accs_trn.append(acc_trn)
        if acc_tst is not None:
            self.accs_tst.append(acc_tst)

        text = ''

        if epoch is not None:
            text += f'# {epoch+1:-5d} | '
        else:
            text += f'ITER  | '
        
        if loss_trn is not None and loss_tst is not None:
            text += f'L > trn: {loss_trn:-8.2e}, tst: {loss_tst:-8.2e} | '
        elif loss_trn is not None:
            text += f'L > trn: {loss_trn:-8.2e} | '

        if acc_trn is not None and acc_tst is not None:
            text += f'A > trn: {acc_trn:-5.2f}, tst: {acc_tst:-5.2f}'
        elif acc_trn is not None:
            text += f'A > trn: {acc_trn:-5.2f}'

        self.log(text)

    def _args_to_dict(self):
        def _check(v):
            return isinstance(v, (bool, int, float, str))
        return {n: v for n, v in vars(self.args).items() if _check(v)}