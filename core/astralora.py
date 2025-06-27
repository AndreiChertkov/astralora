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
            self.nepman, self.url = init_neptune(
                self.args.name, 'set_neptune_env.sh', self.args)
            self.log('Use neptune. See: ' + self.url, 'res')
        else:
            self.nepman = None
            self.url = ''

        save_args_to_markdown(self.args, args_parser, self.path('args.md'))

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

    def path(self, fpath):
        return os.path.join(self.args.folder, fpath)

    def save_model(self, model):
        torch.save(model.state_dict(), self.path('model.pth'))

    def step(self, epoch, loss_trn, loss_tst, acc_trn=None, acc_tst=None):
        text = ''
        
        if epoch is not None:
            text += f'# {epoch+1:-4d} | '
        else:
            text += f'ITER | '
        
        text += f'L > trn: {loss_trn:-8.2e}, tst: {loss_tst:-8.2e} | '
        
        if acc_trn is not None and acc_tst is not None:
            text += f'A > trn: {acc_trn:-5.2f}, tst: {acc_tst:-5.2f}'
        
        self.log(text)