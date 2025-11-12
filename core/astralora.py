import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import warnings


from .bb_layers.bb_layer_mrr import build_weight_from_phase as build_weight_from_phase_for_mrr
from .config import config
from .helpers.utils import init_log
from .helpers.utils import init_neptune
from .helpers.utils import init_path
from .helpers.utils import init_seed
from .helpers.utils import save_args_to_markdown
from .layer import AstraloraLayer
from .sparse_bb_layer import SparseBBLayer


class Astralora:
    def __init__(self, task, with_neptune=False, master_process=True):
        self.task = task

        self.args, args_parser = config(task)

        # torch.backends.cudnn.benchmark = True
        
        init_seed(self.args.seed)
        
        if master_process:
            init_path(self.args.name, self.args.root, self.args.rewrite)
        self.args.folder = os.path.join(self.args.root, self.args.name)

        fpath = os.path.join(self.args.folder, 'log.txt')
        self.log = init_log(fpath=fpath, enable=master_process)

        self.device = torch.device(f'cuda:{self.args.device}')

        if with_neptune and master_process:
            self.nepman, self.nepman_url = init_neptune(
                self.args.name, 'set_neptune_env.sh', self.args)
            self.log('Use neptune. See: ' + self.url, 'res')
        else:
            self.nepman, self.nepman_url = None, ''

        if master_process:
            save_args_to_markdown(self.args, args_parser, self.path('args.md'))

        self.losses_trn = []
        self.losses_tst = []

        self.accs_trn = []
        self.accs_tst = []

        self.times = []

        self.bb_num = 0

        self.steps = 0

        self.layers_bb = []

        if not 'epochs' in self.args:
            raise ValueError('Number of epochs is not provided')

    def build(self, layer, d_inp=None, d_out=None):
        if d_inp is None:
            d_inp = layer.in_features
        if d_out is None:
            d_out = layer.out_features

        if self.args.mode == 'digital':
            return layer

        if self.args.mode == 'bb':
            self.bb_num += 1
            layer_bb = AstraloraLayer(
                d_inp=d_inp,
                d_out=d_out,
                kind=self.args.bb_kind,
                rank=self.args.rank, 
                samples_bb=self.args.samples_bb,
                samples_sm=self.args.samples_sm,
                samples_bb_batch_frac=self.args.samples_bb_batch_frac,
                skip_sm=self.args.skip_sm,
                use_residual=self.args.use_residual,
                quan_x=self.args.quan_x,
                quan_w=self.args.quan_w,
                quan_n_x=self.args.quan_n_x,
                quan_n_w=self.args.quan_n_w,
                quan_lim_x_min=self.args.quan_lim_x_min,
                quan_lim_x_max=self.args.quan_lim_x_max,
                quan_lim_w_min=self.args.quan_lim_w_min,
                quan_lim_w_max=self.args.quan_lim_w_max,
                log=self.log,
                nepman=self.nepman)
            self.layers_bb.append(layer_bb)
            return layer_bb
        
        if self.args.mode == 'sparse_bb':
            self.bb_num += 1
            layer_sparse_bb = SparseBBLayer(
                base_layer=layer,
                d_inp=d_inp,
                d_out=d_out,
                bb_kind=self.args.bb_kind,
                rank=self.args.rank,
                samples_bb=self.args.samples_bb,
                samples_sm=self.args.samples_sm,
                samples_bb_batch_frac=self.args.samples_bb_batch_frac,
                skip_sm=self.args.skip_sm,
                use_residual=self.args.use_residual,
                sparse_top_p=self.args.sparse_top_p,
                log=self.log,
                nepman=self.nepman)
            self.layers_bb.append(layer_sparse_bb)
            return layer_sparse_bb
        
        raise NotImplementedError

    def done(self, model=None):
        if model is not None and self.args.save_model:
            torch.save(model.state_dict(), self.path('model.pth'))

        np.savez_compressed(self.path('result.npz'), res={
            'args': self._args_to_dict(),
            'losses_trn': self.losses_trn,
            'losses_tst': self.losses_tst,
            'accs_trn': self.accs_trn,
            'accs_tst': self.accs_tst,
            'times': self.times,
            'bb_num': self.bb_num})

        self.plot()

        for num in range(self.bb_num):
            if self.args.mode in ['bb', 'sparse_bb']:
                self.plot_bb_w(num)
            if self.args.mode in ['bb', 'sparse_bb'] and self.args.bb_kind == 'mrr':
                self.plot_bb_w_matrix_for_mrr(num)
        
    def path(self, fpath):
        return os.path.join(self.args.folder, fpath)

    def plot(self):
        if len(self.losses_trn) > 0 or len(self.losses_tst) > 0:        
            plt.figure(figsize=(6, 4))
            if len(self.losses_trn) > 0:
                plt.plot(self.losses_trn, label='Train')
            if len(self.losses_tst) > 0:
                plt.plot(self.losses_tst, label='Test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.path('_plot_loss.png'))
            plt.close()

        if len(self.accs_trn) > 0 or len(self.accs_tst) > 0:
            plt.figure(figsize=(6, 4))
            if len(self.accs_trn) > 0:
                plt.plot(np.array(self.accs_trn) * 100, label='Train')
            if len(self.accs_tst) > 0:
                plt.plot(np.array(self.accs_tst) * 100, label='Test')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.path('_plot_acc.png'))
            plt.close()

    def plot_bb_w(self, num=0, epoch=None):
        params = [p for p in self.model.parameters()
            if getattr(p, 'ast_bb_weight', False)]

        w0 = self.w0[num].cpu().numpy()
        w1 = params[num].data.detach().clone().cpu().numpy()

        plt.figure(figsize=(12, 7))

        log_scale = False

        plt.hist(w0, bins=500, alpha=0.7, 
                label='Initial', density=True, log=log_scale, color='green')
        plt.hist(w1, bins=500, alpha=0.4, 
                label='Final', density=True, log=log_scale, color='blue')

        if epoch is not None:
            plt.title(f'BB parameters for epoch: {epoch+1}', fontsize=16)
        else:
            plt.title(f'BB parameters', fontsize=16)
        
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density' if not log_scale else 'Log Density', fontsize=12)
        
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        
        fpath = self.path(f'bb_{num}_w.png')
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bb_w_matrix_for_mrr(self, num=0, epoch=None):
        params = [p for p in self.model.parameters()
            if getattr(p, 'ast_bb_weight', False)]

        w0 = self.w0[num]
        w1 = params[num].data.detach().clone()

        if self.args.bb_kind == 'mrr':
            A0 = build_weight_from_phase_for_mrr(w0).cpu().numpy()
            A1 = build_weight_from_phase_for_mrr(w1).cpu().numpy()
        elif self.args.bb_kind == 'matvec':
            A0 = w0.cpu().numpy()
            A1 = w1.cpu().numpy()
        else:
            warnings.warn(f'BB kind {self.args.bb_kind} not supported for plotting entire matrix')

        plt.figure(figsize=(12, 7))

        log_scale = False
        
        plt.hist(A0.flatten(), bins=500, alpha=0.7, 
                label='Initial', density=True, log=log_scale, color='green')
        plt.hist(A1.flatten(), bins=500, alpha=0.4, 
                label='Final', density=True, log=log_scale, color='blue')

        if epoch is not None:   
            plt.title(f'BB parameters for epoch: {epoch+1}', fontsize=16)
        else:
            plt.title(f'BB parameters', fontsize=16)
        
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density' if not log_scale else 'Log Density', fontsize=12)
        
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        
        fpath = self.path(f'bb_{num}_w_matrix.png')
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close()

    def prepare(self, model):
        self.model = model

        params = [p for p in self.model.parameters()
            if getattr(p, 'ast_bb', False)]

        if self.args.mode == 'digital' and len(params) > 0:
            raise ValueError
        if self.args.mode in ['bb', 'sparse_bb'] and len(params) == 0:
            raise ValueError

        if self.args.mode in ['bb', 'sparse_bb'] and len(params) != self.bb_num * 2:
            raise ValueError

        if self.args.mode == 'digital':
            return

        self.optimizer = torch.optim.Adam(params, lr=self.args.lr_bb)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=1e-6)

        # Initial parameters of the BB layers:
        params = [p for p in self.model.parameters()
            if getattr(p, 'ast_bb_weight', False)]
        self.w0 = [p.data.detach().clone() for p in params]

    def step(self):
        if self.args.mode == 'digital':
            return

        self.steps += 1

        if self.args.step_sm_rebuild > 0:
            if self.steps % self.args.step_sm_rebuild == 0:
                for layer_bb in self.layers_bb:
                    layer_bb.rebuild()

        self.optimizer.step()
        self.scheduler.step()

    def step_before(self):
        if self.args.mode == 'digital':
            return

        self.optimizer.zero_grad()

    def step_end(self, epoch=None, loss_trn=None, loss_tst=None,
                 acc_trn=None, acc_tst=None, t=None):

        if loss_trn is not None:
            self.losses_trn.append(loss_trn)
        if loss_tst is not None:
            self.losses_tst.append(loss_tst)
        if acc_trn is not None:
            if acc_trn > 1:
                self.log('Accuracy (trn) should be <1', 'wrn')
            self.accs_trn.append(acc_trn)
        if acc_tst is not None:
            if acc_tst > 1:
                self.log('Accuracy (tst) should be <1', 'wrn')
            self.accs_tst.append(acc_tst)

        text = ''

        if epoch is not None:
            text += f'# {epoch+1:-5d} | '
        else:
            text += f'ITER    | '
        
        if loss_trn is not None and loss_tst is not None:
            text += f'L > trn: {loss_trn:-8.2e}, tst: {loss_tst:-8.2e}'
        elif loss_trn is not None:
            text += f'L > trn: {loss_trn:-8.2e}'
        elif loss_tst is not None:
            text += f'L > tst: {loss_tst:-8.2e}'

        if acc_trn is not None and acc_tst is not None:
            text += f' | A > trn: {acc_trn*100:-4.2f}, tst: {acc_tst*100:-4.2f}'
        elif acc_trn is not None:
            text += f' | A > trn: {acc_trn*100:-4.2f}'
        elif acc_tst is not None:
            text += f' | A > tst: {acc_tst*100:-4.2f}'

        if t is not None:
            text += f' | T: {t:-8.2e}'

        self.log(text)

    def _args_to_dict(self):
        def _check(v):
            return isinstance(v, (bool, int, float, str))
        return {n: v for n, v in vars(self.args).items() if _check(v)}