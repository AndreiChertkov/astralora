import matplotlib.pyplot as plt
import numpy as np
import os
import torch


from .config import config
from .helpers.utils import init_log
from .helpers.utils import init_neptune
from .helpers.utils import init_path
from .helpers.utils import init_seed
from .helpers.utils import modify_gpu_args_for_cryri
from .helpers.utils import save_args_to_markdown
from .layer import AstraloraLayer


class Astralora:
    def __init__(self, task, with_neptune=False, master_process=True):
        self.task = task

        self.args, args_parser = config(task)
        # self.args = modify_gpu_args_for_cryri(self.args)

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
            self.nepman = None
            self.nepman_url = ''

        if master_process:
            save_args_to_markdown(self.args, args_parser, self.path('args.md'))

        self.losses_trn = []
        self.losses_tst = []

        self.accs_trn = []
        self.accs_tst = []

        self.times = []

    def build(self, layer):
        d_inp = layer.in_features
        d_out = layer.out_features

        if self.args.mode == 'digital':
            return layer

        if self.args.mode == 'bb':
            return AstraloraLayer(
                d_inp=d_inp,
                d_out=d_out,
                kind=self.args.bb_kind,
                rank=self.args.rank, 
                samples_bb=self.args.samples_bb,
                samples_sm=self.args.samples_sm,
                skip_sm=self.args.skip_sm,
                use_residual=self.args.use_residual,
                log=self.log,
                nepman=self.nepman)
        
        raise NotImplementedError

    def done(self, model=None):
        if model is not None:
            torch.save(model.state_dict(), self.path('model.pth'))

        np.savez_compressed(self.path('result.npz'), res={
            'args': self._args_to_dict(),
            'losses_trn': self.losses_trn,
            'losses_tst': self.losses_tst,
            'accs_trn': self.accs_trn,
            'accs_tst': self.accs_tst,
            'times': self.times})

        self.plot()

        if self.args.mode == 'bb':
            self.plot_bb_w()
        
    def path(self, fpath):
        return os.path.join(self.args.folder, fpath)

    def plot(self):
        if len(self.losses_trn) > 0 or len(self.losses_tst) > 0:        
            plt.figure(figsize=(6, 4))
            if len(self.losses_trn) > 0:
                plt.plot(self.losses_trn, label='Train Loss')
            if len(self.losses_tst) > 0:
                plt.plot(self.losses_tst, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Evolution')
            plt.tight_layout()
            plt.savefig(self.path('_plot_loss.png'))

        if len(self.accs_trn) > 0 or len(self.accs_tst) > 0:
            plt.figure(figsize=(6, 4))
            if len(self.accs_trn) > 0:
                plt.plot(np.array(self.accs_trn) * 100, label='Train Accuracy')
            if len(self.accs_tst) > 0:
                plt.plot(np.array(self.accs_tst) * 100, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Evolution')
            plt.tight_layout()
            plt.savefig(self.path('_plot_acc.png'))

    def plot_bb_w(self, epoch=None):
        params = [p for k, p in self.model.named_parameters()
            if hasattr(p, 'ast_bb')]

        w0 = self.w0.cpu().numpy()
        w1 = params[0].data.detach().clone().cpu().numpy()

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
        
        plt.savefig(self.path('bb_w.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def prepare(self, model):
        self.model = model

        params =  [p for p in model.parameters() 
            if getattr(p, 'ast_bb', False)]

        if self.args.mode == 'digital' and len(params) > 0:
            raise ValueError
        if self.args.mode == 'bb' and len(params) == 0:
            raise ValueError

        if self.args.mode == 'digital':
            return

        self.w0 = params[0].data.detach().clone()

        if self.task == 'nanogpt_fineweb': # TODO: note this
            epochs = self.args.num_iterations
        else:
            epochs = self.args.epochs

        self.optimizer = torch.optim.Adam(params,
            lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6)

        # self.optimizer = CustomSGD(params,
        #     lr=1e-2,
        #     momentum=0.95,
        #     nesterov=True,
        #     weight_decay=0.1,
        #     cycle_length=100)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def step(self):
        if self.args.mode == 'digital':
            return

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

        #if self.args.mode == 'bb':
        #    self.plot_bb_w(epoch)

    def _args_to_dict(self):
        def _check(v):
            return isinstance(v, (bool, int, float, str))
        return {n: v for n, v in vars(self.args).items() if _check(v)}


class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.95, nesterov=True, 
                 weight_decay=1e-4, cycle_length=100):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, cycle_length=cycle_length)
        
        super().__init__(params, defaults)
        
        self.step_counter = 0
        
    def step(self, closure=None):
        self.step_counter += 1
        
        for group in self.param_groups:
            if self.step_counter % group['cycle_length'] == 0:
                for p in group['params']:
                    if p in self.state:
                        self.state[p].pop('momentum_buffer', None)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                
                buf.mul_(group['momentum']).add_(grad)
                
                if group['nesterov']:
                    p.data.add_(buf.mul(group['momentum']).add(grad), alpha=-group['lr'])
                else:
                    p.data.add_(buf, alpha=-group['lr'])