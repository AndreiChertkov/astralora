import math
import torch


def bb_appr(bb, d_inp, d_out, w, rank=10, log=print, nepman=None):
    # TODO: add real bb-approximation here
    # TODO: add log + nepman logging of approximation proccess
    A = w.reshape(d_out, d_inp)
    
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    U = U[:, :rank]
    S = torch.diag(torch.sqrt(s[:rank]))
    V = V[:rank, :]

    return U, S, V


def bb_build(d_inp, d_out, d, kind='matvec'):
    if kind == 'matvec':
        assert d_inp * d_out == d
    
        def bb(X, w):
            A = w.reshape(d_out, d_inp)
            return X @ A.T

        w = torch.empty((d_out, d_inp))
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        w = w.reshape(-1)

    else:
        raise NotImplementedError
    
    return bb, w