import torch


def psi_implicit(f_old, f_new, U0, S0, V0, samples=100):
    """A projector-splitting integrator (PSI) for dynamical low-rank appr."""
    V0 = V0.T

    def compute_P1(f_new, f_old, V0): # Compute dA @ V0
        V0_batch = V0.T
        res_old = f_old(V0_batch)
        res_new = f_new(V0_batch)
        return (res_new - res_old).T

    P1 = compute_P1(f_new, f_old, V0)

    K1 = U0 @ S0 + P1
    U1, S0_tld = torch.linalg.qr(K1, mode='reduced')
    S0_hat = S0_tld - U1.T @ P1

    def compute_P2(f_new, f_old, U1, d_inp, samples): # Compute dA^t @ U1
        Z = torch.randn(samples, d_inp, device=U1.device, dtype=U1.dtype)
        AZ = f_new(Z) - f_old(Z)
        AZT_U = AZ @ U1
        return Z.T @ AZT_U / samples

    d_inp = V0.shape[0]
    P2 = compute_P2(f_new, f_old, U1, d_inp, samples)

    L1 = V0 @ S0_hat.T + P2

    V1, S1_T = torch.linalg.qr(L1, mode='reduced')
    S1 = S1_T.T

    return U1, S1, V1.T