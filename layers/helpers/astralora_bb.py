import math
import torch
import numpy as np


def bb_appr(bb, d_inp, d_out, w, rank=10, log=print, nepman=None, n_samples=1000, lr=0.01, max_iter=50):
    # Use ALS to approximate low-rank decomposition from samples
    
    # Get device from input tensor
    device = w.device
    
    # Generate random input samples
    X_samples = torch.randn(n_samples, d_inp, device=device)
    
    # Compute target outputs using the black-box function
    Y_target = bb(X_samples, w)
    
    # Create ALS model and fit (using PyTorch version)
    als_model = MatrixFactorizationALS(
        rank=rank, 
        max_iter=max_iter, 
        tol=1e-6,
        random_state=42,
        device=device
    )
    
    # Fit the model
    als_model.fit(X_samples, Y_target, verbose=False, minibatch_size=min(500, n_samples//10))
    
    # Get the factorization B, C where Y ≈ X @ B @ C
    B, C = als_model.get_factorization()
    
    # Convert back to U, V format:
    # Y_approx = X @ B @ C = X @ V.T @ U.T
    # So: V.T = B, U.T = C
    # Therefore: V = B.T, U = C.T
    # TODO: smth feels wrong here
    V = B.T  # (rank, d_inp)
    U = C.T  # (d_out, rank)
    
    # Log final loss
    if log:
        final_loss = als_model.losses[-1] if als_model.losses else 0.0
        log(f"Final ALS loss: {final_loss:.6f}")
        if nepman:
            nepman.log({"final_loss": final_loss})
    
    # Return the learned parameters
    # Orthogonalize U and V using QR decomposition
    U_q, U_r = torch.linalg.qr(U)
    V_q, V_r = torch.linalg.qr(V.T)
    V_q = V_q.T

    # Move non-orthogonal factors into S
    S = U_r @ V_r.T

    # Return orthogonalized U, S, V 
    return U_q, S, V_q


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




class MatrixFactorizationALS:
    """
    Matrix Factorization using Alternating Least Squares (ALS)
    
    Given dataset X, Y where Y = X @ A, finds A = B @ C
    where B is (n x r) and C is (r x n) with r << n
    """
    
    def __init__(self, rank, max_iter=100, tol=1e-6, random_state=42, device='cpu'):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = device
        self.B = None
        self.C = None
        self.losses = []
        
    def _initialize_factors(self, n, dtype=None):
        """Initialize B and C matrices randomly"""
        torch.manual_seed(self.random_state)
        self.B = torch.randn(n, self.rank, device=self.device, dtype=dtype) * 0.1
        self.C = torch.randn(self.rank, n, device=self.device, dtype=dtype) * 0.1
    
    def _compute_loss(self, X, Y):
        """Compute reconstruction loss"""
        Y_pred = X @ self.B @ self.C
        return torch.mean((Y - Y_pred) ** 2).item()
    
    def _update_B(self, X, Y):
        batch_size, n = X.shape
        dtype = X.dtype
        
        CCT = self.C @ self.C.T  # Shape: (r, r)
        try:
            CCT_inv = torch.linalg.pinv(CCT)
            W = Y @ self.C.T @ CCT_inv  # Shape: (batch_size, r)
        except RuntimeError:
            reg = 1e-6
            CCT_reg = CCT + reg * torch.eye(CCT.shape[0], device=self.device, dtype=dtype)
            W = Y @ self.C.T @ torch.linalg.inv(CCT_reg)

        XXT = X @ X.T  # Shape: (batch_size, batch_size)
        try:
            XXT_inv = torch.linalg.pinv(XXT)
            self.B = X.T @ XXT_inv @ W  # Shape: (n, r)
        except RuntimeError:
            reg = 1e-6
            XXT_reg = XXT + reg * torch.eye(XXT.shape[0], device=self.device, dtype=dtype)
            self.B = X.T @ torch.linalg.inv(XXT_reg) @ W
    
    def _update_C(self, X, Y):
        Z = X @ self.B  # Shape: (batch_size, rank)
        dtype = X.dtype
        
        try:
            ZTZ = Z.T @ Z  # Shape: (r, r)
            ZTY = Z.T @ Y  # Shape: (r, n)
            ZTZ_inv = torch.linalg.pinv(ZTZ)
            self.C = ZTZ_inv @ ZTY  # Shape: (r, n)
        except RuntimeError:
            # Fallback to regularized version
            reg = 1e-6
            ZTZ_reg = ZTZ + reg * torch.eye(ZTZ.shape[0], device=self.device, dtype=dtype)
            self.C = torch.linalg.inv(ZTZ_reg) @ Z.T @ Y

    def _sample_minibatch(self, X, Y, minibatch_size):
        batch_size, n = X.shape
        if minibatch_size < batch_size:
            indices = torch.randperm(batch_size, device=self.device)[:minibatch_size]
            X_batch = X[indices]
            Y_batch = Y[indices]
        else:
            X_batch = X
            Y_batch = Y
        return X_batch, Y_batch

    def fit(self, X, Y, verbose=True, minibatch_size=100):
        batch_size, n = X.shape
        assert Y.shape == (batch_size, n), "X and Y must have the same shape"
        # Initialize factors with same dtype as input
        self._initialize_factors(n, dtype=X.dtype)
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            X_batch, Y_batch = self._sample_minibatch(X, Y, minibatch_size)
            self._update_B(X_batch, Y_batch)

            X_batch, Y_batch = self._sample_minibatch(X, Y, minibatch_size)
            self._update_C(X_batch, Y_batch)
            
            current_loss = self._compute_loss(X, Y)
            self.losses.append(current_loss)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {current_loss:.6f}")
            
            if abs(prev_loss - current_loss) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            prev_loss = current_loss
        
        return self
    
    def predict(self, X):
        """Predict Y given X using the learned factorization"""
        if self.B is None or self.C is None:
            raise ValueError("Model not fitted yet")
        return X @ self.B @ self.C
    
    def get_factorization(self):
        """Return the learned factors B and C"""
        return self.B.clone(), self.C.clone()
    
    def reconstruct_A(self):
        """Reconstruct the original matrix A = B @ C"""
        if self.B is None or self.C is None:
            raise ValueError("Model not fitted yet")
        return self.B @ self.C
