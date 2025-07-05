import math
import torch


def approximation(*args, **kwargs):
    # TODO: note that we always use SVD:
    return bb_appr_w_svd(*args, **kwargs)


def bb_appr_w_als(bb, d_inp, d_out, w, rank=10, log=print, nepman=None, 
                  n_samples=1000, lr=0.01, max_iter=50):
    raise NotImplementedError('Outdated code')

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
        device=device,
        use_stochastic=True,  # Use stochastic updates for large matrices
        learning_rate=lr
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


def bb_appr_w_svd(bb, d_inp, d_out, w, rank=10, log=print, nepman=None):
    device = w.device

    E = torch.eye(d_inp, device=device)
    A = bb(E, w).t()
    
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    
    U = U[:, :rank]
    S = torch.diag(S[:rank])
    V = V[:rank, :]
    
    return U, S, V
    

class MatrixFactorizationALS:
    """
    Matrix Factorization using Alternating Least Squares (ALS)
    
    Given dataset X, Y where Y = X @ A, finds A = B @ C
    where B is (n x r) and C is (r x n) with r << n
    """
    
    def __init__(self, rank, max_iter=100, tol=1e-6, random_state=42, 
                 device='cpu', use_stochastic=True, learning_rate=0.01, 
                 momentum=0.9):

        raise NotImplementedError('Outdated code')
        
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = device
        self.use_stochastic = use_stochastic
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.B = None
        self.C = None
        self.losses = []
        
        # For momentum
        self.B_momentum = None
        self.C_momentum = None
        
    def _initialize_factors(self, d_inp, d_out, dtype=None):
        """Initialize B and C matrices randomly with correct dimensions"""
        torch.manual_seed(self.random_state)
        
        # B: (d_inp, rank), C: (rank, d_out)
        # Xavier initialization
        std_B = math.sqrt(2.0 / (d_inp + self.rank))
        std_C = math.sqrt(2.0 / (self.rank + d_out))
        
        self.B = torch.randn(d_inp, self.rank, device=self.device, dtype=dtype) * std_B
        self.C = torch.randn(self.rank, d_out, device=self.device, dtype=dtype) * std_C
        
        # Initialize momentum buffers
        if self.use_stochastic:
            self.B_momentum = torch.zeros_like(self.B)
            self.C_momentum = torch.zeros_like(self.C)
    
    def _compute_loss(self, X, Y):
        """Compute reconstruction loss"""
        Y_pred = X @ self.B @ self.C
        return torch.mean((Y - Y_pred) ** 2).item()
    
    def _update_B_stochastic(self, X, Y):
        """Stochastic update for B using gradient descent with ALS-inspired direction"""
        batch_size = X.shape[0]
        
        # Current prediction
        Y_pred = X @ self.B @ self.C
        residual = Y - Y_pred  # (batch_size, d_out)
        
        # Compute gradient direction inspired by ALS
        # For ALS: B_new = (X.T @ X)^{-1} @ X.T @ Y @ C.T @ (C @ C.T)^{-1}
        # Stochastic approximation: use gradients but in ALS direction
        
        # Gradient w.r.t. B: -2 * X.T @ residual @ C.T
        grad_B = -2.0 * X.T @ residual @ self.C.T / batch_size
        
        # Use ALS-inspired scaling: multiply by (C @ C.T)^{-1}
        CCT = self.C @ self.C.T  # (rank, rank)
        try:
            # Only invert small (rank x rank) matrix
            CCT_inv = torch.linalg.pinv(CCT + 1e-6 * torch.eye(self.rank, device=self.device))
            direction_B = grad_B @ CCT_inv
        except:
            # Fallback to simple gradient
            direction_B = grad_B
        
        # Momentum update
        self.B_momentum = self.momentum * self.B_momentum + (1 - self.momentum) * direction_B
        
        # Update B
        self.B = self.B - self.learning_rate * self.B_momentum
    
    def _update_C_stochastic(self, X, Y):
        """Stochastic update for C using gradient descent with ALS-inspired direction"""
        batch_size = X.shape[0]
        
        # Current prediction
        Y_pred = X @ self.B @ self.C
        residual = Y - Y_pred  # (batch_size, d_out)
        
        # Gradient w.r.t. C: -2 * B.T @ X.T @ residual
        grad_C = -2.0 * self.B.T @ X.T @ residual / batch_size
        
        # Use ALS-inspired scaling: multiply by (B.T @ B)^{-1}
        BTB = self.B.T @ self.B  # (rank, rank)
        try:
            # Only invert small (rank x rank) matrix
            BTB_inv = torch.linalg.pinv(BTB + 1e-6 * torch.eye(self.rank, device=self.device))
            direction_C = BTB_inv @ grad_C
        except:
            # Fallback to simple gradient
            direction_C = grad_C
        
        # Momentum update
        self.C_momentum = self.momentum * self.C_momentum + (1 - self.momentum) * direction_C
        
        # Update C
        self.C = self.C - self.learning_rate * self.C_momentum
    
    def _update_B_exact(self, X, Y):
        """Exact ALS update for B (expensive for large matrices)"""
        # This is the original expensive version - avoid for large matrices
        batch_size, d_inp = X.shape
        dtype = X.dtype
        
        CCT = self.C @ self.C.T  # Shape: (rank, rank)
        try:
            CCT_inv = torch.linalg.pinv(CCT + 1e-6 * torch.eye(self.rank, device=self.device, dtype=dtype))
            
            # More efficient formulation: avoid computing X @ X.T
            XTY = X.T @ Y  # (d_inp, d_out)
            XTYCT = XTY @ self.C.T  # (d_inp, rank)
            self.B = XTYCT @ CCT_inv  # (d_inp, rank)
            
        except RuntimeError:
            # Fallback to gradient-based update
            self._update_B_stochastic(X, Y)
    
    def _update_C_exact(self, X, Y):
        """Exact ALS update for C"""
        Z = X @ self.B  # Shape: (batch_size, rank)
        dtype = X.dtype
        
        try:
            ZTZ = Z.T @ Z  # Shape: (rank, rank)
            ZTY = Z.T @ Y  # Shape: (rank, d_out)
            ZTZ_inv = torch.linalg.pinv(ZTZ + 1e-6 * torch.eye(self.rank, device=self.device, dtype=dtype))
            self.C = ZTZ_inv @ ZTY  # Shape: (rank, d_out)
        except RuntimeError:
            # Fallback to gradient-based update
            self._update_C_stochastic(X, Y)

    def _sample_minibatch(self, X, Y, minibatch_size):
        batch_size, d_inp = X.shape
        if minibatch_size < batch_size:
            indices = torch.randperm(batch_size, device=self.device)[:minibatch_size]
            X_batch = X[indices]
            Y_batch = Y[indices]
        else:
            X_batch = X
            Y_batch = Y
        return X_batch, Y_batch

    def fit(self, X, Y, verbose=True, minibatch_size=100):
        batch_size, d_inp = X.shape
        _, d_out = Y.shape
        
        # Initialize factors with correct dimensions
        self._initialize_factors(d_inp, d_out, dtype=X.dtype)
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Use same minibatch for both updates for consistency
            X_batch, Y_batch = self._sample_minibatch(X, Y, minibatch_size)
            
            if self.use_stochastic:
                self._update_B_stochastic(X_batch, Y_batch)
                self._update_C_stochastic(X_batch, Y_batch)
            else:
                self._update_B_exact(X_batch, Y_batch)
                self._update_C_exact(X_batch, Y_batch)
            
            # Compute loss on full dataset periodically to track progress
            if iteration % 5 == 0 or iteration == self.max_iter - 1:
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