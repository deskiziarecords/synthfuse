# src/retraining_manifold/geometry/optimal_transport.py
import torch
import numpy as np
from typing import Tuple, Optional
import ot  # Python Optimal Transport

class OptimalTransportCoupler:
    """Optimal Transport Weight Interpolation between parameter distributions."""
    
    @staticmethod
    def compute_wasserstein_coupling(
        mu_star: torch.Tensor,
        mu_dagger: torch.Tensor,
        method: str = "emd",
        reg: float = 0.1,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal transport coupling between parameter samples.
        
        Args:
            mu_star: Samples from incumbent distribution (n_samples x d)
            mu_dagger: Samples from superior distribution (m_samples x d)
            method: "emd" (exact) or "sinkhorn" (regularized)
            reg: Regularization strength for sinkhorn
            
        Returns:
            Tuple of (optimal coupling matrix, Wasserstein distance)
        """
        # Convert to numpy for POT library
        if isinstance(mu_star, torch.Tensor):
            mu_star_np = mu_star.cpu().numpy()
            mu_dagger_np = mu_dagger.cpu().numpy()
        else:
            mu_star_np = mu_star
            mu_dagger_np = mu_dagger
        
        n = mu_star_np.shape[0]
        m = mu_dagger_np.shape[0]
        
        # Cost matrix: squared Euclidean distance
        M = ot.dist(mu_star_np, mu_dagger_np, metric='euclidean')
        M = M ** 2  # Wasserstein-2 uses squared distances
        
        # Uniform distributions
        a = np.ones(n) / n
        b = np.ones(m) / m
        
        # Compute optimal transport
        if method == "emd":
            # Exact Earth Mover's Distance
            pi = ot.emd(a, b, M)
        elif method == "sinkhorn":
            # Regularized OT
            pi = ot.sinkhorn(a, b, M, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute Wasserstein distance
        wasserstein_dist = np.sum(pi * M)
        
        return pi, wasserstein_dist
    
    @staticmethod
    def barycentric_interpolation(
        mu_star: torch.Tensor,
        mu_dagger: torch.Tensor,
        lambda_val: float,
        method: str = "sinkhorn",
        reg: float = 0.1
    ) -> torch.Tensor:
        """
        Compute McCann interpolant: θ_λ = ∫ y dπ*(·,y)
        
        Args:
            lambda_val: Interpolation coefficient (0=theta_star, 1=theta_dagger)
            
        Returns:
            Interpolated parameters
        """
        if lambda_val == 0:
            return mu_star.mean(dim=0)
        elif lambda_val == 1:
            return mu_dagger.mean(dim=0)
        
        # Compute optimal coupling
        pi, _ = OptimalTransportCoupler.compute_wasserstein_coupling(
            mu_star, mu_dagger, method=method, reg=reg
        )
        
        # Convert to torch
        pi_torch = torch.tensor(pi, device=mu_star.device)
        
        # Barycentric projection: ∫ y dπ(x,y) / π(x)
        n = mu_star.shape[0]
        m = mu_dagger.shape[0]
        
        # Compute conditional distribution
        marginal = pi_torch.sum(dim=1, keepdim=True)
        conditional = pi_torch / (marginal + 1e-10)
        
        # Interpolated points
        interpolated = torch.zeros(n, mu_star.shape[1], device=mu_star.device)
        
        for i in range(n):
            # Weighted average of mu_dagger points
            weights = conditional[i]
            interpolated[i] = torch.sum(mu_dagger * weights.view(-1, 1), dim=0)
        
        # Combine with original using McCann interpolation
        result = (1 - lambda_val) * mu_star + lambda_val * interpolated
        
        return result.mean(dim=0)  # Return mean of interpolated distribution
    
    @staticmethod
    def compute_mccann_bound(
        mu_star: torch.Tensor,
        mu_dagger: torch.Tensor,
        lambda_val: float,
        loss_gradient: Optional[torch.Tensor] = None,
        method: str = "sinkhorn",
        reg: float = 0.1
    ) -> float:
        """
        Compute McCann interpolation risk bound:
        
        L(θ_λ) ≤ (1−λ)L(θ*) + λL(θ†) - (λ(1−λ)/2)‖∇L‖²_{L²(π*)}
        """
        # Compute optimal coupling
        pi, _ = OptimalTransportCoupler.compute_wasserstein_coupling(
            mu_star, mu_dagger, method=method, reg=reg
        )
        pi_torch = torch.tensor(pi, device=mu_star.device)
        
        n = mu_star.shape[0]
        m = mu_dagger.shape[0]
        
        if loss_gradient is None:
            # Approximate gradient norm from data
            grad_norm_sq = torch.mean((mu_dagger - mu_star) ** 2)
        else:
            # Compute gradient norm w.r.t. coupling
            # ‖∇L‖²_{L²(π*)} = ∫ ‖∇L(x)‖² dπ*(x,y)
            grad_norm_sq = torch.sum(pi_torch * (loss_gradient ** 2).view(n, 1))
        
        # McCann bound
        bound_term = (lambda_val * (1 - lambda_val) / 2) * grad_norm_sq
        
        return bound_term.item()
    
    @staticmethod
    def sample_parameter_distribution(
        model: torch.nn.Module,
        n_samples: int = 100,
        noise_std: float = 0.1
    ) -> torch.Tensor:
        """
        Sample from parameter distribution by adding noise.
        
        Args:
            model: Neural network model
            n_samples: Number of samples
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Tensor of shape (n_samples, d)
        """
        # Get base parameters
        base_params = torch.cat([p.view(-1) for p in model.parameters()])
        d = base_params.shape[0]
        
        # Sample perturbations
        noise = torch.randn(n_samples, d, device=base_params.device) * noise_std
        
        # Add noise to base parameters
        samples = base_params.unsqueeze(0) + noise
        
        return samples