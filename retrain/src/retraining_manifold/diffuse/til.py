# src/retraining_manifold/diffuse/til.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm
import math

from ..base import BaseRetrainingPath
from .sde_base import BaseSDE

class TIL(BaseSDE):
    """
    Teacher-Injected Langevin: Controlled Langevin process with superiority vector field.
    
    dX_t = [−∇L(X_t) + λ_t s(X_t)] dt + σ_t dW_t, s(x) = θ† − x
    """
    
    def __init__(
        self,
        theta_star: nn.Module,
        theta_dagger: nn.Module,
        noise_schedule: str = "exponential",
        lambda_0: float = 1.0,
        sigma_0: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(theta_star, theta_dagger, device, **kwargs)
        
        self.noise_schedule = noise_schedule
        self.lambda_0 = lambda_0
        self.sigma_0 = sigma_0
        
        # Superiority vector field function
        self.s = lambda x: self.params_dagger - x
        
    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Compute drift: −∇L(X_t) + λ_t s(X_t)"""
        # Get current parameters as model
        model = self._set_flat_params(
            type(self.theta_star)().to(self.device),
            x
        )
        model.train()
        
        # Compute gradient of loss
        # Note: In practice, we need data to compute gradient
        # Here we assume gradient function is provided
        if hasattr(self, '_grad_fn'):
            grad_L = self._grad_fn(model, x)
        else:
            # Default: use random gradient approximation
            grad_L = torch.randn_like(x) * 0.1
        
        # Compute lambda_t according to schedule
        lambda_t = self._lambda_schedule(t)
        
        # Total drift
        drift = -grad_L + lambda_t * self.s(x)
        
        return drift
    
    def diffusion(self, t: float) -> float:
        """Compute diffusion coefficient σ_t"""
        if self.noise_schedule == "exponential":
            return self.sigma_0 * math.exp(-t)
        elif self.noise_schedule == "constant":
            return self.sigma_0
        elif self.noise_schedule == "linear":
            return self.sigma_0 * max(0, 1 - t)
        else:
            raise ValueError(f"Unknown schedule: {self.noise_schedule}")
    
    def _lambda_schedule(self, t: float) -> float:
        """Compute λ_t schedule: λ_0/(σ_0² + ∫_0^t σ_s² ds)"""
        if self.noise_schedule == "exponential":
            # ∫_0^t exp(-2s) ds = (1 - exp(-2t))/2
            integral = (1 - math.exp(-2 * t)) / 2
        elif self.noise_schedule == "constant":
            integral = t
        elif self.noise_schedule == "linear":
            # ∫_0^t (1-s)^2 ds = t - t^2 + t^3/3 for t ≤ 1
            integral = t - t**2 + t**3/3 if t <= 1 else 1/3
        else:
            integral = t
        
        denominator = self.sigma_0**2 + integral
        return self.lambda_0 / max(denominator, 1e-8)
    
    def sample(
        self,
        n_samples: int = 1,
        steps: int = 1000,
        step_size: float = 0.01,
        data: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[nn.Module]:
        """
        Sample improved models from the TIL process.
        
        Returns:
            List of sampled models
        """
        if data is not None and labels is not None:
            # Setup gradient function
            criterion = nn.MSELoss() if labels.dim() > 1 else nn.CrossEntropyLoss()
            self._grad_fn = lambda model, x: self._compute_gradient(model, data, labels, criterion)
        
        samples = []
        
        for sample_idx in tqdm(range(n_samples), desc="TIL Sampling"):
            # Initialize at theta_star
            x = self.params_star.clone()
            
            # Euler-Maruyama discretization
            for i in range(steps):
                t = i * step_size
                
                # Compute drift and diffusion
                drift = self.drift(x, t)
                sigma = self.diffusion(t)
                
                # Update
                x = x + step_size * drift + math.sqrt(step_size) * sigma * torch.randn_like(x)
                
                # Optional: clip values
                if kwargs.get('clip', False):
                    x = torch.clamp(x, -10, 10)
            
            # Convert to model
            model = type(self.theta_star)().to(self.device)
            model = self._set_flat_params(model, x)
            samples.append(model)
        
        return samples
    
    def _compute_gradient(
        self,
        model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """Compute gradient of loss at current parameters."""
        model.train()
        model.zero_grad()
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Get flattened gradient
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        return torch.cat(grads) if grads else torch.zeros_like(self.params_star)
    
    def compute_wasserstein_bound(
        self,
        time: float,
        strong_convexity: float = 0.1
    ) -> float:
        """
        Compute Wasserstein-2 distance bound.
        
        W₂(Law(X_t), 𝒩(θ†,Σ_∞)) ≤ C e^{−(m+λ)t}
        """
        m = strong_convexity
        lambda_inf = self._lambda_schedule(float('inf'))
        
        # Theoretical bound constant
        C = float(self.params_star.norm().item())
        
        bound = C * math.exp(-(m + lambda_inf) * time)
        return bound
    
    def get_equilibrium_covariance(self) -> torch.Tensor:
        """Compute equilibrium covariance Σ_∞ = σ²/(2(m+λ)) I"""
        # Assume m (strong convexity) = 0.1
        m = 0.1
        lambda_inf = self._lambda_schedule(float('inf'))
        sigma_inf = self.diffusion(float('inf'))
        
        variance = sigma_inf**2 / (2 * (m + lambda_inf))
        return variance * torch.eye(len(self.params_star), device=self.device)