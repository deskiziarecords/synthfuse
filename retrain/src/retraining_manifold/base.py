# src/retraining_manifold/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

class BaseRetrainingPath(ABC):
    """Abstract base class for all retraining paths."""
    
    def __init__(
        self,
        theta_star: nn.Module,
        theta_dagger: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize retraining path.
        
        Args:
            theta_star: Incumbent model
            theta_dagger: Superior teacher model
            device: Computation device
            **kwargs: Additional method-specific parameters
        """
        self.theta_star = theta_star.to(device)
        self.theta_dagger = theta_dagger.to(device)
        self.device = device
        
        # Store models in evaluation mode
        self.theta_star.eval()
        self.theta_dagger.eval()
        
        # Extract parameters
        self.params_star = self._get_flat_params(theta_star)
        self.params_dagger = self._get_flat_params(theta_dagger)
        
        # Superiority vector
        self.xi = self.params_dagger - self.params_star
        
    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        """Flatten model parameters into a single vector."""
        return torch.cat([p.view(-1) for p in model.parameters()])
    
    def _set_flat_params(self, model: nn.Module, params: torch.Tensor) -> nn.Module:
        """Set model parameters from flattened vector."""
        idx = 0
        for param in model.parameters():
            param.data = params[idx:idx + param.numel()].view_as(param)
            idx += param.numel()
        return model
    
    @abstractmethod
    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        lr: float = 1e-3,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute retraining path.
        
        Returns:
            Tuple of (improved_model, training_metrics)
        """
        pass
    
    @abstractmethod
    def compute_path_length(self) -> float:
        """Compute geometric length of the retraining path."""
        pass
    
    def compute_risk(self, model: nn.Module, data: torch.Tensor, 
                    labels: torch.Tensor, criterion=None) -> float:
        """Compute empirical risk."""
        if criterion is None:
            criterion = nn.MSELoss() if labels.dim() > 1 else nn.CrossEntropyLoss()
        
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, labels)
        return loss.item()
    
    def compute_generalization_proxy(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        method: str = "rademacher"
    ) -> float:
        """Compute generalization bound proxy."""
        # Implement PAC-Bayes, Rademacher, etc.
        if method == "rademacher":
            return self._rademacher_complexity(model, train_data)
        elif method == "pac_bayes":
            return self._pac_bayes_bound(model, train_data, train_labels)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _rademacher_complexity(self, model: nn.Module, data: torch.Tensor) -> float:
        """Estimate Rademacher complexity."""
        n_samples = data.shape[0]
        n_trials = 10
        
        complexities = []
        for _ in range(n_trials):
            sigma = torch.randint(0, 2, (n_samples,)).float() * 2 - 1  # ±1
            sigma = sigma.to(self.device)
            
            with torch.no_grad():
                outputs = model(data)
                complexity = torch.mean(sigma * outputs).item()
            
            complexities.append(abs(complexity))
        
        return np.mean(complexities)
    
    def _pac_bayes_bound(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        delta: float = 0.05
    ) -> float:
        """Compute PAC-Bayes bound."""
        # Simple KL divergence between prior and posterior
        prior_params = self.params_star
        posterior_params = self._get_flat_params(model)
        
        # KL divergence (assuming Gaussian)
        kl_div = 0.5 * (
            (posterior_params - prior_params).norm() ** 2 / prior_params.numel()
        )
        
        n = train_data.shape[0]
        bound = kl_div / n + np.sqrt((kl_div + np.log(n / delta)) / (2 * n))
        
        return bound.item()