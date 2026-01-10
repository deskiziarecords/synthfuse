# src/retraining_manifold/deterministic/wsft.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np
from tqdm import tqdm

from ..base import BaseRetrainingPath

class WSFT(BaseRetrainingPath):
    """
    Warm-Start Fine-Tuning: Parameter-space geodesic with biased initial velocity.
    
    dη/dt = −∇L(η;𝒟′) + λΠ_{T_η𝒫}(ξ), ξ = θ† − θ*
    """
    
    def __init__(
        self,
        theta_star: nn.Module,
        theta_dagger: nn.Module,
        lambda_coef: float = 0.1,
        projection_type: str = "tangent",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(theta_star, theta_dagger, device, **kwargs)
        self.lambda_coef = lambda_coef
        self.projection_type = projection_type
        
    def _tangent_projection(self, v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """Project vector onto tangent space at eta."""
        if self.projection_type == "tangent":
            # Simple orthogonal projection (Euclidean metric)
            return v
        elif self.projection_type == "fisher":
            # Fisher information metric projection
            # Implement Riemannian projection if needed
            return v
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
    
    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        distill_data: Optional[torch.Tensor] = None,
        distill_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute WSFT retraining.
        
        Args:
            distill_data: Optional distilled dataset for biasing
            distill_labels: Optional labels for distilled data
        """
        # Create trainable copy of theta_star
        theta_double_dagger = type(self.theta_star)().to(self.device)
        theta_double_dagger.load_state_dict(self.theta_star.state_dict())
        theta_double_dagger.train()
        
        # Prepare data
        if distill_data is not None and distill_labels is not None:
            # Use distilled data for biasing term
            bias_data, bias_labels = distill_data, distill_labels
        else:
            # Use original training data
            bias_data, bias_labels = train_data, train_labels
        
        # Setup optimizer
        optimizer = optim.Adam(theta_double_dagger.parameters(), lr=lr)
        
        # Loss function
        if train_labels.dim() > 1 and train_labels.shape[1] > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'bias_term': [],
            'grad_norm': []
        }
        
        for epoch in tqdm(range(num_epochs), desc="WSFT Training"):
            epoch_loss = 0.0
            epoch_bias = 0.0
            epoch_grad_norm = 0.0
            
            # Mini-batch training
            n_batches = len(train_data) // batch_size
            for i in range(n_batches):
                # Get batch
                idx = slice(i * batch_size, (i + 1) * batch_size)
                batch_data = train_data[idx].to(self.device)
                batch_labels = train_labels[idx].to(self.device)
                
                # Get bias batch
                bias_idx = slice(i * batch_size, min((i + 1) * batch_size, len(bias_data)))
                bias_batch_data = bias_data[bias_idx].to(self.device)
                bias_batch_labels = bias_labels[bias_idx].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = theta_double_dagger(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Compute bias term: gradient of loss w.r.t. superiority direction
                bias_outputs = theta_double_dagger(bias_batch_data)
                bias_loss = criterion(bias_outputs, bias_batch_labels)
                
                # Compute gradient
                bias_loss.backward(retain_graph=True)
                
                # Extract and project gradient
                bias_grad = self._get_flat_grad(theta_double_dagger)
                projected_bias = self._tangent_projection(self.xi, 
                                                        self._get_flat_params(theta_double_dagger))
                
                # Scale bias gradient
                bias_term = self.lambda_coef * torch.dot(bias_grad, projected_bias)
                
                # Total loss with bias
                total_loss = loss + bias_term
                total_loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Record metrics
                epoch_loss += loss.item()
                epoch_bias += bias_term.item()
                epoch_grad_norm += torch.norm(bias_grad).item()
            
            # Average metrics
            metrics['train_loss'].append(epoch_loss / n_batches)
            metrics['bias_term'].append(epoch_bias / n_batches)
            metrics['grad_norm'].append(epoch_grad_norm / n_batches)
            
            # Validation
            if val_data is not None and val_labels is not None:
                val_loss = self.compute_risk(theta_double_dagger, val_data, val_labels, criterion)
                metrics['val_loss'].append(val_loss)
            
            # Optional: adjust lambda based on progress
            if kwargs.get('adaptive_lambda', False):
                self._adjust_lambda(metrics, epoch)
        
        return theta_double_dagger, metrics
    
    def _get_flat_grad(self, model: nn.Module) -> torch.Tensor:
        """Get flattened gradient vector."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self.device)
    
    def _adjust_lambda(self, metrics: Dict[str, list], epoch: int):
        """Adaptively adjust lambda coefficient."""
        if epoch > 0 and len(metrics['train_loss']) > 1:
            # Decrease lambda if loss is decreasing well
            loss_decrease = metrics['train_loss'][-2] - metrics['train_loss'][-1]
            if loss_decrease > 0:
                self.lambda_coef *= 0.95
            else:
                self.lambda_coef *= 1.05
            
            # Clamp lambda
            self.lambda_coef = max(0.01, min(1.0, self.lambda_coef))
    
    def compute_path_length(self) -> float:
        """Compute geometric length of WSFT path."""
        # Approximate path length using linear interpolation
        return float(self.xi.norm().item())
    
    def get_convergence_bound(
        self,
        hessian_lipschitz: float = 1.0,
        strong_convexity: float = 0.1
    ) -> float:
        """
        Compute theoretical convergence bound.
        
        Returns:
            Upper bound on distance to θ†
        """
        mu = strong_convexity
        beta = hessian_lipschitz
        
        # Check condition: ‖θ*−θ†‖₂ ≤ μ/(2β)
        condition = float(self.xi.norm().item()) <= mu / (2 * beta)
        
        if condition:
            # Convergence bound: ‖η(t)−θ†‖₂ ≤ ‖θ*−θ†‖₂ e^{−μt} + λ/μ
            # For t → ∞, bound = λ/μ
            bound = self.lambda_coef / mu
        else:
            bound = float('inf')
        
        return bound, condition