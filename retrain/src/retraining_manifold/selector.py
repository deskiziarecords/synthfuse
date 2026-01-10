# src/retraining_manifold/selector.py
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from .deterministic import WSFT, KDT, OTWI, FRR
from .diffuse import TIL, DBSD, FDD
from .geometry import compute_fisher_information, compute_description_length

@dataclass
class PathMetrics:
    """Metrics for evaluating retraining paths."""
    method_name: str
    final_risk: float
    generalization_proxy: float
    path_length: float
    computational_cost: float  # FLOPs or time
    description_length: float
    success_probability: float
    convergence_time: float

class PathSelector:
    """
    Select optimal retraining path based on Minimum Description Length.
    
    η* = argmin_{η∈{WSFT,KDT,OTWI,FRR}} [ ∫₀^{T′}‖dη/dt‖₂ dt + λ Comp(η) ]
    """
    
    def __init__(
        self,
        lambda_coef: float = 0.1,
        generalization_method: str = "rademacher",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        self.lambda_coef = lambda_coef
        self.generalization_method = generalization_method
        self.device = device
        self.verbose = verbose
        
        # Available methods
        self.deterministic_methods = ['WSFT', 'KDT', 'OTWI', 'FRR']
        self.diffuse_methods = ['TIL', 'DBSD', 'FDD']
        
    def select(
        self,
        theta_star: nn.Module,
        theta_dagger: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        budget: float = 1000,  # Max computational budget
        epsilon: float = 1e-4,  # Target improvement
        delta: float = 0.05,  # Confidence level
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[nn.Module, str, Dict[str, Any]]:
        """
        Select and execute optimal retraining path.
        
        Returns:
            Tuple of (best_model, best_method_name, all_metrics)
        """
        if methods is None:
            methods = self.deterministic_methods
        
        # Compute superiority gap
        superiority_gap = self._compute_superiority_gap(
            theta_star, theta_dagger, val_data, val_labels
        )
        
        if self.verbose:
            print(f"Superiority gap: {superiority_gap:.6f}")
            print(f"Target improvement: {epsilon}")
            print(f"Budget: {budget}")
        
        # Evaluate each method
        all_metrics = {}
        candidates = []
        
        for method_name in tqdm(methods, desc="Evaluating methods"):
            try:
                # Instantiate method
                method = self._instantiate_method(
                    method_name, theta_star, theta_dagger, **kwargs
                )
                
                # Estimate convergence time
                T_prime = self._estimate_convergence_time(
                    method, superiority_gap, epsilon, **kwargs
                )
                
                # Check budget constraint
                if T_prime > budget:
                    if self.verbose:
                        print(f"Method {method_name} exceeds budget: {T_prime} > {budget}")
                    continue
                
                # Execute retraining
                if method_name in self.deterministic_methods:
                    # Deterministic methods return single model
                    theta_candidate, metrics = method.train(
                        train_data, train_labels,
                        val_data, val_labels,
                        **kwargs.get(f'{method_name.lower()}_kwargs', {})
                    )
                    
                    # Compute final risk
                    final_risk = self._compute_final_risk(
                        theta_candidate, val_data, val_labels
                    )
                    
                    # Compute path length
                    path_length = method.compute_path_length()
                    
                else:
                    # Diffuse methods return distribution
                    samples = method.sample(
                        n_samples=kwargs.get('n_samples', 10),
                        data=train_data,
                        labels=train_labels,
                        **kwargs.get(f'{method_name.lower()}_kwargs', {})
                    )
                    
                    # Select best sample
                    theta_candidate = self._select_best_sample(
                        samples, val_data, val_labels
                    )
                    
                    final_risk = self._compute_final_risk(
                        theta_candidate, val_data, val_labels
                    )
                    
                    # Estimate path length for diffuse methods
                    path_length = self._estimate_diffuse_path_length(method)
                
                # Compute metrics
                metrics = self._compute_path_metrics(
                    method_name,
                    theta_candidate,
                    theta_star,
                    theta_dagger,
                    train_data,
                    train_labels,
                    path_length,
                    T_prime,
                    final_risk,
                    superiority_gap
                )
                
                # Check improvement condition
                if final_risk <= self._compute_final_risk(theta_dagger, val_data, val_labels) - epsilon:
                    candidates.append((theta_candidate, method_name, metrics))
                    all_metrics[method_name] = metrics
                    
                    if self.verbose:
                        print(f"Method {method_name}: risk={final_risk:.6f}, "
                              f"DL={metrics.description_length:.2f}")
                else:
                    if self.verbose:
                        print(f"Method {method_name} failed improvement test")
            
            except Exception as e:
                if self.verbose:
                    print(f"Method {method_name} failed: {e}")
                continue
        
        if not candidates:
            raise ValueError("No method satisfied constraints")
        
        # Select method with minimum description length
        best_idx = np.argmin([c[2].description_length for c in candidates])
        best_model, best_method, best_metrics = candidates[best_idx]
        
        # Certify improvement
        certification = self._certify_improvement(
            best_model, theta_dagger, val_data, val_labels, epsilon, delta
        )
        
        if not certification['success']:
            if self.verbose:
                print(f"Warning: Certification failed for {best_method}")
        
        return best_model, best_method, {
            'all_metrics': all_metrics,
            'selected_metrics': best_metrics,
            'certification': certification
        }
    
    def _instantiate_method(self, method_name: str, theta_star: nn.Module, 
                           theta_dagger: nn.Module, **kwargs) -> BaseRetrainingPath:
        """Instantiate retraining method."""
        method_classes = {
            'WSFT': WSFT,
            'KDT': KDT,
            'OTWI': OTWI,
            'FRR': FRR,
            'TIL': TIL,
            'DBSD': DBSD,
            'FDD': FDD
        }
        
        if method_name not in method_classes:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Get method-specific kwargs
        method_kwargs = kwargs.get(f'{method_name.lower()}_kwargs', {})
        
        return method_classes[method_name](
            theta_star=theta_star,
            theta_dagger=theta_dagger,
            device=self.device,
            **method_kwargs
        )
    
    def _compute_superiority_gap(self, theta_star: nn.Module, theta_dagger: nn.Module,
                                val_data: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Compute L(θ*) - L(θ†)."""
        criterion = nn.MSELoss() if val_labels.dim() > 1 else nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs_star = theta_star(val_data)
            loss_star = criterion(outputs_star, val_labels)
            
            outputs_dagger = theta_dagger(val_data)
            loss_dagger = criterion(outputs_dagger, val_labels)
        
        return (loss_star - loss_dagger).item()
    
    def _estimate_convergence_time(self, method: BaseRetrainingPath,
                                  superiority_gap: float, epsilon: float,
                                  **kwargs) -> float:
        """Estimate T′ for convergence to ε-excess risk."""
        # Method-specific estimation
        if hasattr(method, 'get_convergence_bound'):
            bound, condition = method.get_convergence_bound()
            if condition:
                # Solve for T: ε = bound * exp(-μT)
                mu = kwargs.get('strong_convexity', 0.1)
                T = np.log(bound / epsilon) / mu if epsilon > 0 else 0
                return max(T, 1)
        
        # Default estimation based on superiority gap
        return superiority_gap / (epsilon + 1e-8)
    
    def _compute_final_risk(self, model: nn.Module,
                           val_data: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Compute final risk on validation data."""
        criterion = nn.MSELoss() if val_labels.dim() > 1 else nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs = model(val_data)
            loss = criterion(outputs, val_labels)
        
        return loss.item()
    
    def _select_best_sample(self, samples: List[nn.Module],
                           val_data: torch.Tensor, val_labels: torch.Tensor) -> nn.Module:
        """Select best sample from diffuse method outputs."""
        best_loss = float('inf')
        best_model = None
        
        for model in samples:
            loss = self._compute_final_risk(model, val_data, val_labels)
            if loss < best_loss:
                best_loss = loss
                best_model = model
        
        return best_model
    
    def _estimate_diffuse_path_length(self, method: BaseSDE) -> float:
        """Estimate path length for diffuse methods."""
        # For SDE methods, path length = ∫‖v_t‖ dt
        # Approximate using expected squared velocity
        if hasattr(method, 'compute_wasserstein_bound'):
            bound = method.compute_wasserstein_bound(time=1.0)
            return bound
        else:
            return 1.0  # Default
    
    def _compute_path_metrics(self, method_name: str, theta_candidate: nn.Module,
                             theta_star: nn.Module, theta_dagger: nn.Module,
                             train_data: torch.Tensor, train_labels: torch.Tensor,
                             path_length: float, convergence_time: float,
                             final_risk: float, superiority_gap: float) -> PathMetrics:
        """Compute comprehensive metrics for a retraining path."""
        
        # Compute generalization proxy
        gen_proxy = self._compute_generalization_proxy(
            theta_candidate, train_data, train_labels
        )
        
        # Estimate computational cost (simplified)
        computational_cost = convergence_time * path_length
        
        # Compute description length
        description_length = compute_description_length(
            theta_star, theta_candidate, path_length
        )
        
        # Estimate success probability (simplified)
        success_prob = np.exp(-final_risk / superiority_gap) if superiority_gap > 0 else 0
        
        return PathMetrics(
            method_name=method_name,
            final_risk=final_risk,
            generalization_proxy=gen_proxy,
            path_length=path_length,
            computational_cost=computational_cost,
            description_length=description_length,
            success_probability=success_prob,
            convergence_time=convergence_time
        )
    
    def _compute_generalization_proxy(self, model: nn.Module,
                                     train_data: torch.Tensor, 
                                     train_labels: torch.Tensor) -> float:
        """Compute generalization bound proxy."""
        # Use Rademacher complexity as proxy
        n = train_data.shape[0]
        
        # Simple estimation
        with torch.no_grad():
            outputs = model(train_data)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                # Classification
                variance = outputs.var(dim=0).mean().item()
            else:
                # Regression
                variance = outputs.var().item()
        
        # Rademacher-like bound
        rademacher_bound = np.sqrt(variance / n)
        
        return rademacher_bound
    
    def _certify_improvement(self, theta_candidate: nn.Module,
                            theta_dagger: nn.Module,
                            val_data: torch.Tensor, val_labels: torch.Tensor,
                            epsilon: float, delta: float) -> Dict[str, Any]:
        """Certify ε-improvement with probability 1-δ."""
        n = val_data.shape[0]
        
        # Compute losses
        criterion = nn.MSELoss() if val_labels.dim() > 1 else nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs_candidate = theta_candidate(val_data)
            loss_candidate = criterion(outputs_candidate, val_labels)
            
            outputs_dagger = theta_dagger(val_data)
            loss_dagger = criterion(outputs_dagger, val_labels)
        
        # Statistical test
        loss_diff = loss_dagger - loss_candidate
        loss_diff_var = loss_diff.var() if hasattr(loss_diff, 'var') else 0
        
        # Bernstein bound
        variance_term = 2 * loss_diff_var * np.log(2/delta) / n
        deviation_term = 7 * np.log(2/delta) / (3 * (n-1))
        
        bound = variance_term + deviation_term
        
        success = (loss_diff.item() - bound) >= epsilon
        
        return {
            'success': success,
            'loss_difference': loss_diff.item(),
            'bound': bound,
            'epsilon': epsilon,
            'confidence': 1 - delta
        }