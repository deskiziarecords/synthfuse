# tests/test_deterministic.py
import pytest
import torch
import torch.nn as nn
import numpy as np

from retraining_manifold.deterministic import (
    WSFT, KDT, OTWI, FRR
)

class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@pytest.fixture
def setup():
    """Setup test fixtures."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 100
    input_dim = 10
    
    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 1)
    
    # Split into train/val
    train_idx = n_samples // 2
    X_train, X_val = X[:train_idx], X[train_idx:]
    y_train, y_val = y[:train_idx], y[train_idx:]
    
    # Create models
    theta_star = SimpleModel(input_dim=input_dim, output_dim=1)
    theta_dagger = SimpleModel(input_dim=input_dim, output_dim=1)
    
    # Train theta_dagger to be slightly better
    optimizer = torch.optim.Adam(theta_dagger.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for _ in range(10):
        optimizer.zero_grad()
        outputs = theta_dagger(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return {
        'theta_star': theta_star,
        'theta_dagger': theta_dagger,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }

def test_wsft_initialization(setup):
    """Test WSFT initialization."""
    wsft = WSFT(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        lambda_coef=0.1
    )
    
    assert wsft.lambda_coef == 0.1
    assert wsft.xi.shape[0] > 0
    assert torch.allclose(wsft.params_star, wsft._get_flat_params(setup['theta_star']))

def test_wsft_training(setup):
    """Test WSFT training."""
    wsft = WSFT(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        lambda_coef=0.1
    )
    
    theta_best, metrics = wsft.train(
        train_data=setup['X_train'],
        train_labels=setup['y_train'],
        val_data=setup['X_val'],
        val_labels=setup['y_val'],
        num_epochs=2,
        lr=1e-3,
        batch_size=16
    )
    
    assert isinstance(theta_best, nn.Module)
    assert 'train_loss' in metrics
    assert len(metrics['train_loss']) == 2
    
    # Check that parameters changed
    original_params = wsft._get_flat_params(setup['theta_star'])
    new_params = wsft._get_flat_params(theta_best)
    assert not torch.allclose(original_params, new_params)

def test_kdt_initialization(setup):
    """Test KDT initialization."""
    kdt = KDT(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        alpha=0.5,
        temperature=2.0
    )
    
    assert kdt.alpha == 0.5
    assert kdt.temperature == 2.0
    assert kdt.xi.shape[0] > 0

def test_kdt_training(setup):
    """Test KDT training."""
    kdt = KDT(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        alpha=0.5,
        temperature=2.0
    )
    
    # For classification test
    # Convert to classification problem
    y_train_cls = (setup['y_train'] > 0).long().squeeze()
    y_val_cls = (setup['y_val'] > 0).long().squeeze()
    
    # Modify model for classification
    theta_star_cls = SimpleModel(input_dim=10, output_dim=2)
    theta_dagger_cls = SimpleModel(input_dim=10, output_dim=2)
    
    kdt_cls = KDT(
        theta_star=theta_star_cls,
        theta_dagger=theta_dagger_cls,
        alpha=0.5,
        temperature=2.0
    )
    
    theta_best, metrics = kdt_cls.train(
        train_data=setup['X_train'],
        train_labels=y_train_cls,
        val_data=setup['X_val'],
        val_labels=y_val_cls,
        num_epochs=2,
        lr=1e-3,
        batch_size=16
    )
    
    assert isinstance(theta_best, nn.Module)
    assert 'train_loss' in metrics
    assert len(metrics['train_loss']) == 2

def test_otwi_initialization(setup):
    """Test OTWI initialization."""
    otwi = OTWI(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        n_samples=10,
        noise_std=0.1
    )
    
    assert otwi.n_samples == 10
    assert otwi.noise_std == 0.1

def test_otwi_interpolation(setup):
    """Test OTWI interpolation."""
    otwi = OTWI(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        n_samples=10,
        noise_std=0.1
    )
    
    # Test interpolation at lambda=0.5
    theta_interp = otwi.interpolate(lambda_val=0.5)
    
    assert isinstance(theta_interp, nn.Module)
    
    # Check that interpolated model is different from both
    params_star = otwi._get_flat_params(setup['theta_star'])
    params_dagger = otwi._get_flat_params(setup['theta_dagger'])
    params_interp = otwi._get_flat_params(theta_interp)
    
    assert not torch.allclose(params_interp, params_star)
    assert not torch.allclose(params_interp, params_dagger)

def test_frr_initialization(setup):
    """Test FRR initialization."""
    frr = FRR(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        rho=0.1
    )
    
    assert frr.rho == 0.1
    assert hasattr(frr, 'ntk_kernel')

def test_frr_training(setup):
    """Test FRR training."""
    frr = FRR(
        theta_star=setup['theta_star'],
        theta_dagger=setup['theta_dagger'],
        rho=0.1
    )
    
    theta_best, metrics = frr.train(
        train_data=setup['X_train'],
        train_labels=setup['y_train'],
        val_data=setup['X_val'],
        val_labels=setup['y_val'],
        num_epochs=2,
        lr=1e-3,
        batch_size=16
    )
    
    assert isinstance(theta_best, nn.Module)
    assert 'train_loss' in metrics
    assert len(metrics['train_loss']) == 2
    
    # Check that functional regularization was applied
    assert 'regularization_loss' in metrics

def test_all_methods_path_length(setup):
    """Test that all methods can compute path length."""
    methods = [
        ('WSFT', WSFT(theta_star=setup['theta_star'], 
                     theta_dagger=setup['theta_dagger'])),
        ('KDT', KDT(theta_star=setup['theta_star'], 
                   theta_dagger=setup['theta_dagger'])),
        ('OTWI', OTWI(theta_star=setup['theta_star'], 
                     theta_dagger=setup['theta_dagger'])),
        ('FRR', FRR(theta_star=setup['theta_star'], 
                   theta_dagger=setup['theta_dagger'])),
    ]
    
    for method_name, method in methods:
        path_length = method.compute_path_length()
        assert isinstance(path_length, float)
        assert path_length >= 0
        print(f"{method_name} path length: {path_length}")

def test_risk_computation(setup):
    """Test risk computation across methods."""
    wsft = WSFT(theta_star=setup['theta_star'], 
               theta_dagger=setup['theta_dagger'])
    
    risk_star = wsft.compute_risk(
        setup['theta_star'],
        setup['X_val'],
        setup['y_val']
    )
    
    risk_dagger = wsft.compute_risk(
        setup['theta_dagger'],
        setup['X_val'],
        setup['y_val']
    )
    
    assert isinstance(risk_star, float)
    assert isinstance(risk_dagger, float)
    
    # theta_dagger should have lower risk (it was trained)
    assert risk_dagger <= risk_star * 1.1  # Allow small tolerance

if __name__ == "__main__":
    pytest.main([__file__, "-v"])