# examples/basic_usage.py
"""
Basic usage example for retraining-manifold.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from retraining_manifold import (
    WSFT, KDT, OTWI, FRR,
    TIL, PathSelector,
    compute_ntk
)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_synthetic_data(n_samples=1000, n_features=20):
    """Create synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_baseline_model(model, X_train, y_train, X_val, y_val, epochs=20):
    """Train a baseline model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_acc = (val_outputs.argmax(1) == y_val).float().mean()
            
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss.item():.4f}, Val Acc = {val_acc.item():.4f}")
    
    return model

def main():
    print("=== Retraining Manifold Demo ===")
    
    # Create synthetic data
    X_train, X_val, y_train, y_val = create_synthetic_data()
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}")
    
    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    theta_star = SimpleMLP(input_dim=20, hidden_dim=64, output_dim=2).to(device)
    theta_dagger = SimpleMLP(input_dim=20, hidden_dim=64, output_dim=2).to(device)
    
    # Move data to device
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    
    # Train incumbent model (theta_star)
    print("\n1. Training incumbent model (theta_star)...")
    theta_star = train_baseline_model(theta_star, X_train, y_train, X_val, y_val, epochs=20)
    
    # Train superior teacher model (theta_dagger)
    print("\n2. Training superior teacher model (theta_dagger)...")
    theta_dagger = train_baseline_model(theta_dagger, X_train, y_train, X_val, y_val, epochs=30)
    
    # Compute baseline risks
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        theta_star.eval()
        theta_dagger.eval()
        
        star_outputs = theta_star(X_val)
        star_loss = criterion(star_outputs, y_val)
        star_acc = (star_outputs.argmax(1) == y_val).float().mean()
        
        dagger_outputs = theta_dagger(X_val)
        dagger_loss = criterion(dagger_outputs, y_val)
        dagger_acc = (dagger_outputs.argmax(1) == y_val).float().mean()
    
    print(f"\n3. Baseline Performance:")
    print(f"   theta_star: Loss = {star_loss.item():.4f}, Acc = {star_acc.item():.4f}")
    print(f"   theta_dagger: Loss = {dagger_loss.item():.4f}, Acc = {dagger_acc.item():.4f}")
    print(f"   Superiority gap: {star_loss.item() - dagger_loss.item():.4f}")
    
    # Method 1: Warm-Start Fine-Tuning
    print("\n4. Testing WSFT method...")
    wsft = WSFT(
        theta_star=theta_star,
        theta_dagger=theta_dagger,
        lambda_coef=0.1,
        device=device
    )
    
    theta_wsft, wsft_metrics = wsft.train(
        train_data=X_train,
        train_labels=y_train,
        val_data=X_val,
        val_labels=y_val,
        num_epochs=10,
        lr=1e-4
    )
    
    with torch.no_grad():
        wsft_outputs = theta_wsft(X_val)
        wsft_loss = criterion(wsft_outputs, y_val)
        wsft_acc = (wsft_outputs.argmax(1) == y_val).float().mean()
    
    print(f"   WSFT: Loss = {wsft_loss.item():.4f}, Acc = {wsft_acc.item():.4f}")
    print(f"   Improvement over theta_dagger: {dagger_loss.item() - wsft_loss.item():.4f}")
    
    # Method 2: Diffuse retraining with TIL
    print("\n5. Testing TIL method...")
    til = TIL(
        theta_star=theta_star,
        theta_dagger=theta_dagger,
        noise_schedule='exponential',
        lambda_0=1.0,
        sigma_0=0.1,
        device=device
    )
    
    # Need to set gradient function
    def grad_fn(model, params):
        # For demo, use random gradient
        return torch.randn_like(params) * 0.01
    
    til._grad_fn = grad_fn
    
    til_samples = til.sample(
        n_samples=5,
        steps=100,
        step_size=0.01,
        data=X_train,
        labels=y_train
    )
    
    # Evaluate best sample
    best_til_loss = float('inf')
    best_til_model = None
    
    for i, model in enumerate(til_samples):
        with torch.no_grad():
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
        
        if loss < best_til_loss:
            best_til_loss = loss
            best_til_model = model
    
    til_acc = (best_til_model(X_val).argmax(1) == y_val).float().mean()
    print(f"   TIL (best sample): Loss = {best_til_loss.item():.4f}, Acc = {til_acc.item():.4f}")
    
    # Method 3: Automatic path selection
    print("\n6. Testing automatic path selection...")
    selector = PathSelector(
        lambda_coef=0.1,
        generalization_method="rademacher",
        device=device,
        verbose=True
    )
    
    best_model, best_method, selection_metrics = selector.select(
        theta_star=theta_star,
        theta_dagger=theta_dagger,
        train_data=X_train,
        train_labels=y_train,
        val_data=X_val,
        val_labels=y_val,
        budget=1000,
        epsilon=0.01,
        methods=['WSFT', 'KDT', 'FRR']  # Test subset of methods
    )
    
    with torch.no_grad():
        best_outputs = best_model(X_val)
        best_loss = criterion(best_outputs, y_val)
        best_acc = (best_outputs.argmax(1) == y_val).float().mean()
    
    print(f"\n7. Final Results:")
    print(f"   Selected method: {best_method}")
    print(f"   Best model: Loss = {best_loss.item():.4f}, Acc = {best_acc.item():.4f}")
    print(f"   Improvement over theta_star: {star_loss.item() - best_loss.item():.4f}")
    print(f"   Improvement over theta_dagger: {dagger_loss.item() - best_loss.item():.4f}")
    
    # Print selection metrics
    print(f"\n8. Selection Metrics:")
    for method_name, metrics in selection_metrics['all_metrics'].items():
        print(f"   {method_name}:")
        print(f"     Final risk: {metrics.final_risk:.4f}")
        print(f"     Path length: {metrics.path_length:.2f}")
        print(f"     Description length: {metrics.description_length:.2f}")
        print(f"     Success probability: {metrics.success_probability:.2%}")
    
    return {
        'theta_star': {'loss': star_loss.item(), 'acc': star_acc.item()},
        'theta_dagger': {'loss': dagger_loss.item(), 'acc': dagger_acc.item()},
        'wsft': {'loss': wsft_loss.item(), 'acc': wsft_acc.item()},
        'til': {'loss': best_til_loss.item(), 'acc': til_acc.item()},
        'selected': {'method': best_method, 'loss': best_loss.item(), 'acc': best_acc.item()}
    }

if __name__ == "__main__":
    results = main()