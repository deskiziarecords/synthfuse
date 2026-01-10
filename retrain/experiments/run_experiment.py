# scripts/run_experiment.py
#!/usr/bin/env python3
"""
Command-line interface for running retraining experiments.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from retraining_manifold import PathSelector
from retraining_manifold.utils.dataloaders import load_dataset
from retraining_manifold.utils.metrics import compute_all_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Run retraining experiment")
    
    # Experiment configuration
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'synthetic', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'simple_cnn', 'mlp'],
                       help='Model architecture')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['WSFT', 'KDT', 'OTWI', 'FRR'],
                       help='Methods to evaluate')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--epochs_star', type=int, default=20,
                       help='Epochs to train incumbent model')
    parser.add_argument('--epochs_dagger', type=int, default=30,
                       help='Epochs to train teacher model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Retraining parameters
    parser.add_argument('--budget', type=float, default=1000,
                       help='Computational budget')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Target improvement')
    parser.add_argument('--delta', type=float, default=0.05,
                       help='Confidence level')
    
    # Technical settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"exp_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return device, output_dir

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_models(model_name, dataset_info, device):
    """Create model instances."""
    if model_name == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=dataset_info['num_classes'])
    elif model_name == 'simple_cnn':
        from retraining_manifold.utils.models import SimpleCNN
        model = SimpleCNN(
            input_channels=dataset_info.get('channels', 3),
            num_classes=dataset_info['num_classes']
        )
    elif model_name == 'mlp':
        from retraining_manifold.utils.models import SimpleMLP
        model = SimpleMLP(
            input_dim=dataset_info.get('input_dim', 784),
            num_classes=dataset_info['num_classes']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, epochs, lr, device, desc="Training"):
    """Train a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        if args.verbose and epoch % 5 == 0:
            print(f"{desc} Epoch {epoch}/{epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Val Acc = {val_acc:.2f}%")
    
    return model, history

def main():
    args = parse_args()
    
    # Setup experiment
    device, output_dir = setup_experiment(args)
    print(f"Experiment directory: {output_dir}")
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        # Update args with config
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset_info, train_loader, val_loader, test_loader = load_dataset(
        args.dataset, batch_size=args.batch_size
    )
    
    # Create models
    print(f"\nCreating {args.model} models...")
    theta_star = create_models(args.model, dataset_info, device)
    theta_dagger = create_models(args.model, dataset_info, device)
    
    # Train incumbent model (theta_star)
    print("\n1. Training incumbent model...")
    theta_star, star_history = train_model(
        theta_star, train_loader, val_loader,
        epochs=args.epochs_star, lr=args.lr, device=device,
        desc="Incumbent"
    )
    
    # Train superior teacher model (theta_dagger)
    print("\n2. Training superior teacher model...")
    theta_dagger, dagger_history = train_model(
        theta_dagger, train_loader, val_loader,
        epochs=args.epochs_dagger, lr=args.lr, device=device,
        desc="Teacher"
    )
    
    # Evaluate baseline models
    print("\n3. Evaluating baseline models...")
    star_metrics = compute_all_metrics(theta_star, test_loader, device)
    dagger_metrics = compute_all_metrics(theta_dagger, test_loader, device)
    
    print(f"   theta_star: Loss = {star_metrics['loss']:.4f}, "
          f"Acc = {star_metrics['accuracy']:.2f}%")
    print(f"   theta_dagger: Loss = {dagger_metrics['loss']:.4f}, "
          f"Acc = {dagger_metrics['accuracy']:.2f}%")
    print(f"   Superiority gap: {star_metrics['loss'] - dagger_metrics['loss']:.4f}")
    
    # Prepare data for retraining
    # Extract data from loaders (simplified - in practice use loaders)
    print("\n4. Preparing data for retraining...")
    train_data, train_labels = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        train_labels.append(batch[1])
    
    train_data = torch.cat(train_data).to(device)
    train_labels = torch.cat(train_labels).to(device)
    
    val_data, val_labels = [], []
    for batch in val_loader:
        val_data.append(batch[0])
        val_labels.append(batch[1])
    
    val_data = torch.cat(val_data).to(device)
    val_labels = torch.cat(val_labels).to(device)
    
    # Run retraining path selection
    print(f"\n5. Running retraining path selection with methods: {args.methods}")
    
    selector = PathSelector(
        lambda_coef=0.1,
        generalization_method="rademacher",
        device=device,
        verbose=args.verbose
    )
    
    try:
        best_model, best_method, selection_metrics = selector.select(
            theta_star=theta_star,
            theta_dagger=theta_dagger,
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            budget=args.budget,
            epsilon=args.epsilon,
            delta=args.delta,
            methods=args.methods,
            # Pass additional kwargs for each method
            wsft_kwargs={'lambda_coef': 0.1},
            kdt_kwargs={'alpha': 0.5, 'temperature': 2.0},
            frr_kwargs={'rho': 0.1}
        )
        
        # Evaluate final model
        print(f"\n6. Evaluating final model from {best_method}...")
        final_metrics = compute_all_metrics(best_model, test_loader, device)
        
        print(f"   Final model: Loss = {final_metrics['loss']:.4f}, "
              f"Acc = {final_metrics['accuracy']:.2f}%")
        print(f"   Improvement over theta_star: "
              f"{star_metrics['loss'] - final_metrics['loss']:.4f}")
        print(f"   Improvement over theta_dagger: "
              f"{dagger_metrics['loss'] - final_metrics['loss']:.4f}")
        
        # Save results
        results = {
            'baseline': {
                'theta_star': star_metrics,
                'theta_dagger': dagger_metrics,
                'superiority_gap': star_metrics['loss'] - dagger_metrics['loss']
            },
            'selected_method': best_method,
            'final_model': final_metrics,
            'selection_metrics': {
                method: {
                    'final_risk': metrics.final_risk,
                    'path_length': metrics.path_length,
                    'description_length': metrics.description_length,
                    'success_probability': metrics.success_probability
                } for method, metrics in selection_metrics['all_metrics'].items()
            },
            'certification': selection_metrics['certification'],
            'config': vars(args)
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save models
        torch.save(theta_star.state_dict(), output_dir / 'theta_star.pth')
        torch.save(theta_dagger.state_dict(), output_dir / 'theta_dagger.pth')
        torch.save(best_model.state_dict(), output_dir / 'theta_best.pth')
        
        print(f"\n7. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during retraining: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        with open(output_dir / 'error.txt', 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    main()