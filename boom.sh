#!/usr/bin/env bash
# bootstrap_synthfuse.sh – Focused, GPU-optimized Hybrid ML
set -euo pipefail

echo "🐣 Deploying Synthfuse ACR v1: The JAX Expert Engine..."

# ---------- 1. Infrastructure ----------
mkdir -p synthfuse tests benchmarks .github/workflows

# ---------- 2. Environment ----------
cat > pyproject.toml <<'POLLO'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "synthfuse"
version = "0.1.0"
description = "Adaptive Complexity Regularization (ACR) via JAX Mixture of Experts"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "jax[cpu]>=0.4.20", 
    "optax>=0.1.7",
    "pandas",
    "matplotlib",
    "psutil"
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
POLLO

# ---------- 3. The ACR Engine ----------
cat > synthfuse/__init__.py <<'POLLO'
from .moe_svr import MixtureOfSVRExperts
__all__ = ["MixtureOfSVRExperts"]
POLLO

cat > synthfuse/moe_svr.py <<'POLLO'
import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MixtureOfSVRExperts(BaseEstimator, RegressorMixin):
    def __init__(self, degree=3, C=1.0, epsilon=0.1, lam_tree=0.1, 
                 learning_rate=0.01, epochs=200, random_state=42):
        self.degree = degree
        self.C = C
        self.epsilon = epsilon
        self.lam_tree = lam_tree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def _poly_features(self, X):
        features = [X]
        for d in range(2, self.degree + 1):
            features.append(jnp.power(X, d))
        return jnp.concatenate(features, axis=1)

    def _get_node_depths(self, tree):
        n_nodes = tree.tree_.node_count
        depths = np.zeros(n_nodes)
        stack = [(0, 0)] 
        while stack:
            node_id, depth = stack.pop()
            depths[node_id] = depth
            if tree.tree_.children_left[node_id] != -1:
                stack.append((tree.tree_.children_left[node_id], depth + 1))
                stack.append((tree.tree_.children_right[node_id], depth + 1))
        return depths

    def _compute_complexity(self, X, y):
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=self.random_state)
        rf.fit(X, y)
        all_depths = []
        for tree in rf.estimators_:
            d = self._get_node_depths(tree)
            all_depths.append([d[i] for i in tree.apply(X)])
        mean_d = jnp.mean(jnp.array(all_depths), axis=0)
        return (mean_d - mean_d.min()) / (mean_d.max() - mean_d.min() + 1e-6)

    def fit(self, X, y, assignments=None):
        X, y = check_X_y(X, y)
        self.complexity_ = self._compute_complexity(X, y)
        
        # Expert partitioning
        if assignments is None:
            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=16)
            self.tree_.fit(X, y)
            assignments = self.tree_.apply(X)
        
        unique_leaves = np.unique(assignments)
        self.leaf_to_idx_ = {leaf: i for i, leaf in enumerate(unique_leaves)}
        idx_assignments = jnp.array([self.leaf_to_idx_[l] for l in assignments])
        
        K = len(unique_leaves)
        X_poly = self._poly_features(jnp.array(X))
        F = X_poly.shape[1]

        self.params_ = {
            "W": jax.random.normal(jax.random.PRNGKey(self.random_state), (K, F)),
            "B": jnp.zeros((K, 1))
        }

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params_)

        @jax.jit
        def loss_fn(params, X_p, y_p, assigned_idxs, comp):
            w_ext = params["W"][assigned_idxs]
            b_ext = params["B"][assigned_idxs]
            preds = jnp.sum(w_ext * X_p, axis=1) + b_ext.squeeze()
            hinge = jnp.mean(jnp.maximum(0, jnp.abs(y_p - preds) - self.epsilon))
            reg = 0.5 * jnp.sum(jnp.square(params["W"])) * jnp.mean(comp)
            return hinge * self.C + self.lam_tree * reg

        @jax.jit
        def update(params, opt_state, X_p, y_p, assigned_idxs, comp):
            loss, grads = jax.value_and_grad(loss_fn)(params, X_p, y_p, assigned_idxs, comp)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(self.epochs):
            self.params_, opt_state, _ = update(self.params_, opt_state, X_poly, jnp.array(y), idx_assignments, self.complexity_)
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X_poly = self._poly_features(jnp.array(X))
        leaf_assignments = self.tree_.apply(X)
        idxs = jnp.array([self.leaf_to_idx_[l] for l in leaf_assignments])
        w_final = self.params_["W"][idxs]
        b_final = self.params_["B"][idxs]
        return jnp.sum(w_final * X_poly, axis=1) + b_final.squeeze()
POLLO

# ---------- 4. The Benchmark Suite ----------
cat > benchmarks/run_speed_test.py <<'POLLO'
import time, gc, psutil, jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from synthfuse import MixtureOfSVRExperts

def get_mem(): return psutil.Process().memory_info().rss / 1024**2

def main():
    N, F = 15000, 5
    K_values = [8, 32, 64]
    X = np.random.randn(N, F)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(N)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    
    results = []
    for K in K_values:
        print(f"Testing K={K}...")
        tree = DecisionTreeRegressor(max_leaf_nodes=K).fit(X_tr, y_tr)
        asgn = tree.apply(X_tr)
        
        # SK-Learn
        t0 = time.time()
        for k in range(K):
            mask = (asgn == k)
            if np.any(mask): SVR(kernel='linear').fit(X_tr[mask], y_tr[mask])
        t_sk = time.time() - t0
        
        # JAX
        model = MixtureOfSVRExperts(epochs=50)
        # Warmup
        model.fit(X_tr[:100], y_tr[:100], asgn[:100])
        t0 = time.time()
        model.fit(X_tr, y_tr, asgn)
        t_jax = time.time() - t0
        
        results.append({"K": K, "SK_Time": t_sk, "JAX_Time": t_jax, "Speedup": t_sk/t_jax})

    df = pd.DataFrame(results)
    print(df)
    df.plot(x='K', y='Speedup', kind='bar', title='Synthfuse Speedup vs SK-Learn')
    plt.savefig('benchmarks/speedup.png')

if __name__ == "__main__": main()
POLLO

# ---------- 5. Finalize ----------
git init -b main
git add .
git commit -m "feat: Synthfuse ACR v1 with JAX Mixture of Experts" || true

echo "🚀 Synthfuse is LIVE."
echo "Run: 'uv sync' then 'uv run python benchmarks/run_speed_test.py'"