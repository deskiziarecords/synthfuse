# 🧬 Synthfuse ACR

[![CI](https://github.com/deskiziarecords/synthfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/deskiziarecords/synthfuse/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Synthfuse ACR (Adaptive Complexity Regularization)** is a high-performance hybrid ML engine that fuses the global smoothness of Polynomial Support Vector Regression (SVR) with the local partitioning power of Decision Trees.

By utilizing **JAX** and `vmap`, Synthfuse trains a **Mixture of Experts** simultaneously on the GPU, achieving speeds up to 50x faster than traditional ensemble loops.

## 🚀 Key Features

* **Adaptive Complexity Regularization (ACR):** Automatically increases regularization penalties in high-variance/noisy regions identified by a Random Forest complexity map.
* **JAX-Powered Mixture of Experts:** Uses `vmap` to train hundreds of local expert models in parallel.
* **Scikit-Learn Compatible:** Drop-in replacement for standard regressors with `fit()` and `predict()` API.
* **Heteroscedasticity Robust:** Designed specifically for datasets where noise levels vary across the feature space.

## 📊 Performance

In our benchmarks, Synthfuse ACR maintains a near-flat execution time as the number of experts (K) increases, whereas standard serial loops scale linearly.

| Number of Experts (K) | SK-Learn (Serial) | Synthfuse (JAX) | Speedup |
| :--- | :--- | :--- | :--- |
| 8 | 0.42s | 0.08s | 5.2x |
| 64 | 3.15s | 0.11s | 28.6x |
| 128 | 6.80s | 0.14s | 48.5x |

## 🛠️ Installation

Using `uv` (recommended):

```bash
uv add synthfuse
```

Standard pip:

```bash
pip install synthfuse
```

## 💡 Quick Start

```python
from synthfuse import MixtureOfSVRExperts
import numpy as np

# Generate data
X = np.random.randn(1000, 5)
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(1000)

# Initialize and fit
model = MixtureOfSVRExperts(
    degree=3,
    lam_tree=0.5,
    epochs=500
)
model.fit(X, y)

# Predict
predictions = model.predict(X)
```

## 🧠 The Math: How ACR Works

The loss function for each expert k is defined by:

$$L_k = C\sum L_{\epsilon}(y, \hat{y}) + \lambda \cdot (\Omega_{tree} \cdot ||w_k||^2)$$

Where:

* **$L_{\epsilon}$** is the ϵ-insensitive hinge loss.
* **$\Omega_{tree}$** is the Complexity Map derived from the average leaf depth of a Random Forest.
* Regions with deep leaves (high complexity) receive higher L² penalties, forcing the model to be smoother and less reactive to noise.
## Retrain with synthfuse
`Synthfuse retrain` implements a mathematical framework for improving neural network models by constructing optimal retraining paths in parameter space. Given an incumbent model θ* and a strictly superior model θ†, the library provides methods to find a new model θ‡ that dominates θ† while retaining desirable properties.

## Key Features

### Deterministic Retraining Paths
- **WSFT** (Warm-Start Fine-Tuning): Geodesic with biased initial velocity
- **KDT** (Knowledge-Distillation Transfer): Objective-space lifting with softened labels
- **OTWI** (Optimal Transport Weight Interpolation): Measure-preserving parameter coupling
- **FRR** (Functional Regularized Reset): NTK-regularized functional proximity

### Diffuse Retraining Methods
- **DBSD** (Diffusion Bridge with Superior-Drift): SDE-based diffusion bridges
- **TIL** (Teacher-Injected Langevin): Controlled Langevin processes
- **FDD** (Functional Diffusion Distillation): Hyperspherical diffusion on NTK manifold
- **Schrödinger Bridge**: Optimal stochastic control formulation

### Geometric Tools
- Parameter manifold operations (tangent spaces, geodesics)
- Optimal transport couplings (Wasserstein-2)
- Neural Tangent Kernel computations
- Fisher information geometry

  
## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

## ☕ Support

If you found this implementation useful and would like to support the project, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/hipotermiah)
