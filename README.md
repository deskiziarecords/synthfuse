# 🧬 Synthfuse ACR

**Adaptive Complexity Regularization (ACR)** via JAX-powered **Mixture of SVR Experts**.

Synthfuse fuses the global smoothness of **Polynomial Support Vector Regression** with the local adaptivity of **tree-based partitioning**—all accelerated on CPU/GPU via **JAX + Optax**.

- 🚀 **Differentiable**: Full gradient flow from loss → regularization → tree prior  
- ⚡ **Fast**: `jax.vmap` trains 100+ local experts in parallel  
- 📏 **Smart Regularization**: Regions with high tree depth → stronger L2 penalty  
- 🔌 **Scikit-learn compatible**: Drop-in replacement for `SVR`

## 📦 Install

```bash
pip install synthfuse
# OR from source
uv sync --all-extras
Drop-in enhanced ML models: Polynomial SVR with tree regularization, KNN-boosted RF, 
Conv-LSTM autoencoder, and more. All models are scikit-learn compatible and JAX-accelerated.

## Quick start
```bash
uv sync --all-extras
from hybrid_ml import PolynomialSVR
model = PolynomialSVR(degree=3).fit(X, y)

EOF

cat > LICENSE <<'EOF' MIT License Copyright (c) 2026 hybrid_ml contributors Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. EOF
