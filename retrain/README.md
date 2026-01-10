# README.md
# Synthfuse retrain: Geometric Paths for Model Improvement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/synthfuse-retrain.svg)](https://badge.fury.io/py/synthfuse-retrain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/deskiziarecords/synthfuse/retrain/actions/workflows/tests.yml/badge.svg)](https://github.com/deskiziarecords/synthfuse/retrain/actions)

## Overview

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

## Installation

```bash
pip install synthfuse-retrain
```

## ☕ Support

If you found this implementation useful and would like to support the project, consider buying me a testing board :):

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/hipotermiah)
