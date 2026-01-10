# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retraining-manifold",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Geometric paths for neural network model improvement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retraining-manifold",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "optax>=0.1.0",
        ],
        "full": [
            "torchvision>=0.15.0",
            "tensorflow>=2.11.0",  # For NTK computations
            "neural-tangents>=0.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "retrain=scripts.run_experiment:main",
            "retrain-benchmark=scripts.benchmark:main",
        ],
    },
)