import pytest
import jax
import jax.numpy as jnp
import numpy as np
from synthfuse import MixtureOfSVRExperts


def test_import():
    """Test that MixtureOfSVRExperts can be imported."""
    assert MixtureOfSVRExperts is not None


def test_instantiation():
    """Test model instantiation with valid parameters."""
    model = MixtureOfSVRExperts(degree=2, lam_tree=0.1, epochs=10, n_experts=4)
    assert model is not None


def test_fit_predict_numpy():
    """Test fit/predict with NumPy arrays (scikit-learn compatibility)."""
    X = np.random.randn(200, 3)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(200)

    model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=50, n_experts=8)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert isinstance(y_pred, np.ndarray)


def test_fit_predict_jax():
    """Test fit/predict with JAX arrays."""
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (150, 4))
    y = X[:, 0] ** 2 + 0.05 * jax.random.normal(key, (150,))

    model = MixtureOfSVRExperts(degree=2, lam_tree=0.3, epochs=30, n_experts=4)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert isinstance(y_pred, jnp.ndarray)


def test_predict_before_fit_raises():
    """Ensure predict() raises an error if fit() hasn't been called."""
    model = MixtureOfSVRExperts()
    X = np.ones((5, 2))
    with pytest.raises(Exception, match="not fitted"):
        model.predict(X)


def test_input_validation():
    """Test that incompatible input shapes raise errors."""
    model = MixtureOfSVRExperts()
    X = np.random.randn(100, 3)
    y = np.random.randn(99)  # mismatched length
    with pytest.raises(ValueError):
        model.fit(X, y)
