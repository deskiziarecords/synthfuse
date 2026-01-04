import pytest
import jax
import jax.numpy as jnp
import numpy as np
from synthfuse import SynthfuseACR  # or whatever your main class is called


def test_import():
    """Test that the package can be imported."""
    assert SynthfuseACR is not None


def test_instantiation():
    """Test that the model can be instantiated with default parameters."""
    model = SynthfuseACR()
    assert model is not None


def test_fit_predict_shapes():
    """Test that fit and predict return expected shapes."""
    # Generate simple synthetic data
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (100, 4))  # 100 samples, 4 features
    y = jnp.sum(X, axis=1) + 0.1 * jax.random.normal(key, (100,))  # linear + noise

    model = SynthfuseACR(n_experts=3, max_iter=10)  # small for speed
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert jnp.issubdtype(y_pred.dtype, jnp.floating)


def test_predict_before_fit_raises():
    """Ensure predict() fails gracefully if fit() hasn't been called."""
    model = SynthfuseACR()
    X = jnp.ones((5, 3))
    with pytest.raises(Exception):  # ideally a more specific error like NotFittedError
        model.predict(X)


def test_jax_array_input():
    """Ensure the model works with JAX arrays (not just NumPy)."""
    X_np = np.random.randn(50, 3)
    y_np = np.sum(X_np, axis=1)

    X_jax = jnp.array(X_np)
    y_jax = jnp.array(y_np)

    model = SynthfuseACR(n_experts=2, max_iter=5)
    model.fit(X_jax, y_jax)
    preds = model.predict(X_jax)

    assert isinstance(preds, jnp.ndarray)
    assert preds.shape == y_jax.shape
