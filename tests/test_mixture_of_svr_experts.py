# tests/test_mixture_of_svr_experts.py
"""Test suite for MixtureOfSVRExperts"""

import pytest
import numpy as np
import jax.numpy as jnp
from synthfuse import MixtureOfSVRExperts


class TestMixtureOfSVRExpertsInit:
    """Test model initialization"""

    def test_default_initialization(self):
        """Test model initializes with default parameters"""
        model = MixtureOfSVRExperts()
        assert model.degree == 3
        assert model.C == 1.0
        assert model.epsilon == 0.1
        assert model.lam_tree == 0.1
        assert model.learning_rate == 0.01
        assert model.epochs == 200
        assert model.random_state == 42

    def test_custom_initialization(self):
        """Test model initializes with custom parameters"""
        model = MixtureOfSVRExperts(
            degree=2,
            C=0.5,
            epsilon=0.05,
            lam_tree=0.2,
            learning_rate=0.001,
            epochs=100,
            random_state=123
        )
        assert model.degree == 2
        assert model.C == 0.5
        assert model.epsilon == 0.05
        assert model.lam_tree == 0.2
        assert model.learning_rate == 0.001
        assert model.epochs == 100
        assert model.random_state == 123


class TestPolyFeatures:
    """Test polynomial feature transformation"""

    def test_poly_features_shape(self):
        """Test that polynomial features have correct shape"""
        model = MixtureOfSVRExperts(degree=3)
        X = jnp.ones((10, 5))
        X_poly = model._poly_features(X)
        
        # degree=3 means: original + degree 2 + degree 3 = 3 sets of features
        expected_features = 5 * 3  # n_features * degree
        assert X_poly.shape == (10, expected_features)

    def test_poly_features_values(self):
        """Test that polynomial features are computed correctly"""
        model = MixtureOfSVRExperts(degree=2)
        X = jnp.array([[2.0, 3.0]])
        X_poly = model._poly_features(X)
        
        # Should have [2, 3, 4, 9] (original and squared)
        expected = jnp.array([[2.0, 3.0, 4.0, 9.0]])
        assert jnp.allclose(X_poly, expected)

    def test_poly_features_degree_1(self):
        """Test polynomial features with degree=1 (no transformation)"""
        model = MixtureOfSVRExperts(degree=1)
        X = jnp.array([[2.0, 3.0]])
        X_poly = model._poly_features(X)
        
        assert X_poly.shape == (1, 2)
        assert jnp.allclose(X_poly, X)


class TestFitPredict:
    """Test fit and predict functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
        return X, y

    def test_fit_basic(self, sample_data):
        """Test that model can fit data"""
        X, y = sample_data
        model = MixtureOfSVRExperts(epochs=10)
        model.fit(X, y)
        
        assert hasattr(model, 'is_fitted_')
        assert model.is_fitted_ is True
        assert hasattr(model, 'params_')
        assert hasattr(model, 'tree_')
        assert hasattr(model, 'complexity_')

    def test_predict_shape(self, sample_data):
        """Test that predictions have correct shape"""
        X, y = sample_data
        model = MixtureOfSVRExperts(epochs=10)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (100,)
        assert len(predictions) == len(y)

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict raises error before fit"""
        X, y = sample_data
        model = MixtureOfSVRExperts()
        
        with pytest.raises((AttributeError, KeyError)):
            model.predict(X)

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for chaining"""
        X, y = sample_data
        model = MixtureOfSVRExperts(epochs=10)
        result = model.fit(X, y)
        
        assert result is model

    def test_predict_on_single_sample(self, sample_data):
        """Test prediction on single sample"""
        X, y = sample_data
        model = MixtureOfSVRExperts(epochs=10)
        model.fit(X, y)
        
        single_sample = X[0:1, :]
        prediction = model.predict(single_sample)
        
        assert prediction.shape == (1,)
        assert np.isfinite(prediction[0])

    def test_predictions_are_finite(self, sample_data):
        """Test that predictions don't contain NaN or inf"""
        X, y = sample_data
        model = MixtureOfSVRExperts(epochs=10)
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert np.all(np.isfinite(predictions))


class TestInputValidation:
    """Test input validation"""

    def test_fit_validates_input_shapes(self):
        """Test that fit validates X and y have same length"""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)  # Mismatched length
        model = MixtureOfSVRExperts()
        
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_fit_requires_2d_X(self):
        """Test that fit requires 2D X"""
        X = np.random.randn(100)  # 1D array
        y = np.random.randn(100)
        model = MixtureOfSVRExperts()
        
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, y)

    def test_predict_requires_fitted_model(self):
        """Test that predict requires model to be fitted first"""
        X = np.random.randn(100, 5)
        model = MixtureOfSVRExperts()
        
        with pytest.raises((AttributeError, KeyError)):
            model.predict(X)


class TestComplexityMap:
    """Test complexity map computation"""

    def test_complexity_shape(self):
        """Test that complexity map has correct shape"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = MixtureOfSVRExperts()
        complexity = model._compute_complexity(X, y)
        
        assert complexity.shape == (50,)

    def test_complexity_in_valid_range(self):
        """Test that complexity values are normalized"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = MixtureOfSVRExperts()
        complexity = model._compute_complexity(X, y)
        
        # Values should be between 0 and 1 (normalized)
        assert np.all(complexity >= 0)
        assert np.all(complexity <= 1)


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to prediction"""
        np.random.seed(42)
        X_train = np.random.randn(150, 4)
        y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.05 * np.random.randn(150)
        
        X_test = np.random.randn(30, 4)
        
        model = MixtureOfSVRExperts(degree=2, epochs=20)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (30,)
        assert np.all(np.isfinite(predictions))

    def test_different_random_states_give_different_results(self):
        """Test that different random states produce different models"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        model1 = MixtureOfSVRExperts(epochs=10, random_state=1)
        model2 = MixtureOfSVRExperts(epochs=10, random_state=2)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        pred1 = model1.predict(X[:10])
        pred2 = model2.predict(X[:10])
        
        # Different random states should give different predictions
        assert not np.allclose(pred1, pred2)
