# tests/__init__.py
"""Test suite for Synthfuse"""

# tests/test_mixture_of_experts.py
import pytest
import numpy as np
from synthfuse import MixtureOfSVRExperts


class TestMixtureOfSVRExperts:
    """Test suite for MixtureOfSVRExperts model"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
        return X, y

    def test_initialization(self):
        """Test model initialization"""
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        assert model.degree == 3
        assert model.lam_tree == 0.5
        assert model.epochs == 10

    def test_fit(self, sample_data):
        """Test model fitting"""
        X, y = sample_data
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        model.fit(X, y)
        assert hasattr(model, 'experts'), "Model should have experts after fitting"

    def test_predict_shape(self, sample_data):
        """Test prediction output shape"""
        X, y = sample_data
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape, "Predictions should match target shape"

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict raises error before fit"""
        X, y = sample_data
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        with pytest.raises(AttributeError):
            model.predict(X)

    def test_input_validation(self, sample_data):
        """Test input validation"""
        X, y = sample_data
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        
        # Test mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, y[:-10])  # y has fewer samples than X

    def test_predict_single_sample(self, sample_data):
        """Test prediction on single sample"""
        X, y = sample_data
        model = MixtureOfSVRExperts(degree=3, lam_tree=0.5, epochs=10)
        model.fit(X, y)
        
        single_sample = X[0:1, :]
        prediction = model.predict(single_sample)
        assert prediction.shape == (1,), "Single sample prediction should be 1D array with 1 element"


# tests/test_integration.py
import numpy as np
from synthfuse import MixtureOfSVRExperts


class TestIntegration:
    """Integration tests for Synthfuse"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to prediction"""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(200)
        
        model = MixtureOfSVRExperts(degree=2, lam_tree=0.5, epochs=20, n_experts=5)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Check basic properties
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions)), "Predictions should not contain NaN or inf"
        
        # Check reasonable accuracy (MSE)
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1.0, "Model should achieve reasonable accuracy on this simple problem"
