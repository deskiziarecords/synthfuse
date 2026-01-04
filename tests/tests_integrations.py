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
