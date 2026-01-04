from .moe_svr import MixtureOfSVRExperts
__all__ = ["MixtureOfSVRExperts"]
class MixtureOfSVRExperts:
    def __init__(self, ...):
        self._is_fitted = False
        # ... other init ...

    def fit(self, X, y):
        # ... your fitting logic ...
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("This MixtureOfSVRExperts instance is not fitted yet. Call 'fit' first.")
        # ... prediction logic ...
