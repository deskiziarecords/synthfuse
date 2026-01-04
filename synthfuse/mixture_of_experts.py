"""Mixture of SVR Experts with Adaptive Complexity Regularization"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MixtureOfSVRExperts(BaseEstimator, RegressorMixin):
    """Mixture of Support Vector Regression Experts with Adaptive Complexity Regularization
    
    A high-performance hybrid ML engine that fuses the global smoothness of Polynomial SVR
    with the local partitioning power of Decision Trees using JAX for GPU acceleration.
    
    Parameters
    ----------
    degree : int, default=3
        Degree of polynomial features to generate
    C : float, default=1.0
        Regularization parameter for hinge loss
    epsilon : float, default=0.1
        Epsilon parameter for epsilon-insensitive hinge loss
    lam_tree : float, default=0.1
        Regularization weight for complexity-based regularization
    learning_rate : float, default=0.01
        Learning rate for Adam optimizer
    epochs : int, default=200
        Number of training epochs
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, degree=3, C=1.0, epsilon=0.1, lam_tree=0.1, 
                 learning_rate=0.01, epochs=200, random_state=42):
        self.degree = degree
        self.C = C
        self.epsilon = epsilon
        self.lam_tree = lam_tree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def _poly_features(self, X):
        """Generate polynomial features up to specified degree
        
        Parameters
        ----------
        X : jax.numpy.ndarray
            Input features of shape (n_samples, n_features)
            
        Returns
        -------
        jax.numpy.ndarray
            Polynomial features of shape (n_samples, n_features * degree)
        """
        features = [X]
        for d in range(2, self.degree + 1):
            features.append(jnp.power(X, d))
        return jnp.concatenate(features, axis=1)

    def _get_node_depths(self, tree):
        """Extract node depths from a decision tree
        
        Parameters
        ----------
        tree : sklearn.tree.DecisionTreeRegressor
            Fitted decision tree
            
        Returns
        -------
        numpy.ndarray
            Depth of each node in the tree
        """
        n_nodes = tree.tree_.node_count
        depths = np.zeros(n_nodes)
        stack = [(0, 0)]  # (node_id, depth)
        while stack:
            node_id, depth = stack.pop()
            depths[node_id] = depth
            if tree.tree_.children_left[node_id] != -1:
                stack.append((tree.tree_.children_left[node_id], depth + 1))
                stack.append((tree.tree_.children_right[node_id], depth + 1))
        return depths

    def _compute_complexity(self, X, y):
        """Compute complexity map using Random Forest leaf depths
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
            
        Returns
        -------
        jax.numpy.ndarray
            Normalized complexity scores for each sample
        """
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=self.random_state)
        rf.fit(X, y)
        all_depths = []
        for tree in rf.estimators_:
            d = self._get_node_depths(tree)
            all_depths.append([d[i] for i in tree.apply(X)])
        mean_d = jnp.mean(jnp.array(all_depths), axis=0)
        return (mean_d - mean_d.min()) / (mean_d.max() - mean_d.min() + 1e-6)

    def fit(self, X, y, assignments=None):
        """Fit the Mixture of Experts model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values
        assignments : array-like of shape (n_samples,), optional
            Expert assignments. If None, will be determined by DecisionTreeRegressor
            
        Returns
        -------
        self : MixtureOfSVRExperts
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        self.complexity_ = self._compute_complexity(X, y)
        
        # Expert partitioning using decision tree
        if assignments is None:
            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=16, random_state=self.random_state)
            self.tree_.fit(X, y)
            assignments = self.tree_.apply(X)
        
        unique_leaves = np.unique(assignments)
        self.leaf_to_idx_ = {leaf: i for i, leaf in enumerate(unique_leaves)}
        idx_assignments = jnp.array([self.leaf_to_idx_[l] for l in assignments])
        
        K = len(unique_leaves)
        X_poly = self._poly_features(jnp.array(X))
        F = X_poly.shape[1]

        # Initialize parameters
        self.params_ = {
            "W": jax.random.normal(jax.random.PRNGKey(self.random_state), (K, F)),
            "B": jnp.zeros((K, 1))
        }

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params_)

        @jax.jit
        def loss_fn(params, X_p, y_p, assigned_idxs, comp):
            """Compute loss with complexity-based regularization"""
            w_ext = params["W"][assigned_idxs]
            b_ext = params["B"][assigned_idxs]
            preds = jnp.sum(w_ext * X_p, axis=1) + b_ext.squeeze()
            hinge = jnp.mean(jnp.maximum(0, jnp.abs(y_p - preds) - self.epsilon))
            reg = 0.5 * jnp.sum(jnp.square(params["W"])) * jnp.mean(comp)
            return hinge * self.C + self.lam_tree * reg

        @jax.jit
        def update(params, opt_state, X_p, y_p, assigned_idxs, comp):
            """Single optimization step"""
            loss, grads = jax.value_and_grad(loss_fn)(params, X_p, y_p, assigned_idxs, comp)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        # Training loop
        for _ in range(self.epochs):
            self.params_, opt_state, _ = update(
                self.params_, opt_state, X_poly, jnp.array(y), 
                idx_assignments, self.complexity_
            )
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using the fitted model
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Model predictions
        """
        check_is_fitted(self, ['params_', 'tree_', 'is_fitted_'])
        X = check_array(X)
        
        X_poly = self._poly_features(jnp.array(X))
        leaf_assignments = self.tree_.apply(X)
        idxs = jnp.array([self.leaf_to_idx_[l] for l in leaf_assignments])
        w_final = self.params_["W"][idxs]
        b_final = self.params_["B"][idxs]
        return np.array(jnp.sum(w_final * X_poly, axis=1) + b_final.squeeze())
