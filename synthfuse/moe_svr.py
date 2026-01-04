import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MixtureOfSVRExperts(BaseEstimator, RegressorMixin):
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
        features = [X]
        for d in range(2, self.degree + 1):
            features.append(jnp.power(X, d))
        return jnp.concatenate(features, axis=1)

    def _get_node_depths(self, tree):
        n_nodes = tree.tree_.node_count
        depths = np.zeros(n_nodes)
        stack = [(0, 0)] 
        while stack:
            node_id, depth = stack.pop()
            depths[node_id] = depth
            if tree.tree_.children_left[node_id] != -1:
                stack.append((tree.tree_.children_left[node_id], depth + 1))
                stack.append((tree.tree_.children_right[node_id], depth + 1))
        return depths

    def _compute_complexity(self, X, y):
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=self.random_state)
        rf.fit(X, y)
        all_depths = []
        for tree in rf.estimators_:
            d = self._get_node_depths(tree)
            all_depths.append([d[i] for i in tree.apply(X)])
        mean_d = jnp.mean(jnp.array(all_depths), axis=0)
        return (mean_d - mean_d.min()) / (mean_d.max() - mean_d.min() + 1e-6)

    def fit(self, X, y, assignments=None):
        X, y = check_X_y(X, y)
        self.complexity_ = self._compute_complexity(X, y)
        
        # Expert partitioning
        if assignments is None:
            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=16)
            self.tree_.fit(X, y)
            assignments = self.tree_.apply(X)
        
        unique_leaves = np.unique(assignments)
        self.leaf_to_idx_ = {leaf: i for i, leaf in enumerate(unique_leaves)}
        idx_assignments = jnp.array([self.leaf_to_idx_[l] for l in assignments])
        
        K = len(unique_leaves)
        X_poly = self._poly_features(jnp.array(X))
        F = X_poly.shape[1]

        self.params_ = {
            "W": jax.random.normal(jax.random.PRNGKey(self.random_state), (K, F)),
            "B": jnp.zeros((K, 1))
        }

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params_)

        @jax.jit
        def loss_fn(params, X_p, y_p, assigned_idxs, comp):
            w_ext = params["W"][assigned_idxs]
            b_ext = params["B"][assigned_idxs]
            preds = jnp.sum(w_ext * X_p, axis=1) + b_ext.squeeze()
            hinge = jnp.mean(jnp.maximum(0, jnp.abs(y_p - preds) - self.epsilon))
            reg = 0.5 * jnp.sum(jnp.square(params["W"])) * jnp.mean(comp)
            return hinge * self.C + self.lam_tree * reg

        @jax.jit
        def update(params, opt_state, X_p, y_p, assigned_idxs, comp):
            loss, grads = jax.value_and_grad(loss_fn)(params, X_p, y_p, assigned_idxs, comp)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(self.epochs):
            self.params_, opt_state, _ = update(self.params_, opt_state, X_poly, jnp.array(y), idx_assignments, self.complexity_)
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X_poly = self._poly_features(jnp.array(X))
        leaf_assignments = self.tree_.apply(X)
        idxs = jnp.array([self.leaf_to_idx_[l] for l in leaf_assignments])
        w_final = self.params_["W"][idxs]
        b_final = self.params_["B"][idxs]
        return jnp.sum(w_final * X_poly, axis=1) + b_final.squeeze()
