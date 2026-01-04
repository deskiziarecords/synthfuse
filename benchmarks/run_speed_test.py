import time, gc, psutil, jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from synthfuse import MixtureOfSVRExperts

def get_mem(): return psutil.Process().memory_info().rss / 1024**2

def main():
    N, F = 15000, 5
    K_values = [8, 32, 64]
    X = np.random.randn(N, F)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(N)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    
    results = []
    for K in K_values:
        print(f"Testing K={K}...")
        tree = DecisionTreeRegressor(max_leaf_nodes=K).fit(X_tr, y_tr)
        asgn = tree.apply(X_tr)
        
        # SK-Learn
        t0 = time.time()
        for k in range(K):
            mask = (asgn == k)
            if np.any(mask): SVR(kernel='linear').fit(X_tr[mask], y_tr[mask])
        t_sk = time.time() - t0
        
        # JAX
        model = MixtureOfSVRExperts(epochs=50)
        # Warmup
        model.fit(X_tr[:100], y_tr[:100], asgn[:100])
        t0 = time.time()
        model.fit(X_tr, y_tr, asgn)
        t_jax = time.time() - t0
        
        results.append({"K": K, "SK_Time": t_sk, "JAX_Time": t_jax, "Speedup": t_sk/t_jax})

    df = pd.DataFrame(results)
    print(df)
    df.plot(x='K', y='Speedup', kind='bar', title='Synthfuse Speedup vs SK-Learn')
    plt.savefig('benchmarks/speedup.png')

if __name__ == "__main__": main()
