import numpy as np
from algorithms.sarsa import sarsa
from algorithms.q_learning import q_learning
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def evaluate_params(params, algorithm, episodes, env):
    alpha, eps = params
    
    if algorithm == "sarsa":
        _, returns = sarsa(env, episodes, alpha, eps)
    else:
        _, returns = q_learning(env, episodes, alpha, eps)
    
    avg_return = np.mean(returns[-100:])
    return (alpha, eps, avg_return)

def hyperparameter_search(algorithm, env, episodes=10000):
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2]
    epsilons = [0.5, 0.8, 1.0]
    results = []

    param_combinations = list(itertools.product(alphas, epsilons))
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations")
    
    with ProcessPoolExecutor(max_workers=7) as executor:
        future_to_params = {
            executor.submit(evaluate_params, params, algorithm, episodes, env): params 
            for params in param_combinations
        }
        
        for i, future in enumerate(as_completed(future_to_params), 1):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed {i}/{len(param_combinations)}: α={result[0]:.2f}, ε={result[1]:.1f} → {result[2]:.2f}")
            except Exception as e:
                params = future_to_params[future]
                print(f"Failed for α={params[0]:.2f}, ε={params[1]:.1f}: {e}")

    results.sort(key=lambda x: x[2], reverse=True)
    return results