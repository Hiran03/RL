import numpy as np
from algorithms.sarsa import sarsa
from algorithms.q_learning import q_learning
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluation import evaluate_policy

def run_seed(seed, algorithm, episodes, alpha, epsilon, eval=False, num_bins=10):

    import gymnasium as gym
    import numpy as np

    np.random.seed(seed)
    env = gym.make("Acrobot-v1")
    obs, _ = env.reset(seed=seed)

    if algorithm == "sarsa":
        Q, returns = sarsa(env, episodes, alpha, epsilon, seed, num_bins=num_bins)
    else:
        Q, returns = q_learning(env, episodes, alpha, epsilon, seed, num_bins=num_bins)

    if eval:
        final_perf = evaluate_policy(env, Q)
    else:
        final_perf = -1000.0

    env.close()

    return returns, final_perf

def run_parallel_seeds(algorithm, alpha, epsilon, episodes=10000, num_seeds=10, eval=False):

    seed_returns = []
    final_perfs = []

    print(f"Running {algorithm} with {num_seeds} seeds")

    with ProcessPoolExecutor(max_workers=7) as executor:

        future_to_seed = {
            executor.submit(run_seed, seed, algorithm, episodes, alpha, epsilon, eval): seed
            for seed in range(num_seeds)
        }

        for i, future in enumerate(as_completed(future_to_seed), 1):

            seed = future_to_seed[future]

            try:
                returns, final_perf = future.result()

                seed_returns.append(returns)
                final_perfs.append(final_perf)

                print(f"Completed seed {seed} ({i}/{num_seeds})")

            except Exception as e:
                print(f"Seed {seed} failed: {e}")

    return np.array(seed_returns), np.array(final_perfs)

def run_bins_experiment(algorithm, bins, alpha, epsilon, episodes=10000, seeds=10):

    all_runs = []

    for b in bins:

        print(f"\nRunning {algorithm} with {b} bins")

        runs = []

        with ProcessPoolExecutor(max_workers=7) as executor:

            futures = [
                executor.submit(run_seed, seed, algorithm, episodes, alpha, epsilon, False, b)
                for seed in range(seeds)
            ]

            for f in as_completed(futures):
                returns, _ = f.result()
                runs.append(returns)

        all_runs.append(np.array(runs))

    return all_runs