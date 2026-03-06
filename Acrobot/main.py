import gymnasium as gym
import numpy as np
from config import *
from algorithms.sarsa import sarsa
from algorithms.q_learning import q_learning
from utils.plotting import *
from hyperparameter_search import *
from experiments import *


if __name__ == "__main__":

    # Finding the optimal episodes count

    env = gym.make("Acrobot-v1")

    _, sarsa_returns = sarsa(env, EPISODES, alpha=0.1, eps_start=1.0)
    _, qlearn_returns = q_learning(env, EPISODES, alpha=0.1, eps_start=1.0)

    plot_episodes(sarsa_returns, qlearn_returns)

    env.close()

####################################################################

    # Hyperparameter tuning

    env = gym.make("Acrobot-v1")

    sarsa_params = hyperparameter_search("sarsa", env)
    qlearn_params = hyperparameter_search("q_learning", env)

    print("Top 5 SARSA configs:")
    print(sarsa_params[:5])

    print("\nTop 5 Q-learning configs:")
    print(qlearn_params[:5])

    env.close()

    sarsa_results = params_to_matrix(sarsa_params, ALPHA_CONFIGS, EPSILON_CONFIGS)
    qlearning_results = params_to_matrix(qlearn_params, ALPHA_CONFIGS, EPSILON_CONFIGS)

    plot_heatmap(sarsa_results, ALPHA_CONFIGS, EPSILON_CONFIGS, "SARSA Hyperparameter Heatmap")
    plot_heatmap(qlearning_results, ALPHA_CONFIGS, EPSILON_CONFIGS, "Q-learning Hyperparameter Heatmap")

    # Best SARSA hyperparameters are alpha=0.05, epsilon=0.8
    # Best Q-learning hyperparameters are alpha=0.1, epsilon=0.5


####################################################################

    # Plot of performance between SARSA vs Q-learning

    sarsa_runs, _ = run_parallel_seeds(
        algorithm="sarsa",
        alpha=0.05,
        epsilon=0.8,
        episodes=EPISODES,
        num_seeds=SEEDS
    )

    q_runs, _ = run_parallel_seeds(
        algorithm="q_learning",
        alpha=0.1,
        epsilon=0.5,
        episodes=EPISODES,
        num_seeds=SEEDS
    )

    plot_sarsa_vs_qlearning(sarsa_runs, q_runs)

    
####################################################################

    # Comparison between online and offline performance

    sarsa_runs, sarsa_eval = run_parallel_seeds(
        algorithm="sarsa",
        alpha=0.05,
        epsilon=1.0,
        episodes=EPISODES,
        num_seeds=SEEDS,
        eval=True
    )

    q_runs, q_eval = run_parallel_seeds(
        algorithm="q_learning",
        alpha=0.1,
        epsilon=1.0,
        episodes=EPISODES,
        num_seeds=SEEDS,
        eval=True
    )

    print("Final greedy policy performance")

    print("SARSA mean:", np.mean(sarsa_eval))
    print("SARSA std:", np.std(sarsa_eval))

    print("\nQ-learning mean:", np.mean(q_eval))
    print("Q-learning std:", np.std(q_eval))

    plot_online_vs_offline(sarsa_runs, sarsa_eval, q_runs, q_eval)


####################################################################

    # Finding effect of binning

    sarsa_bins = run_bins_experiment(
        "sarsa",
        BIN_CONFIGS,
        alpha=0.05,
        epsilon=1.0,
        episodes=EPISODES,
        seeds=SEEDS
    )

    qlearn_bins = run_bins_experiment(
        "q_learning",
        BIN_CONFIGS,
        alpha=0.1,
        epsilon=1.0,
        episodes=EPISODES,
        seeds=SEEDS
    )

    plot_bins(sarsa_bins, qlearn_bins)
