import numpy as np
import matplotlib.pyplot as plt
from config import *

def params_to_matrix(params, alphas, epsilons):
    matrix = np.zeros((len(alphas), len(epsilons)))

    for alpha, eps, reward in params:

        alpha_idx = alphas.index(alpha)
        eps_idx = epsilons.index(eps)

        matrix[alpha_idx][eps_idx] = reward

    return matrix

def smooth(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def compute_stats(data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        ci = 1.96 * std / np.sqrt(data.shape[0])

        return mean, ci


def plot_episodes(sarsa_returns, qlearn_returns):

    plt.figure(figsize=(10, 6))
    plt.plot(smooth(sarsa_returns), label="SARSA")
    plt.plot(smooth(qlearn_returns), label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title("Return vs Episodes: SARSA vs Q-learning")
    plt.legend()
    plt.grid()
    plt.show()

def plot_heatmap(results, alphas, epsilons, title):

    plt.figure(figsize=(7, 5))
    plt.imshow(results, origin="lower", aspect="auto")

    plt.xticks(range(len(epsilons)), epsilons)
    plt.yticks(range(len(alphas)), alphas)

    plt.xlabel("Initial ε")
    plt.ylabel("α (stepsize)")
    plt.colorbar(label="Avg Return (last episodes)")
    plt.title(title)
    plt.show()

def plot_sarsa_vs_qlearning(sarsa_runs, q_runs):

    sarsa_mean, sarsa_ci = compute_stats(sarsa_runs)
    q_mean, q_ci = compute_stats(q_runs)

    sarsa_mean_smooth = moving_average(sarsa_mean, WINDOW)
    q_mean_smooth = moving_average(q_mean, WINDOW)

    episodes = np.arange(WINDOW-1, EPISODES)

    plt.figure(figsize=(8,5))

    plt.plot(episodes, sarsa_mean_smooth, label="SARSA", color="red")
    plt.fill_between(
        episodes,
        moving_average(sarsa_mean - sarsa_ci, WINDOW),
        moving_average(sarsa_mean + sarsa_ci, WINDOW),
        color="red",
        alpha=0.2
    )

    plt.plot(episodes, q_mean_smooth, label="Q-learning", color="blue")
    plt.fill_between(
        episodes,
        moving_average(q_mean - q_ci, WINDOW),
        moving_average(q_mean + q_ci, WINDOW),
        color="blue",
        alpha=0.2
    )

    plt.xlabel("Episode Number")
    plt.ylabel("Episode Return")
    plt.title("SARSA vs Q-learning on Acrobot")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_online_vs_offline(sarsa_runs, sarsa_eval, q_runs, q_eval):

    sarsa_mean = sarsa_runs.mean(axis=0)
    sarsa_std = sarsa_runs.std(axis=0)

    q_mean = q_runs.mean(axis=0)
    q_std = q_runs.std(axis=0)

    episodes = np.arange(EPISODES)

    plt.figure(figsize=(8,5))

    plt.plot(episodes, sarsa_mean, label="SARSA")
    plt.plot(episodes, q_mean, label="Q-Learning")

    sarsa_mean_smooth = moving_average(sarsa_mean)
    q_mean_smooth = moving_average(q_mean)

    episodes = np.arange(len(sarsa_mean_smooth))

    plt.figure(figsize=(9,5))

    plt.plot(episodes, sarsa_mean_smooth, label="SARSA")
    plt.plot(episodes, q_mean_smooth, label="Q-learning")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Online Performance During Training")
    plt.legend()

    plt.grid()
    plt.show()

    plt.figure(figsize=(6,5))

    means = [np.mean(sarsa_eval), np.mean(q_eval)]
    stds = [np.std(sarsa_eval), np.std(q_eval)]

    plt.bar(["SARSA", "Q-Learning"], means, yerr=stds, capsize=5)

    plt.ylabel("Average Return")
    plt.title("Offline Performance (Greedy Policy Evaluation)")

    plt.grid(axis="y")
    plt.show()

def plot_bins(sarsa_bins, qlearn_bins):

    for i, bins in enumerate(BIN_CONFIGS):

        runs = sarsa_bins[i]

        mean = runs.mean(axis=0)
        smooth = moving_average(mean, WINDOW)

        plt.plot(smooth, label=f"SARSA bins={bins}")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Effect of Discretization Bins (SARSA)")
    plt.legend()
    plt.show()


    for i, bins in enumerate(BIN_CONFIGS):

        runs = qlearn_bins[i]

        mean = runs.mean(axis=0)
        smooth = moving_average(mean, WINDOW)

        plt.plot(smooth, label=f"Q-Learning bins={bins}")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Effect of Discretization Bins (Q-Learning)")
    plt.legend()
    plt.show()