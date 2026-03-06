import numpy as np
from collections import defaultdict
from utils.discretization import create_bins, discretize
from utils.exploration import epsilon_greedy, epsilon_decay
from config import GAMMA

def q_learning(env, episodes, alpha, eps_start, seed=0, eps_min=0.01, num_bins=10):
    np.random.seed(seed)
    env.reset(seed=seed)
    
    bins = create_bins(num_bins)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    epsilon = eps_start
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins)

        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins)

            Q[state][action] += alpha * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            total_reward += reward

        epsilon = epsilon_decay(epsilon, eps_min=eps_min)

        returns.append(total_reward)

    return Q, returns