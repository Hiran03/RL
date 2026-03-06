import numpy as np
from collections import defaultdict
from utils.discretization import create_bins, discretize
from utils.exploration import epsilon_greedy, epsilon_decay
from config import GAMMA

def sarsa(env, episodes, alpha, eps_start, seed=0, eps_min=0.01, num_bins=10):
    np.random.seed(seed)
    env.reset(seed=seed)

    bins = create_bins(num_bins)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    epsilon = eps_start
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins)
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)

        done = False
        total_reward = 0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)

            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                Q[state][action] += alpha * (
                    reward + GAMMA * Q[next_state][next_action] - Q[state][action]
                )

            state, action = next_state, next_action
            total_reward += reward

        epsilon = epsilon_decay(epsilon, eps_min=eps_min)

        returns.append(total_reward)

    return Q, returns