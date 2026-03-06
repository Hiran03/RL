import numpy as np
from utils.discretization import discretize

def evaluate_policy(env, Q, episodes=100):

    returns = []

    for _ in range(episodes):

        state, _ = env.reset()
        state = discretize(state)
        done = False
        total_reward = 0

        while not done:

            action = np.argmax(Q[state])

            state, reward, terminated, truncated, _ = env.step(action)
            state = discretize(state)

            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

    return np.mean(returns)