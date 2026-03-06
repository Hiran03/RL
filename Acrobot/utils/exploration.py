import numpy as np

def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

def epsilon_decay(eps, eps_min=0.01, decay=0.995):
    return max(eps_min, eps * decay)

def epsilon_schedule(episode, decay_episodes=5000):
    eps_start = 1.0
    eps_end = 0.1
    
    if episode < decay_episodes:
        return eps_start - (eps_start - eps_end) * (episode / decay_episodes)
    else:
        return eps_end