from main import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def value_iteration(theta=1e-6, save_dir="results", save_every=False):

    os.makedirs(save_dir, exist_ok=True)

    V = np.zeros(n_states)

    history = []
    value_snapshots = []

    iteration = 0

    while True:
        delta = 0
        new_V = np.copy(V)

        for i, state in enumerate(states):

            x, y, w = state

            if is_terminal(x, y, w):
                continue

            action_values = []

            for action in ACTIONS:
                total = 0
                for next_state, prob, reward in get_transitions(state, action):
                    j = state_to_idx[next_state]
                    total += prob * (reward + config.GAMMA * V[j])

                action_values.append(total)

            best_value = max(action_values)
            new_V[i] = best_value
            delta = max(delta, abs(V[i] - best_value))

        V = new_V
        history.append(delta)

        if save_every:
            value_snapshots.append(V.copy())

        iteration += 1

        if delta < theta:
            break

    print(f"Converged in {iteration} iterations")

    # ==============================
    # Save final values
    # ==============================

    np.save(f"{save_dir}/values.npy", V)

    df = pd.DataFrame({
        "state": states,
        "value": V
    })
    df.to_csv(f"{save_dir}/values.csv", index=False)

    # ==============================
    # Save convergence plot
    # ==============================

    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Bellman Residual (delta)")
    plt.title("Value Iteration Convergence")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(f"{save_dir}/convergence.png")
    plt.close()

    # Optionally save all snapshots
    if save_every:
        np.save(f"{save_dir}/value_snapshots.npy",
                np.array(value_snapshots))

    return V, history

def extract_policy(V):

    policy = {}

    for i, state in enumerate(states):

        if is_terminal(*state):
            policy[state] = "T"
            continue

        action_values = []

        for action in ACTIONS:
            total = 0
            for next_state, prob, reward in get_transitions(state, action):
                j = state_to_idx[next_state]
                total += prob * (reward + config.GAMMA * V[j])
            action_values.append(total)

        best_action = ACTIONS[np.argmax(action_values)]
        policy[state] = best_action

    return policy


ACTIONS = ["N", "S", "E", "W", "H"]

DIR = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1)
}

# Create all states
states = []
for x in range(config.GRID_SIZE):
    for y in range(config.GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))

state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

n_states = len(states)
n_actions = len(ACTIONS)

V, history = value_iteration(theta=1e-6)
