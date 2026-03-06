from main import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
                total += prob * (reward + GAMMA * V[j])
            action_values.append(total)

        best_action = ACTIONS[np.argmax(action_values)]
        policy[state] = best_action

    return policy

def plot_policy(policy, water=0, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))

   

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):

            icons = []

            if (x, y) in BOULDERS:
                icons.append("assets/rock.png")

            if (x, y) == FIRE:
                icons.append("assets/fire.png")

            if (x, y) == LAKE:
                icons.append("assets/lake.png")

            if (x, y) in SMOKE:
                icons.append("assets/smoke.png")

            action = policy.get((x, y, water), None)

            icons.append(f"assets/{action}.png")

            if len(icons) == 0:
                continue

            alpha = 0.6 if len(icons) > 1 else 1.0

            for icon in icons:
                add_icon(ax, icon, x, y, alpha=alpha)

    # Grid
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    

    ax.tick_params(axis='both', labelsize=18)

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/policy_water_{water}_4d.png")
    plt.close()

    print(f"Policy plot saved for water={water}")



# Create all states
states = []
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))

state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

n_states = len(states)
n_actions = len(ACTIONS)

V = np.load("results/values.npy")
policy = extract_policy(V)


plot_policy(policy, water=0)
plot_policy(policy, water=1)