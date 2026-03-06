import numpy as np
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

GRID_SIZE = config["GRID_SIZE"]

LAKE = tuple(config["LAKE"])
FIRE = tuple(config["FIRE"])

SMOKE = set(tuple(x) for x in config["SMOKE"])
BOULDERS = set(tuple(x) for x in config["BOULDERS"])

NORMAL_MOVE = config["NORMAL_MOVE"]
SMOKE_MOVE = config["SMOKE_MOVE"]

STEP_COST = config["STEP_COST"]
SMOKE_PENALTY = config["SMOKE_PENALTY"]
CRASH_COST = config["CRASH_COST"]
SUCCESS_REWARD = config["SUCCESS_REWARD"]

GAMMA = config["GAMMA"]
ACTIONS = config["ACTIONS"]

DIR = {
    k: tuple(v)
    for k, v in config["DIR"].items()
}

# Create all states
states = []
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for w in [0, 1]:
            states.append((x, y, w))

def in_grid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def move(x, y, action):
    dx, dy = DIR[action]
    nx, ny = x + dx, y + dy
    if not in_grid(nx, ny):
        return x, y
    return nx, ny

def get_perpendicular(action):
    if action in ["N", "S"]:
        return ["E", "W"]
    return ["N", "S"]
def reward(x, y, nx, ny, water):

    r = STEP_COST

    if (nx, ny) in SMOKE:
        r += SMOKE_PENALTY

    if (nx, ny) in BOULDERS:
        r += CRASH_COST

    if (nx, ny) == FIRE and water == 1:
        r += SUCCESS_REWARD

    return r
def is_terminal(x, y, w):
    if (x, y) in BOULDERS:
        return True
    if (x, y) == FIRE and w == 1:
        return True
    return False
def get_transitions(state, action):
    x, y, w = state

    if is_terminal(x, y, w):
        return [(state, 1.0, 0)]

    if action == "H":

        new_w = w
        if (x, y) == LAKE:
            new_w = 1

        reward = STEP_COST

        if (x, y) in SMOKE:
            reward += SMOKE_PENALTY

        if (x, y) in BOULDERS:
            reward += CRASH_COST

        if (x, y) == FIRE and new_w == 1:
            reward += SUCCESS_REWARD

        return [((x, y, new_w), 1.0, reward)]

    probs = SMOKE_MOVE if (x, y) in SMOKE else NORMAL_MOVE

    outcomes = []

    # Intended
    nx, ny = move(x, y, action)
    outcomes.append((nx, ny, probs["intended"]))

    # Perpendicular
    for p in get_perpendicular(action):
        px, py = move(x, y, p)
        outcomes.append((px, py, probs["perp"]))

    # Stay
    outcomes.append((x, y, probs["stay"]))

    transitions = []

    for nx, ny, prob in outcomes:

        new_w = w
        if (nx, ny) == LAKE:
            new_w = 1

        reward = STEP_COST

        if (nx, ny) in SMOKE:
            reward += SMOKE_PENALTY

        if (nx, ny) in BOULDERS:
            reward += CRASH_COST

        if (nx, ny) == FIRE and new_w == 1:
            reward += SUCCESS_REWARD

        transitions.append(((nx, ny, new_w), prob, reward))

    return transitions



def visualize_subset(state, action):

    trans = get_transitions(state, action)

    cells = [(i, j) for i in range(2, 5) for j in range(2, 5)]

    # Proper independent matrix
    matrix = np.empty((3, 3), dtype=object)

    # Initialize with (0 prob, 0 reward)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = (0.0, 0.0)

    # Accumulate probabilities and expected rewards
    for (nx, ny, _), prob, r in trans:
        if (nx, ny) in cells:

            i = nx - 2
            j = ny - 2
            matrix[i, j] = (prob,r)

    df = pd.DataFrame(
        matrix,
        index=[2, 3, 4],
        columns=[2, 3, 4]
    )

    print(f"\nTransition-Reward tuples for action {action} from (3,3):")
    print(df)

    return df

def plot_transition_subset(df, action, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))

    # checkerboard background
    checker = np.indices((3,3)).sum(axis=0) % 2
    ax.imshow(checker, cmap="gray", alpha=0.15,
              extent=[1.5,4.5,1.5,4.5])

    for i, x in enumerate([2,3,4]):
        for j, y in enumerate([2,3,4]):
            # environment icons
            if (x,y) in BOULDERS:
                add_icon(ax,"assets/rock.png",x,y,alpha=0.4)

            if (x,y) in SMOKE:
                add_icon(ax,"assets/smoke.png",x,y,alpha=0.4)

            if (x,y) == FIRE:
                add_icon(ax,"assets/fire.png",x,y,alpha=0.4)

            if (x,y) == LAKE:
                add_icon(ax,"assets/lake.png",x,y,alpha=0.4)
            prob, reward = df.loc[x,y]

            # text with probability and reward
            if prob > 0:
                txt = f"P={prob:.1f}\nR={reward}"
                ax.text(
                    x, y,
                    txt,
                    ha='center',
                    va='center',
                    fontsize=18,
                    bbox=dict(facecolor="white", alpha=0.6)
                )

            

    # highlight start state
    
    ax.text(3,3.35,"Start",ha='center',color='black',fontsize=18)

    # formatting
    ax.invert_yaxis()
    ax.set_xticks([2, 3, 4])
    ax.set_yticks([2, 3, 4])

    ax.tick_params(axis='both', labelsize=18)

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)


    ax.set_xlim(1.5,4.5)
    ax.set_ylim(1.5,4.5)

    

    plt.tight_layout()

    plt.savefig(f"{save_dir}/transition_{action}.png", dpi=300)
    plt.close()

    print(f"Saved transition graph for action {action}")
    
    
def add_icon(ax, img_path, x, y, zoom=0.15, alpha=0.5):
    # Load with alpha channel
    img = Image.open(img_path).convert("RGBA")
    img = img.resize((256, 256), Image.LANCZOS)

    im = OffsetImage(np.array(img), zoom=zoom)
    im.set_alpha(alpha)

    ab = AnnotationBbox(
        im,
        (x, y),
        frameon=False
    )

    ax.add_artist(ab)
if __name__ == "__main__":
    
    for a in ACTIONS : 
        print("Action:", a)
        df = visualize_subset((3,3,0), a)
        plot_transition_subset(df, a)
        print("------------------------------------------------------")
        print()
    