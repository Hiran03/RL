import numpy as np
import math

NUM_BINS = 10
GAMMA = 0.99

THETA_RANGE = (-np.pi, np.pi)
VEL_RANGE = (-8*np.pi, 8*np.pi)

OBS_LOW = np.array([-1.0, -1.0, -1.0, -1.0, -4 * math.pi, -9 * math.pi])
OBS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 4 * math.pi, 9 * math.pi])

EPISODES = 10000
SEEDS = 10

ALPHA_CONFIGS = [0.02, 0.05, 0.10, 0.15, 0.20]
EPSILON_CONFIGS = [0.5, 0.8, 1.0]

WINDOW = 100

BIN_CONFIGS = [5, 15, 20]