import numpy as np
from config import *

def create_bins(num_bins):
    bins = [
        np.linspace(THETA_RANGE[0], THETA_RANGE[1], num_bins - 1),
        np.linspace(THETA_RANGE[0], THETA_RANGE[1], num_bins - 1),
        np.linspace(VEL_RANGE[0], VEL_RANGE[1], num_bins - 1),
        np.linspace(VEL_RANGE[0], VEL_RANGE[1], num_bins - 1)
    ]
    return bins

def discretize(obs, bins):
    cos1, sin1, cos2, sin2, vel1, vel2 = obs
    
    theta1 = np.arctan2(sin1, cos1)
    theta2 = np.arctan2(sin2, cos2)
    
    state = [
        np.digitize(theta1, bins[0]),
        np.digitize(theta2, bins[1]),
        np.digitize(vel1, bins[2]),
        np.digitize(vel2, bins[3])
    ]
    return tuple(state)