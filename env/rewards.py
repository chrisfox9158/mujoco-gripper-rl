# Library imports
import mujoco
import numpy as np

# Local imports

# Rewards handlers
def reward_drop_penalty(model, data, info):
    """Light penalty for dropping the object after it was lifted."""
    object_z = data.xpos[info["object_body_id"]][2]
    floor_threshold = 0.05

    if info["was_lifted"] and object_z <= floor_threshold:
        return -1.0
    return 0.0