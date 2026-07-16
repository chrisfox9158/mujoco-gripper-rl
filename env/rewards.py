# Library imports
import mujoco
import numpy as np

# Local imports
from config import REWARDS

# Rewards handlers
def reward_drop_penalty(model, data, info):
    """Light penalty for dropping the object after it was lifted."""
    object_z = data.xpos[info["object_body_id"]][2]
    floor_threshold = 0.05

    if info["was_lifted"] and object_z <= floor_threshold:
        return REWARDS["drop_penalty"]
    return 0.0

def reward_crush_penalty(model, data, info):
    """Heavy penalty for a sustained crush, not a single accidental touch."""
    
    if info["object_crushed"]:
        return REWARDS["crush_penalty"]
    return 0.0

def reward_grasp(model, data, info):
    """Medium reward, fires once, when grasp first occurs."""

    if info["just_grasped"]:
        return REWARDS["grasp_reward"]
    return 0.0

def reward_distance(model, data, info):
    """Small reward for positive fingertip-to-object delta."""
    return info["distance_delta"] * REWARDS["distance_scale"]

def reward_lift(model, data, info):
    """Small reward proportional to height gain delta."""
    return info["height_delta"] * REWARDS["lift_scale"]

def reward_success(model, data, info):
    """Large reward for holding the object at target height long enough."""

    if info["success"]:
        return REWARDS["success_reward"]
    return 0.0