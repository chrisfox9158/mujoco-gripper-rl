# Training hyperparameters
RANDOMIZATION = {
    "object_mass": {"min": 0.1, "max": 0.4},
    "crush_threshold": {"min": 5.0, "max": 20.0},
}

TRACKING = {
    "lifted_height": 0.1,
    "success_height": 0.8,
    "hold_steps_required": 50,
    "crush_steps_required": 15,
    "maximum_steps": 5000,
    "grasp_threshold": 0.3
}

REWARDS = {
    "drop_penalty": -1.0,
    "crush_penalty": -5.0,
    "grasp_reward": 10.0,
    "distance_scale": 0.05,
    "lift_scale": 0.05,
    "success_reward": 50.0
}

# XML model-specific parameters
HARDWARE = {
    "joint_names": ["palm_lift", "finger1_joint1", "finger1_joint2",
                     "finger2_joint1", "finger2_joint2",
                     "finger3_joint1", "finger3_joint2"],
    "sensor_names": ["finger1_touch", "finger2_touch", "finger3_touch"],
    "site_names": ["finger1_tip", "finger2_tip", "finger3_tip"],
}