# Training hyperparameters
RANDOMIZATION = {
    "object_mass": {"min": 0.1, "max": 0.4},
    "crush_threshold": {"min": 10.0, "max": 20.0},
}

TRACKING = {
    "lifted_height": 0.1,
    "success_height": 0.8,
    "hold_steps_required": 50,
    "crush_steps_required": 15,
    "maximum_steps": 5e3,
    "grasp_threshold": 0.3,
}

REWARDS = {
    "drop_penalty": -10.0,
    "crush_penalty": -20.0,
    "grasp_reward": 5.0,
    "distance_scale": 10.0, # dampens significantly as distance delta is very small step-to-step
    "lift_scale": 25.0, # dampens significantly as height delta is very small step-to-step
    "success_reward": 100.0,
}

# XML model-specific parameters
HARDWARE = {
    "joint_names": ["palm_lift", "finger1_joint1", "finger1_joint2",
                     "finger2_joint1", "finger2_joint2",
                     "finger3_joint1", "finger3_joint2"],
    "sensor_names": ["finger1_touch", "finger2_touch", "finger3_touch"],
    "site_names": ["finger1_tip", "finger2_tip", "finger3_tip"],
}

AGENT_SPECS = {
    "replay_buffer_capacity": int(1e6), # requires integer
    "replay_buffer_batch_size": int(1e3), # requires integer
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "default_noise_scale": 0.125
}