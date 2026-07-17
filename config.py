# Training hyperparameters
RANDOMIZATION = {
    "object_mass": {"min": 0.1, "max": 0.4}, # Default: 0.1, 0.4 
    "crush_threshold": {"min": 10.0, "max": 20.0}, # Default: 10.0, 20.0
}

TRACKING = {
    "lifted_height": 0.1,
    "success_height": 0.8,
    "hold_steps_required": 50,
    "crush_steps_required": 15,
    "maximum_steps": int(5e3),
    "grasp_threshold": 0.3,
    "control_repeat": 5
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
    "hardware_name": "three_finger_two_joint_gripper.xml",
    "joint_names": ["palm_lift", "finger1_joint1", "finger1_joint2",
                     "finger2_joint1", "finger2_joint2",
                     "finger3_joint1", "finger3_joint2"],
    "sensor_names": ["finger1_touch", "finger2_touch", "finger3_touch"],
    "site_names": ["finger1_tip", "finger2_tip", "finger3_tip"],
}

AGENT_SPECS = {
    "replay_buffer_capacity": int(1e6), # requires integer
    "replay_buffer_batch_size": int(256), # requires integer
    "actor_learning_rate": 3e-4, # known default: 3e-4
    "critic_learning_rate": 3e-4, # known default: 3e-4
    "default_noise_scale": 0.125,
    "gamma": 0.99, # known default: 0.99
    "policy_noise": 0.2, # known default: 0.2
    "noise_clip": 0.5, # known default: 0.5
    "policy_delay": 2, # known default: 2
    "tau": 0.005 # known default: 0.005
}

TRAINING_SPECS = {
    "num_episodes": 2000,
    "warmup_steps": 5000,
    "noise_start": 0.3,
    "noise_end": 0.05,
    "noise_decay_rate": 0.002,
    "train_freq": 4
}