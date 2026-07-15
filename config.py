RANDOMIZATION = {
    "object_mass": {"min": 0.1, "max": 0.4},
    "crush_threshold": {"min": 5.0, "max": 20.0},
}

TRACKING = {
    "lifted_height": 0.1,
    "success_height": 0.8,
    "hold_steps_required": 50,
    "crush_steps_required": 15,
    "maximum_steps": 5000
}

REWARDS = {
    "drop_penalty": -1.0,
    "crush_penalty": -5.0,
    "success_reward": 50.0
}