# Library imports
import numpy as np

# Local imports
from env.gripper_env import GripperEnv
from env import observations, rewards
from agent.td3 import TD3
from agent.noise_schedule import NoiseSchedule
from config import HARDWARE
from config import AGENT_SPECS
from config import TRAINING_SPECS

# Setup
env = GripperEnv(
    xml_path="three_finger_two_joint_gripper.xml",
    obs_extractors=[observations.obs_joint_angles, observations.obs_touch_sensors, observations.obs_joint_velocities],
    reward_terms=[rewards.reward_drop_penalty, rewards.reward_crush_penalty,
                  rewards.reward_grasp, rewards.reward_distance,
                  rewards.reward_lift, rewards.reward_success]
)

initial_obs = env.reset()
obs_dim = env.obs_dim
action_dim = env.action_dim
hardware_name = HARDWARE["hardware_name"]

agent = TD3(obs_dim, action_dim, hardware_name)

# Episode loop
total_steps = 0
for episode in range(TRAINING_SPECS["num_episodes"]):
    obs = env.reset()
    episode_reward = 0

    while True:
        if total_steps < TRAINING_SPECS["warmup_steps"]:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            noise_scale = NoiseSchedule.exponential_noise_decay(episode, TRAINING_SPECS["noise_start"], TRAINING_SPECS["noise_end"], TRAINING_SPECS["noise_decay_rate"])
            action = agent.select_action(obs, noise_scale=noise_scale)
        
        next_obs, reward, done = env.step(action)
        agent.replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        total_steps += 1

        if agent.replay_buffer.size >= AGENT_SPECS["replay_buffer_batch_size"]:
            agent.update()

        if done:
            break
    
    print(f"Episode: {episode}: reward={episode_reward:.2f}, steps={total_steps}")