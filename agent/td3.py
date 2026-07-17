# Library imports
import torch
import numpy as np

# Local imports
from networks import Actor
from networks import Critic
from replay_buffer import ReplayBuffer
from config import AGENT_SPECS

class TD3():
    """The primary agent system: active agent instances, random noise injection, TD3-specific architecture."""
    def __init__(self, obs_dim, action_dim):
        """Set up TD3's six agents, two optimizers, and sync systems."""
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)

        self.critic1 = Critic(obs_dim, action_dim)
        self.critic1_target = Critic(obs_dim, action_dim)
        self.critic2 = Critic(obs_dim, action_dim)
        self.critic2_target = Critic(obs_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=AGENT_SPECS["actor_learning_rate"])
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=AGENT_SPECS["critic_learning_rate"])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def select_action(self, state, noise_scale=AGENT_SPECS["default_noise_scale"]):
        """Select action and add random noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]

        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = action + noise

        return np.clip(action, -1, 1)