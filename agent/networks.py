# Library imports
import torch
import torch.nn as nn

# Class definition for Actor agents
class Actor(nn.Module):
    """Definition for the Actor agent type."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, 256) # Begins series of layer-to-layer nn "transformations" of data (from # observer inputs to # available actions)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x
    
class Critic(nn.Module):
    """Definition for the Critic agent type."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x