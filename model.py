import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())

        self.policy_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size))  # Output layer for the mean of the action distribution

        self.policy_std = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size))  # Output layer for the standard deviation of the action distribution

        self.value_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        z = self.shared_layers(obs)
        mean = self.policy_mean(z)
        std = torch.exp(self.policy_std(z))  # Using the exponential function to ensure std is positive
        return mean, std

    def forward(self, obs):
        z = self.shared_layers(obs)
        mean = self.policy_mean(z)
        std = torch.exp(self.policy_std(z))
        value = self.value_layers(z)
        return mean, std, value