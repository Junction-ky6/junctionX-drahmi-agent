import torch
import numpy as np

from config import action_space_size
from schema import Observation

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def listify(x):
    return [float(i) for i in x]

def get_observation_tensor(observation: Observation):
    return torch.tensor([observation.money] + observation.categories_ratios + observation.user_satisfactions + [observation.goal_achievement, observation.investment_profit])