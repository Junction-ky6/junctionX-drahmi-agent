import torch
from torch import log_
from random import gauss

from schema import Observation, Prediction
from model import ActorCriticNetwork
from config import model_path, obs_space_size, action_space_size, n_categories
from utils import get_observation_tensor, softmax, listify

model = ActorCriticNetwork(obs_space_size, action_space_size)
model.load_state_dict(state_dict=torch.load(model_path))

def prediction(model: ActorCriticNetwork, observation: Observation) -> Prediction:
    obs = get_observation_tensor(observation)
    mean, std, _ = model.forward(obs)
    std = log_(std)
    
    logits = []
    for _ in range(action_space_size):
        prob = gauss(mu=mean.mean(), sigma=std.mean())
        logits.append(prob)
    ratios = softmax(listify(logits))
    
    prediction = {
        "categories_ratios": ratios[:n_categories],
        "goal_ratio": ratios[n_categories],
        "investment_ratio": ratios[n_categories + 1]
    }
    return  Prediction(**prediction)