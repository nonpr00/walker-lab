"""Arquitectura de la política: MLP determinista con salida tanh en [-1, 1]."""

import torch
from torch import nn


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim=24, act_dim=4, hidden=(256, 256)):
        super().__init__()
        layers, last = [], obs_dim
        for size in hidden:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.net(obs))
