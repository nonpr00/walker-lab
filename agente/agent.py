"""Adaptador de inferencia: carga policy.pt (MLP entrenado con SAC) y devuelve acciones."""

import numpy as np
import torch

from walker_lab.agent import BaseAgent

from .architecture import PolicyMLP


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        checkpoint = torch.load(self.model_dir / "policy.pt", map_location=self.device, weights_only=True)
        self.policy = PolicyMLP(hidden=tuple(checkpoint["hidden"])).to(self.device)
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy.eval()

    def reset(self):
        # La política es feed-forward: no hay memoria que reiniciar.
        pass

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        action = self.policy(obs).squeeze(0).cpu().numpy().astype(np.float32)
        return np.clip(action, -1.0, 1.0)
