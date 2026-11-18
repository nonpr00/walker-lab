"""Contrato de inferencia del agente."""

from abc import ABC, abstractmethod
from pathlib import Path
import re

import numpy as np

def normalize_device(device):
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("Use device='cpu', 'cuda' o 'cuda:N'.")
    return "cuda:0" if device == "cuda" else device


class BaseAgent(ABC):
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = Path(model_dir).resolve()
        self.device = normalize_device(device)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de entrega: {self.model_dir}")

    def reset(self):
        """Reiniciar memoria recurrente e historial al comenzar un episodio."""

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Recibir float32(24,) y devolver cuatro acciones finitas en [-1, 1]."""
        raise NotImplementedError
