"""Actor determinista de SAC para WalkerEnv, con dos capas ocultas ReLU.

El actor original se extrajo de models/sac_hardcore/solved.pt de Jared Tambua,
repositorio jaredtambua/sac-bipedal-walker, revisión
72d33889124bb90de11ae9d29946d3e7450cf10b. Usa las 24 observaciones directamente.
"""

import json

import numpy as np

from walker_lab.agent import BaseAgent


class Agent(BaseAgent):
    VARIANTES = ("original", "afinado", "mejorado")

    def __init__(self, model_dir, device="cpu", variante=None):
        super().__init__(model_dir, device)
        seleccion = self.model_dir / "seleccion.txt"
        elegida = variante
        if elegida is None and seleccion.is_file():
            elegida = seleccion.read_text(encoding="utf-8").strip()
        if elegida in ("entrega_sac", "entrega_hibrida") or (
                elegida is None and (self.model_dir / "actor.npz").is_file()):
            self.variante = elegida or self.model_dir.name
            self.ruta_pesos = self.model_dir / "actor.npz"
        else:
            self.variante = elegida
            if self.variante not in self.VARIANTES:
                raise ValueError(f"Variante desconocida: {self.variante}")
            self.ruta_pesos = self.model_dir / f"sac_{self.variante}.npz"
        if self.device != "cpu":
            import torch

            self.torch = torch
        self.capas = self._cargar(self.ruta_pesos)
        self.selector = None
        config = self.model_dir / "hibrido.json"
        if elegida == "entrega_hibrida" or (elegida is None and config.is_file()):
            from .selector_v7 import SelectorV7

            self.capas_v6 = self._cargar(self.model_dir / "actor_v6.npz")
            self.selector = SelectorV7(self.model_dir / "selector.npz", json.loads(config.read_text()))
            if self.device != "cpu":
                self.selector.al_dispositivo(self.device)
        self.reset()

    def _cargar(self, ruta):
        with np.load(ruta, allow_pickle=False) as pesos:
            capas = [(pesos[n + ".weight"], pesos[n + ".bias"]) for n in ("trunk.0", "trunk.2", "mean_layer")]
        if self.device != "cpu":
            capas = [(self.torch.as_tensor(p, device=self.device), self.torch.as_tensor(b, device=self.device))
                     for p, b in capas]
        return capas

    def reset(self):
        if self.selector is not None:
            self.selector.reset()

    def _accion(self, x, capas):
        if self.device == "cpu":
            for peso, sesgo in capas[:-1]:
                x = np.maximum(peso @ x + sesgo, 0)
            peso, sesgo = capas[-1]
            return np.tanh(peso @ x + sesgo)

        with self.torch.inference_mode():
            x = self.torch.as_tensor(x, device=self.device)
            for peso, sesgo in capas[:-1]:
                x = (peso @ x + sesgo).relu()
            peso, sesgo = capas[-1]
            return (peso @ x + sesgo).tanh().cpu().numpy()

    def act(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32)
        mezcla = self.selector.peso_v6(x) if self.selector is not None else 0.0
        if mezcla <= 0.0:
            return self._accion(x, self.capas)
        if mezcla >= 1.0:
            return self._accion(x, self.capas_v6)
        return (1.0 - mezcla) * self._accion(x, self.capas) + mezcla * self._accion(x, self.capas_v6)
