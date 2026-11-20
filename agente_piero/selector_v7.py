"""Detecta la salida de bloques con un historial causal de sensores del bípedo."""

from collections import deque
from pathlib import Path
import numpy as np


# A 50 Hz, estas vistas cubren desde el fotograma actual hasta 1,04 s atrás.
EDADES = (0, 4, 9, 15, 24, 36, 52)
VERTICAL = np.cos(1.5 * np.arange(10, dtype=np.float32) / 10)


class Ventana:
    def __init__(self):
        self.historia = deque(maxlen=max(EDADES) + 1)

    def agregar(self, obs):
        vertical = obs[14:] * VERTICAL
        # Profundidad vertical relativa: ayuda a distinguir un bloque aislado
        # de una sucesión de peldaños aunque cambie la altura del torso.
        cuadro = np.concatenate((vertical[1:] - vertical[0], obs[[14, 0, 2, 3]]))
        if not self.historia:
            self.historia.extend([cuadro] * self.historia.maxlen)
        else:
            self.historia.append(cuadro)

    def caracteristicas(self):
        return np.concatenate([self.historia[-1 - edad] for edad in EDADES])


class SelectorV7:
    def __init__(self, pesos, config):
        with np.load(pesos, allow_pickle=False) as data:
            self.media = data["media"]
            self.escala = data["escala"]
            self.capas = [(data[f"peso{i}"], data[f"sesgo{i}"]) for i in range(3)]
        self.umbral = float(config["umbral"])
        self.intensidad = float(config["intensidad"])
        self.rampa = int(config["rampa"])
        self.persistencia = int(config["persistencia"])
        self.device = "cpu"
        if not 0 < self.umbral < 1 or not 0 <= self.intensidad <= 1 or self.rampa < 1 or self.persistencia < 0:
            raise ValueError("Configuración inválida del selector V7.")
        self.arbitro = None
        if "umbral_beneficio" in config:
            from .arbitro_v7 import ArbitroV7

            self.arbitro = ArbitroV7(Path(pesos).with_name("arbitro.npz"))
            self.umbral_beneficio = float(config["umbral_beneficio"])
            if not 0 <= self.umbral_beneficio <= 1:
                raise ValueError("El umbral de beneficio debe estar entre cero y uno.")
        self.reset()

    def al_dispositivo(self, device):
        import torch

        self.device = device
        self.torch = torch
        self.media = torch.as_tensor(self.media, device=device)
        self.escala = torch.as_tensor(self.escala, device=device)
        self.capas = [(torch.as_tensor(p, device=device), torch.as_tensor(b, device=device))
                      for p, b in self.capas]
        if self.arbitro is not None:
            self.arbitro.al_dispositivo(device)

    def reset(self):
        self.ventana = Ventana()
        self.restante = 0
        self.peso = 0.0
        self.probabilidad = 0.0
        self.armado = True
        self.frames_fuera = 0
        self.pasos_pulso = 0

    def _peso_arbitrado(self, obs):
        # Una sola decisión por salida. Evita aplicar un segundo pulso durante
        # el mismo aterrizaje si el clasificador oscila cerca de su umbral.
        if self.pasos_pulso == 0:
            self.frames_fuera = self.frames_fuera + 1 if self.probabilidad < 0.5 else 0
            if self.frames_fuera >= 8:
                self.armado = True
            if self.armado and self.probabilidad >= self.umbral:
                self.armado = False
                if self.arbitro.puntuacion(obs, self.ventana) >= self.umbral_beneficio:
                    self.pasos_pulso = 15
        if not self.pasos_pulso:
            return 0.0
        fase = 16 - self.pasos_pulso
        peso = self.intensidad * min(1.0, fase / self.rampa, self.pasos_pulso / self.rampa)
        self.pasos_pulso -= 1
        return peso

    def peso_v6(self, obs):
        self.ventana.agregar(obs)
        x = self.ventana.caracteristicas()
        if self.device != "cpu":
            x = self.torch.as_tensor(x, device=self.device)
        x = (x - self.media) / self.escala
        for peso, sesgo in self.capas[:-1]:
            x = peso @ x + sesgo
            x = np.maximum(x, 0) if self.device == "cpu" else x.relu()
        peso, sesgo = self.capas[-1]
        logit = float((peso @ x + sesgo).item())
        self.probabilidad = 1.0 / (1.0 + np.exp(-max(-50.0, min(50.0, logit))))
        if self.arbitro is not None:
            self.peso = self._peso_arbitrado(obs)
            return self.peso
        if self.probabilidad >= self.umbral:
            self.restante = self.persistencia + 1
        delta = self.intensidad / self.rampa
        if self.restante:
            self.peso = min(self.intensidad, self.peso + delta)
            self.restante -= 1
        else:
            self.peso = max(0.0, self.peso - delta)
        return self.peso
