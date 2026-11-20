"""Estima la conveniencia de un pulso del especialista al detectar un bloque."""
import numpy as np


def caracteristicas_arbitro(observacion, ventana):
    # El estado de las patas distingue posturas que un lidar aislado confunde.
    # Los tres fotogramas miden también la evolución del perfil y del torso.
    cuadros = [ventana.historia[-1 - edad] for edad in (0, 9, 24)]
    return np.concatenate((observacion[:14], cuadros[0], cuadros[0] - cuadros[1],
                           cuadros[1] - cuadros[2]))


class ArbitroV7:
    def __init__(self, pesos):
        with np.load(pesos, allow_pickle=False) as data:
            self.media = data["media"]
            self.escala = data["escala"]
            self.w1, self.b1 = data["w1"], data["b1"]
            self.w2, self.b2 = data["w2"], data["b2"]
        self.torch = None

    def al_dispositivo(self, device):
        import torch

        self.torch = torch
        self.device = device
        for k in ("media", "escala", "w1", "b1", "w2", "b2"):
            setattr(self, k, torch.as_tensor(getattr(self, k), device=device))

    def puntuacion(self, observacion, ventana_actual):
        rasgos = caracteristicas_arbitro(observacion, ventana_actual)
        if self.torch is not None:
            with self.torch.inference_mode():
                x = (self.torch.as_tensor(rasgos, device=self.device) - self.media) / self.escala
                h = (self.w1 @ x + self.b1).tanh()
                return (self.w2 @ h + self.b2).sigmoid().item()
        x = (rasgos - self.media) / self.escala
        h = np.tanh(self.w1 @ x + self.b1)
        logit = (self.w2 @ h + self.b2).item()
        return float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
