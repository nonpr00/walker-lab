"""Guardar poses y reproducirlas sin agente ni recalcular el recorrido."""

import argparse
import json
from pathlib import Path

import numpy as np

from .environment import WalkerEnv
from .scenarios import validate_scenario

class TrajectoryRecorder:
    def __init__(self, env, simulation_seed=None):
        self.scene = dict(env.scene)
        if simulation_seed is not None:
            self.scene["simulation_seed"] = simulation_seed
        self.poses, self.distances, self.steps = [], [], []
        self.capture(env)

    def capture(self, env):
        self.poses.append([(float(body.position.x), float(body.position.y), float(body.angle))
                           for body in env.legs + [env.hull]])
        self.distances.append(float(env.max_distance))
        self.steps.append(env.steps)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Abrir el archivo evita que NumPy añada otra extensión al nombre.
        with path.open("wb") as stream:
            np.savez_compressed(stream, scenario_json=json.dumps(self.scene, ensure_ascii=False, allow_nan=False),
                                poses=np.asarray(self.poses, dtype=np.float64),
                                distances=np.asarray(self.distances, dtype=np.float64),
                                steps=np.asarray(self.steps, dtype=np.int64))


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as data:
        scene = json.loads(data["scenario_json"].item())
        poses, distances, steps = (data[key] for key in ("poses", "distances", "steps"))
    validate_scenario(scene)
    count = len(poses)
    if (count < 1 or poses.shape != (count, 5, 3) or distances.shape != (count,)
            or steps.shape != (count,) or not np.isfinite(poses).all()
            or not np.isfinite(distances).all() or not np.isfinite(steps).all()
            or not np.issubdtype(steps.dtype, np.integer)
            or np.any(steps < 0) or np.any(steps > scene["max_steps"])
            or np.any(np.diff(steps) < 0) or np.any(np.diff(distances) < 0)
            or np.any(distances < 0) or np.any(distances > scene["length"])):
        raise ValueError("Datos de trayectoria inválidos.")
    return scene, poses, distances, steps


def play_trajectory(path):
    import pygame

    scene, poses, distances, steps = load_trajectory(path)
    env = WalkerEnv(scenario=scene)
    try:
        env.reset()
        env.render_mode = "human"
        pygame.init()
        for pose, distance, step in zip(poses, distances, steps):
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                break
            for body, (x, y, angle) in zip(env.legs + [env.hull], pose):
                body.position = (float(x), float(y))
                body.angle = float(angle)
            env.max_distance = float(distance)
            env.steps = int(step)
            env.render()
            pygame.display.set_caption(f"Reproducción: {Path(path).name}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Reproducir una trayectoria grabada, sin cargar un agente.")
    parser.add_argument("--trajectory", type=Path, required=True, help="Archivo .npz generado con evaluar.py --record.")
    args = parser.parse_args()
    play_trajectory(args.trajectory)
