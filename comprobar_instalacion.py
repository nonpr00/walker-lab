"""Comprobar la API y las dependencias sin disponer de un agente entrenado."""

import numpy as np

from walker_lab.environment import WalkerEnv


def main():
    env = WalkerEnv(profile="suave")
    try:
        observation, info = env.reset(seed=123)
        rng = np.random.default_rng(123)
        for _ in range(100):
            action = rng.uniform(-1, 1, 4).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(f"Instalación correcta. Observación: {observation.shape}; acciones: (4,); final: {info['reason']}.")
        print("Las acciones aleatorias solo verifican la instalación; no son una solución de RL.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
