"""Entrenar el bípedo con SAC (Stable-Baselines3) y exportar el actor a agente/policy.pt.

Ejemplos:
    python entrenar.py --etapa facil --pasos 600000
    python entrenar.py --etapa mixta --pasos 2000000 --resume entrenamiento/ultimo.zip
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from agente.architecture import PolicyMLP
from walker_lab.environment import WalkerEnv

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "entrenamiento"
POLICY_PATH = ROOT / "agente" / "policy.pt"

# (perfil, longitud, dificultad): un entorno paralelo por entrada.
ETAPAS = {
    "facil": [
        ("plano", 60, "normal"), ("suave", 60, "normal"), ("suave", 60, "normal"),
        ("pendientes", 60, "normal"), ("escalones", 60, "normal"), ("bloques", 60, "normal"),
        ("huecos", 60, "normal"), ("mixto", 60, "normal"),
    ],
    "mixta": [
        ("suave", 60, "normal"), ("pendientes", 60, "normal"), ("escalones", 60, "hardcore"),
        ("bloques", 60, "hardcore"), ("huecos", 60, "hardcore"), ("mixto", 60, "hardcore"),
        ("maraton", 120, "normal"), ("maraton", 120, "hardcore-plus"),
    ],
}

# Pistas reales sobre las que se elige el mejor modelo.
EVAL_SCENARIOS = [
    ROOT / "escenarios" / "practica" / "mixto.json",
    ROOT / "escenarios" / "validacion" / "maraton_01.json",
    ROOT / "escenarios" / "validacion" / "maraton_02.json",
    ROOT / "escenarios" / "hardcore" / "maraton_01.json",
]


def make_env(spec, seed):
    def _init():
        profile, length, difficulty = spec
        env = WalkerEnv(profile=profile, length=length, difficulty=difficulty)
        env = Monitor(env, info_keywords=("distance", "success"))
        env.reset(seed=seed)
        return env
    return _init


def export_actor(model, path):
    """Copia el actor determinista de SAC (latent_pi + mu, salida tanh) a PolicyMLP."""
    actor = model.policy.actor
    linears = [m for m in actor.latent_pi if isinstance(m, torch.nn.Linear)] + [actor.mu]
    hidden = [m.out_features for m in linears[:-1]]
    policy = PolicyMLP(hidden=hidden)
    targets = [m for m in policy.net if isinstance(m, torch.nn.Linear)]
    for src, dst in zip(linears, targets):
        dst.weight.data.copy_(src.weight.data.cpu())
        dst.bias.data.copy_(src.bias.data.cpu())
    torch.save({"hidden": hidden, "state_dict": policy.state_dict()}, path)
    return policy


@torch.no_grad()
def evaluate(policy, scenarios):
    """Distancia normalizada (0..1) media sobre pistas fijas, con el MLP exportado."""
    scores = []
    for path in scenarios:
        env = WalkerEnv(scenario=str(path))
        obs, info = env.reset(seed=env.fixed_scenario["simulation_seed"])
        done = False
        while not done:
            action = policy(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["distance"] / info["length"])
        env.close()
    return float(np.mean(scores)), scores


class ExportCallback(BaseCallback):
    def __init__(self, eval_freq, best=-1.0):
        super().__init__()
        self.eval_freq = eval_freq
        self.next_eval = eval_freq
        self.best = best
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            self._evaluate()
        return True

    def _evaluate(self):
        OUT_DIR.mkdir(exist_ok=True)
        self.model.save(OUT_DIR / "ultimo.zip")
        tmp = OUT_DIR / "candidato.pt"
        policy = export_actor(self.model, tmp)
        score, scores = evaluate(policy, EVAL_SCENARIOS)
        buf = [e for e in self.model.ep_info_buffer if "distance" in e]
        train_dist = np.mean([e["distance"] for e in buf]) if buf else float("nan")
        mark = ""
        if score > self.best:
            self.best = score
            export_actor(self.model, POLICY_PATH)
            self.model.save(OUT_DIR / "mejor.zip")
            mark = "  <-- mejor, guardado en agente/policy.pt"
        self.logger.record("eval/distancia_norm", score)
        print(f"[{(time.time() - self.t0) / 60:6.1f} min] pasos={self.num_timesteps:>8} "
              f"dist_train={train_dist:6.1f} eval={score:.3f} "
              f"({', '.join(f'{s:.2f}' for s in scores)}){mark}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Entrenar con SAC.")
    parser.add_argument("--etapa", choices=ETAPAS, default="facil")
    parser.add_argument("--pasos", type=int, default=600_000, help="Pasos de entorno a entrenar en esta ejecución.")
    parser.add_argument("--resume", type=Path, help="Modelo .zip de SB3 desde el que continuar.")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--device", default="cpu", help="Dispositivo de entrenamiento (cpu o cuda).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-init", type=float, default=-1.0, help="Puntaje a superar para reemplazar agente/policy.pt.")
    parser.add_argument("--learning-starts", type=int, default=20_000, help="Pasos aleatorios iniciales (usar poco solo para pruebas).")
    args = parser.parse_args()

    torch.set_num_threads(4)
    specs = ETAPAS[args.etapa]
    env = SubprocVecEnv([make_env(spec, args.seed * 100 + i) for i, spec in enumerate(specs)], start_method="spawn")
    try:
        if args.resume:
            model = SAC.load(args.resume, env=env, device=args.device, tensorboard_log=str(OUT_DIR / "tb"))
            model.learning_starts = 0
        else:
            model = SAC(
                "MlpPolicy", env, device=args.device, seed=args.seed, verbose=0,
                learning_rate=7e-4, buffer_size=1_000_000, batch_size=256, gamma=0.98, tau=0.02,
                learning_starts=args.learning_starts, train_freq=8, gradient_steps=32,
                policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=str(OUT_DIR / "tb"),
            )
        print(f"Etapa {args.etapa}: {len(specs)} entornos; {args.pasos} pasos.", flush=True)
        model.learn(args.pasos, callback=ExportCallback(args.eval_freq, args.best_init), reset_num_timesteps=not args.resume,
                    log_interval=10_000)
        OUT_DIR.mkdir(exist_ok=True)
        model.save(OUT_DIR / "ultimo.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
