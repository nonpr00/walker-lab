"""Probar el único agente del alumno sobre un recorrido."""

import json
from pathlib import Path

from .agent import normalize_device
from .environment import WalkerEnv
from .replay import TrajectoryRecorder
from .scenarios import load_scenario, scenario_digest


class Evaluator:
    def __init__(self, scenario, render=False, device="cpu", record_dir=None, seed=None):
        self.device = normalize_device(device)
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.path = Path(scenario).resolve()
        if self.path.is_dir():
            raise ValueError("Indique un archivo JSON con un único recorrido, no una carpeta de pistas.")
        self.scenario = load_scenario(self.path)
        self.simulation_seed = self.scenario["simulation_seed"] if seed is None else seed
        if not isinstance(self.simulation_seed, int) or self.simulation_seed < 0:
            raise ValueError("La semilla inicial debe ser un entero no negativo.")
        self.render = render

    def evaluate(self, agent):
        env = WalkerEnv(scenario=self.scenario, render_mode="human" if self.render else None)
        try:
            observation, info = env.reset(seed=self.simulation_seed)
            recorder = TrajectoryRecorder(env, simulation_seed=self.simulation_seed) if self.record_dir is not None else None
            agent.reset()
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = agent.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if recorder is not None:
                    recorder.capture(env)
            report = {**info, "device": self.device, "scenario": self.path.name,
                      "scenario_sha256": scenario_digest(self.scenario), "simulation_seed": self.simulation_seed,
                      "max_steps": self.scenario["max_steps"], "training_return": total_reward}
            if recorder is not None:
                target = self.record_dir / f"{self.path.stem}.npz"
                recorder.save(target)
                report["trajectory"] = str(target.resolve())
            print(f"{self.path.name}: distancia={info['distance']:.3f}, tiempo={info['simulated_seconds']:.2f}s, final={info['reason']}", flush=True)
            return report
        finally:
            env.close()

    @staticmethod
    def save_report(report, output):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        saved = dict(report)
        if "trajectory" in saved:
            target = Path(saved["trajectory"])
            try:
                saved["trajectory"] = target.relative_to(output.parent.resolve()).as_posix()
            except ValueError:
                pass
        output.write_text(json.dumps(saved, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Resultado guardado: {output}")
