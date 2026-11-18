"""Generación y lectura de pistas explícitas. No contiene políticas de control."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_Y = 400 / 30 / 4
PROFILES = ("plano", "suave", "pendientes", "escalones", "bloques", "huecos", "mixto", "maraton")
DIFFICULTIES = ("normal", "hardcore", "hardcore-plus")
HARDCORE_STEP = 14 / 30


def generate_scenario(profile="maraton", seed=0, length=None, max_steps=None, difficulty="normal"):
    if profile not in PROFILES:
        raise ValueError(f"Perfil desconocido: {profile}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Dificultad desconocida: {difficulty}")
    length = float(length if length is not None else (240 if profile == "maraton" else 60))
    max_steps = int(max_steps if max_steps is not None else (6000 if profile == "maraton" else 2400))
    if not 25 <= length <= 600 or not 100 <= max_steps <= 30000:
        raise ValueError("Longitud admitida: 25..600; horizonte: 100..30000 pasos.")
    rng = np.random.default_rng(seed)
    start_x = (14 / 30) * 20 / 2
    finish_x = start_x + length
    ground = [[0.0, BASE_Y], [10.0, BASE_Y]]
    obstacles, events = [], []

    def add(x, y):
        point = [round(float(x), 6), round(float(y), 6)]
        if point != ground[-1]:
            ground.append(point)

    cursor = 10.0
    while cursor < finish_x - 6:
        progress = np.clip((cursor - 10) / (finish_x - 10), 0, 0.999999)
        stage = min(4, int(progress * 5)) if profile == "maraton" else 2
        if profile == "maraton":
            choices = [
                ["suave", "pendientes"],
                ["suave", "pendientes", "escalones"],
                ["pendientes", "escalones", "bloques"],
                ["escalones", "bloques", "huecos"],
                ["pendientes", "escalones", "bloques", "huecos"],
            ][stage]
            if difficulty != "normal" and stage == 4:
                choices = ["bloques", "escalones", "huecos"]
            kind = str(rng.choice(choices))
            intensity = 0.25 + 0.16 * stage
        else:
            kind = str(rng.choice(PROFILES[1:6])) if profile == "mixto" else profile
            intensity = 0.45
        scale = (stage + 1) / 5 if profile == "maraton" else 1.0
        boost = 1.1 if difficulty == "hardcore-plus" else 1.0
        x0 = cursor
        if kind in ("plano", "suave"):
            width = float(rng.uniform(3.0, 5.0))
            n = int(np.ceil(width / 0.4))
            amp = 0 if kind == "plano" else 0.03 + 0.10 * intensity
            waves = int(rng.integers(1, 4))
            for j in range(1, n + 1):
                u = j / n
                add(x0 + width * u, BASE_Y + amp * np.sin(np.pi * u) * np.sin(2 * np.pi * waves * u))
            cursor += width
        elif kind == "pendientes":
            width = float(rng.uniform(4.5, 6.5))
            height = float(rng.uniform(0.2, 0.35 + 0.55 * intensity))
            sign = float(rng.choice([-1, 1]))
            add(x0 + width / 2, BASE_Y + sign * height)
            add(x0 + width, BASE_Y)
            cursor += width
        elif kind == "escalones":
            count = int(rng.integers(2, 4))
            width = float(rng.uniform(0.8, 1.2))
            height = float(rng.uniform(0.08, 0.12 + 0.18 * intensity))
            if difficulty != "normal":
                count = int(rng.integers(3, 5))
                width = 4 * HARDCORE_STEP
                height = HARDCORE_STEP * scale * boost
            for j in range(count):
                add(cursor, BASE_Y + (j + 1) * height)
                cursor += width
                add(cursor, BASE_Y + (j + 1) * height)
            for j in reversed(range(count)):
                add(cursor, BASE_Y + j * height)
                cursor += width
                add(cursor, BASE_Y + j * height)
        elif kind == "bloques":
            width = float(rng.uniform(0.45, 0.85))
            height = float(rng.uniform(0.15, 0.25 + 0.38 * intensity))
            if difficulty != "normal":
                height = float(rng.choice([1, 2])) * HARDCORE_STEP * scale * boost
                width = height
            obstacles.append({"x": round(cursor, 6), "y": BASE_Y, "width": width, "height": height})
            cursor += width
            add(cursor, BASE_Y)
        elif kind == "huecos":
            width = float(rng.uniform(0.30, 0.45 + 0.60 * intensity))
            if difficulty != "normal":
                width = float(rng.choice([2, 3])) * HARDCORE_STEP * scale * boost
            add(cursor, BASE_Y - 2.0)
            cursor += width
            add(cursor, BASE_Y - 2.0)
            add(cursor, BASE_Y)
        event = {"kind": kind, "x_start": round(x0, 6), "x_end": round(cursor, 6), "stage": stage + 1}
        if kind in ("escalones", "bloques"):
            event.update(height=height, width=width)
        elif kind == "huecos":
            event.update(width=width)
        events.append(event)
        cursor += float(rng.uniform(2.5, 4.0))
        add(cursor, BASE_Y)
    add(max(cursor, finish_x + 10), BASE_Y)
    scenario = {
        "profile": profile, "seed": int(seed),
        "difficulty": difficulty,
        "simulation_seed": int(seed), "start_x": start_x, "length": length,
        "finish_x": finish_x, "max_steps": max_steps,
        "ground": ground, "obstacles": obstacles, "events": events,
    }
    validate_scenario(scenario)
    return scenario


def validate_scenario(scene):
    if scene.get("profile") not in PROFILES:
        raise ValueError("Perfil inválido.")
    if scene.get("difficulty", "normal") not in DIFFICULTIES:
        raise ValueError("Dificultad inválida.")
    length, start, finish = (float(scene[k]) for k in ("length", "start_x", "finish_x"))
    if not all(np.isfinite([length, start, finish])) or not 25 <= length <= 600:
        raise ValueError("Longitud inválida.")
    if abs(finish - start - length) > 1e-5 or abs(start - (14 / 30) * 20 / 2) > 1e-5:
        raise ValueError("Salida o llegada incoherente.")
    if not isinstance(scene["max_steps"], int) or not 100 <= scene["max_steps"] <= 30000:
        raise ValueError("Horizonte inválido.")
    points = np.asarray(scene["ground"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Geometría inválida.")
    if np.any(np.diff(points[:, 0]) < 0) or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6):
        raise ValueError("El terreno debe avanzar sin segmentos degenerados.")
    if points[0, 0] > 0 or points[-1, 0] < finish + 2:
        raise ValueError("El terreno no cubre la salida y la llegada.")
    for block in scene["obstacles"]:
        values = [float(block[k]) for k in ("x", "y", "width", "height")]
        if not np.isfinite(values).all() or values[2] <= 0 or values[3] <= 0:
            raise ValueError("Bloque inválido.")


def load_scenario(path):
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_scenario(scene)
    return scene


def save_scenario(scene, path):
    validate_scenario(scene)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scenario_digest(scene):
    data = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generar una pista para entrenar o validar.")
    parser.add_argument("--profile", choices=PROFILES, default="maraton")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="normal", help="Rangos normales, de Hardcore o un 10 %% mayores; maraton aumenta gradualmente hasta ellos.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--length", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scene = generate_scenario(args.profile, args.seed, args.length, args.max_steps, args.difficulty)
    save_scenario(scene, args.output)
    print(f"Pista guardada: {args.output}; longitud: {scene['length']:.1f}; eventos: {len(scene['events'])}")


if __name__ == "__main__":
    main()
