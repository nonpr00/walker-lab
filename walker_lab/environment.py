"""Bípedo y observaciones de Gymnasium; pista y evaluación del curso.

La construcción del robot y partes de step derivan de BipedalWalker,
Gymnasium 1.2.3 (licencia MIT; ver THIRD_PARTY_LICENSES.txt).
"""

import math
from os import PathLike

import gymnasium as gym
import numpy as np
from Box2D.b2 import edgeShape, fixtureDef, polygonShape
from gymnasium.envs.box2d.bipedal_walker import (
    BipedalWalker, FPS, SCALE, SPEED_HIP, SPEED_KNEE, MOTORS_TORQUE,
    LIDAR_RANGE, VIEWPORT_W, VIEWPORT_H,
)

from .scenarios import BASE_Y, generate_scenario, load_scenario, validate_scenario


class WalkerEnv(BipedalWalker):
    """Sin imágenes de entrada. Observation: float32(24); action: float32(4).

    profile controla la distribución de entrenamiento; scenario fija una pista.
    Las métricas de info son diagnósticas y no se entregan a la política final.
    """

    def __init__(self, profile="suave", scenario=None, length=None, max_steps=None, render_mode=None, difficulty="normal"):
        self.profile = profile
        self.difficulty = difficulty
        self.fixed_scenario = load_scenario(scenario) if isinstance(scenario, (str, bytes, PathLike)) else scenario
        if self.fixed_scenario is not None:
            validate_scenario(self.fixed_scenario)
        self.requested_length = length
        self.requested_max_steps = max_steps
        self.scene = None
        self._resetting = True
        self._finished = True
        self.steps = 0
        self.max_distance = 0.0
        self.effort = 0.0
        super().__init__(render_mode=render_mode, hardcore=False)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Configure el perfil al crear el entorno; reset solo acepta seed.")
        gym.Env.reset(self, seed=seed)
        if self.fixed_scenario is None:
            scene_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.scene = generate_scenario(self.profile, scene_seed, self.requested_length, self.requested_max_steps, self.difficulty)
            simulation_seed = None
        else:
            self.scene = self.fixed_scenario
            simulation_seed = self.scene["simulation_seed"] if seed is None else seed
        self.steps = 0
        self.max_distance = 0.0
        self.effort = 0.0
        self._resetting = True
        self._finished = False
        obs, _ = super().reset(seed=simulation_seed)
        self._resetting = False
        self.prev_shaping = 130 * float(self.hull.position.x) / SCALE - 5 * abs(float(self.hull.angle))
        return obs, self._info("running")

    def _generate_clouds(self):
        self.cloud_poly = []

    def _generate_terrain(self, hardcore):
        self.terrain, self.terrain_poly = [], []
        points = self.scene["ground"]
        self.terrain_x = [p[0] for p in points]
        self.terrain_y = [p[1] for p in points]
        for p, q in zip(points[:-1], points[1:]):
            fd = fixtureDef(shape=edgeShape(vertices=[tuple(p), tuple(q)]), friction=2.5, categoryBits=1)
            body = self.world.CreateStaticBody(fixtures=fd)
            body.color1 = (65, 137, 89)
            self.terrain.append(body)
        for block in self.scene["obstacles"]:
            x, y, w, h = (block[k] for k in ("x", "y", "width", "height"))
            vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            fd = fixtureDef(shape=polygonShape(vertices=vertices), friction=2.5, categoryBits=1)
            body = self.world.CreateStaticBody(fixtures=fd)
            body.color1 = (112, 116, 124)
            self.terrain.append(body)

    def _info(self, reason):
        return {
            "distance": float(self.max_distance), "length": float(self.scene["length"]),
            "steps": self.steps, "simulated_seconds": self.steps / FPS,
            "effort": float(self.effort), "success": reason == "success", "reason": reason,
        }

    def step(self, action):
        if self._finished:
            raise RuntimeError("El episodio terminó; llame a reset().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("La acción debe ser un vector finito de cuatro valores en [-1, 1].")
        action = np.clip(action, -1, 1)
        for i, joint in enumerate(self.joints):
            joint.motorSpeed = float((SPEED_HIP if i % 2 == 0 else SPEED_KNEE) * np.sign(action[i]))
            joint.maxMotorTorque = float(MOTORS_TORQUE * abs(action[i]))
        self.world.Step(1 / FPS, 6 * 30, 2 * 30)
        pos, vel = self.hull.position, self.hull.linearVelocity
        for i, ray in enumerate(self.lidar):
            ray.fraction = 1.0
            ray.p1 = pos
            ray.p2 = (pos.x + math.sin(1.5 * i / 10) * LIDAR_RANGE, pos.y - math.cos(1.5 * i / 10) * LIDAR_RANGE)
            self.world.RayCast(ray, ray.p1, ray.p2)
        state = [self.hull.angle, 2 * self.hull.angularVelocity / FPS,
                 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS, 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                 self.joints[0].angle, self.joints[0].speed / SPEED_HIP,
                 self.joints[1].angle + 1, self.joints[1].speed / SPEED_KNEE,
                 float(self.legs[1].ground_contact), self.joints[2].angle,
                 self.joints[2].speed / SPEED_HIP, self.joints[3].angle + 1,
                 self.joints[3].speed / SPEED_KNEE, float(self.legs[3].ground_contact)]
        state += [ray.fraction for ray in self.lidar]
        obs = np.asarray(state, dtype=np.float32)
        shaping = 130 * float(pos.x) / SCALE - 5 * abs(float(self.hull.angle))
        reward = 0.0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= 0.00035 * MOTORS_TORQUE * float(np.abs(action).sum())
        failed = self.game_over or pos.x < 0 or pos.y < -5 or not np.isfinite(obs).all()
        reason = "fall" if failed else "running"
        supported = bool(self.legs[1].ground_contact or self.legs[3].ground_contact)
        if not self._resetting:
            self.steps += 1
            self.effort += float(np.abs(action).sum())
            if not failed and supported:
                advance = float(np.clip(pos.x - self.scene["start_x"], 0, self.scene["length"]))
                self.max_distance = max(self.max_distance, advance)
            if not failed and supported and pos.x >= self.scene["finish_x"]:
                reason = "success"
                self.max_distance = self.scene["length"]
        if failed:
            reward = -100.0
        terminated = reason in ("fall", "success")
        truncated = not terminated and not self._resetting and self.steps >= self.scene["max_steps"]
        if truncated:
            reason = "time_limit"
        self._finished = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, self._info(reason)

    def render(self):
        if self.render_mode is None or self.hull is None:
            return None
        import pygame
        if not pygame.get_init():
            pygame.init()
        surface = pygame.Surface((960, 540))
        surface.fill((245, 248, 251))
        camera_x = float(self.hull.position.x) - 7
        zoom = 40

        def pixel(point):
            return (int((point[0] - camera_x) * zoom), int(400 - (point[1] - BASE_Y) * zoom))

        for body in self.terrain + self.legs + [self.hull]:
            for fixture in body.fixtures:
                vertices = [body.transform * vertex for vertex in fixture.shape.vertices]
                if max(v.x for v in vertices) < camera_x - 1 or min(v.x for v in vertices) > camera_x + 25:
                    continue
                points = [pixel(v) for v in vertices]
                color = getattr(body, "color1", (90, 90, 100))
                if len(points) == 2:
                    pygame.draw.line(surface, color, *points, width=4)
                else:
                    pygame.draw.polygon(surface, color, points)
        if self.scene:
            finish = pixel((self.scene["finish_x"], BASE_Y))
            if 0 <= finish[0] < 960:
                pygame.draw.line(surface, (190, 65, 65), (finish[0], 150), (finish[0], 410), 3)
            font = pygame.font.Font(None, 28)
            hud = f"Distancia: {self.max_distance:.2f} / {self.scene['length']:.0f}    Tiempo: {self.steps / FPS:.1f} s"
            surface.blit(font.render(hud, True, (35, 45, 55)), (20, 20))
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        if self.screen is None:
            self.screen = pygame.display.set_mode((960, 540))
            self.clock = pygame.time.Clock()
        pygame.event.pump()
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)


if "WalkerCompeticion-v0" not in gym.registry:
    gym.register("WalkerCompeticion-v0", entry_point="walker_lab.environment:WalkerEnv")
