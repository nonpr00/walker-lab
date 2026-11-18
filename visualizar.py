"""Visor geométrico: permite recorrer una pista sin disponer de una política."""

import argparse
import pygame
from walker_lab.scenarios import BASE_Y, load_scenario

def main():
    parser = argparse.ArgumentParser(description="Ver una pista; flechas para desplazarse, Home/End para extremos.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--save", help="Guardar un panorama PNG y salir; no abre ventana.")
    args = parser.parse_args()
    scene = load_scenario(args.scenario)
    pygame.init()
    size = (2400, 460) if args.save else (1100, 600)
    screen = pygame.Surface(size) if args.save else pygame.display.set_mode(size)
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    camera = 0.0
    zoom = (size[0] - 60) / (scene["finish_x"] + 6) if args.save else 45
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        camera += (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])) * 0.4
        if keys[pygame.K_HOME]:
            camera = 0
        if keys[pygame.K_END]:
            camera = max(0, scene["finish_x"] - 20)
        camera = max(0, min(camera, scene["finish_x"]))
        screen.fill((255, 255, 255))

        def point(x, y):
            return (int(30 + (x - camera) * zoom), int(size[1] * 0.60 - (y - BASE_Y) * 50))

        baseline = int(size[1] * 0.60)
        if scene["profile"] == "maraton":
            for j in range(5):
                x0 = 10 + (scene["finish_x"] - 10) * j / 5
                x1 = 10 + (scene["finish_x"] - 10) * (j + 1) / 5
                p0, p1 = point(x0, BASE_Y), point(x1, BASE_Y)
                color = [(239, 247, 241), (239, 245, 251), (249, 246, 235), (251, 240, 232), (249, 235, 237)][j]
                pygame.draw.rect(screen, color, (p0[0], 60, p1[0] - p0[0], size[1] - 120))
                screen.blit(font.render(f"Zona {j + 1}", True, (50, 55, 65)), (p0[0] + 8, 72))
        pygame.draw.lines(screen, (54, 121, 76), False, [point(x, y) for x, y in scene["ground"]], 3)
        for block in scene["obstacles"]:
            x, y, w, h = (block[k] for k in ("x", "y", "width", "height"))
            p = point(x, y + h)
            pygame.draw.rect(screen, (112, 116, 124), (p[0], p[1], max(2, w * zoom), h * 50))
        finish = point(scene["finish_x"], BASE_Y)
        pygame.draw.line(screen, (190, 60, 60), (finish[0], 100), (finish[0], baseline + 30), 3)
        for x in range(0, int(scene["finish_x"]) + 1, 20):
            p = point(x, BASE_Y)
            screen.blit(font.render(str(x), True, (65, 65, 75)), (p[0], size[1] - 50))
        text = f"{scene['profile']} | Longitud: {scene['length']:.0f} unidades | {len(scene['events'])} eventos"
        screen.blit(font.render(text, True, (35, 40, 50)), (20, 20))
        if args.save:
            pygame.image.save(screen, args.save)
            print(f"Panorama guardado: {args.save}. Escala horizontal y vertical diferentes.")
            break
        pygame.display.flip()
        clock.tick(50)
    pygame.quit()


if __name__ == "__main__":
    main()
