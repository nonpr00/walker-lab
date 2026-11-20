"""Cargar agente/agent.py y evaluar su modelo entrenado."""

import argparse
from pathlib import Path

from agente.agent import Agent
from walker_lab.evaluation import Evaluator

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Probar el modelo de agente/ una vez sobre un recorrido.")
    parser.add_argument("--scenario", "--escenarios", dest="scenario", type=Path,
                        default=ROOT / "escenarios" / "hardcore" / "maraton_04.json", help="Archivo JSON del recorrido.")
                        #default=ROOT / "escenarios" / "validacion" / "maraton_01.json", help="Archivo JSON del recorrido.")
    parser.add_argument("--seed", type=int, help="Semilla de la condición inicial; por defecto, la del JSON.")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados" / "evaluacion.json", help="Archivo JSON donde guardar el informe.")
    parser.add_argument("--render", action="store_true", help="Visualizar al robot durante la evaluación.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda o cuda:N; el adaptador carga su modelo en ese dispositivo.")
    parser.add_argument("--record", action="store_true", help="Guardar trayectorias .npz para reproducirlas después, sin agente.")
    args = parser.parse_args()
    record_dir = args.salida.parent / f"{args.salida.stem}_trayectorias" if args.record else None
    evaluator = Evaluator(args.scenario, render=args.render, device=args.device, record_dir=record_dir, seed=args.seed)
    agent = Agent(model_dir=ROOT / "agente", device=evaluator.device)
    report = evaluator.evaluate(agent)
    evaluator.save_report(report, args.salida)


if __name__ == "__main__":
    main()
