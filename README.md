# Laboratorio 4 

## Instalación

```bash
conda create -n lab4 python=3.12 -y
conda activate lab4
python -m pip install -r requirements.txt
python comprobar_instalacion.py
```

## Contenido

| Elemento | Función |
|---|---|
| agente/ | Código, pesos y normalización de su modelo |
| agente/agent.py | Clase Agent que deben completar |
| walker_lab/agent.py | Contrato BaseAgent |
| walker_lab/environment.py | Clase WalkerEnv, basada en BipedalWalker |
| walker_lab/scenarios.py | Generación y lectura de pistas |
| walker_lab/evaluation.py | Prueba de su agente sobre un recorrido |
| escenarios/practica/ | Siete pistas cortas de habilidades específicas |
| escenarios/validacion/ | Recorridos largos de ejemplo para elegir |
| escenarios/hardcore/ | Recorridos de ejemplo de dificultad hardcore-plus |
| evaluar.py | Ejecutar su modelo una vez sobre una pista |
| visualizar.py | Recorrer la pista sin agente |
| reproducir.py | Reproducir una evaluación guardada sin agente |
| generar_escenario.py | Guardar una pista nueva |
| comprobar_instalacion.py | Verificar dependencias y API |

## Entrenamiento

```python
from walker_lab.environment import WalkerEnv

env = WalkerEnv(profile="suave")
observation, info = env.reset(seed=123)
# action = política(observation)
# observation, reward, terminated, truncated, info = env.step(action)
```

- `action`: vector de cuatro valores en este orden:
    ```python
    [cadera_1, rodilla_1, cadera_2, rodilla_2]
    ```
    Cada valor está entre −1 y 1. El signo indica el sentido de giro y su magnitud determina el esfuerzo máximo del motor.

- `observation`: vector `float32` de 24 valores, con este orden:

    | Índice | Información |
    |---:|---|
    | 0 | Ángulo del torso |
    | 1 | Velocidad angular del torso |
    | 2 | Velocidad horizontal |
    | 3 | Velocidad vertical |
    | 4 | Ángulo de la cadera 1 |
    | 5 | Velocidad de la cadera 1 |
    | 6 | Ángulo de la rodilla 1, desplazado en +1 |
    | 7 | Velocidad de la rodilla 1 |
    | 8 | Contacto de la pierna 1 con el suelo: 0 o 1 |
    | 9 | Ángulo de la cadera 2 |
    | 10 | Velocidad de la cadera 2 |
    | 11 | Ángulo de la rodilla 2, desplazado en +1 |
    | 12 | Velocidad de la rodilla 2 |
    | 13 | Contacto de la pierna 2 con el suelo: 0 o 1 |
    | 14–23 | Diez lecturas lidar del terreno, en orden |

    Las velocidades y las lecturas lidar utilizan las escalas del entorno; no son todas mediciones físicas directas.

## Entrega

Completen `Agent` en `agente/agent.py`: carguen el modelo en `__init__(model_dir, device)`, reinicien la memoria en `reset()` y devuelvan las acciones en `act(observation)`.
Entreguen `agente/` con su arquitectura, pesos y normalización.

Respeten `self.device` para el modelo y sus entradas. `act()` devuelve las acciones como un array NumPy en CPU, incluso al usar CUDA.

## Evaluar su modelo

Probar un intento sobre el recorrido largo de ejemplo `escenarios/validacion/maraton_01.json`:
```bash
python evaluar.py
```
Para inferencia en GPU, añadan `--device cuda`; por defecto se usa CPU. Su framework debe tener soporte CUDA instalado.

Visualizar al robot durante la evaluación:
```bash
python evaluar.py --render
```

Evaluar una pista específica:
```bash
python evaluar.py --scenario escenarios/practica/plano.json --render
```

El evaluador carga `agente/` y muestra distancia, tiempo y causa de finalización de ese intento. Guarda `resultados/evaluacion.json`. Para conservar otra ejecución:
```bash
python evaluar.py --salida resultados/prueba_02.json
```

Evaluar sin ventana y reproducir después, sin volver a ejecutar el agente:
```bash
python evaluar.py --device cuda --record
python reproducir.py --trajectory resultados/evaluacion_trayectorias/maraton_01.npz
```

Un salto se confirma como avance al aterrizar con apoyo y sin caída del torso. Retroceder no acumula distancia adicional. La suma de recompensas se registra por separado de la distancia.

Pueden usar `--seed 123` para elegir la condición inicial sin cambiar el terreno. Las repeticiones, los promedios y el ranking pertenecen a la evaluación docente.

## Ver o generar pistas

```bash
python visualizar.py --scenario escenarios/validacion/maraton_01.json
python visualizar.py --scenario escenarios/validacion/maraton_01.json --save pista.png
python generar_escenario.py --profile maraton --seed 9017 --output escenarios/otra_pista.json
```

En el visor, use las flechas izquierda/derecha para desplazarse y Home/End para ir a los extremos. 
