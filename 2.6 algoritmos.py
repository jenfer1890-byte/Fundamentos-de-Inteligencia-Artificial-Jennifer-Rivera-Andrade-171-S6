import math
import random
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------
# 1. Configuración de datos del problema (15 Nodos Logísticos)
# ---------------------------------------------------------
NUM_CITIES = 15
random.seed(42)  # Semilla fija para reproducibilidad

# Coordenadas (X, Y) en km dentro de la zona de distribución
CITIES = {
    0: (10, 20), 1: (30, 40), 2: (20, 10), 3: (40, 30), 4: (50, 50),
    5: (80, 20), 6: (60, 10), 7: (70, 30), 8: (60, 60), 9: (50, 80),
    10: (20, 70), 11: (10, 90), 12: (40, 90), 13: (80, 80), 14: (90, 50)
}

def calculate_distance(city1, city2):
    x1, y1 = CITIES[city1]
    x2, y2 = CITIES[city2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ---------------------------------------------------------
# 2. Creación Segura de Clases en DEAP (Solución al Error)
# ---------------------------------------------------------
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NUM_CITIES), NUM_CITIES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_tsp(individual):
    distance = 0.0
    for i in range(len(individual)):
        city_a = individual[i]
        city_b = individual[(i + 1) % len(individual)]
        distance += calculate_distance(city_a, city_b)
    return (distance,)

toolbox.register("evaluate", eval_tsp)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------
# 3. Ejecución de la Simulación Evolutiva
# ---------------------------------------------------------
def run_simulation():
    pop_size = 100
    cx_prob = 0.8
    mut_prob = 0.2
    generations = 30

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(1)

    print("=== INICIANDO SIMULACIÓN DEL ALGORITMO GENÉTICO ===")
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob,
        ngen=generations, stats=stats, halloffame=hof, verbose=True
    )

    best_route = hof[0]
    best_distance = best_route.fitness.values[0]

    print("\n" + "="*50)
    print(f"SIMULACIÓN FINALIZADA")
    print(f"MEJOR RUTA ENCONTRADA: {best_route}")
    print(f"DISTANCIA OPTIMIZADA: {best_distance:.2f} km")
    print("="*50)

    # Visualización de la simulación
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Convergencia
    ax1.plot(gen, fit_mins, 'g-', label="Distancia Mínima")
    ax1.plot(gen, fit_avgs, 'b--', label="Distancia Promedio")
    ax1.set_title("Curva de Convergencia de la Simulación")
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Distancia (km)")
    ax1.grid(True)
    ax1.legend()

    # Gráfico 2: Ruta Simulada
    x_coords = [CITIES[i][0] for i in best_route] + [CITIES[best_route[0]][0]]
    y_coords = [CITIES[i][1] for i in best_route] + [CITIES[best_route[0]][1]]
    ax2.plot(x_coords, y_coords, 'ro-', linewidth=2, markersize=7)
    for city in best_route:
        ax2.text(CITIES[city][0]+1, CITIES[city][1]+1, f"N{city}", fontweight='bold')
    ax2.set_title(f"Ruta Final Simulada (Distancia: {best_distance:.2f} km)")
    ax2.set_xlabel("Coordenada X (km)")
    ax2.set_ylabel("Coordenada Y (km)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("simulacion_ag_tsp.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()