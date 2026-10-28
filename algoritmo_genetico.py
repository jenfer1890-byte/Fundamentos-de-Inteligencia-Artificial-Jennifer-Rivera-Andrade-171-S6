"""
Sección 6 - Algoritmos genéticos (AG) con DEAP
Asignatura: Fundamentos de Inteligencia Artificial - Universidad Politécnica de Atlautla (Grupo 171)

CASO (empresarial / social): una cooperativa de productores de la región Ozumba-Atlautla reparte
pedidos semanales (amaranto, nopal, miel) desde su bodega en Atlautla a 11 cabeceras municipales
y regresa a la bodega. Se busca el orden de visita que minimice los kilómetros recorridos
(menos combustible y menos horas de trabajo). Es una instancia del Problema del Agente Viajero (TSP).

REPRESENTACIÓN
  Cromosoma : permutación de los 11 clientes, genes 0..10 (gen g = localidad g+1; la bodega es fija,
              sale y regresa a ella).
  Población : 100 individuos (cada uno es una ruta completa).
  Fitness   : distancia total de la ruta en km  -> se MINIMIZA.
  Operadores: selección por torneo (k=3), cruza ordenada OX, mutación por intercambio de
              genes, elitismo de 1 individuo.
  Paro      : 30 generaciones.

NOTA SOBRE LOS DATOS: las coordenadas son aproximadas y la distancia por carretera se estima como
distancia haversine x 1.30 (factor de sinuosidad). No son kilometrajes oficiales; para datos reales
reemplazar la matriz D (por ejemplo con una API de mapas); el resto del código no cambia.

VERIFICACIÓN: como son solo 12 puntos, se calcula el óptimo EXACTO (programación dinámica de
Held-Karp) para medir qué tan cerca queda el AG.

USO
  pip install deap matplotlib numpy pillow
  python algoritmo_genetico.py                 # abre la simulación animada y guarda resultados
  python algoritmo_genetico.py --sin-ventana   # solo guarda archivos (útil sin pantalla)
  python algoritmo_genetico.py --velocidad 2   # animación más rápida (0.5 = más lenta)
  python algoritmo_genetico.py --semilla 7 --poblacion 200 --generaciones 30 --ejecuciones 30
La ventana de la simulación se pausa/reanuda con un clic.
Genera la carpeta resultados_s6/ con gráficas (.png), simulacion_ag.gif y metricas.json.
"""
import argparse
import json
import math
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from matplotlib.animation import FuncAnimation, PillowWriter

SALIDA = "resultados_s6"
AZUL, VERDE, NARANJA, ROJO, GRIS = "#1F4E9E", "#2E9E5B", "#E08A1E", "#D1342F", "#9AA3B5"

# ----------------------------------------------------------------------------- datos
DEPOSITO = "Atlautla"
LOCALIDADES = {  # nombre: (latitud, longitud) aproximadas
    "Atlautla": (19.0211, -98.7764),
    "Ozumba": (19.0369, -98.7997),
    "Tepetlixpa": (19.0042, -98.8167),
    "Amecameca": (19.1250, -98.7667),
    "Ecatzingo": (18.9500, -98.7500),
    "Totolapan": (18.9889, -98.9031),
    "Juchitepec": (19.1000, -98.8767),
    "Ayapango": (19.1167, -98.7928),
    "Tlalmanalco": (19.2025, -98.7967),
    "Chalco": (19.2634, -98.8975),
    "Atlatlahucan": (18.9333, -98.9167),
    "Yecapixtla": (18.8833, -98.8667),
}
FACTOR_SINUOSIDAD = 1.30
NOMBRES = [DEPOSITO] + [n for n in LOCALIDADES if n != DEPOSITO]   # índice 0 = bodega
N = len(NOMBRES)


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(d))


D = np.array([[haversine(LOCALIDADES[a], LOCALIDADES[b]) * FACTOR_SINUOSIDAD for b in NOMBRES] for a in NOMBRES])
XY = np.array([(LOCALIDADES[n][1], LOCALIDADES[n][0]) for n in NOMBRES])   # (lon, lat) para graficar


def longitud(ruta):
    """Kilómetros del recorrido bodega -> ruta -> bodega."""
    total = D[0, ruta[0]] + D[ruta[-1], 0]
    for a, b in zip(ruta, ruta[1:]):
        total += D[a, b]
    return float(total)


# ----------------------------------------------------------------------------- DEAP
def nodos(cromosoma):
    """El gen g representa a la localidad con índice g+1 (el índice 0 es la bodega)."""
    return [g + 1 for g in cromosoma]


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))           # minimizar km
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(N - 1), N - 1)      # cromosoma = permutación de 0..10
toolbox.register("individuo", tools.initIterate, creator.Individuo, toolbox.indices)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
toolbox.register("evaluate", lambda ind: (longitud(nodos(ind)),))
toolbox.register("mate", tools.cxOrdered)                              # cruza OX
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.15)       # intercambio de genes
toolbox.register("select", tools.selTournament, tournsize=3)          # torneo


def ejecutar_ag(semilla, tam_pob=100, ngen=30, cxpb=0.8, mutpb=0.2, registrar=False):
    """Corre el AG y devuelve la mejor ruta y el historial por generación."""
    random.seed(semilla)
    pob = toolbox.poblacion(n=tam_pob)
    for ind in pob:
        ind.fitness.values = toolbox.evaluate(ind)
    hof = tools.HallOfFame(1)
    hof.update(pob)
    evaluaciones = len(pob)
    t0 = time.perf_counter()
    h = dict(mejor=[], media=[], peor=[], ruta=[], muestra=[])

    def anotar():
        fits = [i.fitness.values[0] for i in pob]
        h["mejor"].append(min(fits)); h["media"].append(float(np.mean(fits))); h["peor"].append(max(fits))
        if registrar:
            h["ruta"].append(nodos(hof[0]))
            h["muestra"].append([nodos(i) for i in random.sample(pob, 5)])

    anotar()
    for _ in range(ngen):
        hijos = list(map(toolbox.clone, toolbox.select(pob, len(pob))))
        hijos = algorithms.varAnd(hijos, toolbox, cxpb, mutpb)           # cruza + mutación
        invalidos = [i for i in hijos if not i.fitness.valid]
        for ind in invalidos:
            ind.fitness.values = toolbox.evaluate(ind)
        evaluaciones += len(invalidos)
        peor = max(range(len(hijos)), key=lambda k: hijos[k].fitness.values[0])
        hijos[peor] = toolbox.clone(hof[0])                                # elitismo
        pob = hijos
        hof.update(pob)
        anotar()
    return dict(ruta=nodos(hof[0]), costo=hof[0].fitness.values[0], hist=h, evaluaciones=evaluaciones,
                tiempo_s=time.perf_counter() - t0)


# ----------------------------------------------------------------------------- referencias
def optimo_exacto():
    """Held-Karp: óptimo exacto del TSP (12 puntos) para validar al AG."""
    m = N - 1
    INF = float("inf")
    dp = [[INF] * m for _ in range(1 << m)]
    padre = [[-1] * m for _ in range(1 << m)]
    for j in range(m):
        dp[1 << j][j] = D[0, j + 1]
    for mask in range(1 << m):
        for j in range(m):
            if not mask & (1 << j) or dp[mask][j] == INF:
                continue
            for k in range(m):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                c = dp[mask][j] + D[j + 1, k + 1]
                if c < dp[nm][k]:
                    dp[nm][k] = c
                    padre[nm][k] = j
    full = (1 << m) - 1
    j = min(range(m), key=lambda j: dp[full][j] + D[j + 1, 0])
    costo = dp[full][j] + D[j + 1, 0]
    ruta, mask = [], full
    while j != -1:
        ruta.append(j + 1)
        mask, j = mask ^ (1 << j), padre[mask][j]
    return float(costo), ruta[::-1]


def vecino_mas_cercano():
    ruta, actual, libres = [], 0, set(range(1, N))
    while libres:
        sig = min(libres, key=lambda k: D[actual, k])
        ruta.append(sig); libres.remove(sig); actual = sig
    return longitud(ruta), ruta


def aleatoria(n=2000, semilla=1):
    rng = random.Random(semilla)
    costos = [longitud(rng.sample(range(1, N), N - 1)) for _ in range(n)]
    return float(np.mean(costos)), float(min(costos))


# ----------------------------------------------------------------------------- gráficas
def dibujar_ruta(ax, ruta, color, titulo, lw=2.6, numerar=True):
    orden = [0] + list(ruta) + [0]
    ax.plot(XY[orden, 0], XY[orden, 1], color=color, lw=lw, zorder=2, solid_joinstyle="round")
    ax.scatter(XY[1:, 0], XY[1:, 1], s=70, c="white", edgecolors="#444", zorder=3)
    ax.scatter(*XY[0], s=190, marker="s", c=VERDE, edgecolors="black", zorder=4)
    for i, nom in enumerate(NOMBRES):
        ax.text(XY[i, 0], XY[i, 1] + 0.011, nom, fontsize=7, ha="center", va="bottom", zorder=5)
    if numerar:
        for pos, k in enumerate(ruta, 1):
            ax.text(XY[k, 0], XY[k, 1], str(pos), fontsize=6.5, ha="center", va="center", color=color, zorder=6)
    ax.set_title(titulo, fontsize=10); ax.axis("off")


def graficas(principal, corridas, exp, opt, vmc, alea):
    h = principal["hist"]
    gens = np.arange(len(h["mejor"]))
    # 1) convergencia
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axs[0].plot(gens, h["peor"], color=GRIS, label="Peor")
    axs[0].plot(gens, h["media"], color=NARANJA, label="Media de la población")
    axs[0].plot(gens, h["mejor"], color=AZUL, lw=2.5, label="Mejor")
    axs[0].axhline(opt[0], color=VERDE, ls="--", label=f"Óptimo exacto ({opt[0]:.1f} km)")
    axs[0].set_title("Convergencia de una ejecución (semilla principal)")
    axs[1].axhline(opt[0], color=VERDE, ls="--", label="Óptimo exacto")
    M = np.array([c["hist"]["mejor"] for c in corridas])
    for fila in M:
        axs[1].plot(gens, fila, color=AZUL, alpha=0.15)
    axs[1].plot(gens, M.mean(axis=0), color=AZUL, lw=2.6, label=f"Media de {len(corridas)} ejecuciones")
    axs[1].set_title(f"Mejor fitness por generación en {len(corridas)} ejecuciones independientes")
    for a in axs:
        a.set_xlabel("Generación"); a.set_ylabel("Distancia total (km)"); a.legend(fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{SALIDA}/fig1_convergencia.png", dpi=150); plt.close(fig)

    # 2) mapas
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6))
    dibujar_ruta(axs[0], principal["ruta"], AZUL, f"Mejor ruta del AG: {principal['costo']:.1f} km")
    dibujar_ruta(axs[1], opt[1], VERDE, f"Óptimo exacto (Held-Karp): {opt[0]:.1f} km")
    fig.tight_layout(); fig.savefig(f"{SALIDA}/fig2_mapa_rutas.png", dpi=150); plt.close(fig)

    # 3) comparación de métodos
    finales = [c["costo"] for c in corridas]
    et = ["Ruta aleatoria\n(promedio)", "Vecino más\ncercano", f"AG (media de\n{len(corridas)} ejec.)", "AG (mejor)", "Óptimo\nexacto"]
    vals = [alea[0], vmc[0], float(np.mean(finales)), min(finales), opt[0]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    barras = ax.bar(et, vals, color=[GRIS, NARANJA, AZUL, AZUL, VERDE])
    ax.bar_label(barras, fmt="%.1f km", fontsize=9)
    ax.set_ylabel("Distancia total (km)"); ax.set_title("Comparación de métodos de ruteo")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{SALIDA}/fig3_comparacion.png", dpi=150); plt.close(fig)

    # 4) experimento de parámetros
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.boxplot([e["finales"] for e in exp], patch_artist=True,
               boxprops=dict(facecolor="#BFD0EE"), medianprops=dict(color=AZUL, lw=2))
    ax.set_xticks(range(1, len(exp) + 1))
    ax.set_xticklabels([e["nombre"] for e in exp])
    ax.axhline(opt[0], color=VERDE, ls="--", label="Óptimo exacto"); ax.legend()
    ax.set_ylabel("Distancia final (km)"); ax.set_title("Efecto del tamaño de población y de las generaciones")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{SALIDA}/fig4_parametros.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- simulación
def simular(principal, opt, velocidad=1.0, mostrar=True, archivo_gif=None):
    """Animación generación por generación: la mejor ruta (azul) y 5 individuos al azar (gris)
    a la izquierda; la curva de convergencia a la derecha."""
    h = principal["hist"]
    ng = len(h["mejor"])
    total = ng + 8
    fig, (axm, axc) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.suptitle("Simulación del algoritmo genético: ruta de reparto de la cooperativa", fontsize=13,
                 fontweight="bold", color=AZUL)
    muestras = [axm.plot([], [], color=GRIS, lw=1, alpha=0.5, zorder=1)[0] for _ in range(5)]
    mejor, = axm.plot([], [], color=AZUL, lw=3, zorder=2)
    axm.scatter(XY[1:, 0], XY[1:, 1], s=70, c="white", edgecolors="#444", zorder=3)
    axm.scatter(*XY[0], s=190, marker="s", c=VERDE, edgecolors="black", zorder=4)
    for i, nom in enumerate(NOMBRES):
        axm.text(XY[i, 0], XY[i, 1] + 0.011, nom, fontsize=7, ha="center", va="bottom", zorder=5)
    axm.axis("off")
    titulo = axm.set_title("")
    axc.set_xlim(0, ng - 1); axc.set_ylim(opt[0] * 0.9, max(h["peor"]) * 1.03)
    axc.axhline(opt[0], color=VERDE, ls="--", label=f"Óptimo exacto ({opt[0]:.1f} km)")
    lmej, = axc.plot([], [], color=AZUL, lw=2.5, label="Mejor")
    lmed, = axc.plot([], [], color=NARANJA, lw=1.6, label="Media")
    axc.set_xlabel("Generación"); axc.set_ylabel("Distancia total (km)"); axc.legend(fontsize=8)
    axc.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def trazo(ruta):
        o = [0] + list(ruta) + [0]
        return XY[o, 0], XY[o, 1]

    def cuadro(f):
        g = min(f, ng - 1)
        mejor.set_data(*trazo(h["ruta"][g]))
        for ln, r in zip(muestras, h["muestra"][g]):
            ln.set_data(*trazo(r))
        lmej.set_data(range(g + 1), h["mejor"][:g + 1])
        lmed.set_data(range(g + 1), h["media"][:g + 1])
        gap = 100 * (h["mejor"][g] / opt[0] - 1)
        titulo.set_text(f"Generación {g}/{ng - 1}  ·  mejor ruta = {h['mejor'][g]:.1f} km  "
                        f"(+{gap:.1f} % sobre el óptimo)")
        return muestras + [mejor, lmej, lmed, titulo]

    anim = FuncAnimation(fig, cuadro, frames=total, interval=350 / velocidad, blit=False, repeat=True)
    estado = {"pausa": False}

    def al_clic(_):
        anim.resume() if estado["pausa"] else anim.pause()
        estado["pausa"] = not estado["pausa"]

    fig.canvas.mpl_connect("button_press_event", al_clic)
    if archivo_gif:
        anim.save(archivo_gif, writer=PillowWriter(fps=max(1, round(3 * velocidad))), dpi=80)
    marcas = [0, (ng - 1) // 6, (ng - 1) // 2, total - 1]
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 7))
    for ax2, f in zip(axs2.ravel(), marcas):
        cuadro(f)
        fig.canvas.draw()
        foto = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        nb = np.where(foto.min(axis=2) < 250)
        ax2.imshow(foto[nb[0].min():nb[0].max() + 1, nb[1].min():nb[1].max() + 1]); ax2.axis("off")
    fig2.tight_layout(); fig2.savefig(f"{SALIDA}/fig5_fotogramas_simulacion.png", dpi=150); plt.close(fig2)
    if mostrar and plt.get_backend().lower() not in ("agg", "pdf", "svg", "cairo", "ps", "template"):
        plt.show()
    plt.close(fig)
    return anim


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description="Algoritmo genético (DEAP) para ruteo de reparto")
    ap.add_argument("--semilla", type=int, default=42, help="semilla de la ejecución principal")
    ap.add_argument("--poblacion", type=int, default=100)
    ap.add_argument("--generaciones", type=int, default=30)
    ap.add_argument("--ejecuciones", type=int, default=30, help="ejecuciones independientes para estadística")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir la ventana; solo guardar archivos")
    ap.add_argument("--velocidad", type=float, default=1.0, help="velocidad de la animación (1 = normal)")
    a = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)

    opt = optimo_exacto()
    vmc = vecino_mas_cercano()
    alea = aleatoria()
    principal = ejecutar_ag(a.semilla, a.poblacion, a.generaciones, registrar=True)
    corridas = [ejecutar_ag(1000 + s, a.poblacion, a.generaciones) for s in range(a.ejecuciones)]
    finales = np.array([c["costo"] for c in corridas])
    aciertos = int(np.sum(finales <= opt[0] + 1e-6))

    configs = [("pob 20\n30 gen", 20, 30), ("pob 50\n30 gen", 50, 30), ("pob 100\n30 gen", 100, 30),
               ("pob 200\n30 gen", 200, 30), ("pob 100\n100 gen", 100, 100)]
    exp = []
    for nombre, p, g in configs:
        fin = [ejecutar_ag(2000 + s, p, g)["costo"] for s in range(a.ejecuciones)]
        exp.append(dict(nombre=nombre, poblacion=p, generaciones=g, finales=fin, media=float(np.mean(fin)),
                        mejor=float(np.min(fin)), aciertos=int(np.sum(np.array(fin) <= opt[0] + 1e-6))))

    graficas(principal, corridas, exp, opt, vmc, alea)
    simular(principal, opt, a.velocidad, not a.sin_ventana, f"{SALIDA}/simulacion_ag.gif")

    resumen = dict(
        parametros=dict(poblacion=a.poblacion, generaciones=a.generaciones, cruza="OX", prob_cruza=0.8,
                        mutacion="intercambio (indpb=0.15)", prob_mutacion=0.2, seleccion="torneo k=3",
                        elitismo=1, factor_sinuosidad=FACTOR_SINUOSIDAD, localidades=N),
        optimo_exacto=dict(km=opt[0], ruta=[NOMBRES[0]] + [NOMBRES[i] for i in opt[1]] + [NOMBRES[0]]),
        vecino_mas_cercano=dict(km=vmc[0], ruta=[NOMBRES[0]] + [NOMBRES[i] for i in vmc[1]] + [NOMBRES[0]]),
        aleatoria=dict(media_km=alea[0], mejor_km=alea[1]),
        ag_principal=dict(semilla=a.semilla, km=principal["costo"], evaluaciones=principal["evaluaciones"],
                          tiempo_s=principal["tiempo_s"],
                          ruta=[NOMBRES[0]] + [NOMBRES[i] for i in principal["ruta"]] + [NOMBRES[0]],
                          mejor_por_generacion=principal["hist"]["mejor"]),
        ag_ejecuciones=dict(n=a.ejecuciones, media_km=float(finales.mean()), desv_km=float(finales.std()),
                            mejor_km=float(finales.min()), peor_km=float(finales.max()),
                            gap_medio_pct=float(100 * (finales.mean() / opt[0] - 1)),
                            alcanzaron_optimo=aciertos),
        experimento_parametros=[{k: v for k, v in e.items() if k != "finales"} | dict(nombre=e["nombre"].replace("\n", " "))
                                for e in exp])
    with open(f"{SALIDA}/metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Óptimo exacto (Held-Karp) : {opt[0]:7.2f} km")
    print(f"Vecino más cercano        : {vmc[0]:7.2f} km")
    print(f"Ruta aleatoria (promedio) : {alea[0]:7.2f} km")
    print(f"AG ejecución principal    : {principal['costo']:7.2f} km  ({principal['evaluaciones']} evaluaciones)")
    print(f"AG {a.ejecuciones} ejecuciones        : media {finales.mean():.2f} km · mejor {finales.min():.2f} · "
          f"peor {finales.max():.2f} · óptimo alcanzado {aciertos}/{a.ejecuciones}")
    for e in exp:
        print(f"  {e['nombre'].replace(chr(10), ' '):<16} media {e['media']:.2f} km · óptimo {e['aciertos']}/{a.ejecuciones}")
    print("Ruta AG:", " > ".join([NOMBRES[0]] + [NOMBRES[i] for i in principal["ruta"]] + [NOMBRES[0]]))


if __name__ == "__main__":
    main()
