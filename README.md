# Algoritmo genético para ruteo de reparto (TSP) con DEAP

**Fundamentos de Inteligencia Artificial · Actividad Parcial 2 · Sección 6 de 6 (2.6 Algoritmos genéticos)**
Universidad Politécnica de Atlautla · Ing. en Tecnologías de la Información e Innovación Digital · Grupo 171
Docente: Mtro. Javier García Lara

**Equipo**
- Sanchez Fragozo Dulce Maria – 24305056
- Lopez Perez Veronica Juliette – 24305097
- Jennifer Rivera Andrade – 24305072
- Marin Garcia Danna Isla – 24305080

---

## 1. ¿Qué hace este proyecto?

Una cooperativa de productores de la región Ozumba–Atlautla reparte pedidos semanales (amaranto, nopal, miel) desde su bodega en **Atlautla** a **11 cabeceras municipales** y regresa a la bodega. El orden de visita cambia los kilómetros recorridos, y con ellos el combustible y las horas de trabajo.

Es una instancia del **Problema del Agente Viajero (TSP)**: con 11 clientes hay 10! = 3,628,800 rutas posibles. El programa usa un **algoritmo genético (AG)** con la librería **DEAP** para encontrar una ruta corta y la compara contra:

- el **óptimo exacto** (programación dinámica de Held-Karp, posible porque son solo 12 puntos),
- la heurística del **vecino más cercano**,
- **rutas aleatorias**.

Además muestra una **simulación animada** de cómo evoluciona la mejor ruta generación por generación.

## 2. Diseño del algoritmo genético

| Componente | Definición |
|---|---|
| Cromosoma | Permutación de los 11 clientes (genes 0–10; el gen *g* es la localidad *g*+1). La bodega es fija. |
| Población | 100 individuos (cada uno es una ruta completa) |
| Fitness | Distancia total bodega → clientes → bodega, en km (**se minimiza**) |
| Selección | Torneo de tamaño 3 |
| Cruza | Ordenada OX (`cxOrdered`), probabilidad 0.8 |
| Mutación | Intercambio de genes (`mutShuffleIndexes`, `indpb = 0.15`), probabilidad 0.2 |
| Elitismo | Se conserva el mejor individuo de cada generación |
| Paro | 30 generaciones |

## 3. Requisitos

- Python **3.10 o superior**
- Librerías: `deap`, `matplotlib`, `numpy`, `pillow`
- Una pantalla para ver la ventana de la simulación (opcional: con `--sin-ventana` funciona sin pantalla)

## 4. Cómo replicarlo paso a paso

1. **Verifica Python.** En una terminal: `python --version` (debe ser 3.10 o mayor; en algunos sistemas el comando es `python3`).
2. **Crea una carpeta** para el proyecto y copia dentro el archivo `algoritmo_genetico.py` (y este `README.md`). Abre la terminal en esa carpeta.
3. **(Opcional) Crea un entorno virtual:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```
4. **Instala las dependencias:**
   ```bash
   pip install deap matplotlib numpy pillow
   ```
5. **Ejecuta el programa:**
   ```bash
   python algoritmo_genetico.py
   ```
   - Tarda cerca de **25 segundos** en calcular todo (óptimo exacto, 1 ejecución principal, 30 ejecuciones independientes y el experimento de parámetros).
   - Después se abre la **ventana de la simulación**. Un clic sobre la ventana la pausa y la reanuda. Al cerrarla, el programa termina.
6. **Lee la consola.** Imprime la distancia del óptimo exacto, del vecino más cercano, de la ruta aleatoria y del AG, y la ruta encontrada.
7. **Revisa la carpeta `resultados_s6/`** que se crea sola:

   | Archivo | Contenido |
   |---|---|
   | `fig1_convergencia.png` | Mejor, media y peor fitness por generación; y 30 ejecuciones independientes |
   | `fig2_mapa_rutas.png` | Ruta del AG frente a la ruta óptima exacta (con el orden de visita) |
   | `fig3_comparacion.png` | Distancia por método (aleatoria, vecino más cercano, AG, óptimo) |
   | `fig4_parametros.png` | Efecto del tamaño de población y del número de generaciones |
   | `fig5_fotogramas_simulacion.png` | Cuatro fotogramas de la simulación |
   | `simulacion_ag.gif` | La simulación animada guardada |
   | `metricas.json` | Todas las cifras de la ejecución |

## 5. Opciones de ejecución

```bash
python algoritmo_genetico.py --sin-ventana        # no abre ventana; solo guarda archivos (servidor, Colab)
python algoritmo_genetico.py --velocidad 2        # animación más rápida (0.5 = más lenta)
python algoritmo_genetico.py --semilla 7          # otra semilla para la ejecución principal
python algoritmo_genetico.py --poblacion 200      # otro tamaño de población
python algoritmo_genetico.py --generaciones 100   # más generaciones
python algoritmo_genetico.py --ejecuciones 10     # menos ejecuciones independientes (más rápido)
```

Las opciones se pueden combinar, por ejemplo: `python algoritmo_genetico.py --poblacion 200 --generaciones 30 --semilla 7 --sin-ventana`.

## 6. Resultados esperados

Con los parámetros por defecto (semilla 42, población 100, 30 generaciones) se obtuvo:

| Método | Distancia (km) |
|---|---|
| Óptimo exacto (Held-Karp) | 142.66 |
| AG, ejecución principal (semilla 42) | 152.39 (+6.8 %) |
| AG, media de 30 ejecuciones | 148.34 (+4.0 %) |
| Vecino más cercano | 156.48 |
| Ruta aleatoria (promedio) | 273.97 |

- El AG llegó al óptimo exacto en **6 de 30** ejecuciones: es una búsqueda aleatoria guiada, no garantiza el óptimo.
- Efecto de los parámetros (media de 30 ejecuciones por configuración): población 20 → 165.29 km; 50 → 151.64; 100 → 149.73; 200 → 146.37; población 100 con 100 generaciones → 147.34. Más población ayuda; más generaciones ayudan menos. El experimento usa otras semillas que la fila "media de 30 ejecuciones", por eso población 100 con 30 generaciones aparece con 149.73 y no con 148.34.
- Las semillas están fijas, así que al repetir la ejecución se esperan los mismos números. Con otra versión de Python o de las librerías pueden aparecer diferencias pequeñas.

## 7. Cómo adaptarlo a otro caso

- **Otras localidades:** edita el diccionario `LOCALIDADES` (nombre → latitud, longitud) y la constante `DEPOSITO` al inicio del archivo. El óptimo exacto con Held-Karp es práctico hasta unos 15–16 puntos; con más puntos el AG sigue funcionando, pero hay que desactivar la comparación exacta.
- **Distancias reales:** reemplaza el cálculo de la matriz `D` con distancias de una API de mapas u OpenStreetMap. El resto del código no cambia.
- **Otros operadores:** en la sección `DEAP` del código se registran la selección, la cruza y la mutación (`toolbox.register(...)`); se pueden cambiar por otros de `deap.tools`.

## 8. Datos y limitaciones

- Las **coordenadas son aproximadas** y la distancia por carretera se estima como *distancia haversine × 1.30* (factor de sinuosidad). **No son kilometrajes oficiales.**
- El modelo es de un solo vehículo, sin capacidad de carga, ventanas de tiempo ni tráfico.
- Con 30 generaciones el mejor individuo suele estancarse (convergencia prematura). Como mejora futura: más diversidad (más mutación o población) o combinar el AG con una búsqueda local 2-opt.

## 9. Subir el proyecto a GitHub

```bash
git init
git add algoritmo_genetico.py README.md resultados_s6/
git commit -m "Sección 6: algoritmo genético para ruteo de reparto"
git branch -M main
git remote add origin <URL-DE-TU-REPOSITORIO>
git push -u origin main
```

Estructura sugerida del repositorio:

```
.
├── algoritmo_genetico.py
├── README.md
└── resultados_s6/        (se genera al ejecutar)
```

## 10. Referencias (APA 7.ª ed.)

- Anthropic. (2026). *Claude Sonnet 5.5* [Modelo de lenguaje de gran tamaño]. https://claude.ai
- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research, 13*, 2171–2175.
- Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of the Society for Industrial and Applied Mathematics, 10*(1), 196–210.
- Russell, S., & Norvig, P. (2020). *Inteligencia artificial: Un enfoque moderno* (4.ª ed.). Pearson.
- Wirsansky, E. (2020). *Hands-on genetic algorithms with Python*. Packt Publishing.

## 11. Declaración de uso de IA

Herramienta: Claude (Anthropic, 2026). Uso: apoyo en la generación del borrador del código, de las gráficas, de la simulación y de esta documentación. Verificación: el código se ejecutó completo, las cifras de este README provienen de `metricas.json` generado por la ejecución y el resultado del AG se validó contra el óptimo exacto (Held-Karp).
