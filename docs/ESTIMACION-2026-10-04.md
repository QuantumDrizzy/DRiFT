# Estimación 2026-10-04

Calibrada con los ejercicios 11–14, que llevaron unas 7–8 horas de reloj, buena parte esperando runs.

- ADR-006 + predicciones: 30–45 min. Riesgo bajo.
- Fase 1: red de tensores aleatoria, 3 grafos, corte mínimo, mapeo a Ising en DRiFT (R1–R3). 3–5 h. Riesgo medio: el mapeo exacto a Ising para Rényi-2 promediado tiene trampas (promedio del cociente frente a cociente de promedios), y la contracción del teselado hiperbólico puede ponerse cara por la treewidth.
- Runs: minutos en la GPU de Antonio; 1–2 h en CPU. Riesgo bajo.
- QuBLAR (R4): +2–4 h. Riesgo medio-alto: hay que leer la API y adaptar el QUBO.
- LYTH: sin estimar. No visto en esa pasada.
- Docs + PR + merge: ~1 h. Riesgo bajo.

Total realista: la fase 1 cabe en un día de trabajo (el lunes). Con QuBLAR, un segundo día. La fase 2 (reconstruir el grafo del bulk solo a partir de las entropías del borde) es investigación abierta: días o semanas, y puede no salir. Si sale, es lo más interesante.

Hoy hubo dos sorpresas que costaron tiempo: el error de diseño en graphity y el MPS sin converger. En algo nuevo cuenta con al menos una así.

Los qubits los sigue Grok, en LYTH, no en este bloque.
