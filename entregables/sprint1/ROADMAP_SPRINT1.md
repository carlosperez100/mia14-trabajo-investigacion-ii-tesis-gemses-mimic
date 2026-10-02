# Roadmap del Sprint 1 (semanas 1–3) · Pipeline reproducible + EDA + baseline mínimo

**Objetivo medible:** que un tercero con credencial PhysioNet clone el repositorio, ejecute `python src/run_pipeline.py` y obtenga el mismo corpus, la misma partición por paciente y las cifras del modelo de la tesis (sensibilidad 0.7623, especificidad 0.7699, AUC 0.8426) con una tolerancia de ± 0.005.

**Track:** A, con un pipeline previo no reproducible. Venía de Proyecto I con un modelo entrenado, pero repartido en 46 scripts sin orquestador, con rutas absolutas y sin splits guardados.

| # | Tarea (issue) | Responsable | Criterio de aceptación | Estado |
|---|---|---|---|---|
| 1 | `requirements.txt` con versiones exactas | Carlos | Versiones fijadas del entorno que ejecutó el sprint | Hecho |
| 2 | `.env.example` + `config/config.yaml`; ninguna ruta absoluta en `src/` | Carlos | `grep -rn "T:/" src/` no devuelve nada | Hecho |
| 3 | `src/ingesta.py` idempotente, con SHA-256 y bitácora de datos | Carlos | Una segunda ejecución no copia ni vuelve a calcular hashes; `data/bitacora_datos.csv` completo | Hecho |
| 4 | `src/preprocesado.py` con conteos por paso y partición por paciente guardada | Carlos | Corpus idéntico nota por nota al de la Fase 9; 0 pacientes compartidos | Hecho |
| 5 | `src/baseline.py`: dummy + logística + referencia Fase 9 | Carlos | Métricas con IC 95 %; verificación ± 0.005 | Hecho: reproduce con diferencia 0.0000 |
| 6 | `src/run_pipeline.py` con `--desde/--hasta` | Carlos | Ejecuta las 3 etapas en orden | Hecho |
| 7 | Notebook EDA orientado a riesgos | Carlos | Conclusiones accionables, solo agregados (DUA) | Hecho |
| 8 | Demo interna de 5–8 min | Carlos | Guion y figura central | Guion listo; falta ensayar |
| 9 | Crear los issues y el tablero en GitHub Projects | Carlos | Tablero con las tareas 1–8 | Issues #1–#14 y milestones creados; falta el tablero de Projects |

## Fuera del Sprint 1 (pasa al Sprint 2)
- Integrar al orquestador las fases que **producen** las etiquetas: 3 v2 (patrones Tier A), 8 (equivalencias CIE-10-CM) y 7 (mapeo CIE-9). Hoy entran como tablas intermedias registradas con hash; todavía no se regeneran desde MIMIC con un comando.
- Validación cruzada por paciente (GroupKFold) con resultados por fold (semana 5).
- Etapa 2 (naturaleza del evento, multietiqueta) dentro del pipeline.
