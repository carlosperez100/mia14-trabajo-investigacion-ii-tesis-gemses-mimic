# Trabajo de Investigación II (MIA-14) — Tesis GEMSES × MIMIC-IV

**Tesis:** Detección automatizada y priorización de eventos adversos desde notas clínicas no estructuradas mediante un pipeline de NLP y aprendizaje automático — aplicación de la Matriz de Priorización del Modelo GEMSES sobre MIMIC-IV.

| Campo | Dato |
|---|---|
| Autor | Carlos Pérez Pérez · carlosperez100@gmail.com |
| Programa | Maestría en Inteligencia Artificial · FIIS · Universidad Nacional de Ingeniería |
| Curso | **MIA-14 Trabajo de Investigación II** · 2026-2 · Sección B (viernes 19:00-22:00) |
| Docente | **Dr. Ing. Glen Dario Rodríguez Rafael** |
| Curso previo | MIA-303 Trabajo de Investigación I · Mg. María Tejada · cerrado 14-jul-2026 |
| Repositorio de la tesis (código y resultados, privado) | https://github.com/carlosperez100/tesis-gemses-mimic-pipeline |
| Curso de PLN (MIA-10, público) | https://github.com/carlosperez100/PLN_SP |

## Qué es este repositorio

Es el repositorio **del curso**: entregables por semana, tablero de sprints y el artefacto reproducible que exige el docente (estructura `data/raw`, `data/processed`, `notebooks/`, `requirements.txt`, `.env.example`, README ejecutable). Es **una sola tesis, en continuación**: no reemplaza al repositorio de la tesis, lo ordena para este curso e integra las lecciones aprendidas en tres frentes (Tesis I, curso de PLN y GEM·VigIA).

## Estructura

```
.
├── entregables/
│   └── semana2/        Diseño reproducible del flujo experimental (MD, DOCX, PDF, diagrama)
├── data/
│   ├── raw/            (vacío en git) MIMIC-IV y ERSP bajo DUA; ver .env.example
│   └── processed/      (vacío en git) salidas por fase: JSON, parquet, pkl
├── notebooks/          exploración
├── src/                orquestador y scripts por fase (Sprint 1)
├── docs/               showcase público del proyecto (GitHub Pages)
├── requirements.txt    dependencias (versiones por fijar en Sprint 1)
└── .env.example        rutas externas; copiar a .env
```

## Estado de reproducibilidad (verificado 25-set-2026)

| Requisito del curso | Estado |
|---|---|
| `requirements.txt` con versiones fijadas | Parcial: solo DuckDB fijado |
| `.env.example` sin rutas absolutas en scripts | Plantilla creada; los scripts de la tesis aún tienen rutas escritas |
| Un comando que regenere el modelo final | **No** (46 scripts, uno por fase, sin orquestador) |
| `data/raw` y `data/processed` | Creados |
| README ejecutable | Este archivo; sección "Reproducir" se completa en el Sprint 1 |

**Sprint 1 — objetivo medible:** que un tercero con credencial PhysioNet clone este repo, ejecute `python src/run_pipeline.py --hasta fase10` y regenere la Fase 9 con sens 0.7623, esp 0.7699 y AUC 0.8426 (± 0.005).

## Entregables

| Semana | Entregable | Archivo |
|---|---|---|
| 2 | Diseño reproducible del flujo experimental (pipeline supervisado) | `entregables/semana2/Entregable_Semana2_Pipeline_Reproducible_CarlosPerez.pdf` |

## Ética y datos

MIMIC-IV v3.1 y MIMIC-IV-Note v2.2 bajo DUA PhysioNet (no se redistribuyen). CITI Program vigente. El corpus ERSP de EsSalud contiene datos personales y **nunca** entra a este repositorio. Todo archivo de datos, modelo o Excel está excluido por `.gitignore`.
