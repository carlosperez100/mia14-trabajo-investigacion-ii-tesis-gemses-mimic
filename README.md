<p align="left"><img src="docs/sprint1/logo_uni.png" alt="Universidad Nacional de Ingeniería" height="70"></p>

# Trabajo de Investigación II (MIA-14) — Tesis GEMSES × MIMIC-IV

**Universidad Nacional de Ingeniería · Facultad de Ingeniería Industrial y de Sistemas · Unidad de Posgrado**
**Maestría en Inteligencia Artificial**

| Campo | Dato |
|---|---|
| Alumno | **Carlos Pérez Pérez** · carlosperez100@gmail.com |
| Docente | **Dr. Ing. Glen Dario Rodríguez Rafael** |
| Curso | **MIA-14 Trabajo de Investigación II** (sílabo: Proyecto de Investigación II, MIA-403) · ciclo 2026-2 · sección B |
| Tesis | Detección automatizada y priorización de eventos adversos desde notas clínicas no estructuradas mediante un pipeline de NLP y aprendizaje automático: aplicación de la Matriz de Priorización del Modelo GEMSES sobre MIMIC-IV |
| Presentación del Sprint 1 | https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/ |
| Tablero de tareas | [Issues](https://github.com/carlosperez100/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/issues) · [Milestones](https://github.com/carlosperez100/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/milestones) |

## Qué es este repositorio

Es el repositorio del curso: entregables por semana, sprints y el artefacto reproducible que pide el sílabo. Es una sola tesis, en continuación de Trabajo de Investigación I (MIA-303). Este repositorio ordena el código para que cualquiera pueda volver a ejecutarlo.

**Tarea del modelo:** a partir de la nota de alta (epicrisis) de MIMIC-IV, decidir si corresponde a una hospitalización con un evento adverso. La etiqueta sale de los códigos CIE de semántica causal (estrato A1). La partición es por paciente.

## Estructura de carpetas

```
.
├── README.md                 este archivo
├── requirements.txt          dependencias con versiones exactas
├── .env.example              rutas externas (copiar a .env); ningún script tiene rutas escritas
├── .gitignore                excluye datos, modelos y .env (DUA PhysioNet)
├── config/
│   └── config.yaml           parámetros del pipeline y semilla (42)
├── data/
│   ├── raw/                  vacío en git: MIMIC-IV se lee desde la ruta del .env (DUA, no se redistribuye)
│   ├── interim/              vacío en git: etiquetas copiadas y época CIE (salida de la ingesta)
│   ├── processed/            vacío en git: dataset.parquet, splits.csv y modelos .pkl
│   └── bitacora_datos.csv    bitácora de datos: fuente, fecha, SHA-256 y tamaño
├── src/
│   ├── run_pipeline.py       orquestador: ingesta → preprocesado → baseline
│   ├── ingesta.py            etapa 1 (idempotente, con hash)
│   ├── preprocesado.py       etapa 2 (limpieza, emparejamiento, partición por paciente)
│   ├── baseline.py           etapa 3 (dummy, logística, referencia de la tesis)
│   └── utils.py              configuración, logging y hash
├── notebooks/
│   └── 01_EDA_sprint1.ipynb  EDA orientado a riesgos (solo agregados)
├── reportes/                 resultados versionados (JSON) y figuras
├── logs/                     un log por etapa y ejecución
├── entregables/
│   ├── semana2/              diseño reproducible del flujo experimental
│   └── sprint1/              roadmap y guion de la demo
└── docs/                     GitHub Pages: presentación del Sprint 1 y showcase de la tesis
```

## Cómo reproducir

Requisitos:
- Credencial PhysioNet con acceso a MIMIC-IV v3.1 (`hosp/diagnoses_icd.csv.gz`) y a MIMIC-IV-Note v2.2 (`note/discharge.csv.gz`).
- Las tres tablas de etiquetas de la tesis (fases 3 v2, 8 y 7).
- Python 3.13 y unos 16 GB de RAM.

```bash
pip install -r requirements.txt
cp .env.example .env          # completar MIMIC_DIR, MIMIC_NOTE_DIR y ETIQUETAS_DIR
python src/run_pipeline.py    # ~45 min en CPU; --sin-referencia omite el modelo de la tesis (~33 min menos)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_sprint1.ipynb
```

| Etapa | Script | Qué hace | Salidas |
|---|---|---|---|
| Ingesta | `src/ingesta.py` | Verifica las fuentes, calcula SHA-256, copia las etiquetas a `data/interim/` y extrae la época CIE. Es idempotente | `data/bitacora_datos.csv`, `data/interim/` |
| Preprocesado | `src/preprocesado.py` | Positivos A1, negativos emparejados por época, texto de alta, limpieza, submuestreo y **partición por paciente guardada** | `data/processed/`, `reportes/preprocesado_resumen.json` |
| Baseline | `src/baseline.py` | Dummy mayoritaria, dummy estratificada, logística TF-IDF y referencia de la tesis; verifica la reproducción (± 0.005) | `reportes/baseline_resultados.json`, `reportes/figuras/` |
| EDA | `notebooks/01_EDA_sprint1.ipynb` | Balance, confusor de época, atajo por largo, fuga de códigos CIE, fuga por paciente y clases raras | `reportes/figuras/` |

## Resultados del Sprint 1 (ejecución del 30-09-2026)

Corpus: 70 000 epicrisis (55 915 de entrenamiento y 14 085 de test), con 0 pacientes compartidos. La prevalencia real del evento en MIMIC-IV es 20.12 %.

| Modelo | AUC (IC 95 %) | VPP a prevalencia real |
|---|---|---|
| Dummy mayoritaria | 0.500 | 0.201 |
| Dummy estratificada | 0.510 (0.501–0.518) | 0.207 |
| Logística TF-IDF | **0.864 (0.858–0.869)** | **0.485** |
| Referencia: modelo de la tesis (LinearSVC) | 0.843 (0.836–0.848) | 0.455 |

**Criterio de aceptación:** el pipeline reproduce el modelo de la tesis (sensibilidad 0.7623, especificidad 0.7699, AUC 0.8426) con diferencia 0.0000. Fuente: `reportes/baseline_resultados.json`.

**Límite declarado:** las tablas de etiquetas entran como insumo registrado con hash; regenerarlas desde MIMIC dentro del orquestador es el primer issue del Sprint 2.

## Entregables

| Semana | Entregable | Dónde |
|---|---|---|
| 2 | Diseño reproducible del flujo experimental (pipeline supervisado) | `entregables/semana2/` |
| 3 | Repositorio con README y carpetas · Sprint 1: pipeline reproducible, EDA y baseline mínimo | este repositorio · [presentación](https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/) |

## Ética y datos

MIMIC-IV v3.1 y MIMIC-IV-Note v2.2 están bajo el DUA de PhysioNet: no se redistribuyen y no se publica texto clínico. La certificación CITI está vigente. El corpus ERSP de EsSalud contiene datos personales y nunca entra a este repositorio. Todo archivo de datos, modelo o Excel está excluido por `.gitignore`.
