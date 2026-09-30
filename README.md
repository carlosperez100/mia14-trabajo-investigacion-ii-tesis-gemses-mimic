# Trabajo de Investigación II (MIA-14) — Tesis GEMSES × MIMIC-IV

**Presentación del Sprint 1:** https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/

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

## Estado de reproducibilidad (verificado 30-set-2026)

| Requisito del curso | Estado |
|---|---|
| `requirements.txt` con versiones fijadas | Sí |
| `.env.example` sin rutas absolutas en scripts | Sí, en `src/` |
| Un comando que regenere el modelo final | **Sí**: `python src/run_pipeline.py` reproduce la Fase 9 con diferencia 0.0000 (las tablas de etiquetas entran como insumo registrado con hash) |
| `data/raw` y `data/processed` | Creados |
| README ejecutable | Sí: sección «Reproducir el Sprint 1» |

**Sprint 1 — objetivo medible:** que un tercero con credencial PhysioNet clone este repo, ejecute `python src/run_pipeline.py --hasta fase10` y regenere la Fase 9 con sens 0.7623, esp 0.7699 y AUC 0.8426 (± 0.005).

## Reproducir el Sprint 1 (ingesta → preprocesado → baseline)

Requisitos: credencial PhysioNet con acceso a MIMIC-IV v3.1 (`hosp/diagnoses_icd.csv.gz`) y MIMIC-IV-Note v2.2 (`note/discharge.csv.gz`), las tres tablas de etiquetas de la tesis (fases 3 v2, 8 y 7), Python 3.13 y unos 16 GB de RAM.

```bash
pip install -r requirements.txt
cp .env.example .env        # completar MIMIC_DIR, MIMIC_NOTE_DIR y ETIQUETAS_DIR
python src/run_pipeline.py  # ~15-25 min; --sin-referencia omite el modelo de la Fase 9
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_sprint1.ipynb
```

| Etapa | Script | Qué hace | Salidas |
|---|---|---|---|
| Ingesta | `src/ingesta.py` | Verifica fuentes, calcula SHA-256, copia etiquetas a `data/interim/`, extrae la época CIE. Idempotente (caché de hash por tamaño y fecha) | `data/bitacora_datos.csv`, `data/interim/` |
| Preprocesado | `src/preprocesado.py` | Positivos A1, negativos emparejados por época, texto de alta, limpieza, submuestreo, **partición por paciente guardada** | `data/processed/dataset.parquet`, `splits.csv`, `reportes/preprocesado_resumen.json` |
| Baseline | `src/baseline.py` | Dummy mayoritaria, dummy estratificada, logística TF-IDF y **referencia Fase 9**; verifica la reproducción de la Fase 9 (± 0.005) | `reportes/baseline_resultados.json`, `reportes/figuras/fig_baseline_roc_pr.png` |
| EDA | `notebooks/01_EDA_sprint1.ipynb` | Riesgos: balance, confusor de época, atajo por largo, fuga de códigos CIE, fuga por paciente, clases raras | figuras en `reportes/figuras/` |

Parámetros en `config/config.yaml` (semilla 42). Rutas en `.env`. Cada ejecución deja un log en `logs/`.

## Entregables

| Semana | Entregable | Archivo |
|---|---|---|
| 2 | Diseño reproducible del flujo experimental (pipeline supervisado) · nota 18/20 | `entregables/semana2/Entregable_Semana2_Pipeline_Reproducible_CarlosPerez.pdf` |
| 3 | **Sprint 1:** pipeline reproducible + EDA + baseline mínimo · [presentación](https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/) | `entregables/sprint1/`, `src/`, `notebooks/01_EDA_sprint1.ipynb`, `reportes/` |

## Ética y datos

MIMIC-IV v3.1 y MIMIC-IV-Note v2.2 bajo DUA PhysioNet (no se redistribuyen). CITI Program vigente. El corpus ERSP de EsSalud contiene datos personales y **nunca** entra a este repositorio. Todo archivo de datos, modelo o Excel está excluido por `.gitignore`.
