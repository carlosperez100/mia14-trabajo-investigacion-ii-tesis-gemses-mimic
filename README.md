<p align="left"><img src="docs/sprint1/logo_uni.png" alt="Universidad Nacional de Ingeniería" height="70"></p>

# 🏥 Detección de eventos adversos en epicrisis (MIMIC-IV) para su priorización con el Modelo GEMSES

Proyecto de investigación (Maestría en IA – UNI): un pipeline de PLN y aprendizaje automático que detecta, en la nota de alta hospitalaria, si la hospitalización tuvo un evento adverso. Es la base para priorizar esos eventos con la Matriz de Priorización del Modelo GEMSES.

Este repositorio forma parte del curso **Proyecto de Investigación II (MIA 403)**, dictado como **MIA-14 Trabajo de Investigación II**. Universidad Nacional de Ingeniería · Facultad de Ingeniería Industrial y de Sistemas · Unidad de Posgrado · Maestría en Inteligencia Artificial · ciclo 2026-2, sección B.

**Presentación del Sprint 1:** https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/

---

## 👥 Autores
- Carlos Pérez Pérez – [@carlosperez100](https://github.com/carlosperez100) · carlosperez100@gmail.com
- Docente: Dr. Ing. Glen Dario Rodríguez Rafael – [@GlenRodriguez](https://github.com/GlenRodriguez)

**Track:** A (venía de Proyecto I con un modelo entrenado, pero con un pipeline no reproducible; en el Sprint 1 se migró y se ordenó).

---

## 📊 Dataset
| | MIMIC-IV-Note v2.2 (texto) | MIMIC-IV v3.1 (diagnósticos) |
|---|---|---|
| **Fuente** | [PhysioNet – MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/) (acceso con credencial y DUA) | [PhysioNet – MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (acceso con credencial y DUA) |
| **Archivo** | `note/discharge.csv.gz` (1 139.2 MB) | `hosp/diagnoses_icd.csv.gz` (33.6 MB) |
| **Registros** | 331 793 epicrisis de 145 914 pacientes | 545 497 hospitalizaciones con diagnóstico |
| **Variables usadas** | `note_id`, `subject_id`, `hadm_id`, `text` | `hadm_id`, `icd_code`, `icd_version` |
| **Versión usada** | archivo del 11/05/2026 | archivo del 14/05/2026 |
| **Hash (SHA-256)** | `27bc552edaf81c2af322040cb7e8eb36417173b23c79405bb92fcf73bef2dbc5` | `47665a41b2a3ad990d6f314062d5bd6d29d467b0fd402dd94662044ec8b073d2` |

- **Variable objetivo `y` (0/1):** 1 si la hospitalización tiene un código CIE de semántica causal explícita de evento adverso (estrato A1, con equivalencias CIE-10-CM y familias CIE-9). Las tablas de etiquetas se registran con su hash en [`data/bitacora_datos.csv`](data/bitacora_datos.csv).
- **Corpus de trabajo:** 70 000 epicrisis (37 692 positivas) emparejadas por época de codificación. Partición por paciente: 55 915 de entrenamiento y 14 085 de test, con 0 pacientes compartidos.
- **Prevalencia real del evento en MIMIC-IV:** 20.12 % (109 775 de 545 497 hospitalizaciones).
- **Ética:** MIMIC está bajo el DUA de PhysioNet; no se redistribuye y no se publica texto clínico. Todo archivo de datos está excluido por `.gitignore`.

---

## 🗂️ Estructura del repositorio
```
data/
 ├── raw/                     # vacío en git: MIMIC se lee desde la ruta del .env (DUA)
 ├── interim/                 # etiquetas copiadas y época CIE (salida de la ingesta)
 ├── processed/               # dataset.parquet, splits.csv y modelos (fuera de git)
 └── bitacora_datos.csv       # versionado de datos: fuente, fecha, SHA-256 y tamaño
notebooks/
 └── 01_EDA_sprint1.ipynb     # EDA: calidad, distribuciones, riesgos y decisiones accionables
src/
 ├── run_pipeline.py          # orquestador: ingesta → preprocesado → baseline
 ├── ingesta.py               # script de ingesta (idempotente, con hash)
 ├── preprocesado.py          # limpieza, emparejamiento y partición por paciente
 ├── baseline.py              # Dummy + regresión logística (+ referencia de la tesis)
 ├── reporte_baseline.py      # escribe logs/metrics_baseline.txt y las matrices de confusión
 └── utils.py                 # configuración, logging y hash
config/config.yaml            # semilla, split e hiperparámetros
logs/                         # logs por etapa y metrics_baseline.txt
reportes/                     # resultados en JSON y figuras
slides/                       # presentación de resultados del sprint
entregables/                  # entregables por semana (guía de la semana 2, roadmap y guion de la demo)
docs/                         # GitHub Pages
README.md
requirements.txt
.env.example
.gitignore
```

---

## ⚙️ Requisitos
- Python 3.13 y unos 16 GB de RAM.
- Credencial PhysioNet con acceso a MIMIC-IV v3.1 y MIMIC-IV-Note v2.2.

```bash
pip install -r requirements.txt
cp .env.example .env     # completar MIMIC_DIR, MIMIC_NOTE_DIR y ETIQUETAS_DIR
```

---

## 🚀 Cómo ejecutar el pipeline
Todo de una vez (unos 45 min en CPU):
```bash
python src/run_pipeline.py
```
O etapa por etapa:

1. **Ingesta de datos**
   ```bash
   python src/ingesta.py
   ```
   - Verifica las fuentes, calcula SHA-256 y escribe `data/bitacora_datos.csv`. La segunda ejecución no repite nada.

2. **Preprocesamiento**
   ```bash
   python src/preprocesado.py
   ```
   - Arma positivos y negativos emparejados por época, recupera el texto, limpia y hace la partición por paciente (semilla 42).
   - Guarda en `data/processed/` y deja los conteos de cada paso en `logs/preprocesado_*.log`.

3. **Exploración inicial**
   - Abrir y ejecutar `notebooks/01_EDA_sprint1.ipynb`.

4. **Entrenamiento baseline (Dummy / regresión logística)**
   ```bash
   python src/baseline.py                  # incluye la referencia de la tesis (~33 min más)
   python src/baseline.py --sin-referencia # solo los baselines
   ```
   - Genera `logs/metrics_baseline.txt`, `reportes/baseline_resultados.json` y las figuras de `reportes/figuras/`.

---

## 📈 Resultados (Semana 3, ejecución del 30/09/2026)
- **EDA inicial** en `notebooks/01_EDA_sprint1.ipynb`, con tres decisiones accionables para el Sprint 2.
- **Métrica central: PR-AUC**, junto con F1 de la clase positiva y Recall. El dummy que siempre dice «sí» obtiene F1 = 0.698, porque el test tiene 53.7 % de positivas; por eso el F1 solo no basta.

| Modelo (test) | F1 (+) | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Dummy (más frecuente) | 0.698 | 1.000 | 0.537 | 0.500 |
| Dummy (estratificado) | 0.550 | 0.555 | 0.541 | 0.510 |
| **Regresión logística TF-IDF** | **0.791** | **0.771** | **0.879** | **0.864** |
| Referencia: modelo de la tesis (LinearSVC) | 0.778 | 0.762 | 0.861 | 0.843 |

- **Reproducibilidad:** el pipeline vuelve a obtener el modelo de la tesis (sensibilidad 0.7623, especificidad 0.7699, ROC-AUC 0.8426) con diferencia 0.0000.
- **Logs de resultados:** [`logs/metrics_baseline.txt`](logs/metrics_baseline.txt).
- **Gráficos:** [curvas ROC y precisión-recall](reportes/figuras/fig_baseline_roc_pr.png) · [matrices de confusión](reportes/figuras/fig_matrices_confusion.png).
- **Slides de resultados:** [`slides/`](slides/) y la [presentación en línea](https://carlosperez100.github.io/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/sprint1/).

---

## 📌 Roadmap
Las tareas, con responsable y plazo, están en los [issues](https://github.com/carlosperez100/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/issues) y [milestones](https://github.com/carlosperez100/mia14-trabajo-investigacion-ii-tesis-gemses-mimic/milestones).
- [x] Semana 1 → Diagnóstico, estructura del repositorio y plan del Sprint 1.
- [x] Semana 2 → Ingesta + preprocesamiento + logging (entregable de diseño del pipeline: 18/20).
- [x] Semana 3 → EDA + baseline + demo interna.
- [ ] Semanas 4–6 (Sprint 2) → Regenerar las etiquetas dentro del pipeline, GroupKFold por paciente, control del atajo por largo, máscara de códigos CIE y explicación de logística frente a LinearSVC.

---

## 📜 Licencia
Uso académico – Universidad Nacional de Ingeniería (UNI). Los datos de MIMIC-IV se rigen por el acuerdo de uso (DUA) de PhysioNet y no forman parte de este repositorio.
