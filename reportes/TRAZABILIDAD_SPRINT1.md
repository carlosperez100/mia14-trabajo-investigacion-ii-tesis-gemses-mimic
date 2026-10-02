# Trazabilidad de cifras · Sprint 1

Cada cifra del README, de la presentación (`docs/sprint1/`) y del notebook EDA tiene su fuente en un archivo del repositorio que produce el código. Tipos de respaldo:
- **Dato**: valor escrito en el archivo.
- **Derivación**: cálculo con datos del archivo (la operación se indica).
- **Juicio**: criterio del autor, rotulado como tal en el texto.

## Dataset (ingesta)
| Cifra | Tipo | Fuente |
|---|---|---|
| 331 793 epicrisis · 145 914 pacientes | Dato | `reportes/ingesta_resumen.json` → `resumen.discharge` |
| 545 497 hospitalizaciones con diagnóstico | Dato | `reportes/ingesta_resumen.json` → `resumen.diagnoses_icd.hospitalizaciones` |
| 1 139.2 MB y 33.6 MB; fechas 11/05/2026 y 14/05/2026; SHA-256 | Dato | `data/bitacora_datos.csv` (`tamano_bytes`, `fecha_modificacion`, `sha256`) |
| MIMIC-IV-Note v2.2 y MIMIC-IV v3.1 | Dato | páginas de PhysioNet enlazadas en el README (verificadas el 02-10-2026) |

## Corpus y partición (preprocesado)
| Cifra | Tipo | Fuente |
|---|---|---|
| 109 775 hospitalizaciones positivas; prevalencia real 20.12 % | Dato | `reportes/preprocesado_resumen.json` (`prevalencia_real`) y `logs/preprocesado_*.log` |
| 219 550 negativos emparejados; 329 325 hospitalizaciones seleccionadas | Dato | `logs/preprocesado_*.log`, pasos 2 y 3 |
| 213 974 notas con texto; 0 descartadas | Dato | `preprocesado_resumen.json` → `conteos.limpieza` |
| 70 000 notas · 37 692 positivas · 48 669 pacientes | Dato | `preprocesado_resumen.json` → `conteos.corpus_final` |
| 55 915 train · 14 085 test · 0 pacientes compartidos · 53.9 % y 53.7 % positivas | Dato | `preprocesado_resumen.json` → `conteos.particion` |
| 53.8 % de positivas en el corpus | Derivación | 37 692 / 70 000 |
| Corpus idéntico nota por nota al de la Fase 9 | Dato | comparación ejecutada el 30-09-2026 contra `fase9_final/dataset_final.parquet` del repositorio de la tesis (mismas notas, mismo orden, mismas etiquetas) |

## Baseline
| Cifra | Tipo | Fuente |
|---|---|---|
| F1 (+), Recall, Precisión, PR-AUC, ROC-AUC, especificidad, VPP e IC 95 % de cada modelo | Dato | `reportes/baseline_resultados.json` y `logs/metrics_baseline.txt` |
| Matrices de confusión (VN, FP, FN, VP) | Dato | `baseline_resultados.json` → `modelos.*.matriz` |
| Reproduce la Fase 9 con diferencia 0.0000 | Dato | `baseline_resultados.json` → `verificacion_reproducibilidad` |
| 226 s y 1 980 s de entrenamiento | Dato | `baseline_resultados.json` → `segundos` (226.3 y 1 979.7) |
| «9 veces más rápido» | Derivación | 1 979.7 / 226.3 = 8.7 |
| 154 FP y 67 FN menos que la tesis | Derivación | 1 502 − 1 348; 1 796 − 1 729 |
| 79 % de negativos acertados; 77 % de eventos detectados; 23 % no detectados | Derivación | 5 180 / 6 528; 5 828 / 7 557; 1 729 / 7 557 |
| Pipeline completo en unos 45 min | Derivación | marcas de tiempo de `logs/ingesta_20260930_170447.log`, `logs/preprocesado_*.log` y `logs/baseline_*.log` (17:04:47 → 17:50:44) |
| 15.6 GB de RAM | Dato | memoria física de la máquina de ejecución (Win32_ComputerSystem, 02-10-2026) |

## EDA
| Cifra | Tipo | Fuente |
|---|---|---|
| 0 nulos · 0 duplicados · largo 353–58 156 · 1 933 outliers (2.8 %) | Dato | `notebooks/01_EDA_sprint1.ipynb`, sección 0 |
| 71.1 % / 28.9 % CIE-9 / CIE-10 en ambas clases; AUC de la época sola = 0.500 | Dato | notebook, sección 2 |
| Mediana 10 717 frente a 9 185 caracteres; AUC del largo solo = 0.606 | Dato | notebook, sección 3 |
| 44.0 % → 76.5 % de positivas por decil de largo | Dato | notebook, sección 3 (salida impresa) |
| Formato CIE-10 en 4.45 % / 4.95 % | Dato | notebook, sección 4 |
| 1.44 notas por paciente (máx. 27); 0 compartidos | Dato | notebook, sección 5 |
| Sistema/Organización: 130 notas (0.34 %) | Dato | notebook, sección 6 |
| Spearman: largo 0.182 · notas del paciente 0.125 · época 0.000 | Dato | notebook, sección 7 |
| Dummy F1 = 0.698 | Dato | `baseline_resultados.json` → `dummy_mayoritaria.f1` (0.6984) |

## Cifras de la tesis citadas como contexto (fuera de este repositorio)
| Cifra | Fuente |
|---|---|
| 46 scripts sin orquestador | conteo de `*.py` en `04_pipeline_codigo/` del repositorio de la tesis (02-10-2026) |
| AUC 0.973 invalidado por el confusor de época | `fase9_final/resultados_finales.json` → `historial_de_cifras.fase4v4_confundido` del repositorio de la tesis |
| Nota 18/20 del entregable de la semana 2 | Teachlr, calificación del 25/26-09-2026 |

## Juicios del autor (rotulados en el texto)
- **Track A:** autoidentificación según la guía de la semana 1.
- **El evento no detectado es el error más costoso en vigilancia:** criterio profesional del autor.
- **El largo de la nota refleja hospitalizaciones más complejas:** hipótesis, no comprobada; se prueba en el Sprint 2.
- **La causa de que la logística supere al LinearSVC:** no explicada; hipótesis para el Sprint 2.
