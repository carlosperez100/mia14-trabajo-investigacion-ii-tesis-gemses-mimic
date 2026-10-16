# Trazabilidad de cifras · Semana 5 (feature engineering)

| Cifra del README | Tipo | Fuente u operación |
|---|---|---|
| PR-AUC, F1 (+), recall, ROC-AUC, atributos y tiempo por configuración (validación cruzada) | Dato | `logs/metrics_fe.txt`, `reportes/fe_resultados.json` → `cv`, `reportes/fe_folds.csv` |
| Diferencia con A y folds mejores (5/5) | Dato | `fe_resultados.json` → `cv.*.dif_pr_auc_vs_A` |
| A+B+C es la mejor de las cuatro en los 5 folds | Dato | `reportes/fe_folds.csv` (PR-AUC por fold) |
| Métricas en test, IC 95 % por pacientes, VPP | Dato | `fe_resultados.json` → `test` |
| Prevalencia real con epicrisis 27.59 % | Dato | `reportes/eda_resumen.json` → `balance.prevalencia_real_con_epicrisis` |
| Control: A en test = 0.8787, diferencia 0.0000 | Dato | `fe_resultados.json` → `control_reproduccion_baseline` |
| PR-AUC sin texto por grupo | Dato | `fe_resultados.json` → `aporte_grupos_sin_texto` |
| Aporte de cada grupo (+0.0575, +0.0557, +0.0083, +0.0004) | Derivación | resta entre filas consecutivas de la tabla sin texto |
| Mejora casi cuatro veces su variabilidad | Derivación | 0.0065 / 0.0017 = 3.8 |
| A+C conserva el 72 % de la mejora | Derivación | 0.0047 / 0.0065 = 0.72 |
| A+B+C cuesta 1.7 veces más tiempo de entrenamiento | Derivación | 240 s / 141 s (tiempo medio por fold) |
| Faltantes: horas en emergencia 33.2 %, destino al alta 12.5 % | Dato | `fe_resultados.json` → `faltantes_dominio` |
| Coeficientes del dominio (OBS, PSURG) | Dato | `logs/metrics_fe.txt`, sección de coeficientes |
| Largo y estancia, Spearman 0.468 | Dato | EDA de la semana 4 (`reportes/eda_resumen.json`) |
| Duración de la corrida, unas 2 horas | Dato | `logs/experimentos_fe_20261015_192534.log` (19:25:34 → 21:33:52) |

**Juicios del autor, rotulados en el texto:** qué servicios cuentan como quirúrgicos; que parte de la mejora del dominio pueda venir de reconocer el tipo de paciente (hipótesis que se verifica con la decisión 3 del EDA).
