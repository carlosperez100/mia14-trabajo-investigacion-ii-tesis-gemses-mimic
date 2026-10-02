# Guion de la mini-demo del Sprint 1 (5–8 min) · viernes 02-10-2026

Sigue las seis partes del «Guion de mini-demo» de la guía de la semana 3 del Dr. Rodríguez.

> Es una guía para hablar, no un texto para leer. Dilo con tus palabras: el sílabo exige que los productos superen los detectores de texto generado por IA.

| Min | Parte (guía) | Qué se muestra | Qué se dice |
|---|---|---|---|
| 0:00–1:00 | **1. Contexto:** objetivo y dataset | README, sección Dataset | **Objetivo:** detectar en la epicrisis si la hospitalización tuvo un evento adverso, para priorizarlo con GEMSES. **Datos:** MIMIC-IV-Note v2.2, con 331 793 epicrisis y 145 914 pacientes; la etiqueta sale de los códigos CIE causales (A1). Corpus de trabajo: 70 000 notas, con partición por paciente. **Track A:** había un modelo, pero no era reproducible. |
| 1:00–3:00 | **2. EDA:** 2 gráficos + 2 hallazgos + 1 riesgo | Notebook: `fig_eda_largo.png` y la tabla de balance y época | **Hallazgo 1:** el test tiene 53.7 % de positivas, frente a 20.1 % real. **Hallazgo 2:** la época está emparejada (AUC de la época sola = 0.500). **Riesgo:** el largo de la nota por sí solo da AUC 0.606 (las positivas suben de 44 % a 76.5 % por decil): posible atajo. Calidad: 0 nulos, 0 duplicados. |
| 3:00–4:00 | **3. Baseline:** qué es y por qué es el mínimo razonable | `src/baseline.py` | Dummy (más frecuente y estratificado) como piso; **regresión logística TF-IDF** como modelo simple obligatorio para clasificación de texto; el modelo de la tesis como referencia. |
| 4:00–5:30 | **4. Resultados:** 1 métrica central + 1 gráfico | `fig_baseline_roc_pr.png` (o las matrices de confusión) | **Métrica central: PR-AUC.** Logística 0.879; modelo de la tesis 0.861; dummy 0.537. ¿Por qué no F1? Porque el dummy obtiene F1 = 0.698 solo por el balance. F1 (+) de la logística: 0.791; Recall: 0.771. |
| 5:30–6:45 | **5. Reproducibilidad:** dónde están los scripts y logs y cómo se corre | Terminal: `python src/run_pipeline.py --hasta ingesta` (8 s) + `logs/metrics_baseline.txt` + `data/bitacora_datos.csv` | Un comando; semilla y split fijos (`config.yaml`); datos versionados con SHA-256; un log por etapa. **Se reproduce el modelo de la tesis con diferencia 0.0000.** |
| 6:45–7:45 | **6. Próximos pasos:** 2 mejoras para el siguiente sprint | Notebook, «Decisiones accionables» | (1) **Separar la evaluación por deciles de largo** para medir el atajo. (2) **Enmascarar los códigos CIE** en el texto antes de vectorizar y medir la diferencia de PR-AUC. Además: GroupKFold por paciente (semana 5). |

## Preguntas probables
- **¿Por qué la logística le gana al modelo de la tesis?** Todavía no está explicado. Es una hipótesis para el Sprint 2: los n-gramas de carácter añaden ruido en notas largas. Se prueba con una ablación.
- **¿La etiqueta es confiable?** Es supervisión débil (códigos de facturación). El panel experto es la validación secundaria.
- **¿Se puede reproducir en otra máquina?** Sí, con credencial PhysioNet, las tablas de etiquetas y `.env`. Integrar al pipeline las fases que generan las etiquetas es el primer issue del Sprint 2.
