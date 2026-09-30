# Guion de la demo interna del Sprint 1 (5–8 min) · viernes 02-10-2026

> Es una guía para hablar, no un texto para leer. Dilo con tus palabras. El sílabo exige que los productos superen detectores de texto generado por IA.

| Min | Qué se muestra | Qué se dice |
|---|---|---|
| 0:00–0:45 | README del repo | **Problema y tarea.** Detectar si una epicrisis de MIMIC-IV corresponde a una hospitalización con un evento adverso. La etiqueta es débil: sale de los códigos CIE causales (A1). Unidad: la nota de alta. Partición por paciente. |
| 0:45–1:30 | `ROADMAP_SPRINT1.md` | **Punto de partida honesto.** De Proyecto I venía un modelo entrenado, pero repartido en 46 scripts con rutas escritas a mano, sin orquestador y sin splits guardados. Por la definición del sílabo, era un pipeline no reproducible. El objetivo del sprint fue reproducirlo con un comando (± 0.005). |
| 1:30–3:00 | Terminal: `python src/run_pipeline.py --hasta preprocesado` (o el log ya ejecutado) + `data/bitacora_datos.csv` | **Ingesta y preprocesado.** Seis fuentes registradas con SHA-256, tamaño y fecha; MIMIC no se copia (DUA). Ingesta idempotente: la segunda corrida no recalcula nada. El preprocesado deja conteos por paso: 109 775 hospitalizaciones positivas de 545 497 (prevalencia real 20.12 %), 70 000 notas finales, 55 915 de train y 14 085 de test, **0 pacientes compartidos**. La partición queda guardada con su hash. |
| 3:00–4:30 | Notebook EDA, tabla de conclusiones | **EDA orientado a riesgos.** (1) El corpus tiene 53.8 % de positivas frente al 20.1 % real: por eso reporto AUC y VPP corregido, no F1. (2) La época está emparejada: el AUC de la época sola es 0.500, así que el confusor que había inflado un AUC de 0.973 está controlado. (3) El largo de la nota solo da un AUC de 0.606: riesgo moderado de atajo, se controla en el Sprint 2. (4) Fuga de códigos CIE en el texto: 4.45 % frente a 4.95 %, baja. (5) La clase Sistema/Organización tiene 130 notas: no es evaluable. |
| 4:30–6:00 | `reportes/figuras/fig_baseline_roc_pr.png` | **Baseline y gráfica central.** Los dos dummy dan AUC 0.500 y 0.510 (el piso). La logística TF-IDF da **0.864 (IC 0.858–0.869)**. El modelo de la tesis, reproducido, da **0.843 (IC 0.836–0.848)**. |
| 6:00–7:00 | `reportes/baseline_resultados.json`, bloque `verificacion_reproducibilidad` | **Criterio de aceptación cumplido.** Reproduce la Fase 9 con diferencia 0.0000 en sensibilidad (0.7623), especificidad (0.7699) y AUC (0.8426), con la misma matriz de confusión. **Hallazgo:** el baseline simple supera al modelo de la tesis y los IC no se solapan. |
| 7:00–7:45 | Roadmap, sección «Fuera del Sprint 1» | **Qué sigue.** Integrar al orquestador las fases que generan las etiquetas; validación cruzada por paciente (GroupKFold) con resultados por fold; control del atajo por largo; ablación que enmascara códigos CIE. La logística pasa a ser el baseline a batir. |

## Preguntas probables y respuesta corta
- **¿Por qué no F1 como métrica principal?** Porque el test tiene 53.7 % de positivas, pero en la realidad son el 20.1 %. La precisión cambia con la prevalencia; el AUC no. El VPP se corrige con Bayes.
- **¿Por qué la logística gana al SVM?** Todavía no está explicado. Es una hipótesis para el Sprint 2: los n-gramas de carácter del SVM añaden ruido sobre notas largas. Hay que probarlo con una ablación antes de afirmarlo.
- **¿La etiqueta es confiable?** Es supervisión débil (códigos de facturación). El panel experto es la validación secundaria y está documentado en la tesis.
- **¿Se puede reproducir sin tu máquina?** Sí, con credencial PhysioNet, las tres tablas de etiquetas y `.env`. Las tablas de etiquetas todavía no se regeneran dentro del orquestador: es la primera tarea del Sprint 2.
