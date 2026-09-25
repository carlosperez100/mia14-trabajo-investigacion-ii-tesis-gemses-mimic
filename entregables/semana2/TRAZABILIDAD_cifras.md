# Trazabilidad de las cifras del Entregable Semana 2

El documento entregado al docente no lleva rutas de archivos (por indicación del autor). Esta tabla conserva la fuente de cada cifra en el repositorio de la tesis `tesis-gemses-mimic-pipeline` (privado), conforme a la regla de evidencia del proyecto.

| Cifra o afirmación | Archivo fuente (repositorio de la tesis) |
|---|---|
| 331 793 epicrisis · 145 914 pacientes · ablación −63.7 % · Tier A 109 714 ingresos | `04_pipeline_codigo/datos_intermedios/fase3_v2/ablacion_final.json`, `tier_a_resumen.json` |
| 223 códigos + 21 equivalencias · 43 códigos inexistentes · Medicación 209 → 14 295 · A1 18 989 → 35 034 | `04_pipeline_codigo/datos_intermedios/fase8_equivalencias/recuperacion.json` |
| AUC 0.973 → 0.904 → 0.840 (confusor de época) | `04_pipeline_codigo/datos_intermedios/fase7_epoca/resultados_epoca.json` |
| Split por paciente no degrada (F1-macro 0.492 ± 0.018); enmascarar disparadores −1.35 pts | `04_pipeline_codigo/datos_intermedios/fase4/RESULTADOS_FASE4.md` (líneas 54–115) |
| Corpus 70 000 · prevalencia 0.2012 · E1 sens 0.7623 (0.752–0.772), esp 0.7699, AUC 0.8426, VPP 0.455 · E2 F1-macro 0.5358, F1-micro 0.7453 · abstención 6/6 | `04_pipeline_codigo/datos_intermedios/fase9_final/resultados_finales.json` |
| TEST_SIZE 0.20, SEED 42, N_MAX 70000, N_BOOTSTRAP 200, TF-IDF 60000/20000, min_df 3/5, LinearSVC balanced, OneVsRest | `04_pipeline_codigo/fase9_modelo_final.py` (líneas 93–97, 223–255, 298–304) |
| Cascada completa F1-macro 0.3626, F1-micro 0.4926 · IC por paciente ×1.13 | `04_pipeline_codigo/datos_intermedios/fase9_final/metricas_corregidas.json` |
| Kappa A–B 0.6435 (IC95 0.413–0.817) | `04_pipeline_codigo/datos_intermedios/fase6_concordancia/concordancia.json` |
| Sistema vs experto n=68: sens 0.9455, esp 0.5385, VPP 0.897, kappa 0.5307; subregistro 4/16 | `04_pipeline_codigo/datos_intermedios/fase12/sistema_vs_experto.json` |
| Fine-tuning: lineal 0.4592 (48 s); Bio_ClinicalBERT ponderado 0.3541 (4.0 h); Bio_ClinicalBERT 0.2487; BioBERT 0.2097 (2.9 h); ventana 256 = 9 % | `04_pipeline_codigo/datos_intermedios/fase11/resultados_transformers.json` |
| ModernBERT-1024 F1-macro 0.4282 tras 2 552 min · Llama 3.2 3B kappa 0.0 | `04_pipeline_codigo/datos_intermedios/fase13/modernbert_final.json`, `llm_local_vs_experto.json` |
| ERSP 6 336 únicas · T1 0.860 · T2 0.7675 · T3 0.606 (GRAVE 0.148, n=23) | `04_pipeline_codigo/datos_intermedios/oe5_ersp/informe_oe5.json` |
| ERSP por centro 0.422 / 0.284 / 0.095 · triaje 98.4 % y 91.5 % con palabras funcionales · muerte regex F1 0.622 vs ML 0.578 | `GEM_MODELOS/modelos/ersp-tfidf-svc/FICHA.md` (39–68), `P9_SUSALUD/triaje.py` (174–193) |
| Precisión 65.7 % (IC95 54.0–75.8, n=70) | `05_validacion_experta/REPORTE_VALIDACION_70_EVENTOS.md` |
| Umbral kappa ≥ 0.61 de la hipótesis general | `latex_tesis/3_1_CAPITULO_MARCO_TEORICO/Capitulo3.tex` (línea 321) |
| Pesos GEMSES 0.40/0.20/0.15/0.25 y nombre de la norma | PDF de la RGG 402-GG-ESSALUD-2020, Anexo 03 (verificado 20-set-2026; ver `11_proyecto_II/03_PENDIENTES_BORRADOR_TESIS_2026-09-20.md` §2) |
| Estado de reproducibilidad (sin requirements.txt, .env.example, run_pipeline.py, data/) · 46 scripts · checkpoints no guardados | Verificación en disco 25-set-2026 y `11_proyecto_II/01_INVENTARIO_MODELOS_Y_CHECKLIST_2026-09-18.md` §5.4 |
| Pérdida de filas por nota del primer tramo (260 000) | `11_proyecto_II/01_INVENTARIO_MODELOS_Y_CHECKLIST_2026-09-18.md` §5.3 |
| Incidente de datos personales 28-jul y pendientes | `07_bitacoras/INCIDENTE_DATOS_PERSONALES_2026-07-28.md`; `11_proyecto_II/02_ESTADO_DE_AVANCE_COMPLETO_2026-09-18.md` §9 |
| **Proceso MIMIC-IV:** registro, DUA v1.5.0, CITI, acceso 28-abr; 364 627 pacientes, 546 028 hospitalizaciones, 94 458 UCI | `00_lectura_rapida/HISTORIAL_PASOS_TESIS.md` (Etapa 3, 28-abr-2026) |
| CITI: cursos, notas 95/100 y 96/100, Record 76729511 y 54788726, vigencia 28-abr-2029 | Certificados y reportes CITI (PDF) en OneDrive, carpeta del curso MIA-303 |
| DUA MIMIC-IV-Note, descarga 1.96 GB (1.14 GB + 789 KB + 781 MB + 39 MB), entorno 11-may | `00_lectura_rapida/HISTORIAL_PASOS_TESIS.md` (Etapa 7, 10–11-may-2026) |
| 331 793 epicrisis confirmadas por DuckDB (11-may) · SHA-256 33/33 (14-may) | `00_lectura_rapida/HISTORIAL_PASOS_TESIS.md` (Etapa 9) y showcase v7 (recuento ⑤ y ⑦) |
| Conflicto de interés y acceso restringido al investigador principal | `latex_tesis/Anexos/AnexoE_anotadores_etica.tex` (líneas 19–25) |
