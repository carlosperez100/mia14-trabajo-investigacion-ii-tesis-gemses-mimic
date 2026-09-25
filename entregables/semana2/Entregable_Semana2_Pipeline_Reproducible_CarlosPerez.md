---
lang: es
---

<!-- PORTADA-DOCX-INI -->
::: {custom-style="Title"}
![](logo_uni.png){width=3.5cm}
:::

::: {custom-style="Subtitle"}
UNIVERSIDAD NACIONAL DE INGENIERÍA · Facultad de Ingeniería Industrial y de Sistemas · Maestría en Inteligencia Artificial
:::

::: {custom-style="Title"}
ENTREGABLE SEMANA 2
:::

::: {custom-style="Subtitle"}
Curso: MIA-14 · Trabajo de Investigación II · Sección B · 2026-2
:::

::: {custom-style="Author"}
**Proceso:** Diseño reproducible del flujo experimental: pipeline de aprendizaje supervisado para la detección y priorización de eventos adversos
:::

::: {custom-style="Author"}
**Tesis:** *Detección Automatizada y Priorización de Eventos Adversos en Notas Clínicas mediante un Flujo de Procesamiento (Pipeline) de NLP y Aprendizaje Automático — Aplicación de la Matriz de Priorización del Modelo GEMSES sobre MIMIC-IV*
:::

::: {custom-style="Author"}
Autor: Carlos Pérez Pérez · Docente: Dr. Ing. Glen Dario Rodríguez Rafael
:::

::: {custom-style="Date"}
Lima, 25 de septiembre de 2026
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```
<!-- PORTADA-DOCX-FIN -->

**Tipo de tesis y continuidad.** Aprendizaje supervisado (clasificación de texto clínico) con etiqueta obtenida por *supervisión débil*: no la asigna un anotador leyendo cada epicrisis, sino que se deriva de los códigos CIE-10 causales de la misma hospitalización (etiqueta A1). Esto permite entrenar con 70 000 epicrisis sin anotación manual masiva; a cambio la etiqueta tiene ruido, y su calidad se verifica con un panel de expertos y con un corpus en español etiquetado por humanos (ERSP de EsSalud). No se prevé anotar más datos. El título de la tesis se mantiene desde Trabajo de Investigación I; el flujo incorpora lo probado desde julio, lo aprendido en el curso de Procesamiento de Lenguaje Natural (auditoría adversarial) y la aplicación del mismo enfoque a los reportes de seguridad del paciente de EsSalud en el sistema GEM·VigIA.

# 1. Diagrama del flujo principal

El diagrama del pipeline en seis etapas, con la salida reproducible de cada una, el baseline, las alternativas refutadas, los controles contra fugas y el estado de reproducibilidad, se presenta a página completa en el **Anexo A**. El proceso de obtención de la base de datos MIMIC-IV, con sus requisitos y tiempos, se presenta en el **Anexo B**.

# 2. Explicación del flujo

**Datos, unidad de análisis y variable objetivo.** MIMIC-IV v3.1 y MIMIC-IV-Note v2.2 (PhysioNet): 331 793 epicrisis de 145 914 pacientes, leídas con DuckDB. La unidad de análisis es la epicrisis (una por hospitalización). Variable objetivo: en la etapa 1, binaria (¿hubo un evento adverso codificado como causal?); en la etapa 2, la naturaleza del evento según la Tabla de Codificación (Anexo 02 de la Directiva N.º 7-OGCyH-ESSALUD-2020), reducida a 6 clases con n suficiente. Segunda fuente, para transferencia al español: 6 336 reportes ERSP con etiqueta humana. El texto es la única entrada del modelo; no hay faltantes numéricos ni escalamiento, y la fusión con variables tabulares prevista en el proyecto original no se ejecutó.

**Obtención de MIMIC-IV: requisitos cumplidos y tiempos (Anexo B).** El 28 de abril de 2026 se completó en un solo día: registro en PhysioNet con nombre real y afiliación académica; los dos cursos CITI de ética («Data or Specimens Only Research», nota 95/100, y «Ética en Investigación» bajo EsSalud, 96/100; vigentes hasta abril de 2029); la firma del *Credentialed Health Data Use Agreement* v1.5.0 y el acceso confirmado a MIMIC-IV (364 627 pacientes, 546 028 hospitalizaciones). El 10 y 11 de mayo se firmó el acuerdo específico de MIMIC-IV-Note, se descargaron sus 4 archivos (1.96 GB) y se verificó la primera lectura (331 793 epicrisis). El 14 de mayo se descargaron los 33 archivos de MIMIC-IV y se comprobó su integridad por SHA-256 (33/33). Tiempos: requisitos y acceso, 1 día; corpus completo verificado, 17 días desde la decisión del dataset. Costo: S/ 0. Compromisos: no reidentificar, no redistribuir, datos y modelos fuera del control de versiones; acceso al texto restringido al investigador acreditado.

**Etapas del pipeline (qué recibe y qué produce cada una).**

| Etapa | Recibe → produce | Decisiones técnicas |
|---|---|---|
| 1. Ingesta | Archivos MIMIC → 331 793 epicrisis con identificadores de paciente y hospitalización | Lectura con DuckDB sobre los archivos comprimidos, sin cargarlos a memoria |
| 2. Preprocesamiento y etiqueta | Epicrisis + códigos → corpus de modelado de 70 000 epicrisis, prevalencia real 0.2012 | 223 códigos CIE-10 causales + 21 equivalencias OMS→CM → etiqueta A1 no circular; emparejamiento de época CIE-9/CIE-10 en los negativos; NegEx y regex solo como baseline |
| 3. Partición y features | Corpus → matrices TF-IDF | Partición por paciente, test 20 %, semilla 42; validación cruzada de 5 pliegues dentro del train; TF-IDF de palabras (1–2-gramas, 60 000) + caracteres (3–5, 20 000); vocabulario ajustado solo con train |
| 4. Entrenamiento | Matrices → modelo final serializado | Cascada: E1 detección sí/no con LinearSVC balanceado; E2 naturaleza multietiqueta con OneVsRest(LinearSVC), solo sobre positivos y con partición propia; abstención ante textos triviales |
| 5. Evaluación | Test reservado, panel experto y ERSP → archivos JSON de métricas | IC95 por *bootstrap* (200 réplicas) por nota y por paciente; cascada extremo a extremo; sistema contra consenso de expertos |
| 6. Salida | Detecciones → tablero de priorización | Matriz GEMSES (Anexo 03): G = 0.40·c + 0.20·d + 0.15·e + 0.25·f → percentiles P25/P50/P75 → banda Verde/Amarillo/Rojo y responsable |

**Baseline y familia de modelos.** Baseline: reglas regex + NegEx (precisión 65.7 %, IC95 54.0–75.8, n = 70) y TF-IDF + regresión logística. Familia elegida: modelo lineal sobre TF-IDF (LinearSVC balanceado) en cascada. Alternativas probadas y refutadas con la misma partición y semilla: BioBERT y Bio_ClinicalBERT con ajuste fino (F1-macro 0.21–0.35 en 2.9–4.0 h de GPU) y BioClinical ModernBERT de 1024 tokens (0.428 en 42.5 h), frente a 0.459 del lineal en 48 s; Llama 3.2 3B *zero-shot* obtuvo kappa 0.0 al clasificar todo como positivo. Causa medida: la ventana de 256 tokens cubre el 9 % de una epicrisis (mediana 3 148 tokens). Solo se reabriría con Longformer o BigBird y GPU dedicada.

**Criterio preliminar de evaluación.** Métrica principal: F1-macro de la cascada completa con IC95 por paciente (hoy 0.363; la etapa 2 aislada da 0.536, por eso se reporta siempre la cascada). Secundarias: etapa 1 con sensibilidad 0.762, especificidad 0.770, AUC 0.843 y VPP 0.455 a prevalencia real; abstención 6/6. Resultado *bueno*: supera el baseline de reglas y alcanza kappa sistema–experto ≥ 0.61 (hoy 0.531, n = 68, solo infección: no alcanzado). Resultado *aceptable*: supera el baseline lineal en cascada y F1 > 0.5 en toda clase con n ≥ 100. Sistema/Organización, Sangre/Hemoderivados y Nutrición se declaran no viables por tamaño de muestra.

**Reproducibilidad.** Versiones: MIMIC-IV v3.1 (DOI 10.13026/kpb9-mt58), MIMIC-IV-Note v2.2 (DOI 10.13026/1n74-ne17), Python 3.13.5, DuckDB 1.5.2; scikit-learn por fijar. Semilla 42 en partición, muestreo y *bootstrap*. Registro: un archivo JSON de resultados por fase y bitácoras fechadas. Artefactos: modelo final, conjunto de datos en Parquet, JSON de resultados y de métricas corregidas; datos y modelos fuera de git por el acuerdo de uso. Brecha que cierra el Sprint 1: dependencias con versiones fijadas, plantilla de variables de entorno, orquestador que encadene las fases, estructura data/raw y data/processed y README ejecutable; criterio de aceptación: regenerar el modelo final con un solo comando y obtener las mismas métricas (± 0.005).

**Riesgos técnicos previsibles y mitigación.**

| Riesgo (dónde se detectó) | Mitigación en el diseño |
|---|---|
| Fuga por circularidad regex → etiqueta (tesis y curso de PLN; enmascarar disparadores solo baja 1.35 puntos) | Etiqueta A1 derivada de códigos causales, no del regex |
| Confusor de época CIE-9/CIE-10 (AUC 0.973 → 0.840 al emparejar) | Emparejamiento de época en los negativos; se descarta la cifra 0.973 |
| Fuga por paciente y por centro asistencial (ERSP en GEM·VigIA: 0.77 estratificado → 0.42 por centro) | Partición por paciente con verificación; OE5 re-medido por centro antes de citarse |
| Fuga por estilo de redacción (triaje en GEM·VigIA: 98.4 %, y 91.5 % solo con palabras funcionales) | Ablaciones de control; sin metadatos de quién redacta |
| Etiqueta débil y clases pequeñas (kappa 0.531 < 0.61; Sistema/Organización F1 0.0 con n = 66) | Panel experto como validación secundaria; clases no viables declaradas |
| Costo de cómputo, datos personales y pérdida de intermedios (transformers 2.9–42.5 h; ERSP con datos personales; 260 000 filas perdidas al morir un proceso) | Familia lineal; datos fuera de git, repositorio privado y copias del ERSP anonimizadas; orquestador con *checkpoint* por fase |

<!-- ANEXOS-PDF-INI -->
```{=latex}
\newpage
\begin{landscape}
\section*{Anexo A. Diagrama del flujo principal}
\begin{center}\includegraphics[width=0.98\linewidth,height=0.84\textheight,keepaspectratio]{fig_pipeline_semana2.png}\end{center}
\end{landscape}
\begin{landscape}
\section*{Anexo B. Proceso de obtención de la base de datos MIMIC-IV}
\begin{center}\includegraphics[width=0.98\linewidth,height=0.84\textheight,keepaspectratio]{fig_acceso_mimic.png}\end{center}
\end{landscape}
```
<!-- ANEXOS-PDF-FIN -->
<!-- ANEXOS-DOCX-INI -->
```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```
# Anexo A. Diagrama del flujo principal

![](fig_pipeline_semana2.png){width=100%}

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```
# Anexo B. Proceso de obtención de la base de datos MIMIC-IV

![](fig_acceso_mimic.png){width=100%}
<!-- ANEXOS-DOCX-FIN -->
