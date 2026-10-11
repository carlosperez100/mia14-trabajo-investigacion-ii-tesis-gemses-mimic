# -*- coding: utf-8 -*-
"""
ETAPA 3 — EDA (análisis exploratorio de datos) reproducible y con log automático.

Entradas: data/processed/dataset.parquet y splits.csv (salida de src/preprocesado.py);
          data/interim/ (tablas de etiquetas y época que deja src/ingesta.py);
          tablas de MIMIC-IV en MIMIC_DIR (admissions, patients, diagnoses_icd, d_icd_diagnoses,
          drgcodes, services, icustays) y la lista de hospitalizaciones con epicrisis (MIMIC_NOTE_DIR);
          data/catalogos/ (catálogo de 231 eventos del Anexo 02 y tabla evento-código CIE-10).
Salidas : logs/eda_<fecha>.log            log automático (hora, tamaño, pasos, resultados)
          logs/metrics_eda.txt            resumen legible de todo el análisis
          reportes/eda_resumen.json       todas las cifras del EDA (fuente del README y del notebook)
          reportes/eda_eventos_231.csv    una fila por evento del catálogo
          reportes/figuras/fig_eda_*.png  figuras, cada una con su explicación al pie

Regla de datos: MIMIC-IV está bajo DUA de PhysioNet. Este script solo escribe AGREGADOS;
nunca imprime ni guarda texto clínico.

Las cifras describen el corpus completo (train y test). Lo que se ajusta con los datos —el
vocabulario y el modelo de contexto— se ajusta solo con train; el test se usa para medir.

Secciones:
  0 calidad: nulos, tipos, rangos, duplicados, valores atípicos
  1 estadísticas descriptivas (largo, palabras, notas por paciente, naturalezas)
  2 volumen y balance de la variable objetivo por partición
  3 época de codificación CIE-9/CIE-10 por clase
  4 largo de la nota por clase (atajo)
  5 códigos CIE escritos en el texto (fuga directa de la etiqueta)
  6 pacientes compartidos entre train y test (fuga por paciente)
  7 naturaleza del evento entre las positivas, en total y por época
  T contenido del texto: secciones y vocabulario                      (src/eda_texto.py)
  A variables de la hospitalización                                    (src/eda_variables.py)
  P prevalencia real por época y periodo, y drift                      (src/eda_variables.py)
  R ruido de la etiqueta                                               (src/eda_variables.py)
  B eventos del Anexo 02                                               (src/eda_variables.py)
  8 correlaciones de Spearman con la variable objetivo en la ÚLTIMA fila
  9 riesgos y decisiones accionables para el siguiente sprint
  + figura del flujo del EDA con el resultado de cada paso

Uso:  python src/eda.py
"""
import re
import textwrap
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import eda_texto as et
import eda_variables as ev
from utils import LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger, sha256

ETAPA = "eda"
FIG = REPORTES / "figuras"
AZUL, ROJO, VERDE, GRIS = "#1F4E79", "#9B2226", "#1E7B4F", "#9FB8CF"
# cualquier cadena con forma de código CIE-10: letra, dos dígitos, punto y decimales
PAT_CIE10 = re.compile(r"\b[A-TV-Z]\d{2}\.\d{1,4}\b")
# solo las familias de las que sale la etiqueta, con el punto y admitiendo letras después (T81.4XXA). Los efectos adversos de
# medicamentos (T36 a T50) se piden con su forma completa (T38.0X5A), porque T36.5 a T40.0 también son temperaturas en Celsius.
PAT_FAMILIAS = re.compile(r"\b(?:(?:T8[0-8]|Y6[35]|Y8[34]|W[01]\d|L89)\.[0-9A-Z]{1,4}|T(?:3[6-9]|4\d|50)\.[0-9A-Z]{2}5[A-Z]?)\b")
# familias equivalentes de la época CIE-9 (996 a 999, 707, E870 a E888, E930 a E949), escritas con punto
PAT_FAMILIAS_CIE9 = re.compile(r"\b(?:99[6-9]\.\d{1,2}|707\.\d{1,2}|E8[78]\d\.\d|E9[34]\d\.\d)\b")
TEMPERATURAS = ("T95", "T96", "T97", "T98", "T99")
FUENTES_DECISION = ["del paso 14", "del paso 14", "de los pasos 5, 6 y 9"]
NOMBRE_NAT = {"Medicacion": "Medicación", "Infeccion nosocomial": "Infección nosocomial",
              "Dispositivo medico": "Dispositivo médico", "Sistema/Organizacion": "Sistema/Organización"}


def decisiones(R):
    """Tres decisiones accionables: verbo, paso del pipeline, evidencia de esta corrida y control medible."""
    N, C = R["ruido_etiqueta"], R["variables"]["contexto"]
    n10, n9, K = N["negativas"]["cie10"], N["negativas"]["cie9"], N["caidas"]
    texto = f", frente a {C['modelo_de_texto']['roc_auc']:.3f} del modelo de texto" if "modelo_de_texto" in C else ""
    return [
        {"n": 1, "decision": "Excluir de la clase negativa las hospitalizaciones que tengan cualquier código de las familias de causa "
                             "explícita",
         "paso": "Preprocesado",
         "evidencia": f"{n10['con_alguno']} de las {n10['n']} negativas de la época CIE-10 ({n10['pct_con_alguno']:.1f} %) llevan un "
                      "código de esas familias (T80 a T88, Y63, Y65, Y83, Y84, W00 a W19, L89 o efecto adverso de medicamento) que no "
                      f"está en la tabla del proyecto; en la época CIE-9, donde se usan las familias completas, son "
                      f"{n9['con_codigo_de_familia_causal']}",
         "control": f"porcentaje de negativas CIE-10 con código de esas familias: hoy {n10['pct_con_alguno']:.1f} %, debe quedar en 0 % "
                    "al regenerar el corpus; se reporta la PR-AUC antes y después"},
        {"n": 2, "decision": "Dejar como positivas por caída solo las que tienen el hospital como lugar de ocurrencia y sacar las "
                             "demás de las dos clases",
         "paso": "Preprocesado",
         "evidencia": f"{K['positivas_solo_por_caida']} positivas ({K['pct_de_las_positivas']:.1f} %) lo son solo por un código de "
                      f"caída; en la época CIE-10, de {K['cie10']['n']} solo {K['cie10']['en_hospital']} "
                      f"({K['cie10']['pct_en_hospital']:.1f} %) tienen el hospital como lugar de ocurrencia y "
                      f"{K['cie10']['en_residencia_privada']} una residencia privada; {K['sin_codigo_de_lugar']} no tienen código de "
                      "lugar y en la época CIE-9 ningún código de lugar corresponde al hospital",
         "control": f"positivas solo por caída sin el hospital como lugar de ocurrencia: hoy {K['sin_el_hospital_como_lugar']} de "
                    f"{K['positivas_solo_por_caida']} ({K['pct_sin_el_hospital_como_lugar']:.1f} %), deben quedar en 0; se reporta la "
                    "PR-AUC antes y después"},
        {"n": 3, "decision": "Evaluar el modelo dentro de cada servicio, grupo de edad y decil de largo, y compararlo con el modelo "
                             "de contexto sin texto",
         "paso": "Evaluación",
         "evidencia": f"un modelo que no lee el texto alcanza un ROC-AUC de {C['al_ingreso']['roc_auc']:.3f} con las variables del "
                      f"ingreso y de {C['al_alta']['roc_auc']:.3f} con las del alta{texto}; el largo de la nota solo da "
                      f"{R['largo']['auc_largo_solo']:.3f}",
         "control": "ROC-AUC del modelo de texto mayor que el del modelo de contexto en cada estrato con 200 epicrisis o más en "
                    "test, con un intervalo de confianza al 95 % por paciente para la diferencia que no incluya el cero"},
    ]


def acciones(R):
    M, F, P, fam = R["eventos"]["mapeo"], R["variables"]["fuera_de_rango"], R["prevalencia_real"], R["fuga_codigos"]["familias_de_la_etiqueta"]
    U = R["ruido_etiqueta"]["ulcera_cie9"]
    return [
        f"Limitar la familia CIE-9 de úlcera por presión a los códigos 707.0x: {U['sin_codigo_707_0']} de sus "
        f"{U['positivas_de_la_familia']} positivas no tienen ninguno.",
        f"Usar como prevalencia real el {P['todas']['pct_positivas_con_epicrisis']:.1f} % de las hospitalizaciones con epicrisis, y no "
        f"el {P['todas']['pct_positivas']:.1f} % sobre todas las hospitalizaciones, al calcular el valor predictivo positivo.",
        "Reportar las métricas por época de codificación, por periodo y por subgrupo (servicio, tipo de ingreso, sexo, grupo de edad, "
        "raza y seguro): la etiqueta se define con familias completas en CIE-9 y con los códigos de la tabla en CIE-10.",
        f"Enlazar con el catálogo del Anexo 02 las {M['filas_sin_enlace']} filas de la tabla de códigos cuyo identificador no es el del "
        "catálogo, o declararlas sin equivalente, antes de reportar cobertura o frecuencia por evento.",
        "No usar como predictores el número de diagnósticos ni la severidad APR-DRG: salen de la misma codificación que la etiqueta.",
        "Mantener la segunda etapa a nivel de naturaleza: la frecuencia por evento no alcanza para un clasificador por evento.",
        f"Tratar como dato faltante la estancia de las {F['estancia_negativa']} hospitalizaciones con estancia negativa al calcular el "
        "proxy de la dimensión Tiempo.",
        f"No enmascarar códigos CIE en el texto: los códigos de las familias de la etiqueta aparecen escritos en {fam['notas']} notas; el "
        "patrón general usado antes contaba sobre todo temperaturas.",
        "Mantener la PR-AUC como métrica central, adoptada en la semana 3.",
    ]


def describir(serie):
    d = serie.describe(percentiles=[.25, .5, .75])
    return {k: round(float(d[k]), 2) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}


def _mil(n):
    """Miles separados con espacio que no se parte entre renglones (70 000)."""
    return f"{int(round(n)):,}".replace(",", " ")


def _pct(x):
    """Porcentaje con seis decimales: se redondea una sola vez, al mostrarlo."""
    return round(100 * float(x), 6)


def pie(fig, texto, ancho, alto):
    """Explicación corta al pie de la figura, para que se entienda aunque se abra sola."""
    fig.tight_layout(rect=(0, alto, 1, 1))
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(texto, ancho)), ha="left", va="bottom",
             fontsize=8.4, color="#333", linespacing=1.35)


def main():
    t0 = time.time()
    log = logger(ETAPA)
    cfg = cargar_config()
    semilla = cfg["semilla"]
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"etapa": ETAPA, "fecha": datetime.now().isoformat(timespec="seconds"), "semilla": semilla,
         "entrada": {"dataset": "data/processed/dataset.parquet", "splits": "data/processed/splits.csv"},
         "alcance": "las cifras describen el corpus completo (train y test); el vocabulario y el modelo de contexto se ajustan solo "
                    "con train"}

    # ---------- carga
    ruta = PROCESSED / "dataset.parquet"
    df = pd.read_parquet(ruta)
    R["entrada"]["tamano_bytes"] = ruta.stat().st_size
    R["entrada"]["sha256_splits"] = sha256(PROCESSED / "splits.csv")
    R["entrada"]["notas"] = int(len(df))
    R["entrada"]["pacientes"] = int(df.subject_id.nunique())
    df["largo"] = df.text.str.len()
    df["palabras"] = df.text.str.count(r"\S+")
    df["notas_paciente"] = df.groupby("subject_id").note_id.transform("count")
    df["n_naturalezas"] = df.labels.apply(lambda l: len(l) if l is not None else 0)
    log.info(f"carga: {len(df):,} notas · {df.subject_id.nunique():,} pacientes · "
             f"{df.hadm_id.nunique():,} hospitalizaciones · {ruta.stat().st_size/1e6:.0f} MB · "
             f"semilla {semilla} · sha256 splits {R['entrada']['sha256_splits'][:16]}…")
    df = ev.cargar_variables(df, log, R)

    # ---------- 0. calidad
    cols = ["note_id", "subject_id", "hadm_id", "text", "y", "epoca", "split", "labels"]
    calidad = {c: {"tipo": str(df[c].dtype), "nulos": int(df[c].isna().sum()),
                   "unicos": int(df[c].nunique()) if c != "labels" else None} for c in cols}
    q1, q3 = df.largo.quantile([.25, .75])
    lim = float(q3 + 1.5 * (q3 - q1))
    R["calidad"] = {
        "columnas": calidad,
        "duplicados": {"note_id": int(df.note_id.duplicated().sum()), "hadm_id": int(df.hadm_id.duplicated().sum()),
                       "texto_identico": int(df.text.duplicated().sum())},
        "faltantes": {"texto_vacio": int((df.largo == 0).sum()),
                      "positivas_sin_naturaleza": int(((df.y == 1) & (df.n_naturalezas == 0)).sum()),
                      "negativas_con_naturaleza": int(((df.y == 0) & (df.n_naturalezas > 0)).sum())},
        "rango_largo": [int(df.largo.min()), int(df.largo.max())],
        "outliers_largo": {"limite_q3_1_5iqr": round(lim), "n": int((df.largo > lim).sum()),
                           "pct": _pct((df.largo > lim).mean()), "accion": "se conservan: son notas reales"},
    }
    nulos_sin_labels = sum(v["nulos"] for c, v in calidad.items() if c != "labels")
    R["calidad"]["nulos"] = {"columnas_sin_labels": nulos_sin_labels,
                             "labels_none": calidad["labels"]["nulos"],
                             "nota": "labels es None en las negativas por construcción (no tienen naturaleza): no es un faltante"}
    log.info(f"0. calidad: nulos {nulos_sin_labels} (labels None en {calidad['labels']['nulos']} negativas, esperado) · "
             f"duplicados {R['calidad']['duplicados']} · "
             f"faltantes {R['calidad']['faltantes']} · largo {R['calidad']['rango_largo']} · "
             f"outliers {R['calidad']['outliers_largo']['n']} ({R['calidad']['outliers_largo']['pct']:.2f} %)")

    # ---------- 1. estadísticas descriptivas
    R["descriptivas"] = {"largo_caracteres": describir(df.largo), "palabras": describir(df.palabras),
                         "notas_por_paciente": describir(df.groupby("subject_id").size()),
                         "naturalezas_por_positiva": describir(df.loc[df.y == 1, "n_naturalezas"])}
    for k, v in R["descriptivas"].items():
        log.info(f"1. {k}: media {v['mean']} · mediana {v['50%']} · desv {v['std']} · min {v['min']} · max {v['max']}")

    # ---------- 2. volumen y balance
    t = pd.crosstab(df.split, df.y, margins=True)
    t.columns = ["negativa", "positiva", "total"]
    R["balance"] = {idx: {"negativa": int(f.negativa), "positiva": int(f.positiva), "total": int(f.total),
                          "pct_positivas": _pct(f.positiva / f.total)} for idx, f in t.iterrows()}
    log.info("2. balance: " + " · ".join(f"{k}: {v['total']:,} notas, {v['pct_positivas']:.2f} % positivas" for k, v in R["balance"].items()))

    # ---------- 3. época por clase
    ep = pd.crosstab(df.y, df.epoca, normalize="index")
    auc_epoca = float(roc_auc_score(df.y, df.epoca))
    R["epoca"] = {"pct_por_clase": {("positiva" if y == 1 else "negativa"): {f"CIE-{int(c)}": _pct(ep.loc[y, c]) for c in ep.columns}
                                    for y in ep.index},
                  "auc_epoca_sola": round(auc_epoca, 4),
                  "pct_positivas_por_epoca": {f"CIE-{int(e)}": _pct(df.loc[df.epoca == e, "y"].mean()) for e in sorted(df.epoca.unique())}}
    log.info(f"3. época: {R['epoca']['pct_por_clase']} · AUC de la época sola {auc_epoca:.3f}")

    # ---------- 4. largo por clase (atajo)
    auc_largo = float(roc_auc_score(df.y, df.largo))
    dec = pd.qcut(df.largo, 10, labels=False)
    pct_decil = [round(100 * float(v), 4) for v in df.groupby(dec).y.mean()]
    R["largo"] = {"mediana_por_clase": {"negativa": float(df.loc[df.y == 0, "largo"].median()),
                                        "positiva": float(df.loc[df.y == 1, "largo"].median())},
                  "auc_largo_solo": round(auc_largo, 4), "pct_positivas_por_decil": pct_decil}
    log.info(f"4. largo: medianas {R['largo']['mediana_por_clase']} · AUC del largo solo {auc_largo:.3f} · "
             f"positivas por decil {pct_decil[0]:.1f} % → {pct_decil[-1]:.1f} %")

    # ---------- 5. códigos CIE en el texto (fuga directa)
    enc = df.text.str.findall(PAT_CIE10)
    con = enc.str.len() > 0
    todos = enc[con].explode()
    temp = todos.str[:3].isin(TEMPERATURAS)
    solo_temp = enc[con].apply(lambda l: all(x[:3] in TEMPERATURAS for x in l))
    frecuentes = todos.value_counts()
    fam10, fam9 = df.text.str.contains(PAT_FAMILIAS), df.text.str.contains(PAT_FAMILIAS_CIE9)
    fam = fam10 | fam9

    def cuenta(m):
        return {"notas": int(m.sum()), "positivas": int(m[df.y == 1].sum()), "negativas": int(m[df.y == 0].sum()),
                "pct_positivas": _pct(m[df.y == 1].mean()), "pct_negativas": _pct(m[df.y == 0].mean())}
    titulos = ev._AUX["titulos"]
    R["fuga_codigos"] = {
        "patron_general": {"notas": int(con.sum()), "pct_positivas": _pct(con[df.y == 1].mean()), "pct_negativas": _pct(con[df.y == 0].mean()),
                           "coincidencias": int(len(todos)), "pct_coincidencias_T95_a_T99": _pct(temp.mean()),
                           "pct_notas_con_solo_T95_a_T99": _pct(solo_temp.mean()),
                           "cadenas_mas_frecuentes": {k: int(v) for k, v in frecuentes[frecuentes >= 100].head(6).items()},
                           "codigos_T95_a_T99_en_el_diccionario_de_mimic": int(sum(k[1] == 10 and k[0][:3] in TEMPERATURAS
                                                                                     for k in titulos.index)),
                           "nota": "letra, dos dígitos, punto y decimales; también captura valores como una temperatura escrita T98.2"},
        "familias_de_la_etiqueta": {**cuenta(fam), "codigos_cie10": cuenta(fam10), "codigos_cie9": cuenta(fam9),
                                    "nota": "búsqueda por patrón de los códigos escritos con punto; un número con la misma forma "
                                            "(996.5) también cuenta, así que es una cota superior"}}
    G, FAM = R["fuga_codigos"]["patron_general"], R["fuga_codigos"]["familias_de_la_etiqueta"]
    log.info(f"5. códigos en el texto: patrón general en {G['notas']} notas ({G['pct_positivas']:.2f} % de positivas y "
             f"{G['pct_negativas']:.2f} % de negativas); {G['pct_coincidencias_T95_a_T99']:.1f} % de las {G['coincidencias']} coincidencias "
             f"son T95 a T99 ({G['cadenas_mas_frecuentes']}), que no existen en el diccionario de MIMIC "
             f"({G['codigos_T95_a_T99_en_el_diccionario_de_mimic']} códigos) · familias de la etiqueta: {FAM['notas']} notas "
             f"({FAM['positivas']} positivas, {FAM['negativas']} negativas; con forma CIE-10 {FAM['codigos_cie10']['notas']}, con forma "
             f"CIE-9 {FAM['codigos_cie9']['notas']})")

    # ---------- 6. fuga por paciente
    comp = set(df.loc[df.split == "train", "subject_id"]) & set(df.loc[df.split == "test", "subject_id"])
    R["pacientes"] = {"compartidos_train_test": len(comp), "notas_por_paciente_media": R["descriptivas"]["notas_por_paciente"]["mean"]}
    log.info(f"6. pacientes compartidos train/test: {len(comp)} · notas por paciente {R['pacientes']['notas_por_paciente_media']}")

    # ---------- 7. naturaleza del evento, en total y por época
    pos = df[df.y == 1]
    nat = pos.labels.explode().value_counts()
    por_epoca = {}
    for e in sorted(pos.epoca.unique()):
        p = pos[pos.epoca == e]
        por_epoca[f"CIE-{int(e)}"] = {"positivas": int(len(p)), "pct_multietiqueta": _pct((p.n_naturalezas > 1).mean()),
                                      "conteo": {str(k): int(v) for k, v in p.labels.explode().value_counts().items()}}
    R["naturaleza"] = {"conteo": {str(k): int(v) for k, v in nat.items()},
                       "pct_sobre_positivas": {str(k): _pct(v / len(pos)) for k, v in nat.items()},
                       "pct_multietiqueta": _pct((pos.n_naturalezas > 1).mean()),
                       "clases_menores_de_200": [str(k) for k, v in nat.items() if v < 200],
                       "por_epoca": por_epoca,
                       "solo_en_una_epoca": [str(k) for k in nat.index if sum(k in v["conteo"] for v in por_epoca.values()) == 1]}
    log.info(f"7. naturalezas: {R['naturaleza']['conteo']} · multietiqueta {R['naturaleza']['pct_multietiqueta']:.2f} % · por época "
             + " · ".join(f"{k}: {v['conteo']}, multietiqueta {v['pct_multietiqueta']:.2f} %" for k, v in por_epoca.items())
             + f" · solo en una época: {R['naturaleza']['solo_en_una_epoca']}")

    # ---------- T. texto · A. variables · P. prevalencia real y drift · R. ruido de la etiqueta · B. eventos
    et.analizar_texto(df, R, log, FIG, pie)
    npp = df.groupby("subject_id").size()
    palabras = df.palabras.copy()
    df = df.drop(columns=["text"])
    ev.analizar_variables(df, R, log, FIG, pie)
    ev.prevalencia_real(df, R, log, FIG, pie)
    ev.ruido_etiqueta(df, R, log)
    ev.analizar_eventos(df, R, log, FIG, pie)
    P = R["prevalencia_real"]

    # ---------- 8. correlaciones (variable objetivo en la ÚLTIMA fila)
    num = pd.DataFrame({"edad": df.edad, "sexo (hombre)": (df.sexo == "Hombre").astype(float).where(df.sexo.notna()),
                        "estancia (días)": df.estancia_dias, "horas en emergencia": df.horas_emergencia,
                        "n.º de diagnósticos": df.n_diagnosticos, "severidad APR-DRG": df.drg_severidad, "paso por UCI": df.uci,
                        "fallecimiento": df.fallecio, "reingreso a 30 días": df.reingreso_30d, "época CIE": df.epoca,
                        "largo (caracteres)": df.largo, "notas del paciente": df.notas_paciente, "y (evento)": df.y})
    corr = num.corr(method="spearman")
    pares = sorted(((abs(corr.iloc[i, j]), corr.index[i], corr.columns[j], corr.iloc[i, j])
                    for i in range(len(corr) - 1) for j in range(i)), reverse=True)[:4]
    R["correlaciones_spearman"] = {"variables": list(corr.columns), "matriz": corr.round(4).values.tolist(),
                                   "con_y": {c: round(float(corr.loc["y (evento)", c]), 4) for c in corr.columns if c != "y (evento)"},
                                   "pares_mas_correlacionados": [{"a": a, "b": b, "rho": round(float(r), 4)} for _, a, b, r in pares],
                                   "nota": "cada celda usa las epicrisis que tienen los dos datos (entre "
                                           f"{int(num.notna().all(axis=1).sum())} y {len(num)} según el par)"}
    log.info(f"8. Spearman con y: {R['correlaciones_spearman']['con_y']} · pares más correlacionados "
             f"{R['correlaciones_spearman']['pares_mas_correlacionados']}")

    # ---------- figuras
    med = R["largo"]["mediana_por_clase"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
    for y, nombre, c in [(0, "sin evento", GRIS), (1, "con evento", ROJO)]:
        ax[0].hist(np.log10(df.loc[df.y == y, "largo"]), bins=60, alpha=.6, density=True, label=nombre, color=c)
    ax[0].set(xlabel="log10(caracteres)", ylabel="densidad"); ax[0].set_title(f"Largo de la epicrisis (AUC del largo solo = {auc_largo:.3f})", fontsize=10)
    ax[0].legend()
    ax[1].bar(range(1, 11), pct_decil, color=AZUL)
    ax[1].axhline(100 * df.y.mean(), ls="--", c="gray", label=f"promedio {100 * df.y.mean():.1f} %")
    ax[1].set(xlabel="decil de largo (1 = más corto)", ylabel="% de notas con evento", ylim=(0, 100))
    ax[1].set_title("Notas con evento por decil de largo", fontsize=10); ax[1].legend()
    pie(fig, "Cómo leer: a la izquierda, la distribución del largo de las epicrisis en escala logarítmica; en gris las notas sin "
             "evento y en rojo las notas con evento. A la derecha, el porcentaje de notas con evento en cada decil de largo, con "
             "el promedio en línea punteada. Lectura: las notas con evento son más largas (mediana de "
             f"{_mil(med['positiva'])} frente a {_mil(med['negativa'])} caracteres; AUC del largo solo = {auc_largo:.3f}) y el "
             f"porcentaje de positivas sube de {pct_decil[0]:.1f} % en el decil más corto a {pct_decil[-1]:.1f} % en el más largo: riesgo "
             "moderado de que el modelo use el largo como atajo.", 175, 0.17)
    fig.savefig(FIG / "fig_eda_largo.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    ax[0, 0].hist(palabras, bins=60, color=AZUL); ax[0, 0].set(xlabel="palabras", ylabel="notas")
    ax[0, 0].set_title("Palabras por epicrisis", fontsize=10)
    ax[0, 1].bar(*np.unique(np.clip(npp, 1, 6), return_counts=True), color=AZUL)
    ax[0, 1].set(xlabel="notas", ylabel="pacientes"); ax[0, 1].set_title("Notas por paciente (6 = 6 o más)", fontsize=10)
    bal = pd.crosstab(df.split, df.y, normalize="index").mul(100)
    bal.plot(kind="bar", stacked=True, ax=ax[1, 0], color=[GRIS, ROJO], rot=0)
    ax[1, 0].set(ylabel="%", xlabel="partición", ylim=(0, 125)); ax[1, 0].set_title("Balance de la variable objetivo por partición", fontsize=10)
    ax[1, 0].legend(["sin evento", "con evento"], loc="upper right", ncol=2, fontsize=8)
    ax[1, 1].barh([NOMBRE_NAT.get(k, k) for k in R["naturaleza"]["conteo"].keys()][::-1], list(R["naturaleza"]["conteo"].values())[::-1], color=VERDE)
    ax[1, 1].set(xlabel="notas"); ax[1, 1].set_title("Naturaleza del evento (positivas)", fontsize=10)
    menor = R["naturaleza"]["clases_menores_de_200"]
    txt_menor = (f"{NOMBRE_NAT.get(menor[0], menor[0])} ({_mil(R['naturaleza']['conteo'][menor[0]])} notas) es demasiado pequeña para evaluarse."
                 if menor else "todas las naturalezas tienen 200 notas o más.")
    pie(fig, "Cómo leer: arriba a la izquierda, cuántas palabras tiene cada epicrisis (mediana de "
             f"{_mil(R['descriptivas']['palabras']['50%'])}). Arriba a la derecha, cuántas notas aporta cada paciente: el "
             f"{100 * float((npp == 1).mean()):.0f} % aporta una sola. Abajo a la izquierda, la proporción de notas con y sin evento "
             f"en train y en test ({R['balance']['train']['pct_positivas']:.1f} % y {R['balance']['test']['pct_positivas']:.1f} % "
             "de positivas): la partición por paciente conserva el mismo balance. Abajo a la derecha, cuántas notas tiene cada "
             f"naturaleza del evento; {txt_menor}", 175, 0.09)
    fig.savefig(FIG / "fig_eda_distribuciones.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.1))
    epc = pd.crosstab(df.y, df.epoca, normalize="index").mul(100)
    epc.index = ["sin evento", "con evento"]; epc.columns = [f"CIE-{int(c)}" for c in epc.columns]
    epc.plot(kind="bar", stacked=True, ax=ax[0], rot=0, color=[GRIS, AZUL])
    ax[0].set(ylabel="% de la clase", ylim=(0, 125)); ax[0].set_title("Época de codificación en cada clase del corpus", fontsize=10)
    ax[0].legend(loc="upper right", ncol=2, fontsize=8)
    ppe, epocas = R["epoca"]["pct_positivas_por_epoca"], list(P["por_epoca"].keys())
    xx = np.arange(len(epocas))
    v_c, v_r = [ppe[e] for e in epocas], [P["por_epoca"][e]["pct_positivas_con_epicrisis"] for e in epocas]
    ax[1].bar(xx - 0.2, v_c, 0.4, color=GRIS, label="corpus de modelado")
    ax[1].bar(xx + 0.2, v_r, 0.4, color=ROJO, label="MIMIC-IV (hospitalizaciones con epicrisis)")
    for i in range(len(epocas)):
        ax[1].text(i - 0.2, v_c[i] + 1, f"{v_c[i]:.1f} %", ha="center", fontsize=8); ax[1].text(i + 0.2, v_r[i] + 1, f"{v_r[i]:.1f} %", ha="center", fontsize=8)
    ax[1].set_xticks(xx, epocas); ax[1].set(ylabel="% con evento", ylim=(0, 80)); ax[1].set_title("Hospitalizaciones con evento por época", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")
    pc = R["epoca"]["pct_por_clase"]["positiva"]
    pie(fig, "Cómo leer: a la izquierda, qué parte de cada clase del corpus se codificó en CIE-9 y qué parte en CIE-10; las dos barras "
             f"son iguales ({pc['CIE-9']:.1f} % y {pc['CIE-10']:.1f} %) porque las negativas se muestrean por época. A la derecha, el "
             "porcentaje de hospitalizaciones con evento en cada época: en gris, en el corpus; en rojo, en todas las hospitalizaciones de "
             f"MIMIC-IV con epicrisis. Lectura: en el corpus la época no separa las clases (AUC de la época sola = {auc_epoca:.3f}), pero "
             f"en la base la proporción real de eventos es de {v_r[0]:.1f} % en CIE-9 y de {v_r[-1]:.1f} % en CIE-10: la etiqueta no se "
             "define igual en las dos épocas, y el muestreo iguala la proporción, no la definición.", 165, 0.2)
    fig.savefig(FIG / "fig_eda_balance_epoca.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 10.9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right"); ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = round(float(corr.iloc[i, j]), 3)
            ax.text(j, i, f"{abs(v) if v == 0 else v:.3f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Correlación de Spearman entre variables (variable objetivo en la última fila)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    cy = R["correlaciones_spearman"]["con_y"]
    top = sorted(cy, key=lambda k: -abs(cy[k]))[:3]
    _, a0, b0, r0 = pares[0]
    pie(fig, "Cómo leer: cada celda es la correlación de Spearman entre dos variables, de -1 a 1 (rojo, positiva; azul, negativa; "
             "blanco, sin relación), calculada con las epicrisis que tienen los dos datos. La última fila es la variable objetivo. "
             "Lectura: las variables más asociadas con el evento son " + ", ".join(f"{k} ({cy[k]:.3f})" for k in top) + ". El número "
             "de diagnósticos y la severidad APR-DRG se calculan con los mismos códigos de los que sale la etiqueta, así que su "
             f"correlación es en parte por construcción. Entre variables, la correlación más alta es la de {a0} con {b0} ({r0:.3f}), y "
             f"la estancia y el largo de la nota se mueven juntos ({corr.loc['estancia (días)', 'largo (caracteres)']:.3f}): una "
             "hospitalización larga produce una nota larga.", 150, 0.13)
    fig.savefig(FIG / "fig_eda_correlaciones.png", dpi=130); plt.close(fig)
    log.info("figuras: fig_eda_largo.png, fig_eda_distribuciones.png, fig_eda_balance_epoca.png, fig_eda_correlaciones.png")

    # ---------- 9. riesgos y decisiones
    V, E, N = R["variables"], R["eventos"], R["ruido_etiqueta"]
    e9, e10 = P["por_epoca"]["CIE-9"]["pct_positivas_con_epicrisis"], P["por_epoca"]["CIE-10"]["pct_positivas_con_epicrisis"]
    R["riesgos"] = {
        "desbalance": f"corpus de modelado con {R['balance']['All']['pct_positivas']:.1f} % de positivas; prevalencia real de "
                      f"{P['todas']['pct_positivas_con_epicrisis']:.1f} % entre las hospitalizaciones con epicrisis "
                      f"({P['todas']['pct_positivas']:.1f} % sobre todas las hospitalizaciones)",
        "leakage_etiqueta": f"códigos de las familias de la etiqueta escritos en {FAM['notas']} notas ({FAM['pct_positivas']:.2f} % de las "
                            f"positivas, {FAM['pct_negativas']:.2f} % de las negativas)",
        "leakage_paciente": f"{len(comp)} pacientes compartidos entre train y test",
        "drift_de_la_etiqueta": f"prevalencia real de {e9:.1f} % en la época CIE-9 y de {e10:.1f} % en la CIE-10; en el corpus el muestreo "
                                "por época la iguala",
        "atajo_largo": f"AUC del largo solo {auc_largo:.3f}; positivas por decil {pct_decil[0]:.1f} % → {pct_decil[-1]:.1f} %",
        "clases_raras_etapa2": R["naturaleza"]["clases_menores_de_200"],
        "naturalezas_solo_en_una_epoca": R["naturaleza"]["solo_en_una_epoca"],
        "etiqueta_debil": "la etiqueta sale de códigos de diagnóstico de facturación, no de una lectura clínica de la nota",
        "ruido_en_negativas": f"{N['negativas']['cie10']['pct_con_alguno']:.1f} % de las negativas CIE-10 llevan un código de las "
                              "familias de causa explícita que no está en la tabla",
        "ruido_en_positivas": f"{N['caidas']['pct_de_las_positivas']:.1f} % de las positivas lo son solo por un código de caída; en CIE-10 "
                              f"el {N['caidas']['cie10']['pct_en_hospital']:.1f} % de ellas tiene el hospital como lugar de ocurrencia; "
                              f"{N['ulcera_cie9']['sin_codigo_707_0']} positivas de la familia CIE-9 de úlcera no tienen un código de "
                              "úlcera por presión (707.0x)",
        "variables_contaminadas": f"el número de diagnósticos (AUC {V['numericas']['n_diagnosticos']['auc_sola']:.3f}) y la severidad "
                                  f"APR-DRG (AUC {V['numericas']['drg_severidad']['auc_sola']:.3f}) salen de la misma codificación que "
                                  "la etiqueta: no se usan como predictores",
        "faltantes_no_aleatorios": "; ".join(f"{d['variable']} falta en {d['pct_faltantes']:.1f} % ({d['pct_positivas_si_falta']:.1f} % de "
                                             f"positivas cuando falta, {d['pct_positivas_si_no_falta']:.1f} % cuando no)"
                                             for d in V["diccionario"] if d["pct_faltantes"] > 5 and d["variable"] != "alta_voluntaria"),
        "diferencias_por_subgrupo": f"V de Cramér de la etiqueta con el servicio {V['categoricas']['servicio']['cramers_v']:.3f} y con el "
                                    f"tipo de ingreso {V['categoricas']['tipo_ingreso']['cramers_v']:.3f}",
        "mezcla_de_casos": f"un modelo sin texto alcanza ROC-AUC {V['contexto']['al_ingreso']['roc_auc']:.3f} con variables del ingreso "
                           f"y {V['contexto']['al_alta']['roc_auc']:.3f} con las del alta",
        "cobertura_del_catalogo": f"{E['mapeo']['eventos_con_codigo']} de {E['catalogo']['eventos']} eventos con código; "
                                  f"{E['mapeo']['filas_sin_enlace']} de {E['mapeo']['filas']} filas de la tabla sin enlace con el catálogo",
        "origen_de_la_etiqueta": f"de las positivas del corpus, el {E['corpus']['pct_con_evento_del_catalogo']:.1f} % tiene un evento del "
                                 f"catálogo, el {E['corpus']['pct_solo_complicacion_sin_enlace']:.1f} % una complicación sin enlace y el "
                                 f"{E['corpus']['pct_solo_familia_cie9']:.1f} % solo familia CIE-9",
    }
    DECISIONES, ACCIONES = decisiones(R), acciones(R)
    R["decisiones"] = DECISIONES
    R["acciones_registradas"] = ACCIONES
    ev.figura_flujo(R, FIG / "fig_eda_flujo.png", FUENTES_DECISION)
    log.info("figura del flujo del EDA: fig_eda_flujo.png")
    R["duracion_s"] = round(time.time() - t0, 1)
    guardar_json(R, REPORTES / "eda_resumen.json")

    # ---------- metrics_eda.txt (resumen legible)
    T, M, O, C = R["texto"], E["mapeo"], E["observado"], E["corpus"]
    L = [f"# metrics_eda.txt · generado {datetime.now():%Y-%m-%d %H:%M:%S} por src/eda.py (fuente: reportes/eda_resumen.json)",
         f"entrada: {len(df):,} notas · {df.subject_id.nunique():,} pacientes · {ruta.stat().st_size/1e6:.0f} MB · "
         f"semilla {semilla} · split por paciente (sha256 splits.csv {R['entrada']['sha256_splits'][:16]}…)",
         f"alcance: {R['alcance']}", "",
         f"calidad: nulos {R['calidad']['nulos']['columnas_sin_labels']} en las columnas del corpus "
         f"(labels None en {R['calidad']['nulos']['labels_none']} negativas: esperado, no tienen naturaleza) · duplicados note_id/hadm_id/texto "
         f"{R['calidad']['duplicados']['note_id']}/{R['calidad']['duplicados']['hadm_id']}/{R['calidad']['duplicados']['texto_identico']} · "
         f"largo {R['calidad']['rango_largo'][0]}-{R['calidad']['rango_largo'][1]} car. · atípicos por largo {R['calidad']['outliers_largo']['n']} "
         f"({R['calidad']['outliers_largo']['pct']:.2f} %, se conservan)", "",
         "descriptivas (media / mediana / desv):"]
    for k, v in R["descriptivas"].items():
        L.append(f"  {k:<26} {v['mean']:>10} / {v['50%']:>10} / {v['std']:>10}")
    L += ["", "balance de la variable objetivo:"]
    for k in ["train", "test", "All"]:
        v = R["balance"][k]
        L.append(f"  {k:<6} notas {v['total']:>7,} · positivas {v['pct_positivas']:>6.2f} %")
    L += ["", "prevalencia real (hospitalizaciones / % con evento / con epicrisis / % con evento entre las que tienen epicrisis):"]
    for k, v in [("todas", P["todas"])] + list(P["por_epoca"].items()) + list(P["por_periodo"].items()):
        L.append(f"  {k:<10} {v['hospitalizaciones']:>8,} / {v['pct_positivas']:>6.2f} / {v['con_epicrisis']:>8,} / {v['pct_positivas_con_epicrisis']:>6.2f}")
    L += ["  dentro de cada periodo, por época (solo con epicrisis): "
          + " · ".join(f"{k}: " + ", ".join(f"{e} {x['pct_positivas']:.2f} %" for e, x in v.items())
                       for k, v in P["por_periodo_y_epoca_con_epicrisis"].items() if v),
          f"  % con epicrisis de cada fuente de positivas: {P['pct_con_epicrisis_por_fuente_de_positivas']}",
          f"  CIE-10 con epicrisis, si dieran etiqueta las familias completas: {P['cie10_con_familias_completas']}", "",
          f"época por clase (corpus): {R['epoca']['pct_por_clase']} · AUC de la época sola {auc_epoca:.3f}",
          f"largo por clase: medianas {med} · AUC del largo solo {auc_largo:.3f} · positivas por decil "
          f"{[round(x, 1) for x in pct_decil]}",
          f"códigos en el texto: patrón general {G['notas']} notas ({G['pct_positivas']:.2f} % de positivas, {G['pct_negativas']:.2f} % de "
          f"negativas); {G['pct_coincidencias_T95_a_T99']:.1f} % de las {G['coincidencias']} coincidencias son T95 a T99 "
          f"{G['cadenas_mas_frecuentes']}; códigos T95 a T99 en el diccionario de MIMIC: {G['codigos_T95_a_T99_en_el_diccionario_de_mimic']}",
          f"códigos de las familias de la etiqueta en el texto: {FAM['notas']} notas ({FAM['positivas']} positivas, {FAM['negativas']} negativas) · "
          f"con forma CIE-10 {FAM['codigos_cie10']} · con forma CIE-9 {FAM['codigos_cie9']}",
          f"pacientes compartidos train/test: {len(comp)}",
          f"naturalezas (positivas): {R['naturaleza']['conteo']} · multietiqueta {R['naturaleza']['pct_multietiqueta']:.2f} %"]
    for k, v in R["naturaleza"]["por_epoca"].items():
        L.append(f"  {k}: positivas {v['positivas']:,} · multietiqueta {v['pct_multietiqueta']:.2f} % · {v['conteo']}")
    L += [f"Spearman con y: {R['correlaciones_spearman']['con_y']}",
          f"pares de variables más correlacionados: {R['correlaciones_spearman']['pares_mas_correlacionados']}", "",
          "contenido del texto:",
          f"  secciones: {T['secciones']['presentes_en_mas_del_95_pct']} de {len(T['secciones']['por_seccion'])} en más del 95 % de las "
          f"notas · notas con todas {T['secciones']['pct_notas_con_todas']:.2f} % · AUC del número de secciones "
          f"{T['secciones']['auc_numero_de_secciones']:.3f}",
          "  sección (% del corpus / % negativas / % positivas):"]
    for k, v in T["secciones"]["por_seccion"].items():
        L.append(f"    {k:<38} {v['pct']:>6.2f} / {v['pct_negativas']:>6.2f} / {v['pct_positivas']:>6.2f}")
    L += [f"  vocabulario (solo train): {T['vocabulario']['terminos']:,} términos en al menos el 1 % de las notas",
          "  más asociados al evento (% positivas / % negativas): "
          + " · ".join(f"{t['termino']} {t['pct_positivas']}/{t['pct_negativas']}" for t in T["vocabulario"]["mas_asociados_al_evento"]),
          "  más asociados a sin evento (% positivas / % negativas): "
          + " · ".join(f"{t['termino']} {t['pct_positivas']}/{t['pct_negativas']}" for t in T["vocabulario"]["mas_asociados_a_sin_evento"]),
          "", f"variables de la hospitalización ({len(V['diccionario'])} variables; {V['variables_de_tablas_de_mimic']} de tablas de MIMIC-IV):",
          "  variable                 tipo        faltantes %   momento            uso"]
    for d in V["diccionario"]:
        L.append(f"  {d['variable']:<24} {d['tipo']:<11} {d['pct_faltantes']:>10.2f}   {d['momento']:<18} {d['uso']}")
    L += ["", "faltantes según el tipo de ingreso (% sin dato):"]
    for k, t_ in V["faltantes_por_tipo_ingreso"].items():
        L.append(f"  {k:<18} " + " · ".join(f"{a} {b:.1f}" for a, b in sorted(t_.items(), key=lambda kv: -kv[1])))
    L += ["faltantes y etiqueta (% de positivas cuando falta / cuando no falta): "
          + " · ".join(f"{d['variable']} {d['pct_positivas_si_falta']:.2f}/{d['pct_positivas_si_no_falta']:.2f}"
                       for d in V["diccionario"] if "pct_positivas_si_falta" in d),
          f"fuera de rango: estancias negativas {V['fuera_de_rango']['estancia_negativa']} (con fallecimiento "
          f"{V['fuera_de_rango']['estancia_negativa_con_fallecimiento']}) · edad de {V['fuera_de_rango']['edad_min_max'][0]} a "
          f"{V['fuera_de_rango']['edad_min_max'][1]} años",
          "", "numéricas (media / desv / mín / q1 / mediana / q3 / máx / % atípicos IQR):"]
    for k, v in V["numericas"].items():
        L.append(f"  {k:<18} {v['media']:>9.2f} / {v['desv']:>9.2f} / {v['min']:>7.2f} / {v['q1']:>8.2f} / {v['mediana']:>8.2f} / "
                 f"{v['q3']:>8.2f} / {v['max']:>9.2f} / {v['atipicos_iqr']['pct']:>5.2f}")
    L += ["", "numéricas por clase (mediana negativa / mediana positiva / AUC de la variable sola / Spearman con y):"]
    for k, v in V["numericas"].items():
        L.append(f"  {k:<18} {v['negativa']['mediana']:>10.2f} / {v['positiva']['mediana']:>10.2f} / {v['auc_sola']:>6.3f} / {v['spearman_con_y']:>6.3f}")
    L += ["", "categóricas (V de Cramér con la etiqueta; «Sin dato» cuenta como categoría):",
          "  " + " · ".join(f"{k} {v['cramers_v']:.3f}" for k, v in sorted(V["categoricas"].items(), key=lambda kv: -kv[1]["cramers_v"])),
          "  % de positivas por categoría (n):"]
    for k in ["servicio", "tipo_ingreso", "destino_alta", "grupo_edad", "sexo", "raza", "seguro", "uci", "fallecio", "reingreso_30d"]:
        L.append(f"    {k}: " + " · ".join(f"{c} {x['pct_positivas']:.1f} ({x['n']})" for c, x in V["categoricas"][k]["categorias"].items()))
    L += ["  pares de variables categóricas más asociados: "
          + " · ".join(f"{p['a']}-{p['b']} {p['cramers_v']:.3f}" for p in V["asociacion_categoricas"]["pares_mas_asociados"]),
          "  tipo de ingreso por época (% de las epicrisis de la época, CIE-9 / CIE-10): "
          + " · ".join(f"{k} {v['CIE-9']:.1f}/{v['CIE-10']:.1f}" for k, v in V["tipo_ingreso_por_epoca"].items()),
          "", "train frente a test:"]
    for s in ("train", "test"):
        L.append(f"  {s:<6} {V['particion'][s]}")
    L += ["", "drift por periodo de atención en el corpus (n / % positivas / % CIE-10 / mediana del largo):"]
    for k, v in V["drift_periodo"].items():
        L.append(f"  {k:<12} {v['n']:>7,} / {v['pct_positivas']:>6.2f} / {v['pct_cie10']:>6.2f} / {v['mediana_largo']:>8.1f}")
    ctx = V["contexto"]
    L += ["", "modelo de contexto sin texto (ajustado en train, medido en test; IC 95 % por paciente):",
          f"  al ingreso: ROC-AUC {ctx['al_ingreso']['roc_auc']} ({ctx['al_ingreso']['roc_auc_ic95'][0]:.3f} a "
          f"{ctx['al_ingreso']['roc_auc_ic95'][1]:.3f}) · PR-AUC {ctx['al_ingreso']['pr_auc']} ({ctx['al_ingreso']['pr_auc_ic95'][0]:.3f} a "
          f"{ctx['al_ingreso']['pr_auc_ic95'][1]:.3f}) · variables {ctx['al_ingreso']['variables']}",
          f"  al alta:    ROC-AUC {ctx['al_alta']['roc_auc']} ({ctx['al_alta']['roc_auc_ic95'][0]:.3f} a "
          f"{ctx['al_alta']['roc_auc_ic95'][1]:.3f}) · PR-AUC {ctx['al_alta']['pr_auc']} ({ctx['al_alta']['pr_auc_ic95'][0]:.3f} a "
          f"{ctx['al_alta']['pr_auc_ic95'][1]:.3f}) · variables {ctx['al_alta']['variables']}",
          f"  modelo de texto del baseline: {ctx.get('modelo_de_texto', 'sin baseline de la misma partición')}",
          "", "ruido de la etiqueta:",
          f"  negativas CIE-10 con código de las familias de causa explícita: {N['negativas']['cie10']['con_alguno']} de "
          f"{N['negativas']['cie10']['n']} ({N['negativas']['cie10']['pct_con_alguno']:.2f} %; familia causal "
          f"{N['negativas']['cie10']['con_codigo_de_familia_causal']}, efecto adverso de fármaco "
          f"{N['negativas']['cie10']['con_efecto_adverso_de_farmaco']}) · negativas CIE-9: {N['negativas']['cie9']['con_codigo_de_familia_causal']} "
          f"de {N['negativas']['cie9']['n']}",
          "  códigos más frecuentes en esas negativas: "
          + " · ".join(f"{t_['codigo']} {t_['epicrisis']} ({t_['titulo_en_mimic'][:60]})" for t_ in N["negativas"]["cie10"]["codigos_mas_frecuentes"]),
          f"  positivas solo por caída: {N['caidas']['positivas_solo_por_caida']} ({N['caidas']['pct_de_las_positivas']:.2f} %) · CIE-10 "
          f"{N['caidas']['cie10']} · CIE-9 {N['caidas']['cie9']}",
          f"  % de las positivas de cada servicio que lo son solo por caída: {N['caidas']['pct_de_las_positivas_del_servicio']}",
          f"  familia CIE-9 de úlcera (707.x): {N['ulcera_cie9']}",
          "", "eventos del Anexo 02:",
          f"  catálogo: {E['catalogo']['eventos']} eventos en {E['catalogo']['naturalezas']} naturalezas · detectables "
          f"{E['catalogo']['detectables']} · por nivel de detección {E['catalogo']['por_nivel_de_deteccion']}",
          f"  tabla de códigos: {M['filas']} filas · {M['codigos_cie10_distintos']} códigos CIE-10 · por fuente {M['por_fuente']}",
          f"  filas enlazadas al catálogo: {M['filas_enlazadas_al_catalogo']} {M['filas_enlazadas_por_fuente']} · sin enlace: "
          f"{M['filas_sin_enlace']} {M['filas_sin_enlace_detalle']}",
          f"  eventos con código: {M['eventos_con_codigo']} de {E['catalogo']['eventos']} (por la tabla {M['eventos_con_codigo_por_la_tabla']}, "
          f"solo por equivalencias {M['eventos_con_codigo_solo_por_equivalencias']}) · tipo de código {M['eventos_por_tipo_de_codigo']}",
          f"  con código por nivel de detección: {M['con_codigo_por_nivel_de_deteccion']} · del nivel A sin código "
          f"{M['nivel_A_sin_codigo']}, de ellos con su código en filas sin enlace {M['nivel_A_sin_codigo_con_su_codigo_en_filas_sin_enlace']}",
          f"  observado: {O['eventos_observados']} eventos en {O['hospitalizaciones_con_codigo_de_evento']:,} hospitalizaciones (con o sin "
          f"epicrisis) · del catálogo {O['eventos_del_catalogo_observados']}: con código de causa explícita "
          f"{O['eventos_del_catalogo_con_codigo_de_causa_explicita']}, solo de condición {O['eventos_del_catalogo_solo_con_codigo_de_condicion']} · "
          f"{O['eventos_que_concentran_el_80_pct']} eventos concentran el 80 %",
          f"  causa explícita (da la etiqueta): {O['causa_explicita']}",
          f"  condición (no da etiqueta): {O['condicion']}",
          "  por naturaleza (en el catálogo / con código / observados / con código de causa explícita / en las positivas del corpus):"]
    for k, v in E["por_naturaleza"].items():
        L.append(f"    {k:<24} {v['en_el_catalogo']:>3} / {v['con_codigo']:>3} / {v['observados']:>3} / {v['con_codigo_de_causa_explicita']:>3} / "
                 f"{v['en_las_positivas_del_corpus']:>3}")
    L += [f"  corpus: positivas {C['positivas']:,} · con evento del catálogo {C['con_evento_del_catalogo']:,} "
          f"({C['pct_con_evento_del_catalogo']:.2f} %) · con complicación sin enlace {C['solo_complicacion_sin_enlace']:,} "
          f"({C['pct_solo_complicacion_sin_enlace']:.2f} %) · solo con familia CIE-9 {C['solo_familia_cie9']:,} "
          f"({C['pct_solo_familia_cie9']:.2f} %) {C['familias_cie9']}",
          f"  eventos del catálogo en las positivas del corpus: {C['eventos_del_catalogo']}",
          "  una fila por evento: reportes/eda_eventos_231.csv", "",
          "riesgos:"] + [f"  - {k}: {v}" for k, v in R["riesgos"].items()] + ["", "decisiones accionables para el siguiente sprint:"]
    for d in DECISIONES:
        L.append(f"  {d['n']}. {d['decision']} [{d['paso']}] — porque {d['evidencia']} — control: {d['control']}")
    L += ["", "otras acciones registradas:"] + [f"  - {a}" for a in ACCIONES]
    L += ["", f"duración: {R['duracion_s']} s"]
    (LOGS / "metrics_eda.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info(f"9. decisiones: {len(DECISIONES)} · resumen -> reportes/eda_resumen.json · logs/metrics_eda.txt · {R['duracion_s']} s")


if __name__ == "__main__":
    main()
