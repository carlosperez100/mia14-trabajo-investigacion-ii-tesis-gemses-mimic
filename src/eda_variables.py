# -*- coding: utf-8 -*-
"""
EDA de las variables de cada hospitalización, de la etiqueta y de los eventos del Anexo 02
(lo llama src/eda.py).

Bloque A · Variables clínicas y administrativas (una fila por epicrisis del corpus de modelado)
  Se unen por hospitalización las tablas de MIMIC-IV: admissions, patients, diagnoses_icd,
  drgcodes, services e icustays. Para cada variable: diccionario (tipo, fuente, momento en que se
  conoce y uso permitido), datos faltantes, estadísticas y valores atípicos, distribución por
  clase, asociación con la etiqueta y entre variables, diferencias por subgrupo, comparación de
  train y test, y cuánto se acierta solo con estas variables, sin leer el texto.

Bloque P · Prevalencia real y drift
  Proporción de hospitalizaciones con evento en toda la base, por época de codificación y por
  periodo, frente a la proporción del corpus de modelado (que está fijada por el muestreo).

Bloque R · Ruido de la etiqueta
  Negativas que llevan un código de las familias de causa explícita y positivas que lo son solo
  por un código de caída, con el lugar donde ocurrió la caída.

Bloque B · Eventos del Anexo 02 (Directiva N.º 7-OGCyH-ESSALUD-2020)
  Del catálogo de 231 eventos a los datos: cuántos tienen código CIE-10 en las tablas del
  proyecto, cuántos se observan en MIMIC-IV y con qué evidencia, cuáles están en las epicrisis
  positivas del corpus, y una fila por evento en reportes/eda_eventos_231.csv.

Regla de datos: MIMIC-IV está bajo DUA de PhysioNet. Aquí solo se escriben AGREGADOS; las
categorías con menos de MIN_CELDA casos se agrupan en «Otros» y, si aun así no llegan a
MIN_CELDA, no se publican. En la tabla por evento los conteos menores de 10 se escriben «<10».
"""
import json
import os
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import INTERIM, RAIZ, REPORTES, sha256, sha256_cacheado

MIN_CELDA = 50
N_BOOTSTRAP = 500
CATALOGOS = RAIZ / "data" / "catalogos"
AZUL, ROJO, VERDE, AMBAR, GRIS = "#1F4E79", "#9B2226", "#1E7B4F", "#B35C00", "#9FB8CF"
PERIODOS = ["2008-2010", "2011-2013", "2014-2016", "2017-2019", "2020-2022"]

# Familias CIE-10 de causa explícita: complicaciones de la atención (T80 a T88), incidentes en la asistencia (Y63, Y65),
# procedimientos como causa de una reacción (Y83, Y84), caídas (W00 a W19) y úlcera por presión (L89). De estas familias, la
# etiqueta del proyecto solo usa los códigos que están en su tabla; aquí se revisa la familia completa.
PREF_CAUSAL = tuple([f"T8{i}" for i in range(9)] + ["Y63", "Y65", "Y83", "Y84"] + [f"W{i:02d}" for i in range(20)] + ["L89"])
# Familias equivalentes en CIE-9 (las del pipeline de etiquetas)
FAM_CIE9 = {"complicaciones_atencion": r"^99[6-9]", "incidentes_asistenciales": r"^E87[0-9]",
            "farmacos_uso_terapeutico": r"^E9[34][0-9]", "caidas": r"^E88[0-8]", "ulcera_presion": r"^707[0-9]"}

# nombre, etiqueta, tipo, tabla de origen, momento en que se conoce, uso en el proyecto
DICCIONARIO = [
    ("edad", "Edad al ingreso (años)", "numérica", "patients, admissions", "ingreso", "descripción y sesgo"),
    ("sexo", "Sexo", "categórica", "patients", "ingreso", "descripción y sesgo"),
    ("raza", "Raza o etnia (agrupada)", "categórica", "admissions", "ingreso", "descripción y sesgo"),
    ("seguro", "Tipo de seguro", "categórica", "admissions", "ingreso", "descripción y sesgo"),
    ("idioma_ingles", "Idioma inglés", "binaria", "admissions", "ingreso", "descripción y sesgo"),
    ("estado_civil", "Estado civil", "categórica", "admissions", "ingreso", "descripción"),
    ("tipo_ingreso", "Tipo de ingreso", "categórica", "admissions", "ingreso", "control"),
    ("servicio", "Servicio al ingreso", "categórica", "services", "ingreso", "control y evaluación por subgrupo"),
    ("horas_emergencia", "Horas en emergencia", "numérica", "admissions", "ingreso", "control"),
    ("periodo", "Periodo de atención (año aproximado)", "categórica", "patients, admissions", "ingreso", "drift temporal"),
    ("uci", "Paso por UCI", "binaria", "icustays", "estancia", "control (puede ser consecuencia del evento)"),
    ("estancia_dias", "Estancia hospitalaria (días)", "numérica", "admissions", "alta",
     "proxy de la dimensión Tiempo de GEMSES; no es predictor"),
    ("destino_alta", "Destino al alta", "categórica", "admissions", "alta", "descripción"),
    ("fallecio", "Fallecimiento hospitalario", "binaria", "admissions", "alta", "desenlace; no es predictor"),
    ("alta_voluntaria", "Alta contra indicación médica", "binaria", "admissions (del destino al alta)", "alta",
     "proxy de la dimensión Satisfacción de GEMSES"),
    ("reingreso_30d", "Reingreso dentro de 30 días", "binaria", "admissions", "posterior al alta",
     "proxy de la dimensión Satisfacción de GEMSES"),
    ("n_diagnosticos", "Número de diagnósticos codificados", "numérica", "diagnoses_icd", "codificación",
     "contaminada por la etiqueta; no es predictor"),
    ("drg_severidad", "Severidad APR-DRG (1 a 4)", "numérica", "drgcodes", "codificación",
     "insumo del proxy de la dimensión Costos de GEMSES (DRG); contaminada por la etiqueta"),
    ("largo", "Largo de la epicrisis (caracteres)", "numérica", "discharge (nota)", "alta", "control del atajo por largo"),
    ("notas_paciente", "Epicrisis del mismo paciente en el corpus", "numérica", "discharge (nota)", "corpus", "control"),
    ("epoca", "Época de codificación (CIE-9 o CIE-10)", "categórica", "diagnoses_icd", "codificación", "confusor"),
]
NUMERICAS = ["edad", "estancia_dias", "horas_emergencia", "n_diagnosticos", "drg_severidad", "largo", "notas_paciente"]
CATEGORICAS = ["sexo", "grupo_edad", "raza", "seguro", "estado_civil", "tipo_ingreso", "servicio", "destino_alta", "periodo"]
BINARIAS = ["idioma_ingles", "uci", "fallecio", "alta_voluntaria", "reingreso_30d"]
ETIQUETA = {n: e for n, e, *_ in DICCIONARIO}
ETIQUETA["grupo_edad"] = "Grupo de edad"
CORTA = {"edad": "Edad", "estancia_dias": "Estancia", "horas_emergencia": "Horas en emergencia", "n_diagnosticos": "N.º de diagnósticos",
         "drg_severidad": "Severidad APR-DRG", "largo": "Largo de la nota", "notas_paciente": "Notas del paciente",
         "destino_alta": "Destino al alta", "alta_voluntaria": "Alta contra indicación", "sexo": "Sexo", "grupo_edad": "Grupo de edad",
         "raza": "Raza o etnia", "seguro": "Seguro", "estado_civil": "Estado civil", "tipo_ingreso": "Tipo de ingreso",
         "servicio": "Servicio", "periodo": "Periodo", "uci": "Paso por UCI", "fallecio": "Fallecimiento",
         "reingreso_30d": "Reingreso a 30 días", "idioma_ingles": "Idioma inglés", "epoca": "Época CIE", "y": "y (evento)"}

# variables del modelo de contexto (sin texto): lo que se sabe al ingreso y lo que se sabe al alta
CONTEXTO_INGRESO = (["edad"], ["sexo", "raza", "seguro", "estado_civil", "tipo_ingreso", "servicio"])
CONTEXTO_ALTA = (["edad", "estancia_dias", "largo"], ["sexo", "raza", "seguro", "estado_civil", "tipo_ingreso", "servicio", "destino_alta"])

NAT_MAPEO = {"Procedimiento": "PROCEDIMIENTO", "Infeccion nosocomial": "INFECCIÓN ASOCIADA",
             "Dispositivo medico": "DISPOSITIVO MÉDICO", "Medicacion": "MEDICACIÓN",
             "Cuidado del paciente": "CUIDADO DEL PACIENTE", "Comportamiento": "COMPORTAMIENTO", "Diagnostico": "DIAGNÓSTICO",
             "Sangre/Hemoderivados": "SANGRE/HEMODERIVADOS", "Gestion Organizacion": "GESTIÓN ORGANIZACIÓN", "Nutricion": "NUTRICIÓN"}
NIVEL = {"A": "código CIE-10", "B": "texto de las notas", "C": "variables estructuradas",
         "D": "auditoría del registro", "E": "no detectable en MIMIC-IV"}
FUENTE_OMS = "CIE-10 OMS"
_AUX = {}       # tablas de MIMIC-IV que usan varios bloques (se cargan una vez)


def _mil(n):
    return f"{int(round(n)):,}".replace(",", " ")   # espacio que no se parte entre renglones


def _pct(x):
    """Porcentaje con seis decimales: se redondea una sola vez, al mostrarlo."""
    return round(100 * float(x), 6)


def _f3(x):
    """Tres decimales con redondeo convencional (0.5365 -> 0.537)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _min1(s):
    """Primera letra en minúscula sin tocar las siglas (APR-DRG)."""
    return s[:1].lower() + s[1:]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _ruta_mimic(rel):
    """Busca la tabla en MIMIC_DIR y, si no está, en MIMIC_DIR_ALT (opcional)."""
    for var in ("MIMIC_DIR", "MIMIC_DIR_ALT"):
        base = os.environ.get(var)
        if base and (Path(base) / rel).exists():
            return Path(base) / rel
    raise FileNotFoundError(f"No se encontró {rel} en MIMIC_DIR ni en MIMIC_DIR_ALT (ver .env.example)")


def _agrupar_raza(s):
    s = s.fillna("").str.upper()
    out = pd.Series("Otra", index=s.index, dtype=object)
    out[s.str.startswith("WHITE") | s.str.contains("PORTUGUESE")] = "Blanca"
    out[s.str.startswith("BLACK")] = "Negra"
    out[s.str.startswith("HISPANIC") | s.str.contains("SOUTH AMERICAN")] = "Hispana o latina"
    out[s.str.startswith("ASIAN")] = "Asiática"
    out[(s == "") | s.str.contains("UNKNOWN|UNABLE|DECLINED")] = np.nan      # no registrada: es un dato faltante
    return out


def _efecto_adverso(codigos):
    """Efecto adverso de un medicamento en CIE-10-CM: familias T36 a T50 con un 5 en el sexto carácter (T38.0X5A)."""
    return codigos.str[:3].between("T36", "T50") & (codigos.str.len() >= 6) & (codigos.str[5] == "5")


def _con_sin_dato(s):
    return s.astype(object).where(s.notna(), "Sin dato")


def _cramers_v(x, y):
    t = pd.crosstab(x, y)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def _periodo(anchor_year_group, anio_ingreso, anchor_year):
    """Las fechas de MIMIC-IV están desplazadas. El año real se aproxima con el punto medio del grupo de años del paciente
    (anchor_year_group) más los años transcurridos desde su año ancla: precisión de más o menos un año. Los años aproximados
    que caen fuera de 2008-2022 por ese margen se llevan al extremo del rango."""
    anio = anchor_year_group.str.slice(0, 4).astype(float) + 1 + (anio_ingreso - anchor_year)
    return anio, pd.cut(anio.clip(2008, 2022), [2007, 2010, 2013, 2016, 2019, 2022], labels=PERIODOS).astype(object)


# ------------------------------------------------------------------ carga
def cargar_variables(df, log, R):
    """Une a cada epicrisis del corpus las variables clínicas y administrativas de su hospitalización."""
    fuentes = {}

    def leer(rel, **kw):
        ruta = _ruta_mimic(rel)
        fuentes[rel] = {"tamano_bytes": ruta.stat().st_size, "sha256": sha256_cacheado(ruta)}
        return pd.read_csv(ruta, **kw)

    adm = leer("hosp/admissions.csv.gz",
               usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "discharge_location", "insurance",
                        "language", "marital_status", "race", "edregtime", "edouttime", "hospital_expire_flag"],
               parse_dates=["admittime", "dischtime", "edregtime", "edouttime"])
    adm = adm.sort_values(["subject_id", "admittime"])
    siguiente = adm.groupby("subject_id").admittime.shift(-1)
    dias = (siguiente - adm.dischtime).dt.total_seconds() / 86400
    adm["reingreso_30d"] = ((dias >= 0) & (dias <= 30)).astype(int)
    adm["estancia_dias"] = (adm.dischtime - adm.admittime).dt.total_seconds() / 86400
    adm["horas_emergencia"] = (adm.edouttime - adm.edregtime).dt.total_seconds() / 3600
    adm["anio_ingreso"] = adm.admittime.dt.year

    pac = leer("hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"])
    dx = leer("hosp/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    ndx = dx.hadm_id.value_counts().rename("n_diagnosticos")
    drg = leer("hosp/drgcodes.csv.gz", usecols=["hadm_id", "drg_type", "drg_severity"])
    drg = drg[drg.drg_type == "APR"].groupby("hadm_id").drg_severity.max().rename("drg_severidad")
    srv = leer("hosp/services.csv.gz", usecols=["hadm_id", "transfertime", "curr_service"])
    srv = srv.sort_values(["hadm_id", "transfertime"]).drop_duplicates("hadm_id").set_index("hadm_id").curr_service.rename("servicio")
    uci = set(leer("icu/icustays.csv.gz", usecols=["hadm_id"]).hadm_id)
    dic = leer("hosp/d_icd_diagnoses.csv.gz", dtype={"icd_code": str})
    d10 = dx[dx.icd_version == 10]
    _AUX.update(adm=adm[["hadm_id", "subject_id", "anio_ingreso"]], pac=pac[["subject_id", "anchor_year", "anchor_year_group"]],
                dx=dx[dx.hadm_id.isin(set(df.hadm_id))], titulos=dic.set_index(["icd_code", "icd_version"]).long_title,
                hadm_familias_cie10=set(d10.loc[d10.icd_code.str.startswith(PREF_CAUSAL) | _efecto_adverso(d10.icd_code), "hadm_id"]))

    n0 = len(df)
    df = df.merge(adm.drop(columns=["subject_id"]), on="hadm_id", how="left", validate="m:1")
    df = df.merge(pac, on="subject_id", how="left", validate="m:1")
    df = df.join(ndx, on="hadm_id").join(drg, on="hadm_id").join(srv, on="hadm_id")
    assert len(df) == n0, "la unión de variables no debe cambiar el número de filas"

    df["edad"] = df.anchor_age + (df.anio_ingreso - df.anchor_year)
    df["grupo_edad"] = pd.cut(df.edad, [0, 44, 64, 79, 200], labels=["18 a 44", "45 a 64", "65 a 79", "80 o más"]).astype(object)
    df["sexo"] = df.gender.map({"F": "Mujer", "M": "Hombre"})
    df["raza"] = _agrupar_raza(df.race)
    df["seguro"] = df.insurance
    df["idioma_ingles"] = np.where(df.language.isna(), np.nan, (df.language.str.upper() == "ENGLISH").astype(float))
    df["estado_civil"] = df.marital_status
    df["tipo_ingreso"] = df.admission_type
    df["destino_alta"] = df.discharge_location
    df["anio_aprox"], df["periodo"] = _periodo(df.anchor_year_group, df.anio_ingreso, df.anchor_year)
    df["uci"] = df.hadm_id.isin(uci).astype(int)
    df["fallecio"] = df.hospital_expire_flag
    df["alta_voluntaria"] = np.where(df.discharge_location.isna(), np.nan,
                                     (df.discharge_location == "AGAINST ADVICE").astype(float))
    R["variables"] = {"fuentes": fuentes, "hospitalizaciones_sin_admision": int(df.admittime.isna().sum()),
                      "fechas": {"anio_de_ingreso_registrado": [int(df.anio_ingreso.min()), int(df.anio_ingreso.max())],
                                 "anio_real_aproximado": [int(df.anio_aprox.min()), int(df.anio_aprox.max())],
                                 "ajustados_al_rango_2008_2022": int(((df.anio_aprox < 2008) | (df.anio_aprox > 2022)).sum())}}
    log.info(f"variables: {len(DICCIONARIO)} variables; {len(fuentes)} tablas de MIMIC-IV leídas · hospitalizaciones sin registro de "
             f"admisión: {R['variables']['hospitalizaciones_sin_admision']} · años de ingreso registrados "
             f"{R['variables']['fechas']['anio_de_ingreso_registrado']} (desplazados) · año real aproximado "
             f"{R['variables']['fechas']['anio_real_aproximado']}")
    return df


# ------------------------------------------------------------------ bloque A
def _tabla_categorica(df, var, media):
    """% de positivas por categoría. Las categorías con menos de MIN_CELDA casos van a «Otros»; si «Otros» tampoco llega, no se publica."""
    s = _con_sin_dato(df[var])
    cuenta = s.value_counts()
    s = s.where(s.map(cuenta) >= MIN_CELDA, "Otros")
    g = df.groupby(s).y.agg(["size", "mean"]).sort_values("size", ascending=False)
    no_pub = int(g.loc[g["size"] < MIN_CELDA, "size"].sum())
    g = g[g["size"] >= MIN_CELDA]
    tabla = {str(k): {"n": int(r["size"]), "pct_del_corpus": _pct(r["size"] / len(df)), "pct_positivas": _pct(r["mean"]),
                      "diferencia_con_promedio": round(100 * float(r["mean"]) - media, 4)} for k, r in g.iterrows()}
    return tabla, no_pub


def _modelo_contexto(df, num, cat, semilla):
    """Regresión logística sin texto: se ajusta en train y se mide en test (misma partición del pipeline).
    El intervalo de confianza se obtiene remuestreando pacientes del test."""
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                             ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Sin dato"),
                                                   OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CELDA)), cat)])
    m = make_pipeline(pre, LogisticRegression(max_iter=2000))
    tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    m.fit(tr[num + cat], tr.y)
    pr = m.predict_proba(te[num + cat])[:, 1]
    y = te.y.values
    grupos = list(te.groupby("subject_id").indices.values())
    rng = np.random.default_rng(semilla)
    roc, prc = [], []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))])
        roc.append(roc_auc_score(y[idx], pr[idx])); prc.append(average_precision_score(y[idx], pr[idx]))
    return {"variables": num + cat, "roc_auc": round(float(roc_auc_score(y, pr)), 4), "pr_auc": round(float(average_precision_score(y, pr)), 4),
            "roc_auc_ic95": [round(float(np.percentile(roc, 2.5)), 6), round(float(np.percentile(roc, 97.5)), 6)],
            "pr_auc_ic95": [round(float(np.percentile(prc, 2.5)), 6), round(float(np.percentile(prc, 97.5)), 6)]}


def analizar_variables(df, R, log, FIG, pie):
    V = R["variables"]
    media = 100 * float(df.y.mean())
    pos, neg = df[df.y == 1], df[df.y == 0]

    # A1. diccionario y faltantes
    dic = []
    for nombre, etiqueta, tipo, fuente, momento, uso in DICCIONARIO:
        falt = df[nombre].isna()
        d = {"variable": nombre, "etiqueta": etiqueta, "tipo": tipo, "fuente": fuente, "momento": momento, "uso": uso,
             "faltantes": int(falt.sum()), "pct_faltantes": _pct(falt.mean()),
             "pct_faltantes_negativas": _pct(falt[df.y == 0].mean()), "pct_faltantes_positivas": _pct(falt[df.y == 1].mean()),
             "valores_distintos": int(df[nombre].nunique())}
        if falt.sum() >= MIN_CELDA:
            d["pct_positivas_si_falta"] = _pct(df.y[falt].mean())
            d["pct_positivas_si_no_falta"] = _pct(df.y[~falt].mean())
        dic.append(d)
    V["diccionario"] = dic
    V["variables_de_tablas_de_mimic"] = int(sum("nota" not in d["fuente"] for d in dic))
    con_falt = sorted([d for d in dic if d["faltantes"] > 0], key=lambda d: -d["pct_faltantes"])
    log.info("A1. faltantes: " + " · ".join(f"{d['variable']} {d['pct_faltantes']:.2f} %" for d in con_falt))

    # ¿de qué depende el faltante? porcentaje sin dato según el tipo de ingreso (variables con más de 5 % de faltantes)
    tipos = df.tipo_ingreso.value_counts()
    tipos = tipos[tipos >= MIN_CELDA].index
    V["faltantes_por_tipo_ingreso"] = {
        d["variable"]: {t: _pct(df.loc[df.tipo_ingreso == t, d["variable"]].isna().mean()) for t in tipos}
        for d in con_falt if d["pct_faltantes"] > 5 and d["variable"] != "alta_voluntaria"}
    for v, t in V["faltantes_por_tipo_ingreso"].items():
        log.info(f"A1. {v} sin dato según el tipo de ingreso: " + " · ".join(f"{k} {p:.1f} %" for k, p in sorted(t.items(), key=lambda kv: -kv[1])))

    # valores fuera de rango
    negativa = df.estancia_dias < 0
    V["fuera_de_rango"] = {"estancia_negativa": int(negativa.sum()),
                           "estancia_negativa_con_fallecimiento": int((negativa & (df.fallecio == 1)).sum()),
                           "edad_min_max": [int(df.edad.min()), int(df.edad.max())],
                           "nota": "en la estancia negativa la hora de alta registrada es anterior a la de ingreso"}
    log.info(f"A1. fuera de rango: estancias negativas {V['fuera_de_rango']['estancia_negativa']} (de ellas, con fallecimiento "
             f"{V['fuera_de_rango']['estancia_negativa_con_fallecimiento']}) · edad de {V['fuera_de_rango']['edad_min_max'][0]} a "
             f"{V['fuera_de_rango']['edad_min_max'][1]} años")

    # A2. numéricas: estadísticas, atípicos y separación por clase
    V["numericas"] = {}
    for v in NUMERICAS:
        ok = df[v].notna()
        x = df.loc[ok, v]
        q1, q3 = x.quantile([.25, .75])
        fuera = (x > q3 + 1.5 * (q3 - q1)) | (x < q1 - 1.5 * (q3 - q1))
        V["numericas"][v] = {
            "n": int(ok.sum()), "media": round(float(x.mean()), 4), "desv": round(float(x.std()), 4),
            "min": round(float(x.min()), 4), "q1": round(float(q1), 4), "mediana": round(float(x.median()), 4),
            "q3": round(float(q3), 4), "max": round(float(x.max()), 4),
            "atipicos_iqr": {"n": int(fuera.sum()), "pct": _pct(fuera.mean())},
            "negativa": {"mediana": round(float(neg[v].median()), 4), "q1": round(float(neg[v].quantile(.25)), 4),
                         "q3": round(float(neg[v].quantile(.75)), 4)},
            "positiva": {"mediana": round(float(pos[v].median()), 4), "q1": round(float(pos[v].quantile(.25)), 4),
                         "q3": round(float(pos[v].quantile(.75)), 4)},
            "auc_sola": round(float(roc_auc_score(df.y[ok], x)), 4),
            "spearman_con_y": round(float(df.loc[ok, [v, "y"]].corr(method="spearman").iloc[0, 1]), 4)}
    orden = sorted(V["numericas"], key=lambda k: -abs(V["numericas"][k]["auc_sola"] - 0.5))
    log.info("A2. numéricas (AUC de la variable sola): " + " · ".join(f"{k} {V['numericas'][k]['auc_sola']:.3f}" for k in orden))
    log.info("A2. atípicos por la regla de 1.5 veces el rango intercuartílico: "
             + " · ".join(f"{k} {V['numericas'][k]['atipicos_iqr']['pct']:.2f} %" for k in NUMERICAS))

    # A3. categóricas y binarias (el dato faltante cuenta como una categoría más, «Sin dato»)
    V["categoricas"] = {}
    for v in CATEGORICAS + BINARIAS:
        s = df[v] if v in CATEGORICAS else df[v].map({1: "Sí", 0: "No", 1.0: "Sí", 0.0: "No"})
        tabla, no_pub = _tabla_categorica(df.assign(**{v: s}), v, media)
        V["categoricas"][v] = {"cramers_v": round(_cramers_v(_con_sin_dato(s), df.y), 4), "categorias": tabla,
                               "epicrisis_en_categorias_no_publicadas": no_pub}
    ordc = sorted(V["categoricas"], key=lambda k: -V["categoricas"][k]["cramers_v"])
    log.info("A3. categóricas (V de Cramér con la etiqueta): " + " · ".join(f"{k} {V['categoricas'][k]['cramers_v']:.3f}" for k in ordc))

    # A4. asociación entre las variables categóricas (V de Cramér), con la etiqueta en la última fila
    nombres = ["sexo", "grupo_edad", "raza", "seguro", "estado_civil", "tipo_ingreso", "servicio", "destino_alta", "periodo",
               "uci", "fallecio", "reingreso_30d", "epoca", "y"]
    cols = {n: _con_sin_dato(df[n]) for n in nombres}
    mat = pd.DataFrame(1.0, index=nombres, columns=nombres)
    for i, a in enumerate(nombres):
        for b in nombres[:i]:
            mat.loc[a, b] = mat.loc[b, a] = _cramers_v(cols[a], cols[b])
    V["asociacion_categoricas"] = {"variables": nombres, "matriz": mat.round(4).values.tolist()}
    # pares de predictores, sin los que se definen uno a partir del otro (fallecimiento es un destino al alta; el periodo fija la época)
    derivados = {frozenset(("fallecio", "destino_alta")), frozenset(("epoca", "periodo"))}
    pares = sorted(((mat.loc[a, b], a, b) for i, a in enumerate(nombres[:-1]) for b in nombres[:i]
                    if frozenset((a, b)) not in derivados), reverse=True)[:5]
    V["asociacion_categoricas"]["pares_mas_asociados"] = [{"a": a, "b": b, "cramers_v": round(float(v), 4)} for v, a, b in pares]
    log.info("A4. pares de variables categóricas más asociados (V de Cramér): " + " · ".join(f"{a}-{b} {v:.3f}" for v, a, b in pares))

    # el par de predictores más asociado es el tipo de ingreso con la época: reparto del tipo de ingreso en cada época
    te = pd.crosstab(_con_sin_dato(df.tipo_ingreso), df.epoca, normalize="columns")
    V["tipo_ingreso_por_epoca"] = {str(k): {f"CIE-{int(e)}": _pct(te.loc[k, e]) for e in te.columns} for k in te.index
                                   if (df.tipo_ingreso == k).sum() >= MIN_CELDA}
    log.info(f"A4. tipo de ingreso por época (% de las epicrisis de la época): {V['tipo_ingreso_por_epoca']}")

    # A5. ¿train y test se parecen? (la partición es por paciente)
    def resumen(d):
        return {"n": int(len(d)), "pct_positivas": _pct(d.y.mean()), "mediana_edad": float(d.edad.median()),
                "mediana_estancia_dias": round(float(d.estancia_dias.median()), 4), "mediana_largo": float(d.largo.median()),
                "pct_hombres": _pct((d.sexo == "Hombre").mean()), "pct_uci": _pct(d.uci.mean()),
                "pct_cie10": _pct((d.epoca == 10).mean()),
                "pct_cie10_positivas": _pct((d.epoca[d.y == 1] == 10).mean()), "pct_cie10_negativas": _pct((d.epoca[d.y == 0] == 10).mean()),
                "auc_epoca_sola": round(float(roc_auc_score(d.y, d.epoca)), 4)}

    V["particion"] = {s: resumen(df[df.split == s]) for s in ("train", "test")}
    log.info(f"A5. train frente a test: {V['particion']}")

    # A6. ¿cuánto separan las clases las variables de contexto, sin leer el texto?
    V["contexto"] = {"al_ingreso": _modelo_contexto(df, *CONTEXTO_INGRESO, R["semilla"]),
                     "al_alta": _modelo_contexto(df, *CONTEXTO_ALTA, R["semilla"]),
                     "nota": "regresión logística sin texto, ajustada en train y medida en test con la partición del pipeline; no incluye "
                             f"el número de diagnósticos ni la severidad APR-DRG; IC 95 % con {N_BOOTSTRAP} remuestreos de pacientes"}
    ruta_base = REPORTES / "baseline_resultados.json"
    if ruta_base.exists():
        base = json.loads(ruta_base.read_text(encoding="utf-8"))
        misma = base.get("semilla") == R["semilla"] and base.get("n_test") == int((df.split == "test").sum())
        if misma:       # solo se compara si el baseline se midió con la misma semilla y el mismo test
            m = base["modelos"]
            V["contexto"]["modelo_de_texto"] = {"roc_auc": m["logistica_tfidf"]["auc"], "pr_auc": m["logistica_tfidf"]["pr_auc"],
                                                "roc_auc_ic95": m["logistica_tfidf"].get("auc_ic95"),
                                                "fuente": "reportes/baseline_resultados.json (src/baseline.py)"}
            V["contexto"]["dummy"] = {"roc_auc": m["dummy_mayoritaria"]["auc"], "pr_auc": m["dummy_mayoritaria"]["pr_auc"]}
        else:
            log.info("A6. baseline_resultados.json es de otra partición: no se compara con el modelo de texto")
    C = V["contexto"]
    ic = lambda k: " a ".join(f"{x:.3f}" for x in C[k]["roc_auc_ic95"])
    log.info(f"A6. contexto sin texto: al ingreso ROC-AUC {C['al_ingreso']['roc_auc']} (IC 95 % {ic('al_ingreso')}; PR-AUC "
             f"{C['al_ingreso']['pr_auc']}) · al alta ROC-AUC {C['al_alta']['roc_auc']} (IC 95 % {ic('al_alta')}; PR-AUC "
             f"{C['al_alta']['pr_auc']}) · modelo de texto del baseline {C.get('modelo_de_texto')}")

    # ---------------- figura: faltantes
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    d2 = sorted(dic, key=lambda d: d["pct_faltantes"])
    ax.barh([d["etiqueta"] for d in d2], [d["pct_faltantes"] for d in d2], color=[ROJO if d["pct_faltantes"] > 5 else AZUL for d in d2])
    for i, d in enumerate(d2):
        ax.text(d["pct_faltantes"] + 0.6, i, f"{d['pct_faltantes']:.1f} %", va="center", fontsize=8)
    ax.set(xlabel="% de epicrisis sin el dato", title="Datos faltantes por variable", xlim=(0, max(d["pct_faltantes"] for d in d2) + 9))
    partes = []
    for d in [d for d in con_falt if d["variable"] in V["faltantes_por_tipo_ingreso"]][:3]:
        t = V["faltantes_por_tipo_ingreso"][d["variable"]]
        tmax, tmin = max(t, key=t.get), min(t, key=t.get)
        partes.append(f"{d['etiqueta'].split(' (')[0]}: sin dato en el {d['pct_faltantes']:.1f} % de las epicrisis "
                      f"({d['pct_faltantes_positivas']:.1f} % de las positivas y {d['pct_faltantes_negativas']:.1f} % de las negativas); "
                      f"según el tipo de ingreso, de {t[tmin]:.1f} % ({tmin}) a {t[tmax]:.1f} % ({tmax})")
    pie(fig, "Cómo leer: cada barra es el porcentaje de epicrisis del corpus que no tiene el dato de esa variable; en rojo, las que "
             "superan el 5 %. " + ". ".join(partes) + ". Lectura: el faltante depende del tipo de ingreso y no se reparte igual "
             "entre las clases, así que no es aleatorio; en las variables categóricas se conserva como categoría «Sin dato» en lugar "
             "de imputarse.", 150, 0.2)
    fig.savefig(FIG / "fig_eda_faltantes.png", dpi=130); plt.close(fig)

    # ---------------- figura: numéricas
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.8))
    paneles = [("edad", np.arange(18, 106, 2), False, "años"), ("estancia_dias", 50, True, "log10(días)"),
               ("horas_emergencia", 50, True, "log10(horas)"), ("n_diagnosticos", np.arange(0.5, 58.5, 1), False, "diagnósticos"),
               ("drg_severidad", None, None, "nivel")]
    for a, (v, bins, log10, xl) in zip(ax.flat, paneles):
        nv = V["numericas"][v]
        if bins is None:
            t = pd.crosstab(df[v], df.y, normalize="columns").mul(100)
            a.bar(t.index - 0.2, t[0], 0.4, color=GRIS, label="sin evento"); a.bar(t.index + 0.2, t[1], 0.4, color=ROJO, label="con evento")
            a.set_ylabel("% de la clase"); a.set_xticks([1, 2, 3, 4])
        else:
            for y, nombre, c in [(0, "sin evento", GRIS), (1, "con evento", ROJO)]:
                x = df.loc[(df.y == y) & df[v].notna(), v]
                x = np.log10(x[x > 0]) if log10 else x
                a.hist(x, bins=bins, alpha=.6, density=True, color=c, label=nombre)
            a.set_ylabel("densidad")
        a.set_title(f"{CORTA[v]}\nAUC de la variable sola = {nv['auc_sola']:.3f}", fontsize=10); a.set_xlabel(xl); a.legend(fontsize=8)
    a = ax.flat[5]
    a.barh([CORTA[k] for k in orden][::-1], [V["numericas"][k]["auc_sola"] for k in orden][::-1], color=AZUL)
    a.axvline(0.5, ls="--", c="gray"); a.set_title("Separación de las clases\ncon cada variable sola", fontsize=10)
    a.set(xlabel="AUC (la línea punteada, 0.5, es no separar)", xlim=(0, 1)); a.tick_params(axis="y", labelsize=8)
    nd, es = V["numericas"]["n_diagnosticos"], V["numericas"]["estancia_dias"]
    pie(fig, "Cómo leer: cada panel compara la distribución de una variable en las epicrisis sin evento (gris) y con evento (rojo); "
             "el último ordena las variables por cuánto separan las clases solas (AUC). Lectura: las hospitalizaciones con evento "
             f"tienen más diagnósticos codificados (mediana de {nd['positiva']['mediana']:.0f} frente a {nd['negativa']['mediana']:.0f}) "
             f"y estancias más largas (mediana de {es['positiva']['mediana']:.1f} frente a {es['negativa']['mediana']:.1f} días). "
             "El número de diagnósticos y la severidad APR-DRG se calculan con los mismos códigos de los que sale la etiqueta, así que "
             "no pueden usarse como predictores: se reportan para describir la población.", 185, 0.12)
    fig.savefig(FIG / "fig_eda_numericas.png", dpi=130); plt.close(fig)

    # ---------------- figura: categóricas
    fig, ax = plt.subplots(2, 4, figsize=(18, 10.4))
    for a, v in zip(ax.flat, ["sexo", "grupo_edad", "raza", "seguro", "tipo_ingreso", "servicio", "destino_alta"]):
        cat = V["categoricas"][v]["categorias"]
        claves = [k for k in cat if k != "Otros"]
        if v == "grupo_edad":
            claves = [k for k in ["18 a 44", "45 a 64", "65 a 79", "80 o más"] if k in cat]
        elif v in ("tipo_ingreso", "servicio", "destino_alta"):
            claves = sorted(claves, key=lambda k: -cat[k]["pct_positivas"])
        a.barh([f"{k}  (n={_mil(cat[k]['n'])})" for k in claves][::-1], [cat[k]["pct_positivas"] for k in claves][::-1],
               color=[GRIS if k == "Sin dato" else AZUL for k in claves][::-1])
        a.axvline(media, ls="--", c="gray")
        a.set_title(f"{ETIQUETA[v]}\nV de Cramér = {V['categoricas'][v]['cramers_v']:.3f}", fontsize=10)
        a.set(xlabel="% de epicrisis con evento", xlim=(0, 100))
        a.tick_params(axis="y", labelsize=6.5 if len(claves) > 9 else 7.5)
    a = ax.flat[7]
    yy = np.arange(len(BINARIAS)); h = 0.38
    si = [V["categoricas"][v]["categorias"].get("Sí", {}).get("pct_positivas", np.nan) for v in BINARIAS]
    no = [V["categoricas"][v]["categorias"].get("No", {}).get("pct_positivas", np.nan) for v in BINARIAS]
    a.barh(yy + h / 2, no[::-1], h, color=GRIS, label="No"); a.barh(yy - h / 2, si[::-1], h, color=AZUL, label="Sí")
    a.set_yticks(yy, [f"{CORTA[v]}\n(V = {V['categoricas'][v]['cramers_v']:.3f})" for v in BINARIAS][::-1], fontsize=7.5)
    a.axvline(media, ls="--", c="gray"); a.set_title("Variables binarias", fontsize=10)
    a.set(xlabel="% de epicrisis con evento", xlim=(0, 100)); a.legend(fontsize=8, loc="lower right")
    sv = {k: v for k, v in V["categoricas"]["servicio"]["categorias"].items() if k not in ("Otros", "Sin dato")}
    smax, smin = max(sv, key=lambda k: sv[k]["pct_positivas"]), min(sv, key=lambda k: sv[k]["pct_positivas"])
    sx, ge = V["categoricas"]["sexo"]["categorias"], V["categoricas"]["grupo_edad"]["categorias"]
    pie(fig, f"Cómo leer: cada barra es el porcentaje de epicrisis con evento dentro de esa categoría; la línea punteada es el promedio "
             f"del corpus ({media:.1f} %) y entre paréntesis va el número de epicrisis. Los tipos de ingreso, los servicios y los destinos "
             "al alta llevan el código con que figuran en MIMIC-IV; la barra gris es la categoría «Sin dato». La V de Cramér resume la "
             "asociación con la etiqueta (0 = ninguna, 1 = total). "
             f"Lectura: el evento varía poco por sexo ({sx['Mujer']['pct_positivas']:.1f} % en mujeres y "
             f"{sx['Hombre']['pct_positivas']:.1f} % en hombres), sube con la edad (de {ge['18 a 44']['pct_positivas']:.1f} % en 18 a 44 "
             f"años a {ge['80 o más']['pct_positivas']:.1f} % en 80 o más) y cambia mucho según el servicio (de "
             f"{sv[smin]['pct_positivas']:.1f} % en {smin} a {sv[smax]['pct_positivas']:.1f} % en {smax}): las epicrisis con y sin "
             "evento no vienen del mismo tipo de paciente.", 250, 0.1)
    fig.savefig(FIG / "fig_eda_categoricas.png", dpi=120); plt.close(fig)

    # ---------------- figura: asociación entre variables categóricas
    fig, ax = plt.subplots(figsize=(10.5, 10.6))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    etq = [CORTA[n] for n in nombres]
    ax.set_xticks(range(len(etq)), etq, rotation=40, ha="right"); ax.set_yticks(range(len(etq)), etq)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if mat.values[i, j] > 0.6 else "black")
    ax.set_title("V de Cramér entre variables categóricas (variable objetivo en la última fila)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    v0, a0, b0 = pares[0]
    pie(fig, "Cómo leer: cada celda es la V de Cramér entre dos variables categóricas, de 0 (ninguna asociación) a 1 (asociación "
             "total); el dato faltante cuenta como una categoría. La última fila es la variable objetivo. Lectura: aparte de las "
             "variables que se definen una a partir de otra (fallecimiento y destino al alta; época y periodo), las más asociadas "
             f"entre sí son {_min1(CORTA[a0])} y {_min1(CORTA[b0])} ({v0:.2f}); con la etiqueta, las más asociadas son "
             + ", ".join(f"{_min1(CORTA[k])} ({mat.loc['y', k]:.2f})" for k in mat.loc["y"].drop("y").sort_values(ascending=False).index[:3])
             + ". Las variables de contexto están relacionadas entre sí, así que sus efectos no se suman de forma independiente.", 150, 0.1)
    fig.savefig(FIG / "fig_eda_asociacion.png", dpi=130); plt.close(fig)

    # ---------------- figura: contexto frente a texto
    if "modelo_de_texto" in C:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        nombres_m = ["Sin información\n(siempre «sí»)", "Contexto al ingreso\n(sin texto)", "Contexto al alta\n(sin texto)",
                     "Texto de la epicrisis\n(logística TF-IDF)"]
        claves = ["dummy", "al_ingreso", "al_alta", "modelo_de_texto"]
        xx = np.arange(4)
        for dx_, clave, color, nombre in [(-0.2, "roc_auc", AZUL, "ROC-AUC"), (0.2, "pr_auc", ROJO, "PR-AUC")]:
            vals = [C[k][clave] for k in claves]
            ax.bar(xx + dx_, vals, 0.4, color=color, label=nombre)
            for i, (k, v) in enumerate(zip(claves, vals)):
                lim = C[k].get(clave + "_ic95")
                if lim:
                    ax.errorbar(i + dx_, v, yerr=[[v - lim[0]], [lim[1] - v]], color="#333", capsize=3, lw=1)
                ax.text(i + dx_, v + 0.02, _f3(v), ha="center", fontsize=8)
        ax.set_xticks(xx, nombres_m, fontsize=8.5); ax.set(ylim=(0, 1.05), ylabel="valor en test")
        ax.set_title("¿Cuánto se acierta sin leer la nota?", fontsize=11); ax.legend(fontsize=8, loc="upper left")
        pie(fig, "Cómo leer: cada par de barras es un modelo medido en el mismo conjunto de test. «Contexto al ingreso» es una regresión "
                 "logística con edad, sexo, raza, seguro, estado civil, tipo de ingreso y servicio; «contexto al alta» añade estancia, "
                 "largo de la nota y destino al alta; ninguno de los dos lee el texto. La última barra es el modelo de texto del "
                 "baseline. La línea sobre una barra es su intervalo de confianza al 95 %. Lectura: solo con el contexto se llega a un "
                 f"ROC-AUC de {_f3(C['al_ingreso']['roc_auc'])} al ingreso y de {_f3(C['al_alta']['roc_auc'])} al alta, frente a "
                 f"{_f3(C['modelo_de_texto']['roc_auc'])} del modelo que lee la epicrisis. Las dos clases ya se distinguen en parte por "
                 "el tipo de paciente, así que el modelo de texto se evaluará también dentro de cada servicio y grupo de edad.", 135, 0.24)
        fig.savefig(FIG / "fig_eda_contexto.png", dpi=130); plt.close(fig)
    log.info("figuras: fig_eda_faltantes.png, fig_eda_numericas.png, fig_eda_categoricas.png, fig_eda_asociacion.png, fig_eda_contexto.png")


# ------------------------------------------------------------------ bloque P
def _hadm_con_epicrisis(log):
    """Hospitalizaciones que tienen epicrisis en MIMIC-IV-Note. Solo se lee la columna hadm_id; se guarda en data/interim."""
    cache = INTERIM / "hadm_con_epicrisis.parquet"
    if not cache.exists():
        ruta = Path(os.environ["MIMIC_NOTE_DIR"]) / "discharge.csv.gz"
        log.info("P. leyendo la lista de hospitalizaciones con epicrisis (solo la columna hadm_id; unos 2 minutos la primera vez)")
        pd.read_csv(ruta, usecols=["hadm_id"]).drop_duplicates().to_parquet(cache, index=False)
    return set(pd.read_parquet(cache).hadm_id)


def prevalencia_real(df, R, log, FIG, pie):
    """Proporción real de hospitalizaciones con evento, por época y periodo, frente a la del corpus (fijada por el muestreo)."""
    a1 = pd.read_csv(INTERIM / "a1_tier_a.csv", usecols=["hadm_id", "estrato"])
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv", usecols=["hadm_id", "estrato"])
    c9 = pd.read_csv(INTERIM / "positivos_cie9.csv", usecols=["hadm_id", "era"])
    fuentes = {"códigos de la tabla": set(a1.loc[a1.estrato == "A1", "hadm_id"]),
               "equivalencias CIE-10-CM": set(eq.loc[eq.estrato == "A1", "hadm_id"]),
               "familias CIE-9": set(c9.loc[c9.era == 9, "hadm_id"])}
    positivas = set().union(*fuentes.values())
    con_nota = _hadm_con_epicrisis(log)
    u = pd.read_parquet(INTERIM / "epoca.parquet").merge(_AUX["adm"], on="hadm_id", how="left").merge(_AUX["pac"], on="subject_id", how="left")
    _, u["periodo"] = _periodo(u.anchor_year_group, u.anio_ingreso, u.anchor_year)
    u["pos"], u["nota"] = u.hadm_id.isin(positivas), u.hadm_id.isin(con_nota)

    def fila(g):
        n = g[g.nota]
        return {"hospitalizaciones": int(len(g)), "pct_positivas": _pct(g.pos.mean()),
                "con_epicrisis": int(len(n)), "pct_positivas_con_epicrisis": _pct(n.pos.mean()) if len(n) else None}

    P = R["prevalencia_real"] = {
        "todas": fila(u),
        "por_epoca": {f"CIE-{int(k)}": fila(g) for k, g in u.groupby("epoca")},
        "por_periodo": {k: fila(u[u.periodo == k]) for k in PERIODOS},
        "por_periodo_y_epoca_con_epicrisis": {
            k: {f"CIE-{int(e)}": {"con_epicrisis": int(len(g)), "pct_positivas": _pct(g.pos.mean())}
                for e, g in u[(u.periodo == k) & u.nota].groupby("epoca") if len(g) >= 1000} for k in PERIODOS},
        "pct_con_epicrisis_por_fuente_de_positivas": {n: _pct(pd.Series(sorted(s)).isin(con_nota).mean()) for n, s in fuentes.items()},
        "nota": "positiva = hospitalización con código de causa explícita (misma regla del preprocesado); el modelo solo ve "
                "hospitalizaciones con epicrisis, y esa es la base que le corresponde"}
    c10 = u[(u.epoca == 10) & u.nota]
    P["cie10_con_familias_completas"] = {
        "con_epicrisis": int(len(c10)), "pct_positivas_con_la_etiqueta_actual": _pct(c10.pos.mean()),
        "pct_positivas_con_las_familias_completas": _pct((c10.pos | c10.hadm_id.isin(_AUX["hadm_familias_cie10"])).mean()),
        "nota": "si en CIE-10 dieran etiqueta todos los códigos de las familias de causa explícita y los efectos adversos de "
                "medicamentos, como ocurre en CIE-9, y no solo los que están en la tabla del proyecto"}
    R["balance"]["prevalencia_real_mimic"] = round(P["todas"]["pct_positivas"] / 100, 6)
    R["balance"]["prevalencia_real_con_epicrisis"] = round(P["todas"]["pct_positivas_con_epicrisis"] / 100, 6)
    log.info(f"P. prevalencia real: {P['todas']['pct_positivas']:.2f} % de {_mil(len(u))} hospitalizaciones · "
             f"{P['todas']['pct_positivas_con_epicrisis']:.2f} % de las {_mil(P['todas']['con_epicrisis'])} que tienen epicrisis · por época "
             + " · ".join(f"{k}: {v['pct_positivas_con_epicrisis']:.2f} %" for k, v in P["por_epoca"].items())
             + " · por periodo " + " · ".join(f"{k}: {v['pct_positivas_con_epicrisis']:.2f} %" for k, v in P["por_periodo"].items())
             + f" · epicrisis por fuente de positivas {P['pct_con_epicrisis_por_fuente_de_positivas']} · CIE-10 con las familias "
             f"completas: {P['cie10_con_familias_completas']['pct_positivas_con_las_familias_completas']:.2f} %")

    # drift del corpus por periodo
    V = R["variables"]
    V["drift_periodo"] = {}
    for k in PERIODOS:
        d = df[df.periodo == k]
        if len(d) >= MIN_CELDA:
            V["drift_periodo"][k] = {"n": int(len(d)), "pct_positivas": _pct(d.y.mean()), "pct_cie10": _pct((d.epoca == 10).mean()),
                                     "mediana_largo": float(d.largo.median()), "mediana_edad": float(d.edad.median()),
                                     "mediana_n_diagnosticos": float(d.n_diagnosticos.median())}
    D = V["drift_periodo"]
    per = list(D.keys())
    pp, ml = [D[k]["pct_positivas"] for k in per], [D[k]["mediana_largo"] for k in per]
    pr = [P["por_periodo"][k]["pct_positivas_con_epicrisis"] for k in per]
    V["drift_resumen"] = {"rango_pct_positivas_corpus": [min(pp), max(pp)], "rango_mediana_largo": [min(ml), max(ml)],
                          "rango_prevalencia_real_con_epicrisis": [min(pr), max(pr)]}
    log.info(f"P. drift por periodo: en el corpus, positivas {min(pp):.2f} % a {max(pp):.2f} % (fijado por el muestreo) · prevalencia real "
             f"con epicrisis {max(pr):.2f} % a {min(pr):.2f} % · mediana del largo {min(ml):.0f} a {max(ml):.0f} caracteres · "
             + " · ".join(f"{k}: n={D[k]['n']}, CIE-10 {D[k]['pct_cie10']:.2f} %" for k in per))

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.2))
    ax[0].plot(per, pr, "o-", color=ROJO, label="prevalencia real (hospitalizaciones con epicrisis)")
    ax[0].plot(per, pp, "s--", color=GRIS, label="corpus de modelado (fijado por el muestreo)")
    ax[0].set_title("Hospitalizaciones con evento por periodo", fontsize=10); ax[0].set(ylabel="%", ylim=(0, 70))
    ax[0].legend(fontsize=7.5, loc="lower left"); ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(per, [D[k]["pct_cie10"] for k in per], color=AZUL)
    ax[1].set_title("Epicrisis del corpus codificadas en CIE-10", fontsize=10); ax[1].set(ylabel="%", ylim=(0, 105))
    ax[1].tick_params(axis="x", rotation=25)
    ax[2].plot(per, ml, "o-", color=VERDE)
    ax[2].set_title("Mediana del largo de la epicrisis", fontsize=10); ax[2].set(ylabel="caracteres", ylim=(0, max(ml) * 1.15))
    ax[2].tick_params(axis="x", rotation=25)
    f = V["fechas"]["anio_de_ingreso_registrado"]
    e9, e10 = P["por_epoca"]["CIE-9"]["pct_positivas_con_epicrisis"], P["por_epoca"]["CIE-10"]["pct_positivas_con_epicrisis"]
    pie(fig, f"Cómo leer: las fechas de MIMIC-IV están desplazadas (los ingresos del corpus figuran entre los años {f[0]} y {f[1]}), "
             "así que el año real de cada hospitalización se aproxima con el grupo de años que la base asigna al paciente, con una "
             "precisión de más o menos un año. A la izquierda, el porcentaje de hospitalizaciones con evento en cada periodo: en rojo, "
             "en todas las hospitalizaciones de MIMIC-IV que tienen epicrisis; en gris, en el corpus de modelado. Al centro, el "
             "porcentaje del corpus codificado en CIE-10; a la derecha, la mediana del largo de la nota. Lectura: en el corpus el "
             f"porcentaje con evento se mantiene entre {min(pp):.1f} % y {max(pp):.1f} % porque el muestreo toma una proporción fija "
             f"de negativas por cada positiva; en la base, la prevalencia baja de {pr[0]:.1f} % a {pr[-1]:.1f} %. La caída coincide con "
             f"el paso de CIE-9 a CIE-10 ({e9:.1f} % de las hospitalizaciones con epicrisis en CIE-9 y {e10:.1f} % en CIE-10), y la "
             f"mediana del largo de la nota sube de {_mil(ml[0])} a {_mil(ml[-1])} caracteres.", 200, 0.27)
    fig.savefig(FIG / "fig_eda_drift.png", dpi=130); plt.close(fig)
    log.info("figura: fig_eda_drift.png")


# ------------------------------------------------------------------ bloque R
def ruido_etiqueta(df, R, log):
    """¿Qué tan limpia es la etiqueta? Negativas con códigos de las familias causales y positivas que lo son solo por una caída."""
    dx, tit = _AUX["dx"], _AUX["titulos"]
    d10, d9 = dx[dx.icd_version == 10], dx[dx.icd_version == 9]
    causal = d10.icd_code.str.startswith(PREF_CAUSAL)
    farmaco = _efecto_adverso(d10.icd_code)
    neg10, neg9 = df[(df.y == 0) & (df.epoca == 10)], df[(df.y == 0) & (df.epoca == 9)]
    h10 = set(neg10.hadm_id)
    hc, hf = set(d10[causal & d10.hadm_id.isin(h10)].hadm_id), set(d10[farmaco & d10.hadm_id.isin(h10)].hadm_id)
    top = d10[(causal | farmaco) & d10.hadm_id.isin(h10)].drop_duplicates(["hadm_id", "icd_code"]).icd_code.value_counts()
    top = top[top >= 10].head(8)
    rx9 = re.compile("|".join(FAM_CIE9.values()))
    h9 = set(d9[d9.icd_code.str.contains(rx9) & d9.hadm_id.isin(set(neg9.hadm_id))].hadm_id)
    N = R["ruido_etiqueta"] = {"negativas": {
        "cie10": {"n": int(len(neg10)), "con_codigo_de_familia_causal": len(hc), "con_efecto_adverso_de_farmaco": len(hf),
                  "con_alguno": len(hc | hf), "pct_con_alguno": _pct(len(hc | hf) / len(neg10)),
                  "codigos_mas_frecuentes": [{"codigo": c, "epicrisis": int(n), "titulo_en_mimic": str(tit.get((c, 10), ""))}
                                             for c, n in top.items()]},
        "cie9": {"n": int(len(neg9)), "con_codigo_de_familia_causal": len(h9), "pct": _pct(len(h9) / len(neg9))},
        "definicion": "familias de causa explícita: T80 a T88, Y63, Y65, Y83, Y84, W00 a W19 y L89; efecto adverso de medicamento: "
                      "T36 a T50 con 5 en el sexto carácter. En CIE-9 la etiqueta usa las familias completas; en CIE-10, solo los "
                      "códigos de la tabla del proyecto"}}
    c10 = N["negativas"]["cie10"]
    log.info(f"R1. negativas con código de las familias causales: CIE-10 {c10['con_alguno']} de {c10['n']} ({c10['pct_con_alguno']:.2f} %; "
             f"familia causal {c10['con_codigo_de_familia_causal']}, efecto adverso de fármaco {c10['con_efecto_adverso_de_farmaco']}) · "
             f"CIE-9 {len(h9)} de {len(neg9)} · más frecuentes: " + ", ".join(f"{t['codigo']} ({t['epicrisis']})" for t in c10["codigos_mas_frecuentes"]))

    # positivas cuya única evidencia es un código de caída, y dónde ocurrió la caída
    a1 = pd.read_csv(INTERIM / "a1_tier_a.csv", usecols=["hadm_id", "estrato", "icd_code"])
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv", usecols=["hadm_id", "estrato"])
    c9 = pd.read_csv(INTERIM / "positivos_cie9.csv", usecols=["hadm_id", "era", "familia"])
    a1 = a1[a1.estrato == "A1"]
    todas_caida = a1.assign(c=a1.icd_code.str.startswith("W")).groupby("hadm_id").c.all()
    solo10 = set(todas_caida[todas_caida].index) - set(eq.loc[eq.estrato == "A1", "hadm_id"])
    fam_dx9 = {k: set(d9[d9.icd_code.str.contains(v)].hadm_id) for k, v in FAM_CIE9.items()}
    solo9 = fam_dx9["caidas"] - set().union(*(s for k, s in fam_dx9.items() if k != "caidas"))
    pos = df[df.y == 1]
    p10, p9 = pos[(pos.epoca == 10) & pos.hadm_id.isin(solo10)], pos[(pos.epoca == 9) & pos.hadm_id.isin(solo9)]
    l10 = d10[d10.hadm_id.isin(set(p10.hadm_id)) & d10.icd_code.str.startswith("Y92")]
    l9 = d9[d9.hadm_id.isin(set(p9.hadm_id)) & d9.icd_code.str.startswith("E849")]

    def n_hadm(d, prefijo):
        return int(d[d.icd_code.str.startswith(prefijo)].hadm_id.nunique())

    def titulo(prefijo, version):
        t = tit[[k for k in tit.index if k[1] == version and k[0].startswith(prefijo)][:1]]
        return str(t.iloc[0]) if len(t) else ""

    solo = pos.hadm_id.isin(set(p10.hadm_id) | set(p9.hadm_id))
    en_hospital = n_hadm(l10, "Y9223")
    lugares9 = {k[0]: str(v) for k, v in tit.items() if k[1] == 9 and k[0].startswith("E849")}
    por_srv = pos.assign(solo=solo).groupby("servicio").solo.agg(["size", "mean"])
    por_srv = por_srv[por_srv["size"] >= MIN_CELDA].sort_values("mean", ascending=False).head(5)
    N["caidas"] = {
        "positivas_solo_por_caida": int(solo.sum()), "pct_de_las_positivas": _pct(solo.mean()),
        "con_el_hospital_como_lugar": en_hospital, "sin_el_hospital_como_lugar": int(solo.sum()) - en_hospital,
        "pct_sin_el_hospital_como_lugar": _pct((int(solo.sum()) - en_hospital) / int(solo.sum())),
        "sin_codigo_de_lugar": int(solo.sum()) - int(l10.hadm_id.nunique()) - int(l9.hadm_id.nunique()),
        "cie10": {"n": int(len(p10)), "con_codigo_de_lugar": int(l10.hadm_id.nunique()),
                  "en_hospital": n_hadm(l10, "Y9223"), "pct_en_hospital": _pct(n_hadm(l10, "Y9223") / len(p10)),
                  "en_residencia_privada": n_hadm(l10, "Y920"),
                  "titulos_en_mimic": {"Y92.23x": titulo("Y9223", 10), "Y92.0x": titulo("Y920", 10)}},
        "cie9": {"n": int(len(p9)), "con_codigo_de_lugar": int(l9.hadm_id.nunique()), "en_el_hogar": n_hadm(l9, "E8490"),
                 "en_institucion_residencial": n_hadm(l9, "E8497"),
                 "titulos_en_mimic": {"E849.0": str(tit.get(("E8490", 9), "")), "E849.7": str(tit.get(("E8497", 9), ""))},
                 "codigos_de_lugar_en_el_diccionario": lugares9,
                 "algun_codigo_de_lugar_menciona_el_hospital": bool(any("hospital" in t.lower() for t in lugares9.values()))},
        "pct_de_las_positivas_del_servicio": {k: _pct(r["mean"]) for k, r in por_srv.iterrows()},
        "nota": "los códigos de caída (W) son de causa externa: dicen que hubo una caída, no dónde; el lugar va en otro código (Y92 o E849)"}
    # la familia CIE-9 «úlcera por presión» se define como 707.x: ¿cuántas de sus positivas tienen un código 707.0x?
    h_ulc = set(c9.loc[(c9.era == 9) & (c9.familia == "ulcera_presion"), "hadm_id"])
    ulc = pos[(pos.epoca == 9) & pos.hadm_id.isin(h_ulc)]
    sin_upp = ~ulc.hadm_id.isin(set(d9[d9.icd_code.str.startswith("7070")].hadm_id))
    N["ulcera_cie9"] = {"positivas_de_la_familia": int(len(ulc)), "sin_codigo_707_0": int(sin_upp.sum()),
                        "pct_sin_codigo_707_0": _pct(sin_upp.mean()) if len(ulc) else None,
                        "de_ellas_con_codigo_707_1": int((sin_upp & ulc.hadm_id.isin(set(d9[d9.icd_code.str.startswith("7071")].hadm_id))).sum()),
                        "titulos_en_mimic": {"707.0x": titulo("7070", 9), "707.1x": titulo("7071", 9)}}
    U = N["ulcera_cie9"]
    log.info(f"R3. familia CIE-9 de úlcera (707.x): {U['positivas_de_la_familia']} positivas · sin código 707.0x {U['sin_codigo_707_0']} "
             f"({U['pct_sin_codigo_707_0']:.1f} %) · de ellas con 707.1x {U['de_ellas_con_codigo_707_1']} · títulos {U['titulos_en_mimic']}")
    K = N["caidas"]
    log.info(f"R2. positivas solo por caída: {K['positivas_solo_por_caida']} ({K['pct_de_las_positivas']:.2f} % de las positivas) · CIE-10 "
             f"{K['cie10']['n']}: con lugar {K['cie10']['con_codigo_de_lugar']}, en hospital {K['cie10']['en_hospital']} "
             f"({K['cie10']['pct_en_hospital']:.1f} %), en residencia privada {K['cie10']['en_residencia_privada']} · CIE-9 {K['cie9']['n']}: "
             f"con lugar {K['cie9']['con_codigo_de_lugar']}, en el hogar {K['cie9']['en_el_hogar']} · por servicio "
             + ", ".join(f"{k} {v:.1f} %" for k, v in K["pct_de_las_positivas_del_servicio"].items())
             + f" · títulos {K['cie10']['titulos_en_mimic']} {K['cie9']['titulos_en_mimic']}")


# ------------------------------------------------------------------ bloque B
def _catalogos(R):
    """Lee el catálogo de 231 eventos del Anexo 02 y la tabla evento-CIE-10 (data/catalogos/, versionados) y registra su hash."""
    out = {}
    for archivo in ("anexo02_catalogo_231.csv", "anexo02_tabla_cie10.csv"):
        ruta = CATALOGOS / archivo
        out[f"data/catalogos/{archivo}"] = {"tamano_bytes": ruta.stat().st_size, "sha256": sha256(ruta)}
    R["eventos"] = {"fuentes": out}
    return pd.read_csv(CATALOGOS / "anexo02_catalogo_231.csv"), pd.read_csv(CATALOGOS / "anexo02_tabla_cie10.csv")


def analizar_eventos(df, R, log, FIG, pie):
    cat, tabla = _catalogos(R)
    E = R["eventos"]
    a1 = pd.read_csv(INTERIM / "a1_tier_a.csv")
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv")
    c9 = pd.read_csv(INTERIM / "positivos_cie9.csv")

    # B1. el catálogo
    E["catalogo"] = {"eventos": int(len(cat)), "naturalezas": int(cat.nat.nunique()),
                     "por_naturaleza": {k: int(v) for k, v in cat.nat.value_counts().items()},
                     "por_nivel_de_deteccion": {k: int(v) for k, v in cat.tier_a.value_counts().sort_index().items()},
                     "detectables": int((cat.tier_a != "E").sum())}

    # B2. qué eventos del catálogo tienen código. Hay dos tablas: la tabla evento-CIE-10 y las equivalencias CIE-10-CM.
    # Una fila de la tabla queda enlazada si su identificador existe en el catálogo con la misma naturaleza; en las filas tomadas
    # de la CIE-10 de la OMS se exige además el mismo nombre, porque esas filas usan una numeración que no es la del catálogo.
    # Las equivalencias nombran el evento con el nombre del catálogo, y se enlazan por ese nombre.
    nombre_cat, nat_cat = cat.set_index("id_ev").nombre, cat.set_index("id_ev").nat
    id_por_nombre = cat.assign(n=cat.nombre.map(_norm)).drop_duplicates("n").set_index("n").id_ev
    en_catalogo = tabla.id_evento_anexo02.isin(nombre_cat.index)
    misma_nat = tabla.id_evento_anexo02.map(nat_cat) == tabla.categoria_anexo02_essalud.map(NAT_MAPEO)
    mismo_nombre = tabla.evento_anexo02.map(_norm) == tabla.id_evento_anexo02.map(nombre_cat).fillna("").map(_norm)
    es_oms = tabla.fuente_normativa == FUENTE_OMS
    tabla["enlazada"] = misma_nat & (~es_oms | mismo_nombre)
    nombres_ok = set(tabla.loc[tabla.enlazada, "evento_anexo02"])
    assert not (nombres_ok & set(tabla.loc[~tabla.enlazada, "evento_anexo02"])), "un mismo evento no puede estar enlazado y sin enlazar"
    eq["id_cat"] = eq.evento.map(_norm).map(id_por_nombre)
    ids_tabla, ids_eq = set(tabla.loc[tabla.enlazada, "id_evento_anexo02"]), set(eq.id_cat.dropna().astype(int))
    ids_cod = ids_tabla | ids_eq

    # tipo de código de cada evento: propio, compartido con otro evento o aproximado (proxy)
    eventos_por_codigo = tabla.groupby("codigo_icd10").evento_anexo02.nunique()
    enl = tabla[tabla.enlazada].assign(proxy=lambda t: t.fuente_normativa.str.contains("proxy"),
                                       compartido=lambda t: t.codigo_icd10.map(eventos_por_codigo) > 1)
    tipo = {}
    for i, g in enl.groupby("id_evento_anexo02"):
        tipo[i] = ("código propio" if (~g.proxy & ~g.compartido).any()
                   else "código compartido con otro evento" if (~g.proxy).any() else "código aproximado (proxy)")
    for i in ids_eq - ids_tabla:
        tipo[i] = "equivalencia CIE-10-CM"
    cat["con_codigo"] = cat.id_ev.isin(ids_cod)
    cat["tipo_de_codigo"] = cat.id_ev.map(tipo).fillna("sin código")

    # eventos del nivel A que quedan «sin código» pero cuyo código documentado en el catálogo está en una fila sin enlace
    rx = re.compile(r"[A-Z]\d{2}(?:\.\d+)?")
    cod_sin_enlace = set(tabla.loc[~tabla.enlazada, "codigo_icd10"].str.replace(".", "", regex=False))
    a_sin = cat[(cat.tier_a == "A") & ~cat.con_codigo]
    candidatos = int(sum(any(t.startswith(c.replace(".", "")) for c in rx.findall(str(r)) for t in cod_sin_enlace) for r in a_sin.razon))
    M = E["mapeo"] = {
        "filas": int(len(tabla)), "codigos_cie10_distintos": int(tabla.codigo_icd10.nunique()),
        "eventos_nombrados": int(tabla.evento_anexo02.nunique()),
        "por_fuente": {k: int(v) for k, v in tabla.fuente_normativa.value_counts().items()},
        "filas_enlazadas_al_catalogo": int(tabla.enlazada.sum()),
        "filas_enlazadas_por_fuente": {k: int(v) for k, v in tabla.loc[tabla.enlazada, "fuente_normativa"].value_counts().items()},
        "filas_sin_enlace": int((~tabla.enlazada).sum()),
        "filas_sin_enlace_por_fuente": {k: int(v) for k, v in tabla.loc[~tabla.enlazada, "fuente_normativa"].value_counts().items()},
        "filas_sin_enlace_detalle": {"identificador_fuera_del_catalogo": int((~en_catalogo).sum()),
                                     "identificador_de_otra_naturaleza": int((en_catalogo & ~misma_nat).sum()),
                                     "misma_naturaleza_y_otro_evento": int((misma_nat & ~tabla.enlazada).sum())},
        "eventos_con_codigo": len(ids_cod), "eventos_con_codigo_por_la_tabla": len(ids_tabla),
        "eventos_con_codigo_solo_por_equivalencias": len(ids_eq - ids_tabla),
        "eventos_por_tipo_de_codigo": {k: int(v) for k, v in cat.tipo_de_codigo.value_counts().items()},
        "con_codigo_por_nivel_de_deteccion": {k: {"eventos": int(len(g)), "con_codigo": int(g.con_codigo.sum())}
                                              for k, g in cat.groupby("tier_a")},
        "nivel_A_sin_codigo": int(len(a_sin)), "nivel_A_sin_codigo_con_su_codigo_en_filas_sin_enlace": candidatos,
        "codigos_asignados_a_mas_de_un_evento": int((eventos_por_codigo > 1).sum())}

    # B3. lo observado en MIMIC-IV en la época CIE-10 (hospitalizaciones con código, tengan o no epicrisis). Se unen las dos tablas
    # de etiquetas; cada evento se identifica por su número del catálogo si está enlazado, o por su nombre si es una complicación.
    info = tabla.drop_duplicates("evento_anexo02").set_index("evento_anexo02")
    a1["id_cat"] = a1.evento.map(info.id_evento_anexo02).where(a1.evento.isin(nombres_ok))
    a1["nat_catalogo"] = a1.evento.map(info.categoria_anexo02_essalud).map(NAT_MAPEO)
    eq["nat_catalogo"] = eq.id_cat.map(nat_cat).fillna(eq.naturaleza.map(NAT_MAPEO))
    cols = ["hadm_id", "evento", "estrato", "id_cat", "nat_catalogo"]
    todo = pd.concat([a1[cols].assign(tabla="directa"), eq[cols].assign(tabla="equivalencia")], ignore_index=True)
    todo["del_catalogo"] = todo.id_cat.notna()
    todo["nombre"] = todo.id_cat.map(nombre_cat).fillna(todo.evento)
    todo["clave"] = np.where(todo.del_catalogo, "cat:" + todo.id_cat.astype("Int64").astype(str), "comp:" + todo.evento)
    nombre_de = todo.drop_duplicates("clave").set_index("clave").nombre
    causal = todo[todo.estrato == "A1"]
    frec = todo.groupby("clave").hadm_id.nunique().sort_values(ascending=False)
    frec_causal = causal.groupby("clave").hadm_id.nunique()
    acum = frec.cumsum() / frec.sum()
    ids_obs = set(todo.loc[todo.del_catalogo, "id_cat"].astype(int))
    ids_causal = set(causal.loc[causal.del_catalogo, "id_cat"].astype(int))
    assert ids_obs <= ids_cod and ids_causal <= ids_obs, "el embudo de eventos debe ser anidado"

    def estrato(d):
        h, he = d.hadm_id.nunique(), d[d.del_catalogo].hadm_id.nunique()
        return {"hospitalizaciones": int(h), "con_evento_del_catalogo": int(he), "solo_complicaciones_sin_enlace": int(h - he),
                "eventos": int(d.clave.nunique()), "eventos_del_catalogo": int(d[d.del_catalogo].clave.nunique()),
                "por_tabla": {t: int(d[d.tabla == t].hadm_id.nunique()) for t in ("directa", "equivalencia")}}

    O = E["observado"] = {
        "hospitalizaciones_con_codigo_de_evento": int(todo.hadm_id.nunique()),
        "eventos_observados": int(len(frec)),
        "eventos_del_catalogo_observados": len(ids_obs),
        "eventos_del_catalogo_con_codigo_de_causa_explicita": len(ids_causal),
        "eventos_del_catalogo_solo_con_codigo_de_condicion": len(ids_obs - ids_causal),
        "complicaciones_sin_enlace_observadas": int(todo.loc[~todo.del_catalogo, "clave"].nunique()),
        "causa_explicita": estrato(causal), "condicion": estrato(todo[todo.estrato == "A2"]),
        "eventos_que_concentran_el_80_pct": int((acum < 0.80).sum() + 1),
        "top": [{"evento": str(nombre_de[k]), "hospitalizaciones": int(n), "con_codigo_de_causa_explicita": int(frec_causal.get(k, 0)),
                 "en_el_catalogo": bool(k.startswith("cat:"))} for k, n in frec.head(15).items()],
        "top_causa_explicita": [{"evento": str(nombre_de[k]), "hospitalizaciones": int(n), "en_el_catalogo": bool(k.startswith("cat:"))}
                                for k, n in frec_causal.sort_values(ascending=False).head(10).items()],
        "nota": "«causa explícita» es el estrato de códigos que da la etiqueta positiva; «condición» es el de códigos que pueden ser el "
                "motivo de ingreso y no se usan como etiqueta"}

    # B4. qué parte del corpus de modelado conserva el evento
    pos = df[df.y == 1]
    h_etq, h_cat = set(causal.hadm_id), set(causal.loc[causal.del_catalogo, "hadm_id"])
    nombrado, en_cat = pos.hadm_id.isin(h_etq), pos.hadm_id.isin(h_cat)
    fam = c9[c9.hadm_id.isin(pos.loc[~nombrado, "hadm_id"]) & (c9.era == 9)].drop_duplicates(["hadm_id", "familia"]).familia.value_counts()
    en_corpus = causal[causal.del_catalogo & causal.hadm_id.isin(set(pos.hadm_id))].drop_duplicates(["hadm_id", "id_cat"])
    notas_ev = en_corpus.groupby("id_cat").hadm_id.nunique()
    notas_ev_test = en_corpus[en_corpus.hadm_id.isin(set(pos.loc[pos.split == "test", "hadm_id"]))].groupby("id_cat").hadm_id.nunique()
    ids_corpus = set(notas_ev.index.astype(int))
    C = E["corpus"] = {
        "positivas": int(len(pos)),
        "con_evento_del_catalogo": int(en_cat.sum()), "pct_con_evento_del_catalogo": _pct(en_cat.mean()),
        "solo_complicacion_sin_enlace": int((nombrado & ~en_cat).sum()), "pct_solo_complicacion_sin_enlace": _pct((nombrado & ~en_cat).mean()),
        "solo_familia_cie9": int((~nombrado).sum()), "pct_solo_familia_cie9": _pct((~nombrado).mean()),
        "familias_cie9": {k: int(v) for k, v in fam.items()},
        "eventos_del_catalogo": {"presentes": len(ids_corpus), "con_100_epicrisis_o_mas": int((notas_ev >= 100).sum()),
                                 "con_30_epicrisis_o_mas_en_test": int((notas_ev_test >= 30).sum())},
        "equivalencias_cie10cm": {"eventos": int(eq.evento.nunique()),
                                  "eventos_con_el_nombre_del_catalogo": int(eq.loc[eq.id_cat.notna(), "evento"].nunique())}}

    # por naturaleza del Anexo 02 (solo eventos del catálogo)
    def por_nat(ids):
        return cat[cat.id_ev.isin(ids)].nat.value_counts()

    cols_nat = {"con_codigo": por_nat(ids_cod), "observados": por_nat(ids_obs), "con_codigo_de_causa_explicita": por_nat(ids_causal),
                "en_las_positivas_del_corpus": por_nat(ids_corpus)}
    E["por_naturaleza"] = {n: {"en_el_catalogo": int(k), **{c: int(s.get(n, 0)) for c, s in cols_nat.items()}}
                           for n, k in cat.nat.value_counts().items()}
    E["naturalezas_sin_codigo"] = [n for n, v in E["por_naturaleza"].items() if v["con_codigo"] == 0]
    E["naturalezas_sin_eventos_observados"] = [n for n, v in E["por_naturaleza"].items() if v["observados"] == 0]

    # una fila por evento del catálogo (los conteos menores de 10 se escriben «<10»)
    def conteo(s):
        s = pd.Series(s.values, index=s.index.astype(int))
        return cat.id_ev.map(s).fillna(0).astype(int).map(lambda n: "<10" if 0 < n < 10 else str(n))

    cond = todo[(todo.estrato == "A2") & todo.del_catalogo]
    pd.DataFrame({
        "id_evento": cat.id_ev, "evento": cat.nombre, "naturaleza": cat.nat.str.capitalize(),
        "nivel_de_deteccion": cat.tier_a.map(NIVEL), "tipo_de_codigo": cat.tipo_de_codigo,
        "hospitalizaciones_con_codigo_de_causa_explicita": conteo(causal[causal.del_catalogo].groupby("id_cat").hadm_id.nunique()),
        "hospitalizaciones_con_codigo_de_condicion": conteo(cond.groupby("id_cat").hadm_id.nunique()),
        "epicrisis_positivas_en_el_corpus": conteo(notas_ev),
    }).to_csv(REPORTES / "eda_eventos_231.csv", index=False, encoding="utf-8")

    log.info(f"B1. catálogo: {E['catalogo']['eventos']} eventos en {E['catalogo']['naturalezas']} naturalezas · detectables "
             f"{E['catalogo']['detectables']} · por nivel {E['catalogo']['por_nivel_de_deteccion']}")
    log.info(f"B2. tabla de códigos: {M['filas']} filas, {M['codigos_cie10_distintos']} códigos, por fuente {M['por_fuente']} · filas "
             f"enlazadas al catálogo {M['filas_enlazadas_al_catalogo']} {M['filas_enlazadas_por_fuente']} · sin enlace "
             f"{M['filas_sin_enlace']} {M['filas_sin_enlace_detalle']} · eventos con código {M['eventos_con_codigo']} de {len(cat)} "
             f"(por la tabla {M['eventos_con_codigo_por_la_tabla']}, solo por equivalencias {M['eventos_con_codigo_solo_por_equivalencias']}) · "
             f"tipo de código {M['eventos_por_tipo_de_codigo']} · por nivel {M['con_codigo_por_nivel_de_deteccion']} · nivel A sin código "
             f"{M['nivel_A_sin_codigo']}, de ellos con su código en filas sin enlace {candidatos}")
    log.info(f"B3. observado: {O['eventos_observados']} eventos en {_mil(O['hospitalizaciones_con_codigo_de_evento'])} hospitalizaciones · "
             f"del catálogo {len(ids_obs)} (con código de causa explícita {len(ids_causal)}, solo de condición {len(ids_obs - ids_causal)}) · "
             f"causa explícita {O['causa_explicita']} · condición {O['condicion']} · {O['eventos_que_concentran_el_80_pct']} eventos "
             f"concentran el 80 % · naturalezas sin eventos observados {E['naturalezas_sin_eventos_observados']}")
    log.info(f"B4. corpus: positivas {C['positivas']} · con evento del catálogo {C['con_evento_del_catalogo']} "
             f"({C['pct_con_evento_del_catalogo']:.2f} %) · con complicación sin enlace {C['solo_complicacion_sin_enlace']} "
             f"({C['pct_solo_complicacion_sin_enlace']:.2f} %) · solo familia CIE-9 {C['solo_familia_cie9']} "
             f"({C['pct_solo_familia_cie9']:.2f} %) {C['familias_cie9']} · eventos del catálogo {C['eventos_del_catalogo']} · tabla por "
             "evento -> reportes/eda_eventos_231.csv")

    # ---------------- figura: del catálogo a los datos
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"width_ratios": [1, 1.2]})
    pasos = [("En el catálogo del Anexo 02", len(cat), GRIS), ("Con código CIE-10 en las tablas", len(ids_cod), AZUL),
             ("Observados en MIMIC-IV", len(ids_obs), VERDE), ("Observados con código\nde causa explícita", len(ids_causal), AMBAR),
             ("En las positivas del corpus", len(ids_corpus), ROJO)]
    ax[0].barh([p[0] for p in pasos][::-1], [p[1] for p in pasos][::-1], color=[p[2] for p in pasos][::-1])
    for i, p in enumerate(pasos[::-1]):
        ax[0].text(p[1] + 3, i, f"{p[1]}  ({100 * p[1] / len(cat):.0f} %)", va="center", fontsize=9)
    ax[0].set_title("Del catálogo a los datos: eventos del Anexo 02", fontsize=10); ax[0].set(xlabel="eventos", xlim=(0, len(cat) * 1.32))
    nats = list(E["por_naturaleza"].keys())[::-1]
    yy = np.arange(len(nats)); h = 0.2
    for k, (clave, color, nombre) in enumerate([("en_el_catalogo", GRIS, "en el catálogo"), ("con_codigo", AZUL, "con código CIE-10"),
                                                 ("observados", VERDE, "observados en MIMIC-IV"),
                                                 ("en_las_positivas_del_corpus", ROJO, "en las positivas del corpus")]):
        ax[1].barh(yy + (1.5 - k) * h, [E["por_naturaleza"][n][clave] for n in nats], h, color=color, label=nombre)
    ax[1].set_yticks(yy, [n.capitalize() for n in nats], fontsize=8.5); ax[1].set_title("Por naturaleza del Anexo 02", fontsize=10)
    ax[1].set_xlabel("eventos"); ax[1].legend(fontsize=8, loc="lower right")
    niv, det, tp = M["con_codigo_por_nivel_de_deteccion"], M["filas_sin_enlace_detalle"], M["eventos_por_tipo_de_codigo"]
    pie(fig, f"Cómo leer: a la izquierda, cuántos de los {len(cat)} eventos del Anexo 02 llegan a cada etapa: tienen un código CIE-10 en "
             "las tablas del proyecto, aparecen en alguna hospitalización de MIMIC-IV codificada en CIE-10, aparecen con un código de causa "
             "explícita (el único que da la etiqueta positiva) y están entre las epicrisis positivas del corpus de modelado. A la derecha, "
             f"cuatro de esos conteos por naturaleza. Lectura: {len(ids_cod)} eventos tienen código, {len(ids_obs)} se observan, "
             f"{len(ids_causal)} con código de causa explícita, y {len(ids_corpus)} están en el corpus. De los que tienen código, "
             f"{tp.get('código propio', 0)} tienen un código propio, {tp.get('código compartido con otro evento', 0)} solo códigos que "
             f"comparten con otro evento, {tp.get('código aproximado (proxy)', 0)} solo un código aproximado y "
             f"{tp.get('equivalencia CIE-10-CM', 0)} entran por la tabla de equivalencias. Según cómo puede detectarse cada evento, tienen "
             f"código {niv['A']['con_codigo']} de los {niv['A']['eventos']} del nivel de código CIE-10, {niv['B']['con_codigo']} de "
             f"{niv['B']['eventos']} del nivel de texto, {niv['C']['con_codigo']} de {niv['C']['eventos']} del de variables estructuradas, "
             f"{niv['D']['con_codigo']} de {niv['D']['eventos']} del de auditoría del registro y {niv['E']['con_codigo']} de "
             f"{niv['E']['eventos']} del no detectable. Además, {M['filas_sin_enlace']} de las {M['filas']} filas de la tabla llevan un "
             f"identificador que no es el del catálogo (en {det['identificador_fuera_del_catalogo']} no existe, en "
             f"{det['identificador_de_otra_naturaleza']} es de otra naturaleza y en {det['misma_naturaleza_y_otro_evento']} es de otro "
             "evento), así que el conteo de eventos con código es un mínimo.", 205, 0.28)
    fig.savefig(FIG / "fig_eda_eventos_catalogo.png", dpi=130); plt.close(fig)

    # ---------------- figura: frecuencia por evento y origen de la etiqueta
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"width_ratios": [1.9, 1]})
    top = O["top"][::-1]
    nombres = [("■ " if t["en_el_catalogo"] else "") + textwrap.shorten(t["evento"], 46, placeholder="…") for t in top]
    v1 = [t["con_codigo_de_causa_explicita"] for t in top]
    ax[0].barh(nombres, v1, color=ROJO, label="con código de causa explícita (da la etiqueta positiva)")
    ax[0].barh(nombres, [t["hospitalizaciones"] - a for t, a in zip(top, v1)], left=v1, color=GRIS,
               label="solo con código de condición (no da etiqueta)")
    ax[0].set_title("Los 15 eventos más frecuentes en MIMIC-IV (época CIE-10)", fontsize=10); ax[0].set_xlabel("hospitalizaciones")
    ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].tick_params(axis="y", labelsize=8)
    vals = [C["con_evento_del_catalogo"], C["solo_complicacion_sin_enlace"], C["solo_familia_cie9"]]
    ax[1].bar(["evento del\ncatálogo", "complicación sin\nenlace con el catálogo", "solo familia\nCIE-9"], vals, color=[VERDE, AMBAR, GRIS])
    for i, v in enumerate(vals):
        ax[1].text(i, v + max(vals) * 0.015, f"{_mil(v)}\n({100 * v / C['positivas']:.1f} %)", ha="center", fontsize=8.5)
    ax[1].set_title("Epicrisis positivas del corpus\nsegún lo que se sabe del evento", fontsize=10); ax[1].set_ylabel("epicrisis")
    ax[1].set_ylim(0, max(vals) * 1.2); ax[1].tick_params(axis="x", labelsize=7.5)
    ec = C["eventos_del_catalogo"]
    pie(fig, f"Cómo leer: a la izquierda, los 15 eventos con más hospitalizaciones entre las {_mil(O['hospitalizaciones_con_codigo_de_evento'])} "
             "que tienen algún código de evento en la época CIE-10, tengan o no epicrisis; en rojo, la parte con código de causa "
             "explícita, que es la única que se usa como etiqueta positiva; el cuadro ■ marca los eventos del catálogo del Anexo 02 y los "
             "demás son complicaciones de la CIE-10 sin enlace con el catálogo, con el nombre que tienen en la tabla. A la derecha, qué "
             f"se sabe del evento en las {_mil(C['positivas'])} epicrisis positivas del corpus. Lectura: los eventos más frecuentes son "
             "condiciones que no dan etiqueta. En el corpus, "
             f"{_mil(C['con_evento_del_catalogo'])} positivas ({C['pct_con_evento_del_catalogo']:.1f} %) tienen un evento del catálogo, "
             f"{_mil(C['solo_complicacion_sin_enlace'])} ({C['pct_solo_complicacion_sin_enlace']:.1f} %) una complicación sin enlace y "
             f"{_mil(C['solo_familia_cie9'])} ({C['pct_solo_familia_cie9']:.1f} %) vienen de la época CIE-9, donde la etiqueta solo "
             f"distingue familias de códigos. De los {ec['presentes']} eventos del catálogo presentes en el corpus, "
             f"{ec['con_100_epicrisis_o_mas']} tienen 100 epicrisis o más: no alcanza para un clasificador por evento.", 205, 0.25)
    fig.savefig(FIG / "fig_eda_eventos_frecuencia.png", dpi=130); plt.close(fig)
    log.info("figuras: fig_eda_eventos_catalogo.png, fig_eda_eventos_frecuencia.png")


# ------------------------------------------------------------------ flujo
def figura_flujo(R, ruta, fuentes_decision):
    """Diagrama del flujo del EDA: quince pasos en orden, con el resultado de cada uno y su estado.
    Las cifras y el estado de cada paso salen del diccionario de resultados R."""
    ESTADO = {"dato": ("#EAF2F8", AZUL, "Dato"), "ok": ("#E8F5EE", VERDE, "Controlado"), "atencion": ("#FFF3E6", AMBAR, "Atención")}
    V, E, T, P, N = R["variables"], R["eventos"], R["texto"], R["prevalencia_real"], R["ruido_etiqueta"]
    cy, dec, F = R["correlaciones_spearman"]["con_y"], R["largo"]["pct_positivas_por_decil"], R["fuga_codigos"]
    falt = sorted([d for d in V["diccionario"] if d["faltantes"] > 0], key=lambda d: -d["pct_faltantes"])[:2]
    num = sorted(V["numericas"], key=lambda k: -abs(V["numericas"][k]["auc_sola"] - 0.5))[:2]
    catv = sorted([k for k in V["categoricas"] if k not in ("periodo", "destino_alta")], key=lambda k: -V["categoricas"][k]["cramers_v"])[:2]
    cy_top = sorted([k for k in cy], key=lambda k: -abs(cy[k]))[:2]
    e9, e10 = P["por_epoca"]["CIE-9"]["pct_positivas_con_epicrisis"], P["por_epoca"]["CIE-10"]["pct_positivas_con_epicrisis"]
    part, C, O = V["particion"], E["corpus"], E["observado"]
    fam = F["familias_de_la_etiqueta"]
    pasos = [
        ("Datos de entrada", "Corpus de modelado, con partición por paciente",
         f"{_mil(R['entrada']['notas'])} epicrisis · {_mil(R['entrada']['pacientes'])} pacientes · semilla {R['semilla']}", "dato"),
        ("Variables y faltantes", "¿Qué variables hay y cuánto dato falta?",
         f"{len(V['diccionario'])} variables · sin dato: "
         + " y ".join(f"{_min1(CORTA.get(d['variable'], d['etiqueta']))} {d['pct_faltantes']:.0f} %" for d in falt),
         "atencion" if falt and falt[0]["pct_faltantes"] > 5 else "ok"),
        ("Calidad y partición", "Duplicados, fuera de rango y train frente a test",
         f"{R['calidad']['duplicados']['note_id']} duplicados · {V['fuera_de_rango']['estancia_negativa']} estancias negativas · positivas: "
         f"{part['train']['pct_positivas']:.1f} % en train y {part['test']['pct_positivas']:.1f} % en test",
         "ok" if abs(part["train"]["pct_positivas"] - part["test"]["pct_positivas"]) < 2 else "atencion"),
        ("Variables numéricas", "¿Cuáles separan las clases por sí solas?",
         " · ".join(f"{_min1(CORTA[k])}: AUC {V['numericas'][k]['auc_sola']:.3f}" for k in num),
         "atencion" if V["numericas"][num[0]]["auc_sola"] > 0.65 else "dato"),
        ("Variables categóricas", "¿Cambia el evento según el grupo?",
         " · ".join(f"{_min1(CORTA[k])}: V = {V['categoricas'][k]['cramers_v']:.3f}" for k in catv),
         "atencion" if V["categoricas"][catv[0]]["cramers_v"] > 0.1 else "dato"),
        ("Contexto sin texto", "¿Cuánto se acierta sin leer la nota?",
         f"ROC-AUC {V['contexto']['al_ingreso']['roc_auc']:.3f} al ingreso y {V['contexto']['al_alta']['roc_auc']:.3f} al alta"
         + (f" · con texto {V['contexto']['modelo_de_texto']['roc_auc']:.3f}" if "modelo_de_texto" in V["contexto"] else ""),
         "atencion" if V["contexto"]["al_ingreso"]["roc_auc"] > 0.6 else "ok"),
        ("Balance y prevalencia", "¿Cuántas hospitalizaciones tienen evento?",
         f"{R['balance']['All']['pct_positivas']:.1f} % en el corpus · {P['todas']['pct_positivas_con_epicrisis']:.1f} % en las "
         "hospitalizaciones con epicrisis", "dato"),
        ("Época y periodo", "¿Cambia la etiqueta con los años? (drift)",
         f"prevalencia real: {e9:.1f} % en CIE-9 y {e10:.1f} % en CIE-10 · en el corpus queda fija por el muestreo",
         "atencion" if abs(e9 - e10) > 5 else "ok"),
        ("Largo de la nota", "¿El largo predice el evento por sí solo? (atajo)",
         f"AUC = {R['largo']['auc_largo_solo']:.3f} · positivas del decil 1 al 10: {dec[0]:.1f} % → {dec[-1]:.1f} %",
         "atencion" if R["largo"]["auc_largo_solo"] > 0.55 else "ok"),
        ("Contenido del texto", "¿Qué secciones y qué palabras traen las notas?",
         f"{T['secciones']['presentes_en_mas_del_95_pct']} de {len(T['secciones']['por_seccion'])} secciones en más del 95 % · "
         f"{_mil(T['vocabulario']['terminos'])} términos frecuentes", "dato"),
        ("Fugas de información", "¿La nota trae la respuesta? ¿Se repiten pacientes?",
         f"códigos de la etiqueta escritos en {fam['notas']} notas · {R['pacientes']['compartidos_train_test']} pacientes compartidos",
         "ok" if fam["pct_positivas"] < 0.5 and R["pacientes"]["compartidos_train_test"] == 0 else "atencion"),
        ("Catálogo de eventos", f"¿Cuántos de los {E['catalogo']['eventos']} eventos llegan a los datos?",
         f"{E['mapeo']['eventos_con_codigo']} con código · {O['eventos_del_catalogo_observados']} observados · "
         f"{C['eventos_del_catalogo']['presentes']} en el corpus", "atencion"),
        ("Origen de la etiqueta", "¿De qué eventos salen las positivas?",
         f"{C['pct_con_evento_del_catalogo']:.1f} % con evento del catálogo · {C['pct_solo_familia_cie9']:.1f} % solo con familia CIE-9",
         "atencion" if C["pct_con_evento_del_catalogo"] < 50 else "ok"),
        ("Ruido de la etiqueta", "¿Hay casos mal puestos en cada clase?",
         f"{N['negativas']['cie10']['pct_con_alguno']:.1f} % de las negativas CIE-10 con código causal · "
         f"{N['caidas']['pct_de_las_positivas']:.1f} % de las positivas solo por caída",
         "atencion" if N["negativas"]["cie10"]["pct_con_alguno"] > 1 or N["caidas"]["pct_de_las_positivas"] > 5 else "ok"),
        ("Correlaciones", "Spearman de cada variable con la etiqueta",
         " · ".join(f"{k}: {cy[k]:.3f}" for k in cy_top), "atencion" if abs(cy[cy_top[0]]) > 0.3 else "dato"),
    ]
    fig, ax = plt.subplots(figsize=(16, 13.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 84); ax.axis("off")
    ax.text(50, 83.0, "Flujo del análisis exploratorio (EDA) y resultado de cada paso", ha="center", va="center",
            fontsize=14, fontweight="bold", color=AZUL)
    for i, e in enumerate(["Ingesta", "Preprocesado", "EDA (este análisis)", "Baseline"]):
        x = 14 + i * 19
        ax.add_patch(FancyBboxPatch((x, 76.2), 15, 4.4, boxstyle="round,pad=0.25,rounding_size=0.8", fc=AZUL if i == 2 else "white", ec=AZUL, lw=1.6))
        ax.text(x + 7.5, 78.4, e, ha="center", va="center", fontsize=10, fontweight="bold", color="white" if i == 2 else AZUL)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 15.4, 78.4), (x + 18.6, 78.4), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))
    ax.text(1.5, 78.4, "Pipeline:", ha="left", va="center", fontsize=10, color="#444", fontweight="bold")
    for k, (fc, ec, nombre) in enumerate(ESTADO.values()):
        x = 66 + k * 11
        ax.add_patch(FancyBboxPatch((x, 72.6), 2.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.3", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 2.9, 73.4, nombre, ha="left", va="center", fontsize=8.5, color="#333")
    W, H, G = 17.6, 17.6, 2.0
    filas_y = [53.4, 33.0, 12.6]
    cajas = []
    for n, (titulo, que, resultado, estado) in enumerate(pasos):
        fila, col = divmod(n, 5)
        x, y = 1.5 + col * (W + G), filas_y[fila]
        fc, ec, nombre = ESTADO[estado]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.6))
        ax.text(x + 0.9, y + H - 1.2, f"{n + 1} · {titulo}", ha="left", va="top", fontsize=9.4, fontweight="bold", color=ec)
        ax.text(x + 0.9, y + H - 4.0, "\n".join(textwrap.wrap(que, 34)), ha="left", va="top", fontsize=7.6, style="italic", color="#444", linespacing=1.3)
        resultado = resultado.replace(" %", " %").replace("V = ", "V = ").replace("AUC ", "AUC ")
        ax.text(x + 0.9, y + H - 8.0, "\n".join(textwrap.wrap(resultado, 27)), ha="left", va="top", fontsize=8.1, color="#111",
                fontweight="bold", linespacing=1.33)
        ax.add_patch(FancyBboxPatch((x + 0.9, y + 0.7), 6.4, 1.8, boxstyle="round,pad=0.12,rounding_size=0.5", fc=ec, ec=ec))
        ax.text(x + 4.1, y + 1.6, nombre, ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
        cajas.append((x, y))
    for n in range(len(pasos) - 1):
        (xa, ya), (xb, yb) = cajas[n], cajas[n + 1]
        if ya == yb:
            ax.add_patch(FancyArrowPatch((xa + W + 0.35, ya + H / 2), (xb - 0.35, yb + H / 2), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))
        else:
            ym = ya - 1.4
            ax.plot([xa + W / 2, xa + W / 2, xb + W / 2], [ya - 0.4, ym, ym], color=AZUL, lw=1.5)
            ax.add_patch(FancyArrowPatch((xb + W / 2, ym), (xb + W / 2, yb + H + 0.4), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))
    ax.add_patch(FancyBboxPatch((1.5, 0.6), 96.0, 9.4, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=AZUL, lw=1.8))
    ax.text(3.0, 8.7, "Decisiones para el Sprint 2", ha="left", va="center", fontsize=10.5, fontweight="bold", color=AZUL)
    for k, d in enumerate(R["decisiones"]):
        ax.text(3.0 + k * 31.6, 6.6, f"D{d['n']} · " + "\n".join(textwrap.wrap(d["decision"], 50)) + f"\n(sale {fuentes_decision[k]})",
                ha="left", va="top", fontsize=8.2, color="#111", linespacing=1.33)
    xc = cajas[12][0] + W / 2
    ax.add_patch(FancyArrowPatch((xc, cajas[12][1] - 0.4), (xc, 10.4), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))
    ax.text(1.5, -1.3, "Cómo leer: cada caja es un paso del análisis; en cursiva, la pregunta que responde, y en negrita, el resultado de "
            "esta corrida. Las flechas marcan el orden, y cada decisión de abajo indica de qué paso sale.", ha="left", va="top",
            fontsize=9.5, color="#333")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
