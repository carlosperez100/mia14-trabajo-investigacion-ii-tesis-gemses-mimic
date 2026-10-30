# -*- coding: utf-8 -*-
"""
Variantes de los DATOS para el cierre del Sprint 2 (semana 6). Marca, por epicrisis del corpus, los casos
cuya etiqueta es dudosa según el EDA de la semana 4 (decisiones 1 y 2):

  negativa_dudosa  negativa con algún código de las familias de causa explícita (T80 a T88, Y63, Y65, Y83,
                   Y84, W00 a W19, L89) o de efecto adverso de medicamento (T36 a T50 con 5 en el sexto
                   carácter) que no está en la tabla de la etiqueta; en CIE-9, con un código de las familias.
  positiva_dudosa  positiva cuya única evidencia es una caída sin el hospital como lugar de ocurrencia
                   (en CIE-10, sin código Y92.23; en CIE-9 no hay código de lugar para el hospital), o una
                   úlcera de la familia CIE-9 (707.x) sin código de úlcera por presión (707.0x).

Variantes:  D0 datos actuales · D1 sin negativas dudosas · D2 sin negativas ni positivas dudosas.
Las variantes cambian solo las filas con que se ENTRENA; la evaluación se hace sobre las mismas filas.

Salida en caché (fuera de git): data/interim/variantes_datos.parquet (hadm_id y marcas).
"""
import pandas as pd

import eda_variables as ev
from utils import INTERIM

VARIANTES = {"D0": "datos actuales",
             "D1": "sin negativas dudosas",
             "D2": "sin negativas ni positivas dudosas"}


def marcar(df, log, cache=None):
    """Devuelve df con las columnas negativa_dudosa, positiva_dudosa y motivo (caché por defecto: la del corpus)."""
    cache = cache or INTERIM / "variantes_datos.parquet"
    if cache.exists():
        m = pd.read_parquet(cache)
        if set(df.hadm_id) <= set(m.hadm_id):
            log.info(f"variantes de datos: se reutiliza {cache.name}")
            return df.drop(columns=[c for c in ("negativa_dudosa", "positiva_dudosa", "motivo") if c in df], errors="ignore").merge(m, on="hadm_id", how="left")

    dx = pd.read_csv(ev._ruta_mimic("hosp/diagnoses_icd.csv.gz"), usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dx = dx[dx.hadm_id.isin(set(df.hadm_id))]
    d10, d9 = dx[dx.icd_version == 10], dx[dx.icd_version == 9]
    fam9 = {k: set(d9.loc[d9.icd_code.str.contains(v), "hadm_id"]) for k, v in ev.FAM_CIE9.items()}
    causal10 = set(d10.loc[d10.icd_code.str.startswith(ev.PREF_CAUSAL) | ev._efecto_adverso(d10.icd_code), "hadm_id"])
    causal9 = set().union(*fam9.values())

    # negativas dudosas
    neg = (df.y == 0) & (((df.epoca == 10) & df.hadm_id.isin(causal10)) | ((df.epoca == 9) & df.hadm_id.isin(causal9)))

    # positivas cuya única evidencia es una caída sin el hospital como lugar
    a1 = pd.read_csv(INTERIM / "a1_tier_a.csv", usecols=["hadm_id", "estrato", "icd_code"])
    a1 = a1[a1.estrato == "A1"]
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv", usecols=["hadm_id", "estrato"])
    todas_w = a1.assign(w=a1.icd_code.str.startswith("W")).groupby("hadm_id").w.all()
    solo_caida10 = set(todas_w[todas_w].index) - set(eq.loc[eq.estrato == "A1", "hadm_id"])
    en_hospital = set(d10.loc[d10.icd_code.str.startswith("Y9223"), "hadm_id"])
    otras9 = lambda k: set().union(*(s for f, s in fam9.items() if f != k))
    solo_caida9 = fam9["caidas"] - otras9("caidas")
    caida = (df.y == 1) & (((df.epoca == 10) & df.hadm_id.isin(solo_caida10 - en_hospital)) | ((df.epoca == 9) & df.hadm_id.isin(solo_caida9)))

    # positivas cuya única evidencia es una úlcera 707.x sin código de úlcera por presión (707.0x)
    upp = set(d9.loc[d9.icd_code.str.startswith("7070"), "hadm_id"])
    ulcera = (df.y == 1) & (df.epoca == 9) & df.hadm_id.isin((fam9["ulcera_presion"] - otras9("ulcera_presion")) - upp)

    out = pd.DataFrame({"hadm_id": df.hadm_id.values, "negativa_dudosa": neg.values, "positiva_dudosa": (caida | ulcera).values,
                        "motivo": ["negativa con código causal" if n else "caída sin lugar hospital" if c else "úlcera sin 707.0x" if u else ""
                                   for n, c, u in zip(neg, caida, ulcera)]})
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log.info(f"variantes de datos: negativas dudosas {int(neg.sum())} · positivas dudosas {int((caida | ulcera).sum())} "
             f"(caída sin lugar hospital {int(caida.sum())}, úlcera sin 707.0x {int(ulcera.sum())})")
    return df.merge(out, on="hadm_id", how="left")


def filas_de_entrenamiento(d, variante):
    """Máscara de las filas que se usan para ENTRENAR en cada variante."""
    if variante == "D0":
        return pd.Series(True, index=d.index)
    if variante == "D1":
        return ~d.negativa_dudosa
    return ~(d.negativa_dudosa | d.positiva_dudosa)
