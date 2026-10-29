# -*- coding: utf-8 -*-
"""
VARIABLES DE DOMINIO (Sprint 2, semana 5) — paciente, servicio, hospitalización y atención.

Para cada hospitalización del corpus se calculan variables conocidas al momento de escribir la epicrisis,
a partir de tablas de MIMIC-IV que NO son la codificación de diagnósticos de la que sale la etiqueta.

  paciente     edad, sexo, tipo de ingreso, procedencia (admission_location), seguro, horas en emergencia
  servicio     servicio al ingreso, servicio al alta, n.º de servicios distintos, paso por servicio quirúrgico
  hospital     estancia (días), n.º de traslados, n.º de unidades distintas, paso por UCI, días en UCI, destino al alta
  atencion     n.º de medicamentos distintos prescritos, n.º de muestras para cultivo microbiológico

Quedan fuera, a propósito:
  - n.º de diagnósticos, severidad APR-DRG y procedimientos CIE: salen de la codificación de facturación
    (misma regla del EDA de la semana 4);
  - fallecimiento y reingreso: son desenlaces, no predictores.
Las ventanas cubren toda la hospitalización, igual que la epicrisis: sirven para detectar el evento al alta,
no para predecirlo durante la estancia.

Salida en caché: data/interim/dominio_hadm.parquet (una fila por hadm_id del corpus).
"""
import numpy as np
import pandas as pd

import eda_variables
from dotenv import load_dotenv

from utils import INTERIM, RAIZ, sha256_cacheado

# Criterio del autor: servicios quirúrgicos por el nombre del código de MIMIC-IV
SERVICIOS_QUIRURGICOS = {"SURG", "CSURG", "NSURG", "VSURG", "TSURG", "PSURG", "ORTHO", "TRAUM"}

GRUPOS = {
    "paciente": {"num": ["edad", "horas_emergencia"], "cat": ["sexo", "tipo_ingreso", "procedencia", "seguro"]},
    "servicio": {"num": ["n_servicios"], "cat": ["servicio", "servicio_alta"], "bin": ["servicio_quirurgico"]},
    "hospital": {"num": ["estancia_dias", "n_traslados", "n_unidades", "dias_uci"], "cat": ["destino_alta"], "bin": ["uci"]},
    "atencion": {"num": ["n_medicamentos", "n_cultivos"]},
}
CONTEO = ["horas_emergencia", "n_servicios", "estancia_dias", "n_traslados", "n_unidades", "dias_uci",
          "n_medicamentos", "n_cultivos"]          # colas largas: log1p + winsorización


def columnas(grupos):
    num, cat, b = [], [], []
    for g in grupos:
        num += GRUPOS[g].get("num", []); cat += GRUPOS[g].get("cat", []); b += GRUPOS[g].get("bin", [])
    return num, cat, b


def _leer(rel, **kw):
    ruta = eda_variables._ruta_mimic(rel)
    return pd.read_csv(ruta, **kw), {"tabla": rel, "tamano_bytes": ruta.stat().st_size, "sha256": sha256_cacheado(ruta)}


def calcular(df, log, cache=None):
    """df: corpus con hadm_id. Devuelve df con las variables de dominio unidas (y deja la caché, por defecto la del corpus)."""
    load_dotenv(RAIZ / ".env")
    cache = cache or INTERIM / "dominio_hadm.parquet"
    hadm = set(df.hadm_id)
    if cache.exists():
        dom = pd.read_parquet(cache)
        if hadm <= set(dom.hadm_id):
            log.info(f"variables de dominio: se reutiliza {cache.name}")
            return df.merge(dom, on="hadm_id", how="left", validate="m:1"), None

    fuentes = []
    base = eda_variables.cargar_variables(df[["note_id", "subject_id", "hadm_id"]].copy(), log, {})
    dom = base[["hadm_id", "edad", "sexo", "tipo_ingreso", "seguro", "horas_emergencia", "servicio",
                "estancia_dias", "destino_alta", "uci"]].copy()
    dom.loc[dom.estancia_dias < 0, "estancia_dias"] = np.nan       # 11 estancias negativas: faltante (EDA semana 4)
    dom.loc[dom.horas_emergencia < 0, "horas_emergencia"] = np.nan

    adm, f = _leer("hosp/admissions.csv.gz", usecols=["hadm_id", "admission_location"]); fuentes.append(f)
    dom = dom.merge(adm.rename(columns={"admission_location": "procedencia"}), on="hadm_id", how="left")

    srv, f = _leer("hosp/services.csv.gz", usecols=["hadm_id", "transfertime", "curr_service"]); fuentes.append(f)
    srv = srv[srv.hadm_id.isin(hadm)].sort_values(["hadm_id", "transfertime"])
    g = srv.groupby("hadm_id").curr_service
    dom = dom.merge(pd.DataFrame({"servicio_alta": g.last(), "n_servicios": g.nunique(),
                                  "servicio_quirurgico": g.agg(lambda s: int(bool(set(s) & SERVICIOS_QUIRURGICOS)))}),
                    left_on="hadm_id", right_index=True, how="left")

    trf, f = _leer("hosp/transfers.csv.gz", usecols=["hadm_id", "eventtype", "careunit"]); fuentes.append(f)
    trf = trf[trf.hadm_id.isin(hadm)]
    n_tr = trf[trf.eventtype == "transfer"].groupby("hadm_id").size().rename("n_traslados")
    unid = trf[trf.eventtype.isin(["admit", "transfer"]) & trf.careunit.notna() & (trf.careunit != "UNKNOWN")]
    n_un = unid.groupby("hadm_id").careunit.nunique().rename("n_unidades")
    dom = dom.join(n_tr, on="hadm_id").join(n_un, on="hadm_id")

    icu, f = _leer("icu/icustays.csv.gz", usecols=["hadm_id", "los"]); fuentes.append(f)
    dom = dom.join(icu[icu.hadm_id.isin(hadm)].groupby("hadm_id").los.sum().rename("dias_uci"), on="hadm_id")

    log.info("variables de dominio: contando medicamentos prescritos (prescriptions, por lotes) ...")
    ruta = eda_variables._ruta_mimic("hosp/prescriptions.csv.gz")
    fuentes.append({"tabla": "hosp/prescriptions.csv.gz", "tamano_bytes": ruta.stat().st_size, "sha256": sha256_cacheado(ruta)})
    pares = []
    for ch in pd.read_csv(ruta, usecols=["hadm_id", "drug"], chunksize=2_000_000):
        ch = ch[ch.hadm_id.isin(hadm) & ch.drug.notna()]
        pares.append(ch.assign(drug=ch.drug.str.strip().str.lower()).drop_duplicates())
    rx = pd.concat(pares).drop_duplicates()
    dom = dom.join(rx.groupby("hadm_id").drug.nunique().rename("n_medicamentos"), on="hadm_id")

    mic, f = _leer("hosp/microbiologyevents.csv.gz", usecols=["hadm_id", "micro_specimen_id"]); fuentes.append(f)
    mic = mic[mic.hadm_id.isin(hadm)]
    dom = dom.join(mic.groupby("hadm_id").micro_specimen_id.nunique().rename("n_cultivos"), on="hadm_id")

    # ausencia de filas en traslados, UCI, prescripciones o cultivos = 0 eventos registrados
    for c in ["n_traslados", "n_unidades", "dias_uci", "n_medicamentos", "n_cultivos"]:
        dom[c] = dom[c].fillna(0)
    dom = dom.drop_duplicates("hadm_id")
    dom.to_parquet(cache, index=False)
    log.info(f"variables de dominio: {dom.shape[1]-1} variables para {len(dom):,} hospitalizaciones -> {cache.name} · "
             f"tablas: {[x['tabla'] for x in fuentes]}")
    return df.merge(dom, on="hadm_id", how="left", validate="m:1"), fuentes
