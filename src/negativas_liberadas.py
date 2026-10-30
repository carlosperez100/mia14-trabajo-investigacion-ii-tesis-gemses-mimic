# -*- coding: utf-8 -*-
"""
VARIANTE D3 (29-10-2026) — PASO 1: negativas que libera la equivalencia validada.

El autor validó el 29-10-2026 la equivalencia de los 223 códigos CIE-10 de la tabla de la etiqueta con los eventos
del Anexo 02 (data/catalogos/anexo02_equivalencia_cie10_oficial.csv) y marcó varios como «no es evento adverso».
Ninguno de ellos pertenece al estrato A1 ni a las equivalencias CIE-10-CM, de modo que ninguna positiva depende de
ellos (se comprueba aquí). Sin embargo, los que figuran en el estrato A2 sí actúan sobre la etiqueta, porque
preprocesado.py excluye del sorteo de negativas toda hospitalización con un código A2. Quitarlos de la etiqueta
«libera» las hospitalizaciones cuyos únicos códigos de la tabla son de ese grupo, que pasan a poder ser negativas.

Pasos, cada uno con su conteo en el log:
  1. Códigos «no es evento adverso» y comprobación de que ninguna positiva depende solo de ellos.
  2. Hospitalizaciones liberadas: no positivas, con algún código A2 y con todos sus códigos A2 en ese grupo.
  3. Proporción que tendrían entre las negativas de la época CIE-10 si el corpus se sorteara de nuevo: la de las
     liberadas en el marco de la época CIE-10 con epicrisis (preprocesado.py sortea al azar dentro de cada época).
  4. Texto de alta de las liberadas, con los filtros de preprocesado.py, y partición por paciente: la nota de un
     paciente del corpus va a su misma partición; la de un paciente nuevo, a test con probabilidad test_size.
  5. Variables de dominio y marcas de etiqueta dudosa, con las mismas funciones del corpus y cachés aparte.

En el resumen, los conteos por código menores de 10 se escriben «<10», como en la tabla por evento del EDA
(src/eda_variables.py).

Salidas: data/interim/negativas_liberadas.parquet (fuera de git) · reportes/negativas_liberadas_resumen.json
Uso:  python src/negativas_liberadas.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import variables_dominio
import variantes_datos as vd
from preprocesado import MAPA_CIE9
from utils import DATA, INTERIM, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "negativas_liberadas"
SALIDA = INTERIM / "negativas_liberadas.parquet"


def main():
    log = logger(ETAPA)
    cfg = cargar_config()
    S, P = cfg["semilla"], cfg["preprocesado"]
    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S}

    # 1. códigos «no es evento adverso» y positivas que dependen de ellos
    W = pd.read_csv(DATA / "catalogos" / "anexo02_equivalencia_cie10_oficial.csv", dtype=str).fillna("")
    NO = set(W.loc[W.estado_validado == "no es evento adverso", "codigo_cie10"])
    a1 = pd.read_csv(INTERIM / "a1_tier_a.csv", dtype={"icd_code": str})
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv", dtype={"codigo_oms": str})
    c9 = pd.read_csv(INTERIM / "positivos_cie9.csv")
    c9 = c9[c9.era == 9].assign(naturaleza=lambda d: d.familia.map(MAPA_CIE9))
    a1p, eqp = a1[a1.estrato == "A1"].dropna(subset=["naturaleza"]), eq[eq.estrato == "A1"].dropna(subset=["naturaleza"])
    c9p = c9.dropna(subset=["naturaleza"])
    positivas = set(a1p.hadm_id) | set(eqp.hadm_id) | set(c9p.hadm_id)          # igual que preprocesado.py, paso 1
    evid = pd.concat([a1p[["hadm_id", "icd_code"]].rename(columns={"icd_code": "codigo"}),
                      eqp[["hadm_id", "codigo_oms"]].rename(columns={"codigo_oms": "codigo"})])
    solo_no = evid.groupby("hadm_id").codigo.agg(lambda s: set(s) <= NO)
    dependen = set(solo_no[solo_no].index) - set(c9p.hadm_id)
    a2 = a1[a1.estrato == "A2"]
    R["codigos"] = {"no_es_evento_adverso": len(NO), "en_estrato_A1_o_equivalencias": len(NO & (set(a1p.icd_code) | set(eqp.codigo_oms))),
                    "en_estrato_A2": len(NO & set(a2.icd_code)), "sin_registros": len(NO - set(a1.icd_code) - set(eq.codigo_oms))}
    R["positivas_que_dependen_solo_de_ellos"] = len(dependen)
    log.info(f"1. Códigos «no es evento adverso»: {R['codigos']} · positivas que dependen solo de ellos: {len(dependen)}")
    assert not dependen, "hay positivas que dependen solo de códigos «no es evento adverso»"

    # 2. hospitalizaciones liberadas
    g = a2[~a2.hadm_id.isin(positivas)].groupby("hadm_id").icd_code.agg(set)
    libres = set(g[g.map(lambda s: s <= NO)].index)
    por_codigo = a2[a2.hadm_id.isin(libres)].groupby("icd_code").hadm_id.nunique().sort_values(ascending=False)
    epoca = pd.read_parquet(INTERIM / "epoca.parquet").set_index("hadm_id")["epoca"]
    ep10 = set(epoca.index[epoca == 10])
    R["liberadas"] = {"hospitalizaciones": len(libres), "por_epoca": {int(k): int(v) for k, v in epoca.reindex(list(libres)).value_counts().items()},
                      "por_codigo": {k: int(v) if v >= 10 else "<10" for k, v in por_codigo.items()}}
    log.info(f"2. Hospitalizaciones liberadas: {len(libres):,} · por época {R['liberadas']['por_epoca']} · "
             f"códigos más frecuentes {dict(list(R['liberadas']['por_codigo'].items())[:5])}")

    # 3. proporción esperada entre las negativas de la época CIE-10 si se sorteara de nuevo
    con_epicrisis = set(pd.read_parquet(INTERIM / "hadm_con_epicrisis.parquet").hadm_id)
    con_codigo = positivas | set(a2.hadm_id)
    marco10 = (ep10 - con_codigo) & con_epicrisis
    libres10 = libres & ep10 & con_epicrisis
    prop = len(libres10) / (len(marco10) + len(libres10))
    R["proporcion"] = {"marco_epoca10_con_epicrisis": len(marco10), "liberadas_epoca10_con_epicrisis": len(libres10),
                       "proporcion_entre_negativas_epoca10": round(prop, 6)}
    log.info(f"3. Marco de negativas de la época CIE-10 con epicrisis: {len(marco10):,} · liberadas {len(libres10):,} · "
             f"proporción esperada entre las negativas de esa época {prop:.4f}")

    # 4. texto de alta y partición por paciente
    trozos = []
    for i, ch in enumerate(pd.read_csv(cfg["ingesta"]["crudas"]["discharge"], usecols=["note_id", "subject_id", "hadm_id", "text"],
                                       chunksize=20000, encoding="utf-8")):
        hit = ch[ch.hadm_id.isin(libres10)]
        if len(hit):
            trozos.append(hit)
        if i % 50 == 0:
            log.info(f"   lote {i} …")
    df = pd.concat(trozos, ignore_index=True)
    n0 = len(df)
    df = df[df.text.notna() & (df.text.str.len() > P["largo_min_texto"])].dropna(subset=["subject_id"]).reset_index(drop=True)
    df["y"], df["epoca"] = 0, 10
    corpus = pd.read_parquet(PROCESSED / "dataset.parquet", columns=["subject_id", "hadm_id", "split"])
    assert not set(df.hadm_id) & set(corpus.hadm_id), "una liberada ya está en el corpus"
    split_paciente = corpus.drop_duplicates("subject_id").set_index("subject_id").split
    nuevos = np.array(sorted(set(df.subject_id) - set(split_paciente.index)))
    rng = np.random.default_rng(S)
    a_test = pd.Series(np.where(rng.random(len(nuevos)) < P["test_size"], "test", "train"), index=nuevos)
    df["split"] = df.subject_id.map(pd.concat([split_paciente, a_test]))
    df["paciente_en_corpus"] = df.subject_id.isin(split_paciente.index)
    R["notas"] = {"recuperadas": int(n0), "tras_limpieza": int(len(df)), "hospitalizaciones": int(df.hadm_id.nunique()),
                  "pacientes": int(df.subject_id.nunique()), "pacientes_ya_en_corpus": int(df[df.paciente_en_corpus].subject_id.nunique()),
                  "por_particion": {k: int(v) for k, v in df.split.value_counts().items()}}
    log.info(f"4. Notas de las liberadas: {R['notas']}")

    # 5. variables de dominio y marcas de etiqueta dudosa (cachés aparte: no tocan las del corpus)
    df, _ = variables_dominio.calcular(df, log, cache=INTERIM / "dominio_liberadas.parquet")
    df = vd.marcar(df, log, cache=INTERIM / "variantes_liberadas.parquet")
    R["marcas"] = {"negativas_dudosas": int(df.negativa_dudosa.sum()),
                   "por_particion_sin_dudosas": {k: int(v) for k, v in df[~df.negativa_dudosa].split.value_counts().items()}}
    log.info(f"5. Negativas dudosas entre las liberadas (códigos causales fuera de la tabla): {R['marcas']}")

    df.to_parquet(SALIDA, index=False)
    guardar_json(R, REPORTES / "negativas_liberadas_resumen.json")
    log.info(f"Salidas: {SALIDA.relative_to(DATA.parent)} (fuera de git) · reportes/negativas_liberadas_resumen.json")


if __name__ == "__main__":
    main()
