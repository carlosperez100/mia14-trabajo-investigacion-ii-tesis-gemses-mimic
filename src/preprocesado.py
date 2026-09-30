# -*- coding: utf-8 -*-
"""
ETAPA 2 — PREPROCESADO (reproduce la construcción del corpus de la Fase 9)

Entradas: data/interim/ (etiquetas + época) y discharge.csv.gz de MIMIC-IV-Note.
Pasos, cada uno con su conteo en el log:
  1. Positivos = unión de A1 (Tier A), equivalencias CIE-10-CM y familias CIE-9.
  2. Negativos = hospitalizaciones sin código A1 ni A2, emparejadas por época CIE.
  3. Recuperación del texto de alta (por lotes de 20 000 filas).
  4. Limpieza: texto nulo o < 100 caracteres, subject_id o época faltantes.
  5. Reemparejamiento de época a nivel de nota; submuestreo estratificado a N_MAX.
  6. Partición train/test POR PACIENTE (GroupShuffleSplit, semilla fija) — se guarda.
La tokenización y el TF-IDF NO se hacen aquí: se ajustan solo con train en baseline.py.

Salidas (data/processed/, fuera de git): dataset.parquet, splits.csv
Salida versionable (reportes/): preprocesado_resumen.json (conteos, prevalencia, hashes)

Uso:  python src/preprocesado.py
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import INTERIM, PROCESSED, REPORTES, cargar_config, guardar_json, logger, sha256

ETAPA = "preprocesado"
MAPA_CIE9 = {
    "complicaciones_atencion": "Procedimiento",
    "incidentes_asistenciales": "Procedimiento",
    "farmacos_uso_terapeutico": "Medicacion",
    "caidas": "Cuidado del paciente",
    "ulcera_presion": "Cuidado del paciente",
}


def main():
    log = logger(ETAPA)
    cfg_all = cargar_config()
    cfg, SEED = cfg_all["preprocesado"], cfg_all["semilla"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    conteos = {}

    # 1. positivos
    a1_full = pd.read_csv(INTERIM / "a1_tier_a.csv")
    a1 = a1_full[a1_full.estrato == "A1"][["hadm_id", "naturaleza"]]
    eq = pd.read_csv(INTERIM / "equivalencias_cm.csv")
    eq = eq[eq.estrato == "A1"][["hadm_id", "naturaleza"]]
    c9 = pd.read_csv(INTERIM / "positivos_cie9.csv")
    c9 = c9[c9.era == 9][["hadm_id", "familia"]].rename(columns={"familia": "naturaleza"})
    c9["naturaleza"] = c9.naturaleza.map(MAPA_CIE9)
    pos = pd.concat([a1, eq, c9], ignore_index=True).dropna()
    pos["naturaleza"] = pos.naturaleza.replace({"Dispositivo": "Dispositivo medico"})
    etiquetas = pos.groupby("hadm_id")["naturaleza"].agg(lambda s: sorted(set(s)))
    conteos["positivos_hadm"] = {"a1": int(a1.hadm_id.nunique()), "equivalencias": int(eq.hadm_id.nunique()),
                                 "cie9": int(c9.hadm_id.nunique()), "union": int(pos.hadm_id.nunique())}
    log.info(f"1. Positivos (hospitalizaciones): {conteos['positivos_hadm']}")

    # prevalencia real sobre el universo de hospitalizaciones con diagnóstico
    epoca = pd.read_parquet(INTERIM / "epoca.parquet").set_index("hadm_id")["epoca"]
    # mismo universo que la Fase 9: unión de las tres fuentes, antes de descartar
    # familias CIE-9 sin naturaleza asignada
    universo = set(a1.hadm_id) | set(eq.hadm_id) | set(c9.hadm_id)
    prevalencia = len(universo) / len(epoca)
    log.info(f"   prevalencia real {prevalencia:.4f} ({len(universo):,} de {len(epoca):,})")

    # 2. negativos emparejados por época
    con_codigo = set(pos.hadm_id) | set(a1_full[a1_full.estrato == "A2"].hadm_id)
    ep_pos = epoca.loc[list(set(pos.hadm_id) & set(epoca.index))]
    cuota = (ep_pos.value_counts() * cfg["razon_negativos"]).to_dict()
    pool = pd.DataFrame({"hadm_id": sorted(set(epoca.index) - con_codigo)})
    pool["epoca"] = pool.hadm_id.map(epoca)
    neg = pd.concat([pool[pool.epoca == era].sample(n=min(int(n), (pool.epoca == era).sum()),
                                                     random_state=SEED)
                     for era, n in cuota.items()], ignore_index=True)
    conteos["negativos_hadm"] = int(len(neg))
    log.info(f"2. Negativos emparejados: {len(neg):,} (cuota por era {cuota})")

    # 3. texto
    objetivo = set(etiquetas.index) | set(neg.hadm_id)
    log.info(f"3. Recuperando texto de {len(objetivo):,} hospitalizaciones (lotes de 20 000) ...")
    trozos = []
    for i, ch in enumerate(pd.read_csv(cfg_all["ingesta"]["crudas"]["discharge"],
                                       usecols=["note_id", "subject_id", "hadm_id", "text"],
                                       chunksize=20000, encoding="utf-8")):
        hit = ch[ch.hadm_id.isin(objetivo)]
        if len(hit):
            trozos.append(hit)
        if i % 50 == 0:
            log.info(f"   lote {i} …")
    df = pd.concat(trozos, ignore_index=True)
    conteos["notas_recuperadas"] = int(len(df))
    log.info(f"   notas recuperadas: {len(df):,}")

    # 4. limpieza
    df["y"] = df.hadm_id.isin(etiquetas.index).astype(int)
    df["epoca"] = df.hadm_id.map(epoca)
    df["labels"] = df.hadm_id.map(etiquetas)
    n0 = len(df)
    df = df[df.text.notna() & (df.text.str.len() > cfg["largo_min_texto"])]
    n1 = len(df)
    df = df.dropna(subset=["subject_id", "epoca"]).reset_index(drop=True)
    conteos["limpieza"] = {"descartadas_texto_nulo_o_corto": n0 - n1,
                           "descartadas_sin_paciente_o_epoca": n1 - len(df), "quedan": int(len(df))}
    log.info(f"4. Limpieza: {conteos['limpieza']}")

    # 5. reemparejamiento a nivel de nota + submuestreo
    pos_df, neg_df = df[df.y == 1], df[df.y == 0]
    obj = (pos_df.epoca == 10).mean()
    n_neg = min(len(neg_df), int(len(pos_df) * cfg["razon_negativos"]))
    n10 = int(round(n_neg * obj)); n9 = n_neg - n10
    d10, d9 = neg_df[neg_df.epoca == 10], neg_df[neg_df.epoca == 9]
    if len(d10) < n10 or len(d9) < n9:
        k = min(len(d10) / max(n10, 1), len(d9) / max(n9, 1), 1.0)
        n10, n9 = int(n10 * k), int(n9 * k)
    neg_df = pd.concat([d10.sample(n10, random_state=SEED), d9.sample(n9, random_state=SEED)])
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    if len(df) > cfg["n_max"]:
        frac = cfg["n_max"] / len(df)
        df = (df.groupby(["epoca", "y"], group_keys=False)
                .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=SEED))
                .reset_index(drop=True))
    p_pos = float((df[df.y == 1].epoca == 10).mean()); p_neg = float((df[df.y == 0].epoca == 10).mean())
    conteos["corpus_final"] = {"notas": int(len(df)), "positivas": int(df.y.sum()),
                               "pacientes": int(df.subject_id.nunique()),
                               "era10_positivos": round(p_pos, 4), "era10_negativos": round(p_neg, 4)}
    log.info(f"5. Corpus final: {conteos['corpus_final']}")

    # 6. partición por paciente
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"], random_state=SEED)
    tr, te = next(gss.split(df.text, df.y, groups=df.subject_id))
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    compartidos = set(df.subject_id.iloc[tr]) & set(df.subject_id.iloc[te])
    assert not compartidos, "fuga: pacientes en train y test"
    conteos["particion"] = {"train": int(len(tr)), "test": int(len(te)),
                            "pacientes_compartidos": 0,
                            "positivas_train": round(float(df.y.iloc[tr].mean()), 4),
                            "positivas_test": round(float(df.y.iloc[te].mean()), 4)}
    log.info(f"6. Partición por paciente: {conteos['particion']}")

    df.to_parquet(PROCESSED / "dataset.parquet", index=False)
    df[["note_id", "subject_id", "hadm_id", "split"]].to_csv(PROCESSED / "splits.csv", index=False)
    resumen = {"etapa": ETAPA, "semilla": SEED, "parametros": cfg,
               "prevalencia_real": round(prevalencia, 4), "conteos": conteos,
               "sha256_splits": sha256(PROCESSED / "splits.csv")}
    guardar_json(resumen, REPORTES / "preprocesado_resumen.json")
    log.info(f"Salidas: data/processed/dataset.parquet, splits.csv (sha256 {resumen['sha256_splits'][:16]}…)")


if __name__ == "__main__":
    main()
