# -*- coding: utf-8 -*-
"""
SEMANA 6 — Cierre del Sprint 2: variantes de los DATOS con las features nuevas y diagnósticos del modelo.

Modelo: A+B+C de la semana 5 (TF-IDF de palabras y bigramas + variables de dominio + regresión logística), sin cambios.
Variantes de los datos (src/variantes_datos.py):
  D0 datos actuales · D1 sin negativas dudosas · D2 sin negativas ni positivas dudosas.
Protocolo: el de la semana 5 (StratifiedGroupKFold de 5 folds por paciente, semilla 42, solo en train).
  Las variantes cambian solo las filas con que se entrena. Todas se evalúan sobre las mismas filas: las de
  validación (o de test) que ninguna variante quita, para que la comparación sea con los mismos casos
  («evaluación común»). También se informa la evaluación con todas las filas y la etiqueta original.
Diagnósticos (guía de la semana 6): comparación por fold, importancia por permutación y su estabilidad entre
  folds, curva de aprendizaje, calibración (curva y Brier), colinealidad (VIF) y coherencia de los datos.

Salidas: logs/experimentos_datos_<fecha>.log · logs/metrics_datos.txt · reportes/datos_resultados.json ·
         reportes/datos_folds.csv · reportes/figuras/fig_datos_*.png
Uso:  python src/experimentos_datos.py
"""
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

import figuras_datos
import variables_dominio
import variantes_datos as vd
from experimentos_fe import ic_pacientes, metricas
from features import columnas_dominio, pipeline
from utils import LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "experimentos_datos"
MODELO = "A+B+C"
TAMANOS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_PERM = 3


def grupos_de_atributos(nombres, variables):
    """Agrupa las columnas transformadas por la variable de origen (o el bloque de texto)."""
    orden = sorted(variables, key=len, reverse=True)
    g = {}
    for i, n in enumerate(nombres):
        if n.startswith("A_palabras__"):
            k = "texto: palabras"
        elif n.startswith("B_bigramas__"):
            k = "texto: bigramas"
        elif n.startswith("C_dominio__largo__"):
            k = "largo de la nota"
        else:
            resto = n.split("__", 2)[2].replace("missingindicator_", "")
            k = next(v for v in orden if resto == v or resto.startswith(v + "_"))
        g.setdefault(k, []).append(i)
    return g


def importancia(clf, X, y, grupos, rng):
    """Caída de la PR-AUC al barajar cada grupo de atributos. Como el modelo es lineal, barajar un grupo equivale a
    barajar su aporte al logit, así que no hace falta volver a transformar los datos."""
    w, b = clf.coef_[0], clf.intercept_[0]
    z = np.asarray(X @ w).ravel() + b
    base = average_precision_score(y, expit(z))
    out = {}
    for k, cols in grupos.items():
        c = np.asarray(X[:, cols] @ w[cols]).ravel()
        out[k] = float(np.mean([base - average_precision_score(y, expit(z - c + c[rng.permutation(len(c))])) for _ in range(N_PERM)]))
    return out


def resumen(v):
    return {"media": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4)}


def main():
    t_inicio = time.time()
    log = logger(ETAPA)
    cfg = cargar_config()
    S, P = cfg["semilla"], cfg["feature_engineering"]
    rng = np.random.default_rng(S)
    eda = json.loads((REPORTES / "eda_resumen.json").read_text(encoding="utf-8"))
    prev = float(eda["balance"]["prevalencia_real_con_epicrisis"])

    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    X_cols = ["text"] + dom_cols
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    if "--prueba" in sys.argv:          # corrida corta para revisar el flujo completo; no reemplaza resultados
        tr, te = tr.sample(4000, random_state=S).reset_index(drop=True), te.sample(1500, random_state=S).reset_index(drop=True)
    assert not set(tr.subject_id) & set(te.subject_id), "fuga: pacientes en train y test"
    dudosa_tr = (tr.negativa_dudosa | tr.positiva_dudosa).values
    dudosa_te = (te.negativa_dudosa | te.positiva_dudosa).values
    marcas = {s: {"negativas_dudosas": int(d.negativa_dudosa.sum()), "positivas_dudosas": int(d.positiva_dudosa.sum()),
                  "por_motivo": {k: int(v) for k, v in d.motivo[d.motivo != ""].value_counts().items()},
                  "filas": int(len(d)), "filas_de_evaluacion_comun": int((~(d.negativa_dudosa | d.positiva_dudosa)).sum()),
                  "pct_positivas_evaluacion_comun": round(100 * float(d.y[~(d.negativa_dudosa | d.positiva_dudosa)].mean()), 4)}
              for s, d in (("train", tr), ("test", te))}
    entreno = {v: int(vd.filas_de_entrenamiento(tr, v).sum()) for v in vd.VARIANTES}
    log.info(f"marcas: {marcas} · filas de entrenamiento por variante (train completo): {entreno}")

    sgkf = StratifiedGroupKFold(n_splits=P["n_folds"], shuffle=True, random_state=S)
    folds = list(sgkf.split(tr[X_cols], tr.y, groups=tr.subject_id))
    for k, (a, b) in enumerate(folds, 1):
        assert not set(tr.subject_id.iloc[a]) & set(tr.subject_id.iloc[b]), f"fuga en el fold {k}"

    # ---------- validación cruzada: las variantes cambian solo las filas de entrenamiento
    filas, oof, imp = [], {v: np.full(len(tr), np.nan) for v in vd.VARIANTES}, {v: [] for v in vd.VARIANTES}
    for v in vd.VARIANTES:
        usar = vd.filas_de_entrenamiento(tr, v).values
        for k, (a, b) in enumerate(folds, 1):
            a_v = a[usar[a]]
            m = pipeline(MODELO, P, S)
            t0 = time.time(); m.fit(tr.loc[a_v, X_cols], tr.y.iloc[a_v]); t_fit = time.time() - t0
            Xb = m.named_steps["atributos"].transform(tr.loc[b, X_cols])
            clf = m.named_steps["clf"]
            p = clf.predict_proba(Xb)[:, 1]
            oof[v][b] = p
            comun = ~dudosa_tr[b]
            yb = tr.y.values[b]
            rc, rf = metricas(yb[comun], p[comun], prev), metricas(yb, p, prev)
            fila = {"variante": v, "fold": k, "filas_entrenamiento": int(len(a_v)), "seg_entrenamiento": round(t_fit, 1),
                    **{f"{m_}_comun": round(x, 6) for m_, x in rc.items()}, "brier_comun": round(float(brier_score_loss(yb[comun], p[comun])), 6),
                    **{f"{m_}_completa": round(x, 6) for m_, x in rf.items()}}
            filas.append(fila)
            nombres = m.named_steps["atributos"].get_feature_names_out()
            imp[v].append(importancia(clf, Xb[np.where(comun)[0]], yb[comun], grupos_de_atributos(nombres, dom_cols), rng))
            log.info(f"{v} fold {k}: entrena con {len(a_v):,} · PR-AUC común {rc['pr_auc']:.4f} · completa {rf['pr_auc']:.4f} · "
                     f"ROC-AUC común {rc['roc_auc']:.4f} · {t_fit:.0f} s")
            if v == "D0" and k == 1:          # colinealidad de las numéricas del dominio, ya transformadas (train del fold 1)
                Xa = m.named_steps["atributos"].transform(tr.loc[a_v[:20000], X_cols])
                idx = [i for i, n in enumerate(nombres) if ("__conteo__" in n or "__num__" in n) and "missingindicator" not in n]
                M = Xa[:, idx].toarray()
                vif = np.diag(np.linalg.inv(np.corrcoef(M, rowvar=False)))
                vif = {nombres[i].split("__")[-1]: round(float(x), 2) for i, x in zip(idx, vif)}
                log.info(f"VIF de las numéricas del dominio: {vif}")
    folds_df = pd.DataFrame(filas)
    folds_df.to_csv(REPORTES / "datos_folds.csv", index=False, float_format="%.6f")

    cv = {}
    base = folds_df[folds_df.variante == "D0"].set_index("fold")
    for v, g in folds_df.groupby("variante", sort=False):
        g = g.set_index("fold")
        cv[v] = {e: {m_: resumen(g[f"{m_}_{e}"]) for m_ in ("pr_auc", "roc_auc", "f1", "recall", "especificidad")} for e in ("comun", "completa")}
        cv[v]["comun"]["brier"] = resumen(g.brier_comun)
        d = g.pr_auc_comun - base.pr_auc_comun
        cv[v]["dif_pr_auc_comun_vs_D0"] = {**resumen(d), "folds_mejor_que_D0": int((d > 0).sum())}
        d2 = g.pr_auc_completa - base.pr_auc_completa
        cv[v]["dif_pr_auc_completa_vs_D0"] = {**resumen(d2), "folds_mejor_que_D0": int((d2 > 0).sum())}
        cv[v]["filas_entrenamiento_media"] = int(g.filas_entrenamiento.mean())
        cv[v]["seg_entrenamiento"] = resumen(g.seg_entrenamiento)
    elegida = max(cv, key=lambda v: cv[v]["comun"]["pr_auc"]["media"])
    log.info("CV (evaluación común): " + " · ".join(f"{v} {cv[v]['comun']['pr_auc']['media']:.4f} ± {cv[v]['comun']['pr_auc']['std']:.4f}"
                                                  for v in cv) + f" · elegida: {elegida}")

    importancias = {v: {k: resumen([f[k] for f in imp[v]]) for k in imp[v][0]} for v in imp}
    top5 = [sorted(f, key=f.get, reverse=True)[:5] for f in imp[elegida]]
    estabilidad = pd.Series([k for t in top5 for k in t]).value_counts().to_dict()
    log.info(f"importancia ({elegida}): top 5 por fold {top5} · veces en el top 5: {estabilidad}")

    # ---------- calibración con las predicciones fuera de fold (evaluación común)
    comun_tr = ~dudosa_tr
    calib = {}
    for v in sorted({"D0", elegida}):
        frac, media = calibration_curve(tr.y.values[comun_tr], oof[v][comun_tr], n_bins=10, strategy="uniform")
        calib[v] = {"brier": round(float(brier_score_loss(tr.y.values[comun_tr], oof[v][comun_tr])), 4),
                    "prob_media": [round(float(x), 4) for x in media], "frecuencia_real": [round(float(x), 4) for x in frac]}
    log.info(f"calibración: {{v: brier}} = { {v: c['brier'] for v, c in calib.items()} }")

    # ---------- curva de aprendizaje (variante elegida, fold 1)
    a, b = folds[0]
    a_v = a[vd.filas_de_entrenamiento(tr, elegida).values[a]]
    comun_b = b[~dudosa_tr[b]]
    curva = []
    for f_ in TAMANOS:
        sub = np.sort(rng.choice(a_v, size=int(len(a_v) * f_), replace=False)) if f_ < 1 else a_v
        m = pipeline(MODELO, P, S).fit(tr.loc[sub, X_cols], tr.y.iloc[sub])
        muestra_tr = sub if len(sub) <= 10000 else np.sort(rng.choice(sub, size=10000, replace=False))
        pr_tr = average_precision_score(tr.y.iloc[muestra_tr], m.predict_proba(tr.loc[muestra_tr, X_cols])[:, 1])
        pr_va = average_precision_score(tr.y.iloc[comun_b], m.predict_proba(tr.loc[comun_b, X_cols])[:, 1])
        curva.append({"fraccion": f_, "filas": int(len(sub)), "pr_auc_train": round(float(pr_tr), 4), "pr_auc_validacion": round(float(pr_va), 4)})
        log.info(f"curva de aprendizaje: {len(sub):,} filas · PR-AUC train {pr_tr:.4f} · validación {pr_va:.4f}")

    # ---------- test: cada variante se entrena con todo train y se mide una sola vez
    test = {}
    comun_te = ~dudosa_te
    for v in vd.VARIANTES:
        usar = vd.filas_de_entrenamiento(tr, v).values
        m = pipeline(MODELO, P, S).fit(tr.loc[usar, X_cols], tr.y[usar])
        p = m.predict_proba(te[X_cols])[:, 1]
        test[v] = {"comun": {**{k: round(x, 4) for k, x in metricas(te.y.values[comun_te], p[comun_te], prev).items()},
                             "pr_auc_ic95_pacientes": ic_pacientes(te.y.values[comun_te], p[comun_te], te.subject_id.values[comun_te],
                                                                   P["n_bootstrap"], S),
                             "brier": round(float(brier_score_loss(te.y.values[comun_te], p[comun_te])), 4)},
                   "completa": {**{k: round(x, 4) for k, x in metricas(te.y.values, p, prev).items()},
                                "pr_auc_ic95_pacientes": ic_pacientes(te.y.values, p, te.subject_id.values, P["n_bootstrap"], S)}}
        log.info(f"test {v}: PR-AUC común {test[v]['comun']['pr_auc']:.4f} {test[v]['comun']['pr_auc_ic95_pacientes']} · completa "
                 f"{test[v]['completa']['pr_auc']:.4f}")

    # ---------- coherencia de los datos
    cats = [c for c in dom_cols if not pd.api.types.is_numeric_dtype(tr[c])]
    nums = [c for c in dom_cols if pd.api.types.is_numeric_dtype(tr[c])]
    coherencia = {
        "tipos": {"numericas": nums, "categoricas": cats},
        "rangos_train": {c: [round(float(tr[c].min()), 3), round(float(tr[c].max()), 3)] for c in nums},
        "valores_negativos_train": {c: int((tr[c] < 0).sum()) for c in nums if (tr[c] < 0).any()},
        "categorias_solo_en_test": {c: sorted(set(te[c].dropna().astype(str)) - set(tr[c].dropna().astype(str))) for c in cats},
        "pct_nulos_train": {c: round(100 * float(tr[c].isna().mean()), 2) for c in dom_cols if tr[c].isna().any()}}
    log.info(f"coherencia: {coherencia}")

    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "modelo": MODELO,
         "parametros": P, "prevalencia_real_con_epicrisis": prev,
         "protocolo": f"StratifiedGroupKFold({P['n_folds']}, shuffle=True, random_state={S}) por subject_id en train; las variantes "
                      "cambian solo las filas de entrenamiento; evaluación común = filas que ninguna variante quita",
         "variantes": vd.VARIANTES, "marcas": marcas, "filas_de_entrenamiento_train_completo": entreno,
         "cv": cv, "elegida_por_cv": elegida, "test": test, "importancia": importancias,
         "importancia_top5_por_fold": top5, "importancia_veces_en_top5": estabilidad,
         "curva_de_aprendizaje": {"variante": elegida, "fold": 1, "puntos": curva}, "calibracion": calib,
         "vif": vif, "coherencia": coherencia}
    guardar_json(R, REPORTES / "datos_resultados.json")

    # ---------- figuras (con el pie que explica ejes y resultado)
    figuras_datos.dibujar(R, folds_df)

    # ---------- resumen legible
    L = [f"# metrics_datos.txt · {R['generado']} · modelo {MODELO} · semilla {S}", f"protocolo: {R['protocolo']}",
         f"marcas: {marcas}", f"filas de entrenamiento (train completo): {entreno}", "",
         "validación cruzada (media ± desv. de 5 folds) · evaluación común | completa:"]
    for v in cv:
        c = cv[v]
        L.append(f"  {v} ({vd.VARIANTES[v]}): PR-AUC {c['comun']['pr_auc']['media']:.4f} ± {c['comun']['pr_auc']['std']:.4f} | "
                 f"{c['completa']['pr_auc']['media']:.4f} ± {c['completa']['pr_auc']['std']:.4f} · ROC-AUC común {c['comun']['roc_auc']['media']:.4f} · "
                 f"F1 común {c['comun']['f1']['media']:.4f} · Brier común {c['comun']['brier']['media']:.4f} · dif. con D0 (común) "
                 f"{c['dif_pr_auc_comun_vs_D0']['media']:+.4f} ± {c['dif_pr_auc_comun_vs_D0']['std']:.4f} "
                 f"({c['dif_pr_auc_comun_vs_D0']['folds_mejor_que_D0']}/5 folds mejores)")
    L += [f"elegida por validación cruzada: {elegida}", "", "test (una sola medición):"]
    for v, t in test.items():
        L.append(f"  {v}: común PR-AUC {t['comun']['pr_auc']:.4f} {t['comun']['pr_auc_ic95_pacientes']} · ROC-AUC {t['comun']['roc_auc']:.4f} · "
                 f"Brier {t['comun']['brier']:.4f} | completa PR-AUC {t['completa']['pr_auc']:.4f} {t['completa']['pr_auc_ic95_pacientes']}")
    L += ["", f"importancia por permutación ({elegida}), caída de PR-AUC media ± desv.:"]
    for k, x in sorted(importancias[elegida].items(), key=lambda kv: -kv[1]["media"]):
        L.append(f"  {k:<24} {x['media']:.4f} ± {x['std']:.4f} · top 5 en {estabilidad.get(k, 0)}/5 folds")
    L += ["", f"curva de aprendizaje: {curva}", f"calibración: {calib}", f"VIF: {vif}", f"coherencia: {coherencia}",
          "", f"duración: {round(time.time() - t_inicio)} s"]
    (LOGS / "metrics_datos.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Salidas: logs/metrics_datos.txt · reportes/datos_resultados.json · reportes/datos_folds.csv · reportes/figuras/fig_datos_*.png")


if __name__ == "__main__":
    main()
