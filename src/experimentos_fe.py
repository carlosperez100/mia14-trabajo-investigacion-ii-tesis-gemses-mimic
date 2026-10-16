# -*- coding: utf-8 -*-
"""
EXPERIMENTOS DE FEATURE ENGINEERING (Sprint 2, semana 5)

Protocolo
  1. Validación cruzada en TRAIN: StratifiedGroupKFold de 5 folds por paciente (subject_id), semilla fija.
     En cada fold el Pipeline completo (atributos + modelo) se ajusta solo con la parte de entrenamiento.
  2. Selección: la configuración con mayor PR-AUC media en validación cruzada.
  3. Holdout: cada configuración se reentrena con todo train y se mide UNA vez en TEST (la misma
     partición por paciente del Sprint 1). El test no interviene en ninguna decisión.
  4. Controles de fuga: folds sin pacientes compartidos; train y test sin pacientes compartidos;
     el baseline A en test debe reproducir la logística del Sprint 1 (PR-AUC 0.8787).

Métrica central: PR-AUC. Secundarias: F1 y recall de la clase positiva (umbral 0.5), ROC-AUC,
valor predictivo positivo a la prevalencia real con epicrisis (EDA semana 4) y costo en segundos.

Salidas: logs/metrics_fe.txt · reportes/fe_resultados.json · reportes/fe_folds.csv ·
         reportes/figuras/fig_fe_pr.png

Uso:  python src/experimentos_fe.py
"""
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

import variables_dominio
from features import CONFIGURACIONES, DESCRIPCION, columnas_dominio, dominio, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "experimentos_fe"


def metricas(y, p, prev):
    pred = (p >= 0.5).astype(int)
    rec = recall_score(y, pred)
    esp = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    vpp = rec * prev / (rec * prev + (1 - esp) * (1 - prev)) if rec + esp else 0.0
    return {"pr_auc": float(average_precision_score(y, p)), "f1": float(f1_score(y, pred)), "recall": float(rec),
            "especificidad": float(esp), "roc_auc": float(roc_auc_score(y, p)), "vpp_prevalencia_real": float(vpp)}


def ic_pacientes(y, p, grupos, n, semilla):
    """IC 95 % de la PR-AUC remuestreando pacientes del test."""
    idx_g = list(pd.Series(range(len(y))).groupby(grupos).indices.values())
    rng = np.random.default_rng(semilla)
    v = []
    for _ in range(n):
        idx = np.concatenate([idx_g[i] for i in rng.integers(0, len(idx_g), len(idx_g))])
        v.append(average_precision_score(y[idx], p[idx]))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main():
    log = logger(ETAPA)
    cfg = cargar_config()
    S, P = cfg["semilla"], cfg["feature_engineering"]
    eda = json.loads((REPORTES / "eda_resumen.json").read_text(encoding="utf-8"))
    prev = float(eda["balance"]["prevalencia_real_con_epicrisis"])
    log.info(f"parámetros: {json.dumps(P, ensure_ascii=False)} · semilla {S} · prevalencia real con epicrisis {prev:.4f}")

    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, fuentes_dom = variables_dominio.calcular(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    falt = {c: round(float(df[c].isna().mean()), 4) for c in dom_cols}
    firma = {c: {"media": round(float(df.loc[df.split == "train", c].mean()), 3),
                 "p50": round(float(df.loc[df.split == "train", c].median()), 3),
                 "p90": round(float(df.loc[df.split == "train", c].quantile(0.9)), 3)}
             for c in dom_cols if pd.api.types.is_numeric_dtype(df[c])}
    log.info(f"variables de dominio unidas · faltantes: {falt}")
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    assert not set(tr.subject_id) & set(te.subject_id), "fuga: pacientes en train y test"
    log.info(f"train {len(tr):,} · test {len(te):,} · 0 pacientes compartidos")

    X_cols = ["text"] + dom_cols
    sgkf = StratifiedGroupKFold(n_splits=P["n_folds"], shuffle=True, random_state=S)
    folds = list(sgkf.split(tr[X_cols], tr.y, groups=tr.subject_id))
    for k, (a, b) in enumerate(folds, 1):
        assert not set(tr.subject_id.iloc[a]) & set(tr.subject_id.iloc[b]), f"fuga en el fold {k}"
        log.info(f"fold {k}: entrenamiento {len(a):,} · validación {len(b):,} · positivas en validación "
                 f"{tr.y.iloc[b].mean():.3f} · 0 pacientes compartidos")

    filas, oof = [], {}
    for nombre in CONFIGURACIONES:
        oof[nombre] = np.zeros(len(tr))
        for k, (a, b) in enumerate(folds, 1):
            m = pipeline(nombre, P, S)
            t0 = time.time(); m.fit(tr.loc[a, X_cols], tr.y.iloc[a]); t_fit = time.time() - t0
            t0 = time.time(); p = m.predict_proba(tr.loc[b, X_cols])[:, 1]; t_pred = time.time() - t0
            oof[nombre][b] = p
            r = metricas(tr.y.iloc[b].values, p, prev)
            r.update(config=nombre, fold=k, n_atributos=int(m.named_steps["clf"].coef_.shape[1]),
                     seg_entrenamiento=round(t_fit, 1), seg_prediccion=round(t_pred, 1))
            filas.append(r)
            log.info(f"{nombre:<6} fold {k}: PR-AUC {r['pr_auc']:.4f} · F1 {r['f1']:.4f} · recall {r['recall']:.4f} · "
                     f"ROC-AUC {r['roc_auc']:.4f} · {r['n_atributos']:,} atributos · {t_fit:.0f} s")
    folds_df = pd.DataFrame(filas)
    folds_df.to_csv(REPORTES / "fe_folds.csv", index=False, float_format="%.6f")

    cols = ["pr_auc", "f1", "recall", "roc_auc", "vpp_prevalencia_real", "seg_entrenamiento", "seg_prediccion", "n_atributos"]
    cv = {}
    base = folds_df[folds_df.config == "A"].set_index("fold").pr_auc
    for nombre, g in folds_df.groupby("config", sort=False):
        d = g.set_index("fold").pr_auc - base
        cv[nombre] = {c: {"media": round(float(g[c].mean()), 4), "std": round(float(g[c].std(ddof=1)), 4)} for c in cols}
        cv[nombre]["dif_pr_auc_vs_A"] = {"media": round(float(d.mean()), 4), "std": round(float(d.std(ddof=1)), 4),
                                         "folds_mejor_que_A": int((d > 0).sum())}
    elegida = max(cv, key=lambda c: cv[c]["pr_auc"]["media"])

    # aporte de cada grupo de dominio: modelos SIN texto que suman un grupo a la vez (validación cruzada)
    escalera, acumulado = {}, []
    for g in P["C"]["grupos_escalera"]:
        acumulado.append(g)
        v = []
        for a, b in folds:
            cols = columnas_dominio(acumulado)
            m = Pipeline([("dom", dominio(P["C"], acumulado)),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=S))])
            m.fit(tr.loc[a, cols], tr.y.iloc[a])
            v.append(average_precision_score(tr.y.iloc[b], m.predict_proba(tr.loc[b, cols])[:, 1]))
        clave = " + ".join(acumulado)
        escalera[clave] = {"media": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4)}
        log.info(f"sin texto · {clave}: PR-AUC {escalera[clave]['media']:.4f} ± {escalera[clave]['std']:.4f}")
    log.info(f"configuración con mayor PR-AUC media en validación cruzada: {elegida}")

    # holdout: cada configuración se reentrena con todo train y se mide una vez en test
    test, coef_dom = {}, {}
    for nombre in CONFIGURACIONES:
        m = pipeline(nombre, P, S)
        t0 = time.time(); m.fit(tr[X_cols], tr.y); t_fit = time.time() - t0
        p = m.predict_proba(te[X_cols])[:, 1]
        r = metricas(te.y.values, p, prev)
        r = {k: round(v, 4) for k, v in r.items()}
        r["pr_auc_ic95_pacientes"] = ic_pacientes(te.y.values, p, te.subject_id.values, P["n_bootstrap"], S)
        r["seg_entrenamiento"] = round(t_fit, 1)
        test[nombre] = r
        if "C" in CONFIGURACIONES[nombre]:
            nombres = m.named_steps["atributos"].get_feature_names_out()
            coef = m.named_steps["clf"].coef_[0]
            sel = [i for i, n in enumerate(nombres) if n.startswith("C_dominio")]
            orden = sorted(sel, key=lambda i: -abs(coef[i]))[:12]
            coef_dom[nombre] = [(str(nombres[i]).replace("C_dominio__", ""), round(float(coef[i]), 4)) for i in orden]
        log.info(f"TEST {nombre:<6} PR-AUC {r['pr_auc']:.4f} {r['pr_auc_ic95_pacientes']} · F1 {r['f1']:.4f} · "
                 f"recall {r['recall']:.4f} · ROC-AUC {r['roc_auc']:.4f} · VPP {r['vpp_prevalencia_real']:.4f}")

    ref = json.loads((REPORTES / "baseline_resultados.json").read_text(encoding="utf-8"))["modelos"]["logistica_tfidf"]
    control = {"pr_auc_sprint1": ref["pr_auc"], "pr_auc_A_test": test["A"]["pr_auc"],
               "diferencia": round(abs(ref["pr_auc"] - test["A"]["pr_auc"]), 4)}
    control["reproduce"] = control["diferencia"] <= 0.005
    log.info(f"control: el baseline A en test reproduce la logística del Sprint 1: {control}")

    # figura clave: curvas precisión-sensibilidad en validación cruzada (predicciones fuera de fold)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for nombre in CONFIGURACIONES:
        pr, rc, _ = precision_recall_curve(tr.y, oof[nombre])
        a1.plot(rc, pr, label=f"{DESCRIPCION[nombre]} (PR-AUC {cv[nombre]['pr_auc']['media']:.3f})")
    a1.axhline(tr.y.mean(), color="k", ls=":", lw=0.8, label=f"proporción de positivas ({tr.y.mean():.3f})")
    a1.set(xlabel="Sensibilidad (recall)", ylabel="Precisión", ylim=(0.5, 1.0),
           title="Curva precisión-sensibilidad · validación cruzada por paciente (5 folds)")
    a1.legend(fontsize=7.5, loc="lower left"); a1.grid(alpha=0.3)
    for i, nombre in enumerate(CONFIGURACIONES):
        v = folds_df[folds_df.config == nombre].pr_auc.values
        a2.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=18, alpha=0.7)
        a2.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="o", color="k", capsize=5)
    a2.set_xticks(range(len(CONFIGURACIONES)), list(CONFIGURACIONES))
    a2.set(ylabel="PR-AUC por fold", title="PR-AUC en cada fold (media ± desviación estándar)")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    (REPORTES / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTES / "figuras" / "fig_fe_pr.png", dpi=160)

    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "parametros": P,
         "prevalencia_real_con_epicrisis": prev, "faltantes_dominio": falt,
         "validacion": f"StratifiedGroupKFold({P['n_folds']}, shuffle=True, random_state={S}) por subject_id, solo en train",
         "n_train": len(tr), "n_test": len(te), "configuraciones": DESCRIPCION, "cv": cv, "elegida_por_cv": elegida,
         "test": test, "coeficientes_dominio": coef_dom, "aporte_grupos_sin_texto": escalera,
         "variables_dominio": dom_cols, "fuentes_dominio": fuentes_dom, "firma_datos_train": firma, "control_reproduccion_baseline": control}
    guardar_json(R, REPORTES / "fe_resultados.json")

    L = [f"# metrics_fe.txt · {R['generado']}",
         f"validación: {R['validacion']} · test: holdout por paciente del Sprint 1 (n={len(te)}), usado una vez",
         f"semilla {S} · parámetros {json.dumps(P, ensure_ascii=False)}",
         f"prevalencia real con epicrisis (VPP): {prev:.4f}", "",
         "VALIDACIÓN CRUZADA (media ± std de 5 folds)",
         f"{'config':<7}{'PR-AUC':>17}{'F1(+)':>17}{'Recall':>17}{'ROC-AUC':>17}{'dif PR-AUC vs A':>18}{'seg fit':>10}{'atributos':>11}"]
    for n, c in cv.items():
        f = lambda k: f"{c[k]['media']:.4f} ± {c[k]['std']:.4f}"
        L.append(f"{n:<7}{f('pr_auc'):>17}{f('f1'):>17}{f('recall'):>17}{f('roc_auc'):>17}"
                 f"{c['dif_pr_auc_vs_A']['media']:>+11.4f} ({c['dif_pr_auc_vs_A']['folds_mejor_que_A']}/5)"
                 f"{c['seg_entrenamiento']['media']:>10.0f}{c['n_atributos']['media']:>11.0f}")
    L += ["", f"elegida por validación cruzada: {elegida}", "", "TEST (una sola medición)",
          f"{'config':<7}{'PR-AUC':>9}{'IC95 pacientes':>20}{'F1(+)':>8}{'Recall':>8}{'ROC-AUC':>9}{'VPP':>8}{'seg fit':>9}"]
    for n, r in test.items():
        L.append(f"{n:<7}{r['pr_auc']:>9.4f}{str(r['pr_auc_ic95_pacientes']):>20}{r['f1']:>8.4f}{r['recall']:>8.4f}"
                 f"{r['roc_auc']:>9.4f}{r['vpp_prevalencia_real']:>8.4f}{r['seg_entrenamiento']:>9.0f}")
    L += ["", "APORTE DE CADA GRUPO DE DOMINIO SIN TEXTO (validación cruzada, PR-AUC media ± std)"]
    L += [f"  {k}: {v['media']:.4f} ± {v['std']:.4f}" for k, v in escalera.items()]
    L += ["", f"variables de dominio ({len(dom_cols)}): {dom_cols}", f"faltantes: {falt}",
          "", f"control de reproducción del baseline: {control}", "", "coeficientes del bloque de dominio (mayor |coef|):"]
    for n, c in coef_dom.items():
        L.append(f"  {n}: " + "; ".join(f"{a} {b:+.3f}" for a, b in c))
    (LOGS / "metrics_fe.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Salidas: logs/metrics_fe.txt · reportes/fe_resultados.json · reportes/fe_folds.csv · reportes/figuras/fig_fe_pr.png")


if __name__ == "__main__":
    main()
