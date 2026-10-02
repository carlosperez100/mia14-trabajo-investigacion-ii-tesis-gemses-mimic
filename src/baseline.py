# -*- coding: utf-8 -*-
"""
ETAPA 3 — BASELINE MÍNIMO + VERIFICACIÓN DE REPRODUCIBILIDAD (detección, etapa 1)

Tarea: ¿la epicrisis contiene un evento adverso? (binaria; etiqueta débil A1 por CIE).
Modelos, del más simple al de la tesis:
  dummy_mayoritaria    siempre predice la clase más frecuente de train
  dummy_estratificada  azar con la proporción de train
  logistica_tfidf      TF-IDF de palabras (1-gramas) + regresión logística balanceada
  referencia_fase9     TF-IDF palabra (1-2) + carácter (3-5) + LinearSVC balanceado,
                       configuración exacta del modelo final de la tesis
Todas las transformaciones se ajustan SOLO con train (sin fuga). Partición leída de splits.

Métricas: sensibilidad, especificidad, precisión, F1, ROC-AUC, PR-AUC, VPP corregido a la
prevalencia real (Bayes) e IC 95 % por bootstrap para sensibilidad, especificidad y AUC.

Salidas versionables: reportes/baseline_resultados.json, reportes/figuras/fig_baseline_roc_pr.png
Salidas fuera de git: data/processed/modelos/*.pkl

Uso:  python src/baseline.py [--sin-referencia]
"""
import argparse
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, roc_auc_score, roc_curve)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from utils import PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "baseline"
# Cifras del modelo final de la tesis (repo de la tesis, fase9_final/resultados_finales.json, 29-07-2026)
ESPERADO_FASE9 = {"sensibilidad": 0.7623, "especificidad": 0.7699, "auc": 0.8426}
TOLERANCIA = 0.005


def modelos(cfg, seed):
    lg, rf = cfg["logistica"], cfg["referencia_fase9"]
    return {
        "dummy_mayoritaria": Pipeline([("vec", TfidfVectorizer(max_features=1)),
                                       ("clf", DummyClassifier(strategy="most_frequent"))]),
        "dummy_estratificada": Pipeline([("vec", TfidfVectorizer(max_features=1)),
                                         ("clf", DummyClassifier(strategy="stratified", random_state=seed))]),
        "logistica_tfidf": Pipeline([
            ("vec", TfidfVectorizer(max_features=lg["max_features"], min_df=3, sublinear_tf=True,
                                    strip_accents="unicode", lowercase=True)),
            ("clf", LogisticRegression(C=lg["C"], class_weight="balanced", max_iter=2000,
                                       solver="liblinear", random_state=seed))]),
        "referencia_fase9": Pipeline([
            ("vec", FeatureUnion([
                ("palabra", TfidfVectorizer(max_features=rf["palabra_max_features"], ngram_range=(1, 2),
                                            min_df=3, sublinear_tf=True, strip_accents="unicode",
                                            lowercase=True)),
                ("caracter", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                             max_features=rf["caracter_max_features"], min_df=5,
                                             sublinear_tf=True, strip_accents="unicode",
                                             lowercase=True))])),
            ("clf", LinearSVC(class_weight="balanced", random_state=seed))]),
    }


def puntaje(pipe, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        return pipe.decision_function(X), 0.0
    return pipe.predict_proba(X)[:, 1], 0.5


def ic(y, s, pred, n, seed):
    rng = np.random.default_rng(seed)
    y, s, pred = np.asarray(y), np.asarray(s), np.asarray(pred)
    se, es, au = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, pi = y[i], pred[i]
        se.append(pi[yi == 1].mean()); es.append((pi[yi == 0] == 0).mean())
        au.append(roc_auc_score(yi, s[i]) if len(np.unique(s[i])) > 1 else 0.5)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return q(se), q(es), q(au)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-referencia", action="store_true",
                    help="omite el modelo de la Fase 9 (el más lento, ~10 min)")
    args = ap.parse_args()

    log = logger(ETAPA)
    cfg_all = cargar_config()
    cfg, SEED = cfg_all["baseline"], cfg_all["semilla"]
    prev = json.loads((REPORTES / "preprocesado_resumen.json").read_text(encoding="utf-8"))["prevalencia_real"]

    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    log.info(f"train {len(tr):,} · test {len(te):,} · positivas test {te.y.mean():.1%} · prevalencia real {prev}")

    (PROCESSED / "modelos").mkdir(exist_ok=True)
    resultados, curvas = {}, {}
    for nombre, pipe in modelos(cfg, SEED).items():
        if nombre == "referencia_fase9" and args.sin_referencia:
            continue
        t0 = time.time()
        pipe.fit(tr.text, tr.y.values)
        s, umbral = puntaje(pipe, te.text)
        # la Fase 9 decide con margen >= 0; los dummy usan su propia regla de predicción
        pred = pipe.predict(te.text) if nombre.startswith("dummy") else (s >= umbral).astype(int)
        y = te.y.values
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens, esp = tp / (tp + fn), tn / (tn + fp)
        auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
        ic_s, ic_e, ic_a = ic(y, s, pred, cfg["n_bootstrap"], SEED)
        vpp = sens * prev / (sens * prev + (1 - esp) * (1 - prev)) if (sens + esp) else 0.0
        r = {"sensibilidad": round(sens, 4), "sensibilidad_ic95": ic_s,
             "especificidad": round(esp, 4), "especificidad_ic95": ic_e,
             "precision": round(precision_score(y, pred, zero_division=0), 4),
             "f1": round(f1_score(y, pred, zero_division=0), 4),
             "auc": round(auc, 4), "auc_ic95": ic_a,
             "pr_auc": round(average_precision_score(y, s), 4),
             "vpp_prevalencia_real": round(vpp, 4),
             "matriz": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
             "segundos": round(time.time() - t0, 1)}
        resultados[nombre] = r
        curvas[nombre] = (roc_curve(y, s), precision_recall_curve(y, s))
        with open(PROCESSED / "modelos" / f"{nombre}.pkl", "wb") as f:
            pickle.dump(pipe, f)
        log.info(f"{nombre:<20} sens {sens:.4f} · esp {esp:.4f} · F1 {r['f1']:.4f} · AUC {auc:.4f} "
                 f"· PR-AUC {r['pr_auc']:.4f} · VPP {vpp:.4f} ({r['segundos']} s)")

    verificacion = None
    if "referencia_fase9" in resultados:
        rf = resultados["referencia_fase9"]
        dif = {k: round(abs(rf[k] - v), 4) for k, v in ESPERADO_FASE9.items()}
        ok = all(d <= TOLERANCIA for d in dif.values())
        verificacion = {"esperado": ESPERADO_FASE9, "obtenido": {k: rf[k] for k in ESPERADO_FASE9},
                        "diferencia_abs": dif, "tolerancia": TOLERANCIA, "reproduce": ok}
        log.info(f"VERIFICACIÓN Fase 9: {'REPRODUCE' if ok else 'NO REPRODUCE'} · diferencias {dif}")

    # figura central: ROC y PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for nombre, ((fpr, tpr, _), (p, r_, _)) in curvas.items():
        a1.plot(fpr, tpr, label=f"{nombre} (AUC {resultados[nombre]['auc']:.3f})")
        a2.plot(r_, p, label=f"{nombre} (AP {resultados[nombre]['pr_auc']:.3f})")
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    a1.set(xlabel="1 − especificidad", ylabel="Sensibilidad", title="Curva ROC (test, partición por paciente)")
    a2.axhline(te.y.mean(), color="k", ls=":", lw=0.8)
    a2.set(xlabel="Sensibilidad (recall)", ylabel="Precisión", title="Curva precisión-sensibilidad")
    for a in (a1, a2):
        a.legend(fontsize=8, loc="lower right" if a is a1 else "lower left"); a.grid(alpha=0.3)
    fig.suptitle("Sprint 1 · Detección de eventos adversos en epicrisis MIMIC-IV: baselines vs. modelo de la tesis",
                 fontsize=10)
    fig.tight_layout()
    (REPORTES / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTES / "figuras" / "fig_baseline_roc_pr.png", dpi=160)

    guardar_json({"etapa": ETAPA, "semilla": SEED, "tarea": "deteccion binaria (etapa 1)",
                  "n_train": int(len(tr)), "n_test": int(len(te)), "prevalencia_real": prev,
                  "modelos": resultados, "verificacion_reproducibilidad": verificacion},
                 REPORTES / "baseline_resultados.json")
    log.info("Salidas: reportes/baseline_resultados.json · reportes/figuras/fig_baseline_roc_pr.png")
    import reporte_baseline
    reporte_baseline.main()
    log.info("Métricas -> logs/metrics_baseline.txt · reportes/figuras/fig_matrices_confusion.png")


if __name__ == "__main__":
    main()
