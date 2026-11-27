# -*- coding: utf-8 -*-
"""
SEMANA 11 · Sprint 4: mejora del modelo con otro algoritmo (E1) y otra arquitectura (E2), frente a la configuración actual.

Enunciado (Teachlr, «Entregable semana 11»): mejorar el modelo con mejores datasets, más feature engineering o más optimización de
hiperparámetros; si los modelos no mejoran, probar otro algoritmo u otra arquitectura. En la semana 8 la optimización de
hiperparámetros no mejoró el test (PR-AUC 0.8670 frente a 0.8671), de modo que esta semana prueba otro algoritmo y otra
arquitectura con los mismos datos y atributos. El diseño está en config/semana11_mejora.yaml, escrito antes de correr.

  Referencia  D2 con A+B+C y la configuración actual (regresión logística liblinear, balanced, C 1.0; A y B 20 000/3; sublinear_tf).
  E1          Otro algoritmo: regresión logística sobre la presencia de los términos de A y B multiplicada por la razón de
              log-conteos de Naive Bayes, calculada en las filas de entrenamiento de cada fold; el bloque C entra sin cambios.
  E2          Otra arquitectura, en dos etapas: la regresión logística con solo el texto (A+B) da un puntaje, fuera de fold dentro de
              las filas de entrenamiento (3 folds internos por paciente), y un HistGradientBoostingClassifier combina ese puntaje
              con el bloque C.

Datos y partición: los de src/experimentos_hp.py (los 5 folds del proyecto, StratifiedGroupKFold por paciente, semilla 42; entrenan
las filas de D2; la PR-AUC se mide en las epicrisis de validación sin etiqueta dudosa). Usa la caché de conteos de los 5 folds de
confirmación (data/interim/cache_hp/confirmacion_k5_f*). El test no se mide ni se lee: sus filas se descartan al cargar los datos.
Control: la referencia reproduce, fold por fold, la confirmación de la semana 8 (reportes/hp_pruning_resultados.json).
Regla: la de las semanas 7 y 8 (al menos 4 de 5 folds mejores y mejora media mayor que 1 desviación estándar de las diferencias).

Salidas: logs/experimentos_semana11_<fecha>.log · reportes/semana11_folds.csv · reportes/semana11_resultados.json ·
         reportes/figuras/fig_semana11_folds.png
Uso:  python src/experimentos_semana11.py            (requiere la caché de python src/experimentos_hp.py --solo-conteos)
      python src/experimentos_semana11.py --prueba   (un fold y una muestra de filas; escribe en data/interim/prueba_semana11)
Si la corrida se interrumpe, al repetirla se toman de data/interim/semana11_folds.json los folds ya evaluados.
"""
import json
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import experimentos_hp as hp
import variables_dominio
import variantes_datos as vd
from utils import INTERIM, PROCESSED, RAIZ, REPORTES, cargar_config, guardar_json, logger

ETAPA = "experimentos_semana11"
VERSION = 1                      # cambia si cambia el cálculo: invalida la caché de folds
EXPERIMENTOS = ("referencia", "E1_nblr", "E2_fusion_tardia")


# ---------------------------------------------------------------- atributos desde la caché de conteos
def bloques_texto(d, prm, presencia=False, filas_tr=None, filas_va=None):
    """A y B de la configuración, como en hp.matrices: TF-IDF ajustado en train del fold o presencia (0/1)."""
    tr, va = [], []
    for b in hp.BLOQUES_TEXTO:
        cols = hp.seleccion(hp.cargar_meta(d / f"{b}_meta.npz"), prm[f"{b}_min_df"], prm[f"{b}_max_features"])
        Xtr = hp.a_float(sparse.load_npz(d / f"{b}_tr.npz")[:, cols])
        Xva = hp.a_float(sparse.load_npz(d / f"{b}_va.npz")[:, cols])
        if filas_tr is not None:
            Xtr, Xva = Xtr[filas_tr], Xva[filas_va]
        if presencia:
            Xtr.data[:] = 1.0
            Xva.data[:] = 1.0
        else:
            t = TfidfTransformer(sublinear_tf=prm["sublinear_tf"]).fit(Xtr)
            Xtr, Xva = t.transform(Xtr, copy=False), t.transform(Xva, copy=False)
        tr.append(Xtr)
        va.append(Xva)
    return hp.apilar(tr), hp.apilar(va)


def bloque_c(d, filas_tr=None, filas_va=None):
    Ctr, Cva = sparse.load_npz(d / "C_tr.npz").tocsr(), sparse.load_npz(d / "C_va.npz").tocsr()
    if filas_tr is not None:
        Ctr, Cva = Ctr[filas_tr], Cva[filas_va]
    return Ctr, Cva


# ---------------------------------------------------------------- modelos
def referencia(d, y_tr, clf, prm, ftr, fva):
    if ftr is None:
        Xtr, Xva, _ = hp.matrices(d, prm)
    else:
        Ttr, Tva = bloques_texto(d, prm, filas_tr=ftr, filas_va=fva)
        Ctr, Cva = bloque_c(d, ftr, fva)
        Xtr, Xva = hp.apilar([Ttr, Ctr]), hp.apilar([Tva, Cva])
    return clone(clf).fit(Xtr, y_tr).predict_proba(Xva)[:, 1], {}


def e1_nblr(d, y_tr, clf, prm, ftr, fva, alfa):
    Ptr, Pva = bloques_texto(d, prm, presencia=True, filas_tr=ftr, filas_va=fva)
    p = alfa + np.asarray(Ptr[y_tr == 1].sum(axis=0)).ravel()
    q = alfa + np.asarray(Ptr[y_tr == 0].sum(axis=0)).ravel()
    r = np.log((p / p.sum()) / (q / q.sum()))
    Ctr, Cva = bloque_c(d, ftr, fva)
    Xtr = hp.apilar([sparse.csr_matrix(Ptr.multiply(r)), Ctr])
    Xva = hp.apilar([sparse.csr_matrix(Pva.multiply(r)), Cva])
    extra = {"r_min": round(float(r.min()), 4), "r_max": round(float(r.max()), 4), "r_abs_medio": round(float(np.abs(r).mean()), 4)}
    return clone(clf).fit(Xtr, y_tr).predict_proba(Xva)[:, 1], extra


def e2_fusion(d, y_tr, g_tr, clf, prm, ftr, fva, E2, semilla):
    Ttr, Tva = bloques_texto(d, prm, filas_tr=ftr, filas_va=fva)
    s_oof = np.zeros(Ttr.shape[0])
    for ia, ib in StratifiedGroupKFold(n_splits=E2["folds_internos"], shuffle=True, random_state=semilla).split(Ttr, y_tr, groups=g_tr):
        assert not set(g_tr[ia]) & set(g_tr[ib]), "fuga de pacientes en un fold interno"
        s_oof[ib] = clone(clf).fit(Ttr[ia], y_tr[ia]).decision_function(Ttr[ib])
    m_texto = clone(clf).fit(Ttr, y_tr)
    s_va = m_texto.decision_function(Tva)
    Ctr, Cva = bloque_c(d, ftr, fva)
    Ztr, Zva = np.column_stack([Ctr.toarray(), s_oof]), np.column_stack([Cva.toarray(), s_va])
    hgb = HistGradientBoostingClassifier(**E2["hgb"]).fit(Ztr, y_tr)
    extra = {"n_columnas_etapa2": int(Ztr.shape[1]), "iteraciones_hgb": int(hgb.n_iter_)}
    return hgb.predict_proba(Zva)[:, 1], extra, s_va


# ---------------------------------------------------------------- figura
def figura(folds, ruta):
    tab = pd.DataFrame(folds).pivot(index="fold", columns="experimento", values="pr_auc_comun")
    etiquetas = {"referencia": "Referencia (logística A+B+C)", "E1_nblr": "E1 · NB-LR", "E2_fusion_tardia": "E2 · fusión tardía"}
    estilos = {"referencia": dict(color="#4a4a4a", marker="o"), "E1_nblr": dict(color="#1f77b4", marker="s"),
               "E2_fusion_tardia": dict(color="#d62728", marker="^")}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for e in [x for x in EXPERIMENTOS if x in tab]:
        ax.plot(tab.index, tab[e], label=etiquetas[e], linewidth=1.6, **estilos[e])
    ax.set_xticks(tab.index)
    ax.set_xlabel("Fold de validación (5 folds por paciente, solo train)")
    ax.set_ylabel("PR-AUC sin etiqueta dudosa")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    t_inicio = time.time()
    prueba = "--prueba" in sys.argv
    log = logger(ETAPA + ("_prueba" if prueba else ""))
    cfg = cargar_config()
    S, P, H = cfg["semilla"], cfg["feature_engineering"], cfg["hiperparametros"]
    D = yaml.safe_load((RAIZ / "config" / "semana11_mejora.yaml").read_text(encoding="utf-8"))["semana11_mejora"]
    E1, E2, REGLA = D["experimentos"]["E1_nblr"], D["experimentos"]["E2_fusion_tardia"], D["regla"]
    assert D["variante"] == H["variante"] and D["n_folds"] == H["n_folds_confirmacion"] and D["test"] is False
    ACT = hp.actual(P)
    clf = hp.modelo_oficial(ACT, P, S, H["modelo"]).named_steps["clf"]
    log.info(f"referencia: D2 con A+B+C · {hp.texto_prm(ACT)} · clasificador {clf}")
    log.info(f"E1: NB-LR con alfa {E1['alfa']} · E2: fusión tardía, {E2['folds_internos']} folds internos, HGB {E2['hgb']}")
    log.info(f"REGLA (semanas 7 y 8): al menos {REGLA['folds_mejores_minimo']} de 5 folds mejores y mejora media mayor que "
             f"{REGLA['desviaciones_minimas']} desviación estándar de las diferencias por fold · el test no se mide")

    # ---------- datos: los mismos pasos de experimentos_hp.py; las filas de test se descartan sin usarse
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    tr = df[df.split == "train"].reset_index(drop=True)
    n_test = int((df.split == "test").sum())
    del df
    usar = vd.filas_de_entrenamiento(tr, H["variante"]).values
    comun = ~(tr.negativa_dudosa | tr.positiva_dudosa).values
    y, grupos = tr.y.values.astype(int), tr.subject_id.values
    log.info(f"train {len(tr):,} (entrenan {usar.sum():,} filas de D2; evaluación común {comun.sum():,}) · "
             f"{n_test:,} filas de test descartadas sin usarse")
    part = {}
    for i, (a, b) in enumerate(StratifiedGroupKFold(n_splits=D["n_folds"], shuffle=True, random_state=S).split(tr, tr.y, groups=grupos), 1):
        assert not set(grupos[a]) & set(grupos[b]), f"fuga de pacientes en el fold {i}"
        part[i] = (a[usar[a]], b)
    del tr

    # ---------- control: la referencia de la semana 8, fold por fold
    s8 = json.loads((REPORTES / "hp_pruning_resultados.json").read_text(encoding="utf-8"))
    ref_s8 = {int(f["particion"].rsplit("_f", 1)[1]): f["pr_auc_comun"] for f in s8["confirmacion_5_folds"]["actual"]["folds"]}

    rng = np.random.default_rng(S)
    if prueba:
        part = {1: part[1]}
    salida = INTERIM / "prueba_semana11" if prueba else REPORTES
    memo_f = INTERIM / ("prueba_semana11_folds.json" if prueba else "semana11_folds.json")
    memo = json.loads(memo_f.read_text(encoding="utf-8")) if memo_f.exists() and not prueba else {}
    folds = []
    for i, (a, b) in part.items():
        d = INTERIM / "cache_hp" / f"confirmacion_k{D['n_folds']}_f{i}"
        assert (d / "listo.json").exists(), f"falta la caché {d}; correr python src/experimentos_hp.py --solo-conteos"
        listo = json.loads((d / "listo.json").read_text(encoding="utf-8"))
        assert listo["filas_train"] == len(a) and listo["filas_validacion"] == len(b), f"la caché del fold {i} no corresponde"
        ftr = fva = None
        if prueba:                                   # muestra de filas para revisar el flujo
            ftr, fva = np.sort(rng.choice(len(a), 4000, replace=False)), np.sort(rng.choice(len(b), 2000, replace=False))
            a, b = a[ftr], b[fva]
        y_tr, yb, cb = y[a], y[b], comun[b]
        for e in EXPERIMENTOS:
            k = json.dumps({"v": VERSION, "fold": i, "experimento": e})
            if k in memo:
                folds.append(memo[k])
                log.info(f"fold {i} · {e}: tomado de la caché ({memo[k]['pr_auc_comun']:.6f})")
                continue
            t0 = time.time()
            if e == "referencia":
                p, extra = referencia(d, y_tr, clf, ACT, ftr, fva)
            elif e == "E1_nblr":
                p, extra = e1_nblr(d, y_tr, clf, ACT, ftr, fva, E1["alfa"])
            else:
                p, extra, s_va = e2_fusion(d, y_tr, grupos[a], clf, ACT, ftr, fva, E2, S)
                extra["pr_auc_comun_etapa1_solo_texto"] = float(average_precision_score(yb[cb], s_va[cb]))
            f = {"fold": i, "experimento": e, "pr_auc_comun": float(average_precision_score(yb[cb], p[cb])),
                 "roc_auc_comun": float(roc_auc_score(yb[cb], p[cb])), "pr_auc_completa": float(average_precision_score(yb, p)),
                 "filas_entrenamiento": int(len(a)), "filas_validacion": int(len(b)), "filas_validacion_comun": int(cb.sum()),
                 "segundos": round(time.time() - t0, 1), **extra}
            if e == "referencia" and not prueba:
                f["control_semana8"] = ref_s8[i]
                f["control_dif"] = abs(f["pr_auc_comun"] - ref_s8[i])
                assert f["control_dif"] <= 1e-9, f"la referencia no reproduce la semana 8 en el fold {i}: {f['control_dif']:.2e}"
            folds.append(f)
            log.info(f"fold {i} · {e}: PR-AUC {f['pr_auc_comun']:.6f} · ROC-AUC {f['roc_auc_comun']:.6f} · {f['segundos']:.0f} s"
                     + (f" · control semana 8 {f['control_dif']:.1e}" if "control_dif" in f else ""))
            if not prueba:
                memo[k] = f
                memo_f.write_text(json.dumps(memo), encoding="utf-8")

    # ---------- resumen y regla
    tab = pd.DataFrame(folds)
    salida.mkdir(parents=True, exist_ok=True)
    tab.to_csv(salida / "semana11_folds.csv", index=False)
    resumen = {}
    for e in EXPERIMENTOS:
        t = tab[tab.experimento == e].sort_values("fold")
        resumen[e] = {m: {"media": round(float(t[m].mean()), 6), "std": round(float(t[m].std(ddof=1)), 6) if len(t) > 1 else 0.0}
                      for m in ("pr_auc_comun", "roc_auc_comun", "pr_auc_completa")}
        resumen[e]["segundos"] = round(float(t.segundos.sum()), 1)
    ref = tab[tab.experimento == "referencia"].sort_values("fold").pr_auc_comun.values
    reglas = {}
    for e in EXPERIMENTOS[1:]:
        dif = tab[tab.experimento == e].sort_values("fold").pr_auc_comun.values - ref
        reglas[e] = hp.regla_de_decision(dif, REGLA) if len(dif) > 1 else {"diferencias_por_fold": dif.round(6).tolist()}
        dif_roc = (tab[tab.experimento == e].sort_values("fold").roc_auc_comun.values
                   - tab[tab.experimento == "referencia"].sort_values("fold").roc_auc_comun.values)
        reglas[e]["roc_auc_diferencias_por_fold"] = [round(float(x), 6) for x in dif_roc]
        log.info(f"{e}: diferencias de PR-AUC por fold {reglas[e]['diferencias_por_fold']} · "
                 f"cumple la regla: {reglas[e].get('adopta_candidata')}")
    candidatos = [e for e in reglas if reglas[e].get("adopta_candidata")]
    decision = (f"candidato(s) del curso: {', '.join(candidatos)}; su test se mide después del contraste de la HE1, como análisis no "
                "prerregistrado" if candidatos else "ningún experimento cumple la regla; se mantiene la referencia")
    log.info(f"DECISIÓN: {decision}")
    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "prueba": prueba, "semilla": S,
         "sklearn": sklearn.__version__, "diseno": "config/semana11_mejora.yaml", "referencia": ACT,
         "protocolo": "StratifiedGroupKFold(5, shuffle=True, random_state=42) por subject_id en train (los folds de las semanas 5 a 8); "
                      "entrenan las filas de D2; métrica: PR-AUC en las epicrisis de validación sin etiqueta dudosa; test sin medir",
         "filas_test_descartadas": n_test, "resumen": resumen, "regla": reglas, "decision": decision,
         "segundos_corrida": round(time.time() - t_inicio, 1)}
    guardar_json(R, salida / "semana11_resultados.json")
    figura(folds, (salida / "figuras" / "fig_semana11_folds.png") if not prueba else salida / "fig_semana11_folds.png")
    log.info(f"listo en {(time.time() - t_inicio) / 60:.1f} min → {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
