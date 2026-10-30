# -*- coding: utf-8 -*-
"""
VARIANTE D3 (29-10-2026) — PASO 2: D2 frente a D3, con las negativas que libera la equivalencia validada.

D2 es la variante vigente (semana 6). D3 entrena con las mismas filas que D2 y el mismo número de ellas, pero reemplaza
una parte de las negativas de la época CIE-10 por hospitalizaciones liberadas (src/negativas_liberadas.py), en la
proporción que tendrían si el corpus se sorteara de nuevo sin los códigos «no es evento adverso» en el estrato A2.
Modelo: el vigente, A+B+C con la configuración de config.yaml, sin cambios.

Protocolo: los 5 folds de la semana 6 (StratifiedGroupKFold por paciente, semilla 42, solo en train). La hospitalización
liberada de un paciente del corpus va al fold de ese paciente; la de un paciente nuevo, a un fold al azar por paciente.
Las negativas dudosas (códigos causales fuera de la tabla) no entran al entrenamiento ni a la evaluación, como en D2.
Evaluaciones, las mismas para las dos variantes:
  común     las filas de validación (o de test) sin etiquetas dudosas, las de la semana 6;
  ampliada  las mismas filas más liberadas de los mismos folds (o del test), en la proporción esperada;
  liberadas proporción de hospitalizaciones liberadas que el modelo marca como evento (p >= 0.5), sobre todas las del fold
            (o del test), frente a la de las negativas de la época CIE-10 de la evaluación común.
Regla, fijada antes de correr: D3 se considera mejor que D2 si su PR-AUC en la evaluación ampliada supera a la de D2 en al
menos 4 de los 5 folds y la diferencia media supera una desviación estándar de las diferencias. Aun así, HE2 y HE3 siguen
con D2, porque el lote de la revisión ciega se sorteó con esa variante (decisión del autor, 29-10-2026).

Salidas: logs/experimentos_d3_<fecha>.log · logs/metrics_d3.txt · reportes/d3_resultados.json · reportes/d3_folds.csv ·
         reportes/figuras/fig_d3_resultados.png
Uso:  python src/experimentos_d3.py
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
from sklearn.model_selection import StratifiedGroupKFold

import variables_dominio
import variantes_datos as vd
from experimentos_fe import ic_pacientes, metricas
from features import columnas_dominio, pipeline
from utils import INTERIM, LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "experimentos_d3"
MODELO = "A+B+C"
VARIANTES = {"D2": "vigente: sin negativas ni positivas dudosas",
             "D3": "D2 con negativas liberadas por la equivalencia validada"}


def resumen(v):
    return {"media": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4)}


def tasa_marcadas(p):
    return round(float(np.mean(p >= 0.5)), 4)


def main():
    t_inicio = time.time()
    log = logger(ETAPA)
    cfg = cargar_config()
    S, P = cfg["semilla"], cfg["feature_engineering"]
    rng = np.random.default_rng(S)
    prev = float(json.loads((REPORTES / "eda_resumen.json").read_text(encoding="utf-8"))["balance"]["prevalencia_real_con_epicrisis"])
    lib_res = json.loads((REPORTES / "negativas_liberadas_resumen.json").read_text(encoding="utf-8"))

    # ---------- corpus (igual que la semana 6) y liberadas
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    X_cols = ["text"] + dom_cols
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    if "--prueba" in sys.argv:          # corrida corta para revisar el flujo completo; escribe en las mismas salidas, así que después se repite la corrida completa
        tr, te = tr.sample(4000, random_state=S).reset_index(drop=True), te.sample(1500, random_state=S).reset_index(drop=True)
    lib = pd.read_parquet(INTERIM / "negativas_liberadas.parquet")
    lib = lib[~lib.negativa_dudosa].reset_index(drop=True)
    assert not set(lib.hadm_id) & set(df.hadm_id), "una liberada ya está en el corpus"
    assert not set(lib[lib.split == "train"].subject_id) & set(te.subject_id), "fuga: liberada de train con paciente en test"
    assert not set(lib[lib.split == "test"].subject_id) & set(tr.subject_id), "fuga: liberada de test con paciente en train"

    # proporción esperada de liberadas entre las negativas de la época CIE-10 sin etiqueta dudosa
    pr = lib_res["proporcion"]
    neg10 = df[(df.y == 0) & (df.epoca == 10)]
    d_marco = float(neg10.negativa_dudosa.mean())                       # el corpus es una muestra al azar del marco
    d_lib = lib_res["marcas"]["negativas_dudosas"] / lib_res["notas"]["tras_limpieza"]
    a, b = pr["liberadas_epoca10_con_epicrisis"] * (1 - d_lib), pr["marco_epoca10_con_epicrisis"] * (1 - d_marco)
    s = a / (a + b)
    log.info(f"proporción esperada de liberadas entre las negativas CIE-10 sin etiqueta dudosa: {s:.4f} "
             f"(dudosas: {d_marco:.4f} en el corpus, {d_lib:.4f} en las liberadas)")

    # ---------- folds de la semana 6 y fold de cada liberada de train
    sgkf = StratifiedGroupKFold(n_splits=P["n_folds"], shuffle=True, random_state=S)
    folds = list(sgkf.split(tr[X_cols], tr.y, groups=tr.subject_id))
    fold_paciente = {}
    for k, (_, val) in enumerate(folds, 1):
        fold_paciente.update(dict.fromkeys(tr.subject_id.iloc[val].unique(), k))
    lib_tr = lib[lib.split == "train"].reset_index(drop=True)
    lib_te = lib[lib.split == "test"].reset_index(drop=True)
    nuevos = np.array(sorted(set(lib_tr.subject_id) - set(fold_paciente)))
    fold_paciente.update(zip(nuevos, rng.integers(1, P["n_folds"] + 1, len(nuevos))))
    lib_tr["fold"] = lib_tr.subject_id.map(fold_paciente).astype(int)
    log.info(f"liberadas sin etiqueta dudosa: train {len(lib_tr):,} (por fold {lib_tr.fold.value_counts().sort_index().to_dict()}) · test {len(lib_te):,}")

    dudosa_tr = (tr.negativa_dudosa | tr.positiva_dudosa).values
    dudosa_te = (te.negativa_dudosa | te.positiva_dudosa).values
    usar_d2 = vd.filas_de_entrenamiento(tr, "D2").values
    neg10_tr = ((tr.y == 0) & (tr.epoca == 10)).values

    def entrenamiento_d3(idx, disponibles):
        """Filas de D3: las de D2, con una parte de las negativas CIE-10 reemplazada por liberadas (mismo total)."""
        cand = idx[neg10_tr[idx]]
        n_rep = int(round(s * len(cand)))
        assert n_rep <= len(disponibles), "no alcanzan las liberadas para el reemplazo"
        quitar = set(rng.choice(cand, size=n_rep, replace=False))
        poner = rng.choice(disponibles, size=n_rep, replace=False)
        base = tr.loc[[i for i in idx if i not in quitar], X_cols + ["y"]]
        return pd.concat([base, lib_tr.loc[poner, X_cols + ["y"]]], ignore_index=True), n_rep

    def sorteo_ampliada(n_neg10, n_disponibles):
        """Liberadas que se agregan a la evaluación común: las mismas para D2 y D3."""
        n_add = int(round(s / (1 - s) * n_neg10))
        assert n_add <= n_disponibles, "no alcanzan las liberadas para la evaluación ampliada"
        return rng.permutation(n_disponibles)[:n_add]

    def evaluar(m, filas, y, extra, sel):
        """Predicciones de la evaluación común, de las liberadas y de la ampliada (común + liberadas sorteadas)."""
        p = m.predict_proba(filas)[:, 1]
        pl = m.predict_proba(extra)[:, 1]
        return p, pl, np.concatenate([y, np.zeros(len(sel), dtype=int)]), np.concatenate([p, pl[sel]])

    # ---------- validación cruzada
    filas_cv = []
    for k, (a_idx, b_idx) in enumerate(folds, 1):
        a_d2 = a_idx[usar_d2[a_idx]]
        b_c = b_idx[~dudosa_tr[b_idx]]
        disp = lib_tr.index[lib_tr.fold != k].values
        extra = lib_tr.loc[lib_tr.fold == k, X_cols]
        y_c = tr.y.values[b_c]
        neg10_c = neg10_tr[b_c]
        sel = sorteo_ampliada(neg10_c.sum(), len(extra))
        for v in VARIANTES:
            if v == "D2":
                Xa, ya_, n_rep = tr.loc[a_d2, X_cols], tr.y.values[a_d2], 0
            else:
                d3, n_rep = entrenamiento_d3(a_d2, disp)
                Xa, ya_ = d3[X_cols], d3.y.values
            m = pipeline(MODELO, P, S)
            t0 = time.time(); m.fit(Xa, ya_); t_fit = time.time() - t0
            p, pl, ya, pa = evaluar(m, tr.loc[b_c, X_cols], y_c, extra, sel)
            n_add = len(sel)
            rc, ra = metricas(y_c, p, prev), metricas(ya, pa, prev)
            filas_cv.append({"variante": v, "fold": k, "filas_entrenamiento": int(len(ya_)), "liberadas_en_entrenamiento": n_rep,
                             "seg_entrenamiento": round(t_fit, 1), "liberadas_en_ampliada": n_add,
                             **{f"{m_}_comun": round(x, 6) for m_, x in rc.items()},
                             **{f"{m_}_ampliada": round(x, 6) for m_, x in ra.items()},
                             "marcadas_liberadas": tasa_marcadas(pl), "liberadas_evaluadas": int(len(pl)),
                             "marcadas_negativas_cie10_comun": tasa_marcadas(p[neg10_c])})
            f = filas_cv[-1]
            log.info(f"{v} fold {k}: entrena con {f['filas_entrenamiento']:,} ({n_rep:,} liberadas) · PR-AUC común "
                     f"{f['pr_auc_comun']:.4f} · ampliada {f['pr_auc_ampliada']:.4f} · liberadas marcadas "
                     f"{f['marcadas_liberadas']:.4f} de {len(pl):,} · negativas CIE-10 marcadas {f['marcadas_negativas_cie10_comun']:.4f} · {t_fit:.0f} s")
    folds_df = pd.DataFrame(filas_cv)
    folds_df.to_csv(REPORTES / "d3_folds.csv", index=False, float_format="%.6f")

    cv = {}
    g2 = folds_df[folds_df.variante == "D2"].set_index("fold")
    for v, g in folds_df.groupby("variante", sort=False):
        g = g.set_index("fold")
        cv[v] = {e: {m_: resumen(g[f"{m_}_{e}"]) for m_ in ("pr_auc", "roc_auc", "f1", "recall", "especificidad")} for e in ("comun", "ampliada")}
        cv[v]["marcadas_liberadas"] = resumen(g.marcadas_liberadas)
        cv[v]["marcadas_negativas_cie10_comun"] = resumen(g.marcadas_negativas_cie10_comun)
        for e in ("comun", "ampliada"):
            d = g[f"pr_auc_{e}"] - g2[f"pr_auc_{e}"]
            cv[v][f"dif_pr_auc_{e}_vs_D2"] = {**resumen(d), "folds_mejor_que_D2": int((d > 0).sum())}
    d = cv["D3"]["dif_pr_auc_ampliada_vs_D2"]
    regla = d["folds_mejor_que_D2"] >= 4 and d["media"] > d["std"]
    log.info(f"CV: D2 ampliada {cv['D2']['ampliada']['pr_auc']['media']:.4f} · D3 ampliada {cv['D3']['ampliada']['pr_auc']['media']:.4f} · "
             f"dif. {d['media']:+.4f} ± {d['std']:.4f} ({d['folds_mejor_que_D2']}/5 folds) · regla cumplida: {regla}")

    # ---------- test: cada variante se entrena con todo train y se mide una sola vez
    test = {}
    c_te = ~dudosa_te
    y_te, neg10_te = te.y.values[c_te], ((te.y == 0) & (te.epoca == 10)).values[c_te]
    a_all = np.arange(len(tr))[usar_d2]
    sel = sorteo_ampliada(neg10_te.sum(), len(lib_te))
    sub = te.subject_id.values[c_te]
    a2 = pd.read_csv(INTERIM / "a1_tier_a.csv", usecols=["hadm_id", "estrato", "icd_code"], dtype={"icd_code": str})
    a2 = a2[(a2.estrato == "A2") & a2.hadm_id.isin(set(lib_te.hadm_id))]
    codigos = lib_te.hadm_id.map(a2.groupby("hadm_id").icd_code.agg(set))
    for v in VARIANTES:
        if v == "D2":
            Xa, ya_, n_rep = tr.loc[a_all, X_cols], tr.y.values[a_all], 0
        else:
            d3, n_rep = entrenamiento_d3(a_all, lib_tr.index.values)
            Xa, ya_ = d3[X_cols], d3.y.values
        m = pipeline(MODELO, P, S).fit(Xa, ya_)
        p, pl, ya, pa = evaluar(m, te.loc[c_te, X_cols], y_te, lib_te[X_cols], sel)
        n_add = len(sel)
        por_codigo = {}
        for c in ("N17.9", "J96.0", "B95", "S22", "N17.0"):
            msk = codigos.map(lambda s_: c in s_).values
            if msk.sum():
                por_codigo[c] = {"n": int(msk.sum()), "marcadas": tasa_marcadas(pl[msk])}
        test[v] = {"filas_entrenamiento": int(len(ya_)), "liberadas_en_entrenamiento": n_rep,
                   "comun": {**{k_: round(x, 4) for k_, x in metricas(y_te, p, prev).items()},
                             "pr_auc_ic95_pacientes": ic_pacientes(y_te, p, sub, P["n_bootstrap"], S)},
                   "ampliada": {**{k_: round(x, 4) for k_, x in metricas(ya, pa, prev).items()}, "liberadas_agregadas": n_add},
                   "marcadas_liberadas": tasa_marcadas(pl), "liberadas_evaluadas": int(len(pl)),
                   "marcadas_negativas_cie10_comun": tasa_marcadas(p[neg10_te]), "marcadas_liberadas_por_codigo": por_codigo}
        log.info(f"test {v}: PR-AUC común {test[v]['comun']['pr_auc']:.4f} {test[v]['comun']['pr_auc_ic95_pacientes']} · ampliada "
                 f"{test[v]['ampliada']['pr_auc']:.4f} · liberadas marcadas {test[v]['marcadas_liberadas']:.4f} de {len(pl):,} · "
                 f"negativas CIE-10 marcadas {test[v]['marcadas_negativas_cie10_comun']:.4f} · por código {por_codigo}")

    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "modelo": MODELO,
         "parametros": P, "prevalencia_real_con_epicrisis": prev, "variantes": VARIANTES,
         "liberadas": {"sin_etiqueta_dudosa_train": int(len(lib_tr)), "sin_etiqueta_dudosa_test": int(len(lib_te)),
                       "proporcion_esperada_entre_negativas_cie10": round(s, 4), "dudosas_en_el_corpus_cie10": round(d_marco, 4),
                       "dudosas_en_las_liberadas": round(d_lib, 4)},
         "protocolo": f"StratifiedGroupKFold({P['n_folds']}, shuffle=True, random_state={S}) por subject_id en train (folds de la semana 6); "
                      "D3 reemplaza negativas CIE-10 de D2 por liberadas en la proporción esperada, con el mismo total de filas; "
                      "evaluación común (semana 6), ampliada (común + liberadas en la proporción esperada) y tasa de liberadas marcadas",
         "regla": "D3 mejor que D2 si su PR-AUC ampliada supera a la de D2 en >= 4 de 5 folds y la diferencia media supera 1 desv. estándar",
         "regla_cumplida": bool(regla), "cv": cv, "test": test}
    guardar_json(R, REPORTES / "d3_resultados.json")
    figura(folds_df, R)

    L = [f"# metrics_d3.txt · {R['generado']} · modelo {MODELO} · semilla {S}", f"protocolo: {R['protocolo']}",
         f"liberadas: {R['liberadas']}", f"regla: {R['regla']} -> cumplida: {regla}", "",
         "validación cruzada (media ± desv. de 5 folds):"]
    for v, c in cv.items():
        L.append(f"  {v}: PR-AUC común {c['comun']['pr_auc']['media']:.4f} ± {c['comun']['pr_auc']['std']:.4f} · ampliada "
                 f"{c['ampliada']['pr_auc']['media']:.4f} ± {c['ampliada']['pr_auc']['std']:.4f} · liberadas marcadas "
                 f"{c['marcadas_liberadas']['media']:.4f} · negativas CIE-10 marcadas {c['marcadas_negativas_cie10_comun']['media']:.4f} · "
                 f"dif. ampliada con D2 {c['dif_pr_auc_ampliada_vs_D2']['media']:+.4f} ({c['dif_pr_auc_ampliada_vs_D2']['folds_mejor_que_D2']}/5)")
    L += ["", "test (una sola medición):"]
    for v, t in test.items():
        L.append(f"  {v}: común PR-AUC {t['comun']['pr_auc']:.4f} {t['comun']['pr_auc_ic95_pacientes']} · ampliada {t['ampliada']['pr_auc']:.4f} · "
                 f"liberadas marcadas {t['marcadas_liberadas']:.4f} de {t['liberadas_evaluadas']:,} · negativas CIE-10 marcadas "
                 f"{t['marcadas_negativas_cie10_comun']:.4f} · por código {t['marcadas_liberadas_por_codigo']}")
    L += ["", f"duración: {round(time.time() - t_inicio)} s"]
    (LOGS / "metrics_d3.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Salidas: logs/metrics_d3.txt · reportes/d3_resultados.json · reportes/d3_folds.csv · reportes/figuras/fig_d3_resultados.png")


def figura(folds_df, R):
    """PR-AUC por fold en las dos evaluaciones y proporción de liberadas marcadas como evento, D2 frente a D3."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colores = {"D2": "#0F7A6B", "D3": "#C2410C"}
    for v, g in folds_df.groupby("variante", sort=False):
        ax[0].plot(g.fold, g.pr_auc_comun, "o-", color=colores[v], label=f"{v} común")
        ax[0].plot(g.fold, g.pr_auc_ampliada, "s--", color=colores[v], label=f"{v} ampliada")
    ax[0].set(xlabel="fold", ylabel="PR-AUC", title="PR-AUC por fold", xticks=range(1, 6))
    ax[0].legend(fontsize=8)
    t = R["test"]
    x = np.arange(2)
    for i, v in enumerate(("D2", "D3")):
        ax[1].bar(x + (i - 0.5) * 0.38, [t[v]["marcadas_liberadas"], t[v]["marcadas_negativas_cie10_comun"]], 0.38,
                  color=colores[v], label=v)
    ax[1].set(xticks=x, xticklabels=["liberadas", "negativas CIE-10\nde la evaluación común"], ylabel="proporción con p ≥ 0.5",
              title="Marcadas como evento en test")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    salida = REPORTES / "figuras" / "fig_d3_resultados.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
