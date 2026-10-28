# -*- coding: utf-8 -*-
"""
SEMANA 7 · Sprint 3: importancia de variables del modelo vigente (D2 con A+B+C, configuración confirmada en la búsqueda
de hiperparámetros: C 1, A y B con 20 000 términos y min_df 3, class_weight balanced y sublinear_tf).

Partición: solo los 5 folds del proyecto dentro de train (StratifiedGroupKFold por paciente, semilla 42, los mismos de las
semanas 5 a 7). En cada fold el modelo se entrena con las filas de D2 y se evalúa en las epicrisis de validación sin etiqueta
dudosa. No se usa test, y no se calcula ninguna importancia ni desempeño por estratos (servicio, edad o largo de la nota).

Métodos
  1. Por bloque: importancia por permutación de grupo. Cada bloque (A palabras, B bigramas, C contexto, y A+B como texto
     completo) se permuta en conjunto entre las filas de validación del fold, n_repeticiones veces, y se mide la caída de la
     PR-AUC. Como el modelo es lineal, permutar las filas de un grupo de atributos equivale a permutar su aporte al logit.
  2. Dentro de C: la misma permutación por variable original (las columnas one-hot y las marcas de faltante se permutan
     juntas, con la agrupación de src/experimentos_datos.py).
  3. Coeficientes de la regresión: términos de A y de B con mayor coeficiente positivo y negativo en cada fold y su
     estabilidad entre los 5 folds (intersección del top, Jaccard y correlación de rangos de Spearman).
Filtro de privacidad (licencia de PhysioNet) antes de mostrar o guardar cualquier término: solo letras (sin dígitos ni ___),
al menos min_pacientes pacientes distintos y min_pct_notas % de las epicrisis de train, y sin aspecto de nombre propio
(mayúscula inicial en más de max_pct_mayuscula_inicial % de sus apariciones en una muestra de train).

Reutiliza la caché de conteos de data/interim/cache_hp/ (src/experimentos_hp.py) y comprueba, fold por fold, que las matrices
coinciden con las que produce el pipeline de src/features.py al ajustarse en las mismas filas.

Salidas: logs/importancia_variables_<fecha>.log · logs/metrics_importancia.txt · reportes/importancia_resultados.json ·
         reportes/importancia_grupos.csv · reportes/importancia_terminos.csv · reportes/figuras/fig_imp_*.png
Uso:  python src/importancia_variables.py           (requiere la caché de python src/experimentos_hp.py --solo-conteos)
      python src/importancia_variables.py --prueba  (2 folds y 2 permutaciones; escribe en data/interim/prueba_imp)
"""
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import experimentos_hp as hp
import figuras_imp
import variables_dominio
import variantes_datos as vd
from experimentos_datos import grupos_de_atributos
from features import columnas_dominio
from utils import INTERIM, LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "importancia_variables"
BLOQUES = {"A": "A_palabras", "B": "B_bigramas", "C": "C_dominio"}


def huella(M):
    """Huella exacta de una matriz dispersa (valores e índices, con los índices ordenados)."""
    M = M.tocsr().copy()
    M.sort_indices()
    h = hashlib.sha256()
    for a in (M.data, M.indices, M.indptr, np.array(M.shape)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def pr_auc_permutada(y, z, z_g, n, rng):
    return [average_precision_score(y, expit(z - z_g + z_g[rng.permutation(len(z_g))])) for _ in range(n)]


def resumen(v):
    v = np.asarray(v, dtype=float)
    return {"media": round(float(v.mean()), 5), "std": round(float(v.std(ddof=1)), 5) if len(v) > 1 else 0.0}


def main():
    t_inicio = time.time()
    prueba = "--prueba" in sys.argv          # 2 folds y 2 permutaciones, para revisar el flujo; escribe en data/interim/prueba_imp
    log = logger(ETAPA + ("_prueba" if prueba else ""))
    cfg = cargar_config()
    S, P, I = cfg["semilla"], cfg["feature_engineering"], dict(cfg["importancia"])
    if prueba:
        I["n_repeticiones"] = 2
    salida = INTERIM / "prueba_imp" if prueba else REPORTES
    (salida / "figuras").mkdir(parents=True, exist_ok=True)
    F = I["filtro_privacidad"]
    rng = np.random.default_rng(S)
    prm = hp.actual(P)
    modelo = hp.modelo_oficial(prm, P, S, "A+B+C")
    log.info(f"modelo: D2 con A+B+C · {hp.texto_prm(prm)} · partición: {I['particion']} (5 folds dentro de train; no se usa test)")
    log.info(f"parámetros: {json.dumps(I, ensure_ascii=False)}")

    # ---------- datos: solo train (las mismas filas y folds de experimentos_hp.py)
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    tr = df[df.split == "train"].reset_index(drop=True)
    del df
    usar = vd.filas_de_entrenamiento(tr, "D2").values
    comun = ~(tr.negativa_dudosa | tr.positiva_dudosa).values
    y = tr.y.values.astype(int)
    k_folds = cfg["hiperparametros"]["n_folds_confirmacion"]
    folds = list(StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=S).split(tr, tr.y, groups=tr.subject_id))
    s6 = pd.read_csv(REPORTES / "datos_folds.csv")
    s6 = s6[s6.variante == "D2"].sort_values("fold").pr_auc_comun.values
    if prueba:
        folds, s6 = folds[:2], s6[:2]
    log.info(f"train {len(tr):,} epicrisis · entrenan {usar.sum():,} filas de D2 · {k_folds} folds por paciente")

    atr = modelo.named_steps["atributos"]
    vec = {b: dict((t[0], t[1]) for t in atr.transformers)[BLOQUES[b]] for b in ("A", "B")}
    _, tf_c, cols_c = [t for t in atr.transformers if t[0] == BLOQUES["C"]][0]
    filas, coefs, equivalencia, base = [], [], [], []
    t_modelos = t_nombres = t_perm = 0.0
    for k, (a, b) in enumerate(folds, 1):
        assert not set(tr.subject_id.iloc[a]) & set(tr.subject_id.iloc[b]), f"fuga en el fold {k}"
        a = a[usar[a]]
        d = INTERIM / "cache_hp" / f"{I['particion']}_k{k_folds}_f{k}"
        listo = json.loads((d / "listo.json").read_text(encoding="utf-8"))
        assert listo["filas_train"] == len(a) and listo["filas_validacion"] == len(b), f"la caché del fold {k} no corresponde a estas filas"
        t0 = time.time()
        Xtr, Xva, _ = hp.matrices(d, prm)
        clf = clone(modelo.named_steps["clf"]).fit(Xtr, y[a])
        t_modelos += time.time() - t0
        w, b0 = clf.coef_[0], clf.intercept_[0]
        # nombres y control de equivalencia: los bloques del pipeline de features.py ajustados en las mismas filas
        t0 = time.time()
        anchos, nombres = {}, []
        for blq in ("A", "B"):
            v = clone(vec[blq])
            M = v.fit_transform(tr.text.values[a])
            anchos[blq] = M.shape[1]
            ini = sum(anchos.values()) - M.shape[1]
            equivalencia.append({"fold": k, "bloque": blq, "iguales": huella(M) == huella(Xtr[:, ini:ini + M.shape[1]])})
            nombres += [f"{BLOQUES[blq]}__{t}" for t in v.get_feature_names_out()]
            del M, v
        c = clone(tf_c)
        M = sparse.csr_matrix(c.fit_transform(tr.loc[a, cols_c]))
        ini = anchos["A"] + anchos["B"]
        equivalencia.append({"fold": k, "bloque": "C", "iguales": huella(M) == huella(Xtr[:, ini:])})
        nombres += [f"{BLOQUES['C']}__{t}" for t in c.get_feature_names_out()]
        assert len(nombres) == Xtr.shape[1] == len(w)
        del M, c, Xtr
        t_nombres += time.time() - t0
        log.info(f"fold {k}: caché igual al pipeline de features.py en A, B y C: "
                 f"{all(e['iguales'] for e in equivalencia if e['fold'] == k)}")

        # ---------- permutación por grupo, sobre las epicrisis de validación sin etiqueta dudosa
        cb = comun[b]
        Xv = Xva[np.where(cb)[0]]
        yv = y[b][cb]
        z = np.asarray(Xv @ w).ravel() + b0
        pr0 = average_precision_score(yv, expit(z))
        base.append(pr0)
        g = grupos_de_atributos(np.array(nombres), dom_cols)
        idx = np.arange(len(nombres))
        grupos = {"A · palabras": idx[:anchos["A"]], "B · bigramas": idx[anchos["A"]:anchos["A"] + anchos["B"]],
                  "C · contexto": idx[anchos["A"] + anchos["B"]:], "A+B · texto completo": idx[:anchos["A"] + anchos["B"]]}
        grupos.update({k_: np.array(v_) for k_, v_ in g.items() if not k_.startswith("texto")})
        t0 = time.time()
        for nombre, cols in grupos.items():
            z_g = np.asarray(Xv[:, cols] @ w[cols]).ravel()
            caidas = pr0 - np.array(pr_auc_permutada(yv, z, z_g, I["n_repeticiones"], rng))
            filas.append({"fold": k, "grupo": nombre, "tipo": "bloque" if "·" in nombre else "variable de C",
                          "n_columnas": int(len(cols)), "pr_auc_base": round(float(pr0), 6),
                          "caida_media": round(float(caidas.mean()), 6), "caida_std_repeticiones": round(float(caidas.std(ddof=1)), 6)})
        t_perm += time.time() - t0
        log.info(f"fold {k}: PR-AUC de validación {pr0:.6f} (semana 6: {s6[k - 1]:.6f}) · {len(grupos)} grupos permutados "
                 f"{I['n_repeticiones']} veces en {time.time() - t0:.0f} s")
        coefs.append({"A": dict(zip([n.split("__", 1)[1] for n in nombres[:anchos["A"]]], w[:anchos["A"]])),
                      "B": dict(zip([n.split("__", 1)[1] for n in nombres[anchos["A"]:anchos["A"] + anchos["B"]]],
                                    w[anchos["A"]:anchos["A"] + anchos["B"]]))})
        del Xva, Xv, clf

    G = pd.DataFrame(filas)
    imp = {}
    for nombre, gr in G.groupby("grupo", sort=False):
        imp[nombre] = {"tipo": gr.tipo.iloc[0], "n_columnas_media": round(float(gr.n_columnas.mean()), 1), **resumen(gr.caida_media),
                       "por_fold": [round(float(x), 6) for x in gr.caida_media],
                       "std_repeticiones_media": round(float(gr.caida_std_repeticiones.mean()), 6)}
    bloques = {k: v for k, v in imp.items() if v["tipo"] == "bloque"}
    contexto = dict(sorted(((k, v) for k, v in imp.items() if v["tipo"] != "bloque"), key=lambda kv: -kv[1]["media"]))
    log.info("bloques: " + " · ".join(f"{k} {v['media']:.4f} ± {v['std']:.4f}" for k, v in bloques.items()))
    log.info("variables de C (las 5 primeras): " + " · ".join(f"{k} {v['media']:.4f}" for k, v in list(contexto.items())[:5]))

    # ---------- coeficientes: estabilidad entre folds (sin mostrar términos todavía)
    TOP, NC = I["top_terminos"], I["candidatos_por_fold"]
    estabilidad, candidatos = {}, set()
    for blq in ("A", "B"):
        series = [pd.Series(c[blq]) for c in coefs]
        rho = [float(spearmanr(s1.reindex(s1.index.intersection(s2.index)), s2.reindex(s1.index.intersection(s2.index)))[0])
               for s1, s2 in combinations(series, 2)]
        vocab_comun = set.intersection(*(set(s.index) for s in series))
        estabilidad[blq] = {"terminos_por_fold": [int(len(s)) for s in series], "terminos_en_los_5_folds": int(len(vocab_comun)),
                            "spearman_coeficientes": {"media": round(float(np.mean(rho)), 4), "min": round(float(np.min(rho)), 4),
                                                      "max": round(float(np.max(rho)), 4)}}
        for signo, asc in (("positivo", False), ("negativo", True)):
            tops = [set(s.sort_values(ascending=asc).index[:TOP]) for s in series]
            jac = [len(t1 & t2) / len(t1 | t2) for t1, t2 in combinations(tops, 2)]
            cuenta = pd.Series([t for tp in tops for t in tp]).value_counts()
            estabilidad[blq][signo] = {"top": TOP, "en_los_5_folds": int((cuenta == len(series)).sum()),
                                       "en_al_menos_3_folds": int((cuenta >= 3).sum()), "terminos_distintos_en_los_top": int(len(cuenta)),
                                       "jaccard_medio_entre_pares": round(float(np.mean(jac)), 4)}
            for s in series:
                candidatos |= {(blq, signo, t) for t in s.sort_values(ascending=asc).index[:NC]}
        log.info(f"estabilidad {blq}: {estabilidad[blq]}")

    # ---------- filtro de privacidad de los términos candidatos
    t0 = time.time()
    terminos = sorted({t for _, _, t in candidatos})
    letras = {t: all(re.fullmatch(r"[a-z]+", p) for p in t.split(" ")) for t in terminos}
    pacientes, notas = {}, {}
    sujetos = pd.factorize(tr.subject_id)[0]
    Sm = sparse.csr_matrix((np.ones(len(tr)), (sujetos, np.arange(len(tr)))))
    for ng in ((1, 1), (2, 2)):
        voc = [t for t in terminos if len(t.split(" ")) == ng[0]]
        if not voc:
            continue
        base_v = {k_: v_ for k_, v_ in vec["A"].get_params().items() if k_ in CountVectorizer().get_params()
                  and k_ not in ("min_df", "max_features", "dtype", "ngram_range", "vocabulary", "max_df", "binary")}
        X = CountVectorizer(**base_v, ngram_range=ng, vocabulary=voc, binary=True).transform(tr.text.values)
        notas.update(dict(zip(voc, np.asarray(X.sum(axis=0)).ravel().astype(int))))
        pacientes.update(dict(zip(voc, np.asarray(((Sm @ X) > 0).sum(axis=0)).ravel().astype(int))))
    palabras = sorted({p for t in terminos for p in t.split(" ") if letras[t]})
    muestra = tr.text.sample(min(F["muestra_mayusculas"], len(tr)), random_state=S).values
    patron = re.compile(r"\b(" + "|".join(map(re.escape, palabras)) + r")\b", re.IGNORECASE)
    total, titulo = dict.fromkeys(palabras, 0), dict.fromkeys(palabras, 0)
    for texto in muestra:
        for m in patron.finditer(texto):
            p = m.group(0)
            total[p.lower()] += 1
            titulo[p.lower()] += p[0].isupper() and p[1:].islower()
    pct_titulo = {p: (100 * titulo[p] / total[p] if total[p] else 0.0) for p in palabras}
    min_notas = int(np.ceil(F["min_pct_notas"] / 100 * len(tr)))
    motivo = {}
    for t in terminos:
        m = []
        if F["solo_letras"] and not letras[t]:
            m.append("dígitos o símbolos")
        if pacientes.get(t, 0) < F["min_pacientes"]:
            m.append(f"menos de {F['min_pacientes']} pacientes")
        if notas.get(t, 0) < min_notas:
            m.append(f"menos del {F['min_pct_notas']:g} % de las epicrisis")
        if letras[t] and max(pct_titulo[p] for p in t.split(" ")) > F["max_pct_mayuscula_inicial"]:
            m.append("posible nombre propio")
        motivo[t] = m
    pasan = {t for t in terminos if not motivo[t]}
    filtro = {"candidatos": len(terminos), "excluidos": len(terminos) - len(pasan), "publicables": len(pasan),
              "min_notas": min_notas, "por_motivo": {r: sum(r in m for m in motivo.values()) for r in
                                                    ("dígitos o símbolos", f"menos de {F['min_pacientes']} pacientes",
                                                     f"menos del {F['min_pct_notas']:g} % de las epicrisis", "posible nombre propio")},
              "segundos": round(time.time() - t0, 1)}
    log.info(f"filtro de privacidad: {filtro}")

    # ---------- términos publicables: media del coeficiente en los 5 folds
    filas_t = []
    for blq in ("A", "B"):
        tabla = pd.DataFrame([c[blq] for c in coefs]).T          # término × fold (NaN si el término no está en ese fold)
        for signo, asc in (("positivo", False), ("negativo", True)):
            tops = [set(tabla[i].dropna().sort_values(ascending=asc).index[:TOP]) for i in tabla.columns]
            cand = sorted(t for (bb, ss, t) in candidatos if bb == blq and ss == signo and t in pasan and tabla.loc[t].notna().all())
            sel = tabla.loc[cand].mean(axis=1).sort_values(ascending=asc).index[:I["mostrar_terminos"]]
            for r, t in enumerate(sel, 1):
                filas_t.append({"bloque": blq, "signo": signo, "puesto": r, "termino": t,
                               "coef_media": round(float(tabla.loc[t].mean()), 4), "coef_std": round(float(tabla.loc[t].std(ddof=1)), 4),
                               "folds_en_top": int(sum(t in tp for tp in tops)), "pacientes_train": int(pacientes[t]),
                               "pct_notas_train": round(100 * notas[t] / len(tr), 1)})
    T = pd.DataFrame(filas_t)
    assert set(T.termino) <= pasan
    T.to_csv(salida / "importancia_terminos.csv", index=False)
    G.to_csv(salida / "importancia_grupos.csv", index=False, float_format="%.6f")

    # ---------- comparación con las semanas 5 y 6 (leídas, no recalculadas)
    fe = json.loads((REPORTES / "fe_resultados.json").read_text(encoding="utf-8"))
    s6imp = json.loads((REPORTES / "datos_resultados.json").read_text(encoding="utf-8"))["importancia"]["D2"]
    cv5 = {k: v["pr_auc"]["media"] for k, v in fe["cv"].items()}
    comparacion = {
        "ablacion_semana5": {"nota": "D0, todas las filas de validación; caída de la PR-AUC al quitar el bloque y reentrenar",
                             "A": None, "B": round(cv5["A+B+C"] - cv5["A+C"], 4), "C": round(cv5["A+B+C"] - cv5["A+B"], 4),
                             "A+B+C": cv5["A+B+C"], "A+C": cv5["A+C"], "A+B": cv5["A+B"],
                             "C_sin_texto": fe["aporte_grupos_sin_texto"][list(fe["aporte_grupos_sin_texto"])[-1]]["media"]},
        "permutacion_semana6": {"nota": "D2, evaluación común, 3 repeticiones por variable; no calculó el bloque C completo",
                                "A": s6imp["texto: palabras"]["media"], "B": s6imp["texto: bigramas"]["media"],
                                "variables_de_C": {k: v["media"] for k, v in s6imp.items() if not k.startswith("texto")}}}
    control = {"pr_auc_base_esta_corrida": [round(float(x), 6) for x in base], "pr_auc_semana6": [round(float(x), 6) for x in s6],
               "dif_maxima": round(float(np.max(np.abs(np.array(base) - s6))), 6)}
    control["reproduce"] = control["dif_maxima"] <= 1e-6
    log.info(f"control: la PR-AUC de cada fold reproduce la semana 6: {control}")

    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "modelo": "A+B+C", "variante": "D2",
         "configuracion": prm, "parametros": I, "particion": f"StratifiedGroupKFold({k_folds}, shuffle=True, random_state={S}) por "
         "subject_id en train; entrenan las filas de D2; se evalúa en las epicrisis de validación sin etiqueta dudosa; no se usa test",
         "n_train": int(len(tr)), "equivalencia_cache_pipeline": {"todos_iguales": bool(all(e["iguales"] for e in equivalencia)),
                                                                  "detalle": equivalencia},
         "control_reproduccion": control, "importancia_bloques": bloques, "importancia_contexto": contexto,
         "comparacion": comparacion, "estabilidad_coeficientes": estabilidad, "filtro_privacidad": filtro,
         "tiempos_segundos": {"modelos": round(t_modelos, 1), "nombres_y_equivalencia": round(t_nombres, 1),
                              "permutaciones": round(t_perm, 1), "filtro_privacidad": filtro["segundos"],
                              "total": round(time.time() - t_inicio, 1)}}
    guardar_json(R, salida / "importancia_resultados.json")
    escribir_metrics(R, T, salida if prueba else LOGS)
    figuras_imp.dibujar(R, T, salida / "figuras")
    d = "data/interim/prueba_imp" if prueba else "reportes"
    log.info(f"Salidas: {d + '/' if prueba else 'logs/'}metrics_importancia.txt · {d}/importancia_resultados.json · {d}/importancia_grupos.csv · "
             f"{d}/importancia_terminos.csv · {d}/figuras/fig_imp_*.png · duración {R['tiempos_segundos']['total']:.0f} s")


def escribir_metrics(R, T, carpeta):
    L = [f"# metrics_importancia.txt · {R['generado']} · D2 con A+B+C · semilla {R['semilla']}",
         f"partición: {R['particion']}", f"configuración: {hp.texto_prm(R['configuracion'])}",
         f"repeticiones por grupo y fold: {R['parametros']['n_repeticiones']}",
         f"caché igual al pipeline de features.py en los 5 folds: {R['equivalencia_cache_pipeline']['todos_iguales']}",
         f"control (PR-AUC de cada fold = semana 6): {R['control_reproduccion']}", "",
         "IMPORTANCIA POR BLOQUE (caída de la PR-AUC al permutar; media ± desv. entre folds)"]
    L += [f"  {k:<22} {v['media']:.4f} ± {v['std']:.4f} · por fold {v['por_fold']}" for k, v in R["importancia_bloques"].items()]
    L += ["", "IMPORTANCIA DE LAS VARIABLES DE C"]
    L += [f"  {k:<22} {v['media']:.4f} ± {v['std']:.4f}" for k, v in R["importancia_contexto"].items()]
    L += ["", f"comparación con las semanas 5 y 6: {R['comparacion']}", "", "ESTABILIDAD DE LOS COEFICIENTES ENTRE FOLDS"]
    L += [f"  {k}: {v}" for k, v in R["estabilidad_coeficientes"].items()]
    L += ["", f"filtro de privacidad: {R['filtro_privacidad']}", "", "TÉRMINOS PUBLICABLES (ya filtrados)"]
    L += [f"  {r.bloque} {r.signo:<9} {r.puesto:>2} {r.termino:<28} coef {r.coef_media:+.4f} ± {r.coef_std:.4f} · top {R['parametros']['top_terminos']} "
          f"en {r.folds_en_top}/5 folds · {r.pacientes_train} pacientes · {r.pct_notas_train} % de las notas" for r in T.itertuples()]
    L += ["", f"tiempos (s): {R['tiempos_segundos']}"]
    (carpeta / "metrics_importancia.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
