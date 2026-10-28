# -*- coding: utf-8 -*-
"""
SEMANA 7 · Sprint 3: búsqueda de hiperparámetros del modelo vigente (D2 con A+B+C), aleatoria frente a bayesiana.

Modelo y datos: A+B+C de la semana 5 (src/features.py) entrenado con las filas de la variante D2 de la semana 6
  (src/variantes_datos.py), igual que en src/experimentos_datos.py. El diseño no cambia; solo se buscan hiperparámetros.
Protocolo (prerregistrado en config/config.yaml, sección hiperparametros, antes de correr la búsqueda):
  1. Búsqueda solo en train: StratifiedGroupKFold de 3 folds por paciente, semilla 42. Métrica: PR-AUC media de los folds,
     medida en las epicrisis de validación sin etiqueta dudosa (la evaluación común de la semana 6, con la que se eligió D2).
     Se registran también la ROC-AUC y el tiempo de cada ensayo.
  2. Dos métodos con el mismo presupuesto (n_ensayos cada uno), los dos con Optuna y semilla 42: búsqueda aleatoria
     (RandomSampler) y bayesiana (TPESampler multivariado, que empieza con ensayos_iniciales_tpe ensayos al azar). Con la
     misma semilla, esos primeros ensayos coinciden con los primeros de la aleatoria: se evalúan una vez y se reutilizan.
     El TPE es multivariado: en Optuna 5.0.0, multivariate=None equivale a True con un solo objetivo. La corrida del
     28-10-2026 usó ese valor por defecto; después se hizo explícito (hiperparametros.tpe_multivariado) y se verificó que
     la secuencia de los 15 ensayos bayesianos es idéntica a la registrada en reportes/hp_ensayos.csv.
  3. Candidata: la configuración con mayor PR-AUC media entre los ensayos de los dos métodos. Se confirma frente a la
     configuración actual con los 5 folds del proyecto (los mismos de las semanas 5 y 6).
  4. Regla: la candidata reemplaza a la actual solo si mejora la PR-AUC en al menos 4 de los 5 folds y si la mejora media
     supera una desviación estándar de las diferencias por fold. Si no, se mantiene la actual.
  5. Test: la actual y la candidata se reentrenan con todo train (filas de D2) y se miden UNA sola vez en test.
  Controles: ningún paciente en dos partes; la configuración actual reproduce, fold por fold y en test, la PR-AUC de D2 de la
  semana 6 (reportes/datos_folds.csv y reportes/datos_resultados.json).

Cálculo exacto y más rápido: en cada partición las palabras y los bigramas se cuentan una sola vez (CountVectorizer con la misma
tokenización del TfidfVectorizer de features.py) y los conteos se guardan en data/interim/cache_hp/ (fuera de git). Cada ensayo
elige el vocabulario con la misma regla de scikit-learn (min_df y, entre esos términos, los max_features más frecuentes) y aplica
TfidfTransformer, de modo que obtiene las mismas matrices que el pipeline. Con --prueba se comprueba esa equivalencia contra el
pipeline de features.py.

Salidas: logs/experimentos_hp_<fecha>.log · logs/metrics_hp.txt · reportes/hp_ensayos.csv · reportes/hp_resultados.json ·
         reportes/figuras/fig_hp_busqueda.png, fig_hp_parametros.png y fig_hp_folds.png
Uso:  python src/experimentos_hp.py                 # búsqueda, confirmación y test (1.8 horas de CPU el 28-10-2026, incluida la caché)
      python src/experimentos_hp.py --solo-conteos  # solo arma la caché de conteos
      python src/experimentos_hp.py --prueba        # corrida corta para revisar el flujo (escribe en data/interim/prueba_hp)
"""
import copy
import json
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_array

import figuras_hp
import variables_dominio
import variantes_datos as vd
from experimentos_fe import ic_pacientes, metricas
from features import columnas_dominio, pipeline
from utils import INTERIM, LOGS, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "experimentos_hp"
VERSION = 1                      # cambia si cambia el cálculo: invalida la caché de resultados
BLOQUES_TEXTO = {"A": "A_palabras", "B": "B_bigramas"}


# ---------------------------------------------------------------- configuraciones
def actual(P):
    """Configuración vigente: la de feature_engineering (semanas 5 y 6)."""
    return {"C": float(P["logistica"]["C"]), "A_max_features": P["A"]["max_features"], "A_min_df": P["A"]["min_df"],
            "B_max_features": P["B"]["max_features"], "B_min_df": P["B"]["min_df"], "class_weight": "balanced",
            "sublinear_tf": True}


def modelo_oficial(prm, P, S, modelo):
    """Pipeline de features.py para una configuración: mismos bloques, solo cambian los hiperparámetros."""
    p = copy.deepcopy(P)
    p["A"].update(max_features=prm["A_max_features"], min_df=prm["A_min_df"])
    p["B"].update(max_features=prm["B_max_features"], min_df=prm["B_min_df"])
    p["logistica"]["C"] = prm["C"]
    return pipeline(modelo, p, S).set_params(
        clf__class_weight=None if prm["class_weight"] == "ninguno" else prm["class_weight"],
        **{f"atributos__{n}__sublinear_tf": prm["sublinear_tf"] for n in BLOQUES_TEXTO.values()})


def parametros(fila, ACT):
    """Parámetros de una fila de ensayos con tipos de Python (para el JSON y para volver a evaluar)."""
    tipo = {"C": float, "class_weight": str, "sublinear_tf": bool}
    return {k: tipo.get(k, int)(fila[k]) for k in ACT}


def texto_prm(prm):
    return (f"C={prm['C']:.4g} · A {prm['A_max_features']}/{prm['A_min_df']} · B {prm['B_max_features']}/{prm['B_min_df']} · "
            f"class_weight={prm['class_weight']} · sublinear_tf={prm['sublinear_tf']}")


def espacio(trial, E):
    return {"C": trial.suggest_float("C", E["C"][0], E["C"][1], log=True),
            **{k: trial.suggest_categorical(k, E[k]) for k in ("A_max_features", "A_min_df", "B_max_features", "B_min_df",
                                                               "class_weight", "sublinear_tf")}}


# ---------------------------------------------------------------- caché de conteos
def guardar_csr(ruta, M):
    sparse.save_npz(ruta, M.tocsr(), compressed=False)


def contar(vec, textos_tr, textos_va, E, b):
    """Cuenta términos en train del fold y guarda solo las columnas que alguna combinación (min_df, max_features) puede elegir.
    Para cada min_df guarda el orden de frecuencia que usa scikit-learn (CountVectorizer._limit_features)."""
    assert vec.max_df == 1.0, "la selección replica la regla de scikit-learn solo con max_df = 1.0"
    base = {k: v for k, v in vec.get_params().items() if k in CountVectorizer().get_params() and k not in ("min_df", "max_features", "dtype")}
    cv = CountVectorizer(**base, min_df=min(E[f"{b}_min_df"]), dtype=np.int32)
    Xtr = cv.fit_transform(textos_tr)
    Xva = cv.transform(textos_va)
    dfs = np.bincount(Xtr.indices, minlength=Xtr.shape[1])
    tfs = np.asarray(Xtr.sum(axis=0), dtype=np.float64).ravel()
    tope = max(E[f"{b}_max_features"])
    orden = {}
    for m in E[f"{b}_min_df"]:
        idx = np.where(dfs >= m)[0]
        orden[m] = idx[(-tfs[idx]).argsort()][:tope]       # mismo argsort que scikit-learn, cortado al tope de la grilla
    union = np.unique(np.concatenate(list(orden.values())))
    meta = {"n_terminos_min_df": {int(m): int((dfs >= m).sum()) for m in orden},
            "pos": {int(m): np.searchsorted(union, o) for m, o in orden.items()}}
    return Xtr[:, union], Xva[:, union], meta


def seleccion(meta, min_df, max_features):
    """Columnas (en orden alfabético, como el vocabulario de scikit-learn) para un par (min_df, max_features)."""
    pos, n = meta["pos"][min_df], meta["n_terminos_min_df"][min_df]
    return np.sort(pos[:max_features] if n > max_features else pos)


def preparar(nombre, textos, X_dom, a_idx, b_idx, actual_modelo, E, carpeta, log):
    """Arma (o reutiliza) la caché de una partición: conteos de A y B y el bloque C ya transformado (ajustado en train).
    a_idx y b_idx son posiciones en `textos` y `X_dom` (train seguido de test)."""
    d = carpeta / nombre
    if (d / "listo.json").exists():
        return json.loads((d / "listo.json").read_text(encoding="utf-8"))["segundos"]
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    atr = actual_modelo.named_steps["atributos"]
    for b, n in BLOQUES_TEXTO.items():
        vec = dict((t[0], t[1]) for t in atr.transformers)[n]
        Xtr, Xva, meta = contar(vec, textos[a_idx], textos[b_idx], E, b)
        guardar_csr(d / f"{b}_tr.npz", Xtr); guardar_csr(d / f"{b}_va.npz", Xva)
        np.savez(d / f"{b}_meta.npz", **{f"pos_{m}": p for m, p in meta["pos"].items()},
                 n=np.array([[m, k] for m, k in meta["n_terminos_min_df"].items()]))
        del Xtr, Xva
    _, tf_c, cols_c = [t for t in atr.transformers if t[0] == "C_dominio"][0]
    tf_c = clone(tf_c)
    guardar_csr(d / "C_tr.npz", sparse.csr_matrix(tf_c.fit_transform(X_dom.iloc[a_idx][cols_c])))
    guardar_csr(d / "C_va.npz", sparse.csr_matrix(tf_c.transform(X_dom.iloc[b_idx][cols_c])))
    seg = round(time.time() - t0, 1)
    (d / "listo.json").write_text(json.dumps({"segundos": seg, "filas_train": int(len(a_idx)), "filas_validacion": int(len(b_idx))}),
                                  encoding="utf-8")
    log.info(f"caché de conteos {nombre}: {len(a_idx):,} filas de entrenamiento · {len(b_idx):,} de evaluación · {seg:.0f} s")
    return seg


def cargar_meta(ruta):
    z = np.load(ruta)
    return {"pos": {int(k[4:]): z[k] for k in z.files if k.startswith("pos_")}, "n_terminos_min_df": {int(m): int(k) for m, k in z["n"]}}


def a_float(M):
    """Pasa los conteos a float64 sin reordenar los índices de cada fila (astype los ordena). El orden importa: el
    TfidfVectorizer del pipeline deja los términos de train en el orden en que aparecieron, y la norma l2 y liblinear
    suman en ese orden; con otro orden el resultado difiere en el último decimal y la regresión, en unos 1e-5."""
    return sparse.csr_matrix((M.data.astype(np.float64), M.indices, M.indptr), shape=M.shape)


def apilar(bloques):
    """Igual que ColumnTransformer._hstack de scikit-learn con salida dispersa."""
    return sparse.hstack([check_array(X, accept_sparse=True, ensure_all_finite=False) for X in bloques]).tocsr()


def matrices(d, prm):
    """Atributos de train y de evaluación de una partición para una configuración (A, B y C en el orden del pipeline)."""
    tr, va, n = [], [], 0
    for b in BLOQUES_TEXTO:
        cols = seleccion(cargar_meta(d / f"{b}_meta.npz"), prm[f"{b}_min_df"], prm[f"{b}_max_features"])
        Xtr = a_float(sparse.load_npz(d / f"{b}_tr.npz")[:, cols])          # la selección de columnas conserva el orden
        Xva = a_float(sparse.load_npz(d / f"{b}_va.npz")[:, cols])
        t = TfidfTransformer(sublinear_tf=prm["sublinear_tf"]).fit(Xtr)    # norm l2, use_idf y smooth_idf: como TfidfVectorizer
        tr.append(t.transform(Xtr, copy=False)); va.append(t.transform(Xva, copy=False)); n += len(cols)
    Ctr, Cva = sparse.load_npz(d / "C_tr.npz"), sparse.load_npz(d / "C_va.npz")
    tr.append(Ctr); va.append(Cva)
    return apilar(tr), apilar(va), n + Ctr.shape[1]


# ---------------------------------------------------------------- evaluación
class Evaluador:
    """Evalúa configuraciones en una partición; guarda los resultados en una caché por si la corrida se repite."""

    def __init__(self, carpeta, particiones, y, comun, P, S, modelo, log):
        self.carpeta, self.part, self.y, self.comun = carpeta, particiones, y, comun
        self.P, self.S, self.modelo, self.log = P, S, modelo, log
        self.memo_f = carpeta / "resultados.json"
        self.memo = json.loads(self.memo_f.read_text(encoding="utf-8")) if self.memo_f.exists() else {}
        self.ahora, self.previas = set(), set()      # evaluaciones hechas en esta corrida / tomadas de corridas previas

    def clave(self, particion, prm):
        return json.dumps({"v": VERSION, "particion": particion, **prm}, sort_keys=True)

    def __call__(self, particion, prm, guardar_p=False):
        k = self.clave(particion, prm)
        if k in self.memo and not guardar_p:
            if k not in self.ahora:
                self.previas.add(k)
            return self.memo[k]
        self.ahora.add(k)
        clf = self.modelo_clf(prm)
        folds, total = [], 0.0
        for nombre, (a, b) in self.part[particion].items():
            t0 = time.time()
            Xtr, Xva, n_atr = matrices(self.carpeta / nombre, prm)
            m = clone(clf).fit(Xtr, self.y[a])
            p = m.predict_proba(Xva)[:, 1]
            seg = time.time() - t0
            total += seg
            yb, cb = self.y[b], self.comun[b]
            f = {"particion": nombre, "pr_auc_comun": float(average_precision_score(yb[cb], p[cb])),
                 "roc_auc_comun": float(roc_auc_score(yb[cb], p[cb])), "pr_auc_completa": float(average_precision_score(yb, p)),
                 "n_atributos": int(n_atr), "segundos": round(seg, 1)}
            if guardar_p:
                f["p"] = p
            folds.append(f)
            del Xtr, Xva, m
        pr = [f["pr_auc_comun"] for f in folds]
        r = {"pr_auc_media": float(np.mean(pr)), "pr_auc_std": float(np.std(pr, ddof=1)) if len(pr) > 1 else 0.0,
             "roc_auc_media": float(np.mean([f["roc_auc_comun"] for f in folds])),
             "roc_auc_std": float(np.std([f["roc_auc_comun"] for f in folds], ddof=1)) if len(pr) > 1 else 0.0,
             "segundos": round(total, 1), "n_atributos": folds[0]["n_atributos"], "folds": folds}
        if not guardar_p:
            self.memo[k] = r
            self.memo_f.write_text(json.dumps(self.memo), encoding="utf-8")
        return r

    def modelo_clf(self, prm):
        return modelo_oficial(prm, self.P, self.S, self.modelo).named_steps["clf"]


def regla_de_decision(d, R):
    """d: diferencias de PR-AUC por fold (candidata − actual)."""
    mejores, media, sd = int((d > 0).sum()), float(np.mean(d)), float(np.std(d, ddof=1))
    c1, c2 = mejores >= R["folds_mejores_minimo"], media > R["desviaciones_minimas"] * sd
    return {"diferencias_por_fold": [round(float(x), 6) for x in d], "folds_mejores": mejores, "n_folds": int(len(d)),
            "dif_media": round(media, 6), "dif_std": round(sd, 6), "razon_media_sobre_std": round(media / sd, 3) if sd > 0 else None,
            "criterio_folds": c1, "criterio_desviacion": c2, "adopta_candidata": bool(c1 and c2)}


def por_10000(t, prev):
    """Sensibilidad y especificidad de test (umbral 0.5) aplicadas a 10 000 hospitalizaciones con la prevalencia real."""
    ev, sin = 10_000 * prev, 10_000 * (1 - prev)
    det, fal = t["recall"] * ev, (1 - t["especificidad"]) * sin
    return {"eventos": round(ev), "detectados_vp": round(det), "se_escapan_fn": round(ev) - round(det), "falsas_alarmas_fp": round(fal),
            "vpp_de_cada_100_alertas": round(100 * det / (det + fal))}


# ---------------------------------------------------------------- principal
def main():
    t_inicio = time.time()
    prueba, solo_conteos = "--prueba" in sys.argv, "--solo-conteos" in sys.argv
    log = logger(ETAPA + ("_prueba" if prueba else ""))
    cfg = cargar_config()
    S, P, H = cfg["semilla"], cfg["feature_engineering"], copy.deepcopy(cfg["hiperparametros"])
    E, REGLA, MODELO = H["espacio"], H["regla"], H["modelo"]
    if prueba:
        H.update(n_ensayos=3, ensayos_iniciales_tpe=2, n_bootstrap=20)
    salida = INTERIM / "prueba_hp" if prueba else REPORTES
    carpeta = INTERIM / ("cache_hp_prueba" if prueba else "cache_hp")
    carpeta.mkdir(parents=True, exist_ok=True)
    eda = json.loads((REPORTES / "eda_resumen.json").read_text(encoding="utf-8"))
    prev = float(eda["balance"]["prevalencia_real_con_epicrisis"])
    ACT = actual(P)
    log.info(f"modelo {MODELO} · variante {H['variante']} · semilla {S} · configuración actual: {texto_prm(ACT)}")
    log.info(f"espacio de búsqueda: {json.dumps(E, ensure_ascii=False)}")
    log.info(f"presupuesto: {H['n_ensayos']} ensayos por método · bayesiana con {H['ensayos_iniciales_tpe']} ensayos iniciales al azar"
             f" · TPE multivariado: {H['tpe_multivariado']}")
    log.info(f"REGLA PRERREGISTRADA: la candidata reemplaza a la actual solo si mejora la PR-AUC (5 folds) en al menos "
             f"{REGLA['folds_mejores_minimo']} de 5 folds y la mejora media supera {REGLA['desviaciones_minimas']} desviación "
             "estándar de las diferencias por fold")

    # ---------- datos (los mismos pasos de experimentos_datos.py)
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    del df
    if prueba:
        tr, te = tr.sample(4000, random_state=S).reset_index(drop=True), te.sample(1500, random_state=S).reset_index(drop=True)
    assert not set(tr.subject_id) & set(te.subject_id), "fuga: pacientes en train y test"
    usar = vd.filas_de_entrenamiento(tr, H["variante"]).values
    comun_tr = ~(tr.negativa_dudosa | tr.positiva_dudosa).values
    comun_te = ~(te.negativa_dudosa | te.positiva_dudosa).values
    log.info(f"train {len(tr):,} (entrenan {usar.sum():,} filas de {H['variante']}; evaluación común {comun_tr.sum():,}) · "
             f"test {len(te):,} (evaluación común {comun_te.sum():,}) · 0 pacientes compartidos")

    # ---------- particiones: búsqueda (3 folds), confirmación (5 folds del proyecto) y test
    part = {}
    for clave, k in (("busqueda", H["n_folds_busqueda"]), ("confirmacion", H["n_folds_confirmacion"])):
        part[clave] = {}
        for i, (a, b) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=S).split(tr, tr.y, groups=tr.subject_id), 1):
            assert not set(tr.subject_id.iloc[a]) & set(tr.subject_id.iloc[b]), f"fuga en {clave}, fold {i}"
            part[clave][f"{clave}_k{k}_f{i}"] = (a[usar[a]], b)
            log.info(f"{clave} fold {i}/{k}: entrena con {usar[a].sum():,} · valida con {len(b):,} ({comun_tr[b].sum():,} sin etiqueta dudosa)")
    n_tr = len(tr)
    part["test"] = {"test": (np.where(usar)[0], np.arange(n_tr, n_tr + len(te)))}

    # ---------- fase 1: conteos (único paso que lee el texto)
    act_modelo = modelo_oficial(ACT, P, S, MODELO)
    textos = np.concatenate([tr.text.values, te.text.values])
    dominio_df = pd.concat([tr[["text"] + dom_cols], te[["text"] + dom_cols]], ignore_index=True)
    t0 = time.time()
    seg_conteos = {}
    for clave in part:
        for nombre, (a, b) in part[clave].items():
            seg_conteos[nombre] = preparar(nombre, textos, dominio_df, a, b, act_modelo, E, carpeta, log)
    log.info(f"caché de conteos lista en {time.time() - t0:.0f} s (cálculo original: {sum(seg_conteos.values()):.0f} s)")
    y_all = np.concatenate([tr.y.values, te.y.values]).astype(int)
    comun_all = np.concatenate([comun_tr, comun_te])
    grupos_te = te.subject_id.values
    if prueba:
        control_equivalencia(tr, dom_cols, part, carpeta, E, P, S, MODELO, log)
    del textos, dominio_df, tr, te
    if solo_conteos:
        log.info("--solo-conteos: caché lista; no se corre la búsqueda")
        return

    ev = Evaluador(carpeta, part, y_all, comun_all, P, S, MODELO, log)

    # ---------- referencia: la configuración actual en los 3 folds de búsqueda (fuera del presupuesto)
    ref = ev("busqueda", ACT)
    log.info(f"referencia (actual) en {H['n_folds_busqueda']} folds: PR-AUC {ref['pr_auc_media']:.4f} ± {ref['pr_auc_std']:.4f} · "
             f"ROC-AUC {ref['roc_auc_media']:.4f} · {ref['segundos']:.0f} s")

    # ---------- fase 2: búsqueda aleatoria y bayesiana con el mismo presupuesto
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    filas, vistos, seg_metodo = [], {}, {}
    for metodo, sampler in (("aleatoria", optuna.samplers.RandomSampler(seed=S)),
                            ("bayesiana", optuna.samplers.TPESampler(seed=S, n_startup_trials=H["ensayos_iniciales_tpe"],
                                                                     multivariate=H["tpe_multivariado"]))):
        t_m = time.time()

        def objetivo(trial):
            prm = espacio(trial, E)
            k = ev.clave("busqueda", prm)
            reutilizado = vistos.get(k, "")
            r = ev("busqueda", prm)
            vistos.setdefault(k, f"{metodo} #{trial.number + 1}")
            fila = {"metodo": metodo, "ensayo": trial.number + 1, **prm,
                    "pr_auc_media": round(r["pr_auc_media"], 6), "pr_auc_std": round(r["pr_auc_std"], 6),
                    **{f"pr_auc_fold{i}": round(f["pr_auc_comun"], 6) for i, f in enumerate(r["folds"], 1)},
                    "roc_auc_media": round(r["roc_auc_media"], 6), "roc_auc_std": round(r["roc_auc_std"], 6),
                    "n_atributos": r["n_atributos"], "segundos": r["segundos"], "reutilizado_de": reutilizado}
            filas.append(fila)
            pd.DataFrame(filas).to_csv(salida / "hp_ensayos.csv", index=False)
            log.info(f"{metodo} #{trial.number + 1:02d} · {texto_prm(prm)} · PR-AUC {r['pr_auc_media']:.4f} ± {r['pr_auc_std']:.4f} · "
                     f"ROC-AUC {r['roc_auc_media']:.4f} · {r['segundos']:.0f} s" + (f" (igual a {reutilizado}: se reutiliza)" if reutilizado else ""))
            return r["pr_auc_media"]

        salida.mkdir(parents=True, exist_ok=True)
        optuna.create_study(direction="maximize", sampler=sampler, study_name=metodo).optimize(objetivo, n_trials=H["n_ensayos"])
        seg_metodo[metodo] = round(time.time() - t_m, 1)
    ens = pd.DataFrame(filas)
    mejores = {m: g.sort_values(["pr_auc_media", "ensayo"], ascending=[False, True]).iloc[0] for m, g in ens.groupby("metodo", sort=False)}
    cand_m = max(mejores, key=lambda m: (mejores[m].pr_auc_media, m == "aleatoria"))
    CAND = parametros(mejores[cand_m], ACT)
    log.info("mejor por método: " + " · ".join(f"{m} #{r.ensayo} PR-AUC {r.pr_auc_media:.4f}" for m, r in mejores.items())
             + f" · candidata: la de la {cand_m} ({texto_prm(CAND)})")

    # ---------- confirmación con los 5 folds del proyecto
    t0 = time.time()
    conf = {"actual": ev("confirmacion", ACT), "candidata": ev("confirmacion", CAND)}
    seg_conf = round(time.time() - t0, 1)
    pa = np.array([f["pr_auc_comun"] for f in conf["actual"]["folds"]])
    pc = np.array([f["pr_auc_comun"] for f in conf["candidata"]["folds"]])
    regla = regla_de_decision(pc - pa, REGLA)
    log.info(f"confirmación (5 folds): actual {pa.mean():.4f} ± {pa.std(ddof=1):.4f} · candidata {pc.mean():.4f} ± {pc.std(ddof=1):.4f} · "
             f"diferencias {regla['diferencias_por_fold']} · regla: {regla}")

    # control: la configuración actual reproduce D2 de la semana 6, fold por fold
    s6 = pd.read_csv(REPORTES / "datos_folds.csv")
    s6 = s6[s6.variante == H["variante"]].sort_values("fold").pr_auc_comun.values
    control_cv = {"pr_auc_semana6": [round(float(x), 6) for x in s6], "pr_auc_esta_corrida": [round(float(x), 6) for x in pa],
                  "dif_maxima": round(float(np.max(np.abs(s6 - pa))), 6) if len(s6) == len(pa) else None}
    control_cv["reproduce"] = bool(control_cv["dif_maxima"] is not None and control_cv["dif_maxima"] <= 1e-5)
    log.info(f"control de reproducción (validación cruzada): {control_cv}"
             + (" · no aplica en --prueba: muestra de 4 000 filas de train, distinta de la semana 6" if prueba else ""))

    # ---------- test: una sola medición de la actual y de la candidata
    t0 = time.time()
    test = {}
    _, b_te = part["test"]["test"]
    yt, ct, gt = y_all[b_te], comun_all[b_te], grupos_te
    for nombre, prm in (("actual", ACT), ("candidata", CAND)):
        r = ev("test", prm, guardar_p=True)
        p = r["folds"][0]["p"]
        test[nombre] = {
            "comun": {**{k: round(v, 4) for k, v in metricas(yt[ct], p[ct], prev).items()},
                      "pr_auc_ic95_pacientes": ic_pacientes(yt[ct], p[ct], gt[ct], H["n_bootstrap"], S),
                      "brier": round(float(brier_score_loss(yt[ct], p[ct])), 4)},
            "completa": {**{k: round(v, 4) for k, v in metricas(yt, p, prev).items()},
                         "pr_auc_ic95_pacientes": ic_pacientes(yt, p, gt, H["n_bootstrap"], S)},
            "n_atributos": r["n_atributos"], "segundos_entrenamiento_y_prediccion": r["segundos"]}
        test[nombre]["por_10000_comun"] = por_10000(test[nombre]["comun"], prev)
        test[nombre]["por_10000_completa"] = por_10000(test[nombre]["completa"], prev)
        log.info(f"TEST {nombre}: PR-AUC común {test[nombre]['comun']['pr_auc']:.4f} {test[nombre]['comun']['pr_auc_ic95_pacientes']} · "
                 f"completa {test[nombre]['completa']['pr_auc']:.4f} · ROC-AUC común {test[nombre]['comun']['roc_auc']:.4f}")
    seg_test = round(time.time() - t0, 1)
    s6t = json.loads((REPORTES / "datos_resultados.json").read_text(encoding="utf-8"))["test"][H["variante"]]
    control_test = {"pr_auc_comun_semana6": s6t["comun"]["pr_auc"], "pr_auc_comun_esta_corrida": test["actual"]["comun"]["pr_auc"],
                    "pr_auc_completa_semana6": s6t["completa"]["pr_auc"], "pr_auc_completa_esta_corrida": test["actual"]["completa"]["pr_auc"]}
    control_test["reproduce"] = bool(abs(control_test["pr_auc_comun_semana6"] - control_test["pr_auc_comun_esta_corrida"]) <= 1e-4
                                     and abs(control_test["pr_auc_completa_semana6"] - control_test["pr_auc_completa_esta_corrida"]) <= 1e-4)
    log.info(f"control de reproducción (test): {control_test}"
             + (" · no aplica en --prueba: muestra de 1 500 filas de test" if prueba else ""))

    # ---------- resultados
    sin_p = lambda r: {**{k: v for k, v in r.items() if k != "folds"},
                       "folds": [{k: v for k, v in f.items() if k != "p"} for f in r["folds"]]}
    duracion = round(time.time() - t_inicio, 1)
    seg_busqueda_calculo = round(float(ens.loc[ens.reutilizado_de == "", "segundos"].sum()) + ref["segundos"], 1)
    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "modelo": MODELO,
         "variante": H["variante"], "prueba": prueba, "parametros": H, "optuna": optuna.__version__,
         "protocolo": (f"búsqueda: StratifiedGroupKFold({H['n_folds_busqueda']}, shuffle=True, random_state={S}) por subject_id en train; "
                       f"confirmación: StratifiedGroupKFold({H['n_folds_confirmacion']}, shuffle=True, random_state={S}) (los folds de las "
                       f"semanas 5 y 6); entrenan las filas de {H['variante']}; métrica: PR-AUC media en las epicrisis de validación sin "
                       "etiqueta dudosa; test usado una vez"),
         "prevalencia_real_con_epicrisis": prev, "configuracion_actual": ACT, "referencia_actual_busqueda": sin_p(ref),
         "mejor_por_metodo": {m: {"ensayo": int(r.ensayo), "parametros": parametros(r, ACT), "pr_auc_media": float(r.pr_auc_media),
                                  "pr_auc_std": float(r.pr_auc_std), "roc_auc_media": float(r.roc_auc_media), "segundos": float(r.segundos)}
                              for m, r in mejores.items()},
         "candidata": {"metodo": cand_m, "parametros": CAND},
         "ensayos_reutilizados": int((ens.reutilizado_de != "").sum()),
         "confirmacion_5_folds": {k: sin_p(v) for k, v in conf.items()}, "regla": {**REGLA, **regla},
         "decision": "se adopta la candidata" if regla["adopta_candidata"] else "se mantiene la configuración actual",
         "test": test, "control_reproduccion_cv": control_cv, "control_reproduccion_test": control_test,
         "tiempos_segundos": {"conteos_calculo_original": round(sum(seg_conteos.values()), 1), "busqueda_aleatoria": seg_metodo["aleatoria"],
                              "busqueda_bayesiana": seg_metodo["bayesiana"], "referencia_actual_busqueda": ref["segundos"],
                              "busqueda_calculo_original": seg_busqueda_calculo, "confirmacion": seg_conf, "test": seg_test,
                              "esta_corrida": duracion,
                              "computo_total": round(sum(seg_conteos.values()) + seg_busqueda_calculo
                                                     + sum(r["segundos"] for r in conf.values()) + seg_test, 1)},
         "evaluaciones_tomadas_de_corridas_previas": len(ev.previas)}
    guardar_json(R, salida / "hp_resultados.json")
    figuras_hp.dibujar(R, ens, salida / "figuras")
    escribir_metrics(R, ens, salida if prueba else LOGS)
    log.info(f"Salidas: metrics_hp.txt · hp_ensayos.csv · hp_resultados.json · figuras fig_hp_*.png · duración {duracion:.0f} s")


def escribir_metrics(R, ens, carpeta):
    f = lambda x: f"{x:.4f}"
    L = [f"# metrics_hp.txt · {R['generado']} · modelo {R['modelo']} con {R['variante']} · semilla {R['semilla']} · Optuna {R['optuna']}",
         f"protocolo: {R['protocolo']}",
         f"regla prerregistrada: adopta la candidata si mejora la PR-AUC en ≥ {R['regla']['folds_mejores_minimo']} de 5 folds y "
         f"la mejora media > {R['regla']['desviaciones_minimas']} desviación estándar de las diferencias por fold",
         f"configuración actual: {texto_prm(R['configuracion_actual'])}",
         f"referencia (actual, {R['parametros']['n_folds_busqueda']} folds): PR-AUC {f(R['referencia_actual_busqueda']['pr_auc_media'])} ± "
         f"{f(R['referencia_actual_busqueda']['pr_auc_std'])} · {R['referencia_actual_busqueda']['segundos']:.0f} s", "",
         "ENSAYOS (uno por línea): método · n.º · parámetros · PR-AUC media ± desv. (3 folds) · ROC-AUC media · tiempo"]
    for r in ens.itertuples():
        L.append(f"{r.metodo:<9} #{r.ensayo:02d} · {texto_prm(r._asdict())} · PR-AUC {f(r.pr_auc_media)} ± {f(r.pr_auc_std)} · "
                 f"ROC-AUC {f(r.roc_auc_media)} · {r.segundos:.0f} s" + (f" · igual a {r.reutilizado_de} (reutilizado)" if r.reutilizado_de else ""))
    L += ["", "MEJOR POR MÉTODO"]
    for m, r in R["mejor_por_metodo"].items():
        L.append(f"  {m}: ensayo #{r['ensayo']} · {texto_prm(r['parametros'])} · PR-AUC {f(r['pr_auc_media'])} ± {f(r['pr_auc_std'])}")
    L.append(f"  candidata: la de la búsqueda {R['candidata']['metodo']}")
    c, g = R["confirmacion_5_folds"], R["regla"]
    L += ["", "CONFIRMACIÓN (5 folds del proyecto; PR-AUC en las epicrisis sin etiqueta dudosa)"]
    for k in ("actual", "candidata"):
        L.append(f"  {k:<9} " + " · ".join(f(x["pr_auc_comun"]) for x in c[k]["folds"])
                 + f" · media {f(c[k]['pr_auc_media'])} ± {f(c[k]['pr_auc_std'])} · ROC-AUC {f(c[k]['roc_auc_media'])}")
    L += [f"  diferencias (candidata − actual): {g['diferencias_por_fold']} · folds mejores {g['folds_mejores']}/{g['n_folds']} · "
          f"media {g['dif_media']:+.4f} · desv. {g['dif_std']:.4f}",
          f"  regla: criterio de folds {g['criterio_folds']} · criterio de desviación {g['criterio_desviacion']} → {R['decision']}", "",
          "TEST (una sola medición)"]
    for k, t in R["test"].items():
        L.append(f"  {k:<9} común PR-AUC {f(t['comun']['pr_auc'])} {t['comun']['pr_auc_ic95_pacientes']} · ROC-AUC {f(t['comun']['roc_auc'])} · "
                 f"sensibilidad {f(t['comun']['recall'])} · especificidad {f(t['comun']['especificidad'])} | completa PR-AUC "
                 f"{f(t['completa']['pr_auc'])} {t['completa']['pr_auc_ic95_pacientes']}")
        L.append(f"            por 10 000 hospitalizaciones (común): {t['por_10000_comun']}")
    L += ["", f"control de reproducción (validación cruzada): {R['control_reproduccion_cv']}",
          f"control de reproducción (test): {R['control_reproduccion_test']}", f"tiempos (s): {R['tiempos_segundos']}"]
    (carpeta / "metrics_hp.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def control_equivalencia(tr, dom_cols, part, carpeta, E, P, S, modelo, log):
    """--prueba: la ruta con conteos en caché da las mismas probabilidades que el pipeline de features.py."""
    prm = {"C": 3.7, "A_max_features": E["A_max_features"][0], "A_min_df": E["A_min_df"][-1], "B_max_features": E["B_max_features"][0],
           "B_min_df": E["B_min_df"][0], "class_weight": "ninguno", "sublinear_tf": False}
    nombre, (a, b) = next(iter(part["busqueda"].items()))
    X_cols = ["text"] + dom_cols
    m = modelo_oficial(prm, P, S, modelo).fit(tr.loc[a, X_cols], tr.y.iloc[a])
    p_ref = m.predict_proba(tr.loc[b, X_cols])[:, 1]
    Xtr, Xva, _ = matrices(carpeta / nombre, prm)
    p = clone(m.named_steps["clf"]).fit(Xtr, tr.y.values[a]).predict_proba(Xva)[:, 1]
    dif = float(np.abs(p - p_ref).max())
    log.info(f"control de equivalencia con el pipeline de features.py ({texto_prm(prm)}): diferencia máxima de probabilidad {dif:.2e}")
    assert dif == 0.0, "la ruta con caché no reproduce el pipeline"


if __name__ == "__main__":
    main()
