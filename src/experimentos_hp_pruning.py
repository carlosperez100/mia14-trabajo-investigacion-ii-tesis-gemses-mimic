# -*- coding: utf-8 -*-
"""
SEMANA 8 · Sprint 3: la búsqueda de hiperparámetros de la semana 7 con pruning, más ensayos, artefactos y curva de aprendizaje.

Responde a la lámina 21 del material de la semana 7 (Búsqueda de hiperparámetros): experimentos comparables aleatoria/bayesiana con
pruning o early stopping, logs y artefactos guardados, tabla top-k, gráfico de evolución y resumen del espacio, el presupuesto y la
decisión. El diseño está en config/config.yaml (sección hiperparametros_pruning), escrito antes de correr.

Modelo, datos y partición: los de src/experimentos_hp.py, sin cambios (D2 con A+B+C, el mismo espacio, StratifiedGroupKFold de 3
folds por paciente en train, semilla 42 y PR-AUC en las epicrisis de validación sin etiqueta dudosa). Reutiliza su caché de conteos.
Lo nuevo:
  1. Pruning: cada ensayo se evalúa fold por fold y, después de cada fold, informa a Optuna la PR-AUC media de los folds ya hechos.
     Un MedianPruner corta el ensayo si su mejor valor intermedio queda por debajo de la mediana de los ensayos completos de la misma
     búsqueda en ese fold. La regresión logística no tiene early stopping nativo (la lámina 13 lo asocia a XGB, LGBM y redes), de
     modo que el corte se hace por fold. El ensayo también se corta si supera max_segundos_por_ensayo.
  2. Presupuesto: 30 ensayos por método (15 en la semana 7) y 10 al azar antes del TPE (5 en la semana 7). Con la misma semilla, los
     15 primeros ensayos de la aleatoria repiten las configuraciones de la semana 7, lo que permite medir el efecto del pruning sobre
     los mismos ensayos, y los 10 primeros de la bayesiana repiten los 10 primeros de la aleatoria: se evalúan una vez y se reutilizan.
  3. Se registra, en cada fold, el número de iteraciones de la regresión (liblinear) y si convergió.
  4. Decisión: la regla de la semana 7. La candidata (mayor PR-AUC media entre los ensayos completos) se confirma con los 5 folds del
     proyecto, y el test se mide una sola vez y solo si la regla la adopta.
  5. Respaldo: las n_respaldo configuraciones de mayor PR-AUC media en 3 folds, distintas de la ganadora, entre los ensayos completos
     de las semanas 7 y 8, confirmadas con los 5 folds.
  6. Importancia de los hiperparámetros: evaluador por defecto de Optuna 5.0.0 (PED-ANOVA) sobre los ensayos completos de las cuatro
     búsquedas, sin repetir configuraciones.
  7. Curva de aprendizaje de la ganadora: PR-AUC en las filas de entrenamiento y en validación con fracciones de las filas de
     entrenamiento de cada uno de los 5 folds del proyecto. Las fracciones menores que 1 se ajustan con el pipeline de features.py
     sobre el texto; la fracción 1 usa la caché de conteos, que da las mismas matrices que el pipeline.
  8. Artefactos: la ganadora y los respaldos se reentrenan con el pipeline de features.py en todo train (filas de D2) y se guardan
     en models/ (fuera de git), con su hash SHA-256. Se comprueba que cada modelo guardado da las mismas probabilidades que el de la
     caché en 2 000 filas de train.
Controles: la configuración actual reproduce, fold por fold, su referencia de la semana 7 en los 3 folds, y los folds evaluados de los
15 primeros ensayos de la aleatoria reproducen los de la semana 7 (reportes/hp_ensayos.csv).

Salidas: logs/experimentos_hp_pruning_<fecha>.log · logs/metrics_hp_pruning.txt · logs/hpo_runs.csv (todos los ensayos de las semanas
         7 y 8) · reportes/hp_pruning_ensayos.csv · reportes/hp_pruning_topk.csv · reportes/hp_pruning_curva.csv ·
         reportes/hp_pruning_resultados.json · reportes/figuras/fig_hp_pruning_*.png · models/hp_pruning_*.joblib (fuera de git)
Uso:  python src/experimentos_hp_pruning.py           (requiere la caché de python src/experimentos_hp.py --solo-conteos)
      python src/experimentos_hp_pruning.py --prueba  (corrida corta para revisar el flujo; escribe en data/interim/prueba_hp_pruning)
Si la corrida se interrumpe, al repetirla se toman de las cachés los folds de la búsqueda (con su tiempo original) y las
confirmaciones con 5 folds; el test, la curva y los modelos se calculan de nuevo.
"""
import copy
import json
import sys
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import experimentos_hp as hp
import figuras_hp_pruning
import variables_dominio
import variantes_datos as vd
from experimentos_fe import ic_pacientes, metricas
from features import columnas_dominio
from utils import INTERIM, LOGS, PROCESSED, RAIZ, REPORTES, cargar_config, guardar_json, logger, sha256

ETAPA = "experimentos_hp_pruning"
VERSION = 1                      # cambia si cambia el cálculo: invalida la caché de folds de la semana 8
HIPER = ("C", "A_max_features", "A_min_df", "B_max_features", "B_min_df", "class_weight", "sublinear_tf")


# ---------------------------------------------------------------- utilidades
def prm_de(fila):
    """Parámetros de una fila (CSV o diccionario) con tipos de Python; sublinear_tf puede venir como texto."""
    s = fila["sublinear_tf"]
    return {"C": float(fila["C"]), "A_max_features": int(fila["A_max_features"]), "A_min_df": int(fila["A_min_df"]),
            "B_max_features": int(fila["B_max_features"]), "B_min_df": int(fila["B_min_df"]),
            "class_weight": str(fila["class_weight"]), "sublinear_tf": bool(s) if isinstance(s, (bool, np.bool_)) else str(s) == "True"}


def clave_prm(prm):
    return json.dumps({k: (bool(v) if k == "sublinear_tf" else v) for k, v in prm.items()}, sort_keys=True)


def leer_ensayos(ruta):
    """CSV de ensayos. Con float_precision="round_trip", C se lee con todos sus decimales: el lector por defecto de pandas puede
    cambiar el último dígito, y entonces una configuración de la semana 7 no se reconoce como igual a la misma de la semana 8."""
    df = pd.read_csv(ruta, keep_default_na=False, na_values=[""], float_precision="round_trip")
    df["reutilizado_de"] = df.reutilizado_de.fillna("")
    return df


def resumen_folds(folds):
    pr = [f["pr_auc_comun"] for f in folds]
    return {"pr_auc_media": float(np.mean(pr)), "pr_auc_std": float(np.std(pr, ddof=1)) if len(pr) > 1 else None,
            "roc_auc_media": float(np.mean([f["roc_auc_comun"] for f in folds])),
            "segundos": round(sum(f["segundos"] for f in folds), 1), "folds": folds}


class EvaluadorPorFold:
    """Evalúa una configuración en un fold (para informar al pruner después de cada fold). Guarda cada fold en una caché propia de la
    semana 8, con su tiempo de cálculo, de modo que una corrida interrumpida se retoma sin perder los tiempos."""

    def __init__(self, carpeta, y, comun, P, S, modelo):
        self.carpeta, self.y, self.comun, self.P, self.S, self.modelo = carpeta, y, comun, P, S, modelo
        self.memo_f = carpeta / "resultados_semana8.json"
        self.memo = json.loads(self.memo_f.read_text(encoding="utf-8")) if self.memo_f.exists() else {}
        self.ahora, self.previas = set(), set()      # folds calculados en esta corrida / tomados de una corrida anterior

    def clave(self, nombre, prm):
        return json.dumps({"v": VERSION, "fold": nombre, **prm}, sort_keys=True)

    def __call__(self, nombre, a, b, prm):
        k = self.clave(nombre, prm)
        if k in self.memo:
            if k not in self.ahora:
                self.previas.add(k)
            return {**self.memo[k], "calculado_ahora": False}
        t0 = time.time()
        Xtr, Xva, n_atr = hp.matrices(self.carpeta / nombre, prm)
        clf = clone(hp.modelo_oficial(prm, self.P, self.S, self.modelo).named_steps["clf"])
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Xtr, self.y[a])
        p = clf.predict_proba(Xva)[:, 1]
        seg = time.time() - t0
        yb, cb = self.y[b], self.comun[b]
        f = {"particion": nombre, "pr_auc_comun": float(average_precision_score(yb[cb], p[cb])),
             "roc_auc_comun": float(roc_auc_score(yb[cb], p[cb])), "n_atributos": int(n_atr), "segundos": round(seg, 1),
             "n_iter": int(np.max(clf.n_iter_)),
             "convergio": not any(issubclass(w.category, ConvergenceWarning) for w in avisos)}
        self.memo[k] = f
        self.ahora.add(k)
        self.memo_f.write_text(json.dumps(self.memo), encoding="utf-8")
        return {**f, "calculado_ahora": True}


# ---------------------------------------------------------------- piezas de la semana 8
def importancia(pool, E, minimo):
    """Importancia de los hiperparámetros con el evaluador por defecto de Optuna 5.0.0 (PED-ANOVA) sobre ensayos completos."""
    dist = {"C": optuna.distributions.FloatDistribution(E["C"][0], E["C"][1], log=True),
            **{k: optuna.distributions.CategoricalDistribution(E[k]) for k in HIPER if k != "C"}}
    est = optuna.create_study(direction="maximize")
    for r in pool:
        est.add_trial(optuna.trial.create_trial(params=r["prm"], distributions=dist, value=r["pr_auc_media"]))
    n = len(pool)
    q = max(0.1, minimo / n)
    imp = optuna.importance.get_param_importances(est, evaluator=optuna.importance.PedAnovaImportanceEvaluator(target_quantile=q))
    return {"evaluador": "PedAnovaImportanceEvaluator (por defecto en Optuna 5.0.0)", "target_quantile": round(q, 4),
            "ensayos_sobre_el_cuantil": int(np.ceil(q * n)), "n_configuraciones": n,
            "importancias": {k: round(float(v), 4) for k, v in imp.items()}}


def efecto_mismos_ensayos(s7, ens, n_folds):
    """Los primeros ensayos de la aleatoria de la semana 8 frente a los de la semana 7. Con la misma semilla, el RandomSampler
    propone las mismas configuraciones, de modo que la diferencia en folds evaluados y en tiempo es la que produce el pruning."""
    s7a = s7[s7.metodo == "aleatoria"].sort_values("ensayo")
    s8a = ens[(ens.metodo == "aleatoria") & (ens.ensayo <= len(s7a))].sort_values("ensayo")
    mismas = len(s8a) == len(s7a) and all(clave_prm(prm_de(x)) == clave_prm(prm_de(y)) for x, y in
                                          zip(s7a.to_dict("records"), s8a.to_dict("records")))
    difs = [abs(float(x[f"pr_auc_fold{i}"]) - float(y[f"pr_auc_fold{i}"])) for x, y in zip(s7a.to_dict("records"), s8a.to_dict("records"))
            for i in range(1, int(y["folds_evaluados"]) + 1)] if mismas else []
    top5_7 = set(s7a.nlargest(5, "pr_auc_media").ensayo)
    efecto = {"mismas_configuraciones_que_la_semana_7": bool(mismas), "n_ensayos": int(len(s8a)),
              "cortados": int((s8a.estado != "completo").sum()),
              "folds_evaluados": int(s8a.folds_evaluados.sum()), "folds_sin_pruning": int(n_folds * len(s8a)),
              "segundos_semana7": round(float(s7a.segundos.sum()), 1), "segundos_semana8": round(float(s8a.segundos.sum()), 1),
              "dif_maxima_pr_auc_por_fold": round(max(difs), 9) if difs else None,
              "reproduce_folds": bool(difs) and max(difs) <= 1e-6,
              "top5_semana7_cortados": sorted(int(e) for e in s8a[(s8a.ensayo.isin(top5_7)) & (s8a.estado != "completo")].ensayo),
              "mejor_semana7": int(s7a.sort_values(["pr_auc_media", "ensayo"], ascending=[False, True]).iloc[0].ensayo)}
    efecto["mejor_semana7_estado_en_semana8"] = str(s8a[s8a.ensayo == efecto["mejor_semana7"]].estado.iloc[0]) if mismas else None
    return efecto


def armar_pool(s7, comp):
    """Configuraciones completas y distintas de las cuatro búsquedas (semanas 7 y 8): cada una con su primera aparición y las otras."""
    pool, vistas = [], {}
    for semana, df_ in ((7, s7[s7.reutilizado_de == ""]), (8, comp[comp.reutilizado_de == ""])):
        for r in df_.to_dict("records"):
            prm = prm_de(r)
            k = clave_prm(prm)
            origen = f"semana {semana} · {r['metodo']} #{int(r['ensayo'])}"
            if k in vistas:
                vistas[k]["tambien"].append(origen)
                continue
            vistas[k] = {"origen": origen, "tambien": [], "prm": prm, "pr_auc_media": float(r["pr_auc_media"]),
                         "pr_auc_std": float(r["pr_auc_std"]), "segundos": float(r["segundos"]), "n_atributos": int(r["n_atributos"])}
            pool.append(vistas[k])
    return pool


def ordenar_respaldo(pool, GAN, ACT, ref):
    """Posibles respaldos de mayor a menor PR-AUC media en 3 folds, sin la ganadora; si se adoptó la candidata, entra la anterior."""
    excluir = clave_prm(GAN)
    orden = sorted([r for r in pool if clave_prm(r["prm"]) != excluir], key=lambda r: -r["pr_auc_media"])
    if excluir != clave_prm(ACT):
        orden = sorted(orden + [{"origen": "configuración anterior (referencia)", "tambien": [], "prm": ACT,
                                 "pr_auc_media": ref["pr_auc_media"], "pr_auc_std": ref["pr_auc_std"],
                                 "segundos": ref["segundos"], "n_atributos": ref["folds"][0]["n_atributos"]}],
                       key=lambda r: -r["pr_auc_media"])
    return orden


def tabla_topk(pool, ACT, ref, GAN, CAND, respaldos, adopta):
    """Tabla top-k: configuraciones completas de las semanas 7 y 8 y la referencia, de mayor a menor PR-AUC media en 3 folds."""
    filas_k = [{"origen": r["origen"], "tambien": r["tambien"], "prm": r["prm"], "pr_auc_media": r["pr_auc_media"],
                "pr_auc_std": r["pr_auc_std"], "segundos": r["segundos"], "n_atributos": r["n_atributos"]} for r in pool]
    filas_k.append({"origen": "configuración actual (referencia, fuera del presupuesto)", "tambien": [], "prm": ACT,
                    "pr_auc_media": ref["pr_auc_media"], "pr_auc_std": ref["pr_auc_std"], "segundos": ref["segundos"],
                    "n_atributos": ref["folds"][0]["n_atributos"]})
    rol_de = {clave_prm(GAN): "ganadora", **{clave_prm(r["parametros"]): r["rol"].replace("_", " ") for r in respaldos}}
    if not adopta:
        rol_de.setdefault(clave_prm(CAND), "candidata de la semana 8, no adoptada")
    topk = pd.DataFrame([{"origen": r["origen"], "parametros": hp.texto_prm(r["prm"]), "pr_auc_media": round(r["pr_auc_media"], 4),
                          "pr_auc_std": round(r["pr_auc_std"], 4), "segundos": r["segundos"], "n_atributos": r["n_atributos"],
                          "rol": rol_de.get(clave_prm(r["prm"]), ""), "tambien_en": "; ".join(r["tambien"])}
                         for r in filas_k]).sort_values("pr_auc_media", ascending=False).reset_index(drop=True)
    topk.insert(0, "puesto", range(1, len(topk) + 1))
    return topk


def curva_aprendizaje(tr, y_tr, comun_tr, part_conf, carpeta, prm, P, S, modelo, fracciones, X_cols, log):
    filas = []
    for k, (nombre, (a, b)) in enumerate(part_conf.items(), 1):
        cb = comun_tr[b]
        for fr in fracciones:
            t0 = time.time()
            if fr >= 1:
                Xtr, Xva, _ = hp.matrices(carpeta / nombre, prm)
                clf = clone(hp.modelo_oficial(prm, P, S, modelo).named_steps["clf"]).fit(Xtr, y_tr[a])
                filas_tr, ruta = a, "caché de conteos"
                p_tr, p_va = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]
                del Xtr, Xva
            else:
                filas_tr = np.sort(np.random.default_rng(S + k).choice(a, size=int(round(fr * len(a))), replace=False))
                pipe = hp.modelo_oficial(prm, P, S, modelo)
                atr, clf = pipe.named_steps["atributos"], pipe.named_steps["clf"]
                Xtr = atr.fit_transform(tr.loc[filas_tr, X_cols])
                clf.fit(Xtr, y_tr[filas_tr])
                p_tr = clf.predict_proba(Xtr)[:, 1]
                p_va = clf.predict_proba(atr.transform(tr.loc[b, X_cols]))[:, 1]
                ruta = "pipeline de features.py"
                del Xtr
            f = {"fold": k, "fraccion": fr, "filas_entrenamiento": int(len(filas_tr)),
                 "pr_auc_entrenamiento": round(float(average_precision_score(y_tr[filas_tr], p_tr)), 6),
                 "pr_auc_validacion": round(float(average_precision_score(y_tr[b][cb], p_va[cb])), 6),
                 "roc_auc_validacion": round(float(roc_auc_score(y_tr[b][cb], p_va[cb])), 6),
                 "segundos": round(time.time() - t0, 1), "ruta": ruta}
            filas.append(f)
            log.info(f"curva de aprendizaje · fold {k} · fracción {fr:g} ({f['filas_entrenamiento']:,} filas) · PR-AUC entrenamiento "
                     f"{f['pr_auc_entrenamiento']:.4f} · validación {f['pr_auc_validacion']:.4f} · {f['segundos']:.0f} s ({ruta})")
    return pd.DataFrame(filas)


def guardar_modelos(roles, tr, y_tr, usar, X_cols, carpeta, P, S, modelo, destino, log):
    """Reentrena cada configuración con el pipeline de features.py en todo train (filas de D2), la guarda con joblib y comprueba que
    da las mismas probabilidades que la ruta de la caché (partición «test», que entrena con las mismas filas) en 2 000 filas de train."""
    destino.mkdir(parents=True, exist_ok=True)
    idx = np.where(usar)[0]
    muestra = np.sort(np.random.default_rng(S).choice(len(idx), size=min(2000, len(idx)), replace=False))
    salida = {}
    for rol, prm in roles.items():
        t0 = time.time()
        pipe = hp.modelo_oficial(prm, P, S, modelo).fit(tr.loc[idx, X_cols], y_tr[idx])
        seg = round(time.time() - t0, 1)
        ruta = destino / f"hp_pruning_{rol}.joblib"
        joblib.dump(pipe, ruta, compress=3)
        Xtr, _, _ = hp.matrices(carpeta / "test", prm)
        clf = clone(hp.modelo_oficial(prm, P, S, modelo).named_steps["clf"]).fit(Xtr, y_tr[idx])
        # joblib.load solo abre el archivo que esta misma función acaba de escribir, para comprobar que el artefacto guardado
        # es el modelo evaluado; nunca se cargan modelos de otra procedencia
        dif = float(np.abs(clf.predict_proba(Xtr[muestra])[:, 1]
                           - joblib.load(ruta).predict_proba(tr.loc[idx[muestra], X_cols])[:, 1]).max())
        del Xtr, clf, pipe
        salida[rol] = {"archivo": ruta.relative_to(RAIZ).as_posix() if ruta.is_relative_to(RAIZ) else str(ruta),
                       "sha256": sha256(ruta), "megabytes": round(ruta.stat().st_size / 2**20, 2), "parametros": prm,
                       "filas_entrenamiento": int(len(idx)), "segundos_entrenamiento": seg,
                       "control_mismas_probabilidades_que_la_cache": {"filas": int(len(muestra)), "dif_maxima": dif, "iguales": dif <= 1e-9}}
        log.info(f"modelo {rol}: {salida[rol]['archivo']} · {salida[rol]['megabytes']} MB · sha256 {salida[rol]['sha256'][:16]}… · "
                 f"diferencia máxima con la caché {dif:.2e} en {len(muestra):,} filas de train")
    return salida


def hpo_runs(s7, ens, S, H, Hp):
    """Registro único de ensayos con las columnas de la lámina 16 (trial_id, params, seed, split_cfg, metric_mean, metric_std, tiempo)."""
    split = (f"StratifiedGroupKFold(n_splits={H['n_folds_busqueda']}, shuffle=True, random_state={S}) por subject_id en train; "
             f"entrenan las filas de {H['variante']}; métrica en la validación sin etiqueta dudosa")
    filas = []
    for semana, df, prun in ((7, s7, "no"), (8, ens, "sí (MedianPruner por fold)")):
        for r in df.to_dict("records"):
            prm = prm_de(r)
            filas.append({"trial_id": f"s{semana}-{r['metodo']}-{int(r['ensayo']):02d}", "semana": semana, "metodo": r["metodo"],
                          "pruning": prun, "ensayo": int(r["ensayo"]), "params": hp.texto_prm(prm), **prm, "seed": S,
                          "split_cfg": split, "metric": "PR-AUC media", "metric_mean": r["pr_auc_media"],
                          "metric_std": r["pr_auc_std"], "folds_evaluados": int(r.get("folds_evaluados", H["n_folds_busqueda"])),
                          "estado": r.get("estado", "completo"), "n_iter_max": r.get("n_iter_max", ""),
                          "tiempo_s": r["segundos"], "reutilizado_de": r["reutilizado_de"]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- principal
def main():
    t_inicio = time.time()
    prueba = "--prueba" in sys.argv
    log = logger(ETAPA + ("_prueba" if prueba else ""))
    cfg = cargar_config()
    S, P, H, Hp = cfg["semilla"], cfg["feature_engineering"], cfg["hiperparametros"], copy.deepcopy(cfg["hiperparametros_pruning"])
    E, REGLA, MODELO = H["espacio"], H["regla"], H["modelo"]
    if prueba:
        Hp.update(n_ensayos=4, ensayos_iniciales_tpe=2, curva_fracciones=[0.5, 1.0], n_respaldo=1)
        Hp["pruner"]["n_startup_trials"] = 1
    salida = INTERIM / "prueba_hp_pruning" if prueba else REPORTES
    carpeta = INTERIM / ("cache_hp_prueba" if prueba else "cache_hp")
    destino_modelos = salida / "models" if prueba else RAIZ / "models"
    (salida / "figuras").mkdir(parents=True, exist_ok=True)
    eda = json.loads((REPORTES / "eda_resumen.json").read_text(encoding="utf-8"))
    prev = float(eda["balance"]["prevalencia_real_con_epicrisis"])
    s7 = leer_ensayos(REPORTES / "hp_ensayos.csv")
    R7 = json.loads((REPORTES / "hp_resultados.json").read_text(encoding="utf-8"))
    ACT = hp.actual(P)
    pr = Hp["pruner"]
    log.info(f"modelo {MODELO} · variante {H['variante']} · semilla {S} · configuración actual: {hp.texto_prm(ACT)}")
    log.info(f"espacio de búsqueda (el de la semana 7): {json.dumps(E, ensure_ascii=False)}")
    log.info(f"presupuesto: {Hp['n_ensayos']} ensayos por método · bayesiana con {Hp['ensayos_iniciales_tpe']} ensayos iniciales al azar · "
             f"TPE multivariado: {H['tpe_multivariado']} · máximo {Hp['max_segundos_por_ensayo']} s por ensayo y "
             f"{Hp['max_minutos_por_busqueda']} min por búsqueda")
    log.info(f"PRUNING PRERREGISTRADO: MedianPruner(n_startup_trials={pr['n_startup_trials']}, n_warmup_steps={pr['n_warmup_steps']}, "
             f"interval_steps={pr['interval_steps']}); valor intermedio: PR-AUC media de los folds ya evaluados, después de cada fold")
    log.info(f"REGLA PRERREGISTRADA (la de la semana 7): la candidata reemplaza a la actual solo si mejora la PR-AUC (5 folds) en al "
             f"menos {REGLA['folds_mejores_minimo']} de 5 folds y la mejora media supera {REGLA['desviaciones_minimas']} desviación "
             f"estándar de las diferencias por fold; el test se mide solo si la regla adopta la candidata. Respaldo: las "
             f"{Hp['n_respaldo']} de mayor PR-AUC media en 3 folds entre los ensayos completos de las semanas 7 y 8")

    # ---------- datos y particiones (los mismos pasos de experimentos_hp.py)
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    dom_cols = [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    X_cols = ["text"] + dom_cols
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
             f"test {len(te):,} · 0 pacientes compartidos")
    part = {}
    for clave, k in (("busqueda", H["n_folds_busqueda"]), ("confirmacion", H["n_folds_confirmacion"])):
        part[clave] = {}
        for i, (a, b) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=S).split(tr, tr.y, groups=tr.subject_id), 1):
            assert not set(tr.subject_id.iloc[a]) & set(tr.subject_id.iloc[b]), f"fuga en {clave}, fold {i}"
            part[clave][f"{clave}_k{k}_f{i}"] = (a[usar[a]], b)
    n_tr = len(tr)
    part["test"] = {"test": (np.where(usar)[0], np.arange(n_tr, n_tr + len(te)))}
    act_modelo = hp.modelo_oficial(ACT, P, S, MODELO)
    textos = np.concatenate([tr.text.values, te.text.values])
    dominio_df = pd.concat([tr[X_cols], te[X_cols]], ignore_index=True)
    for clave in part:
        for nombre, (a, b) in part[clave].items():
            hp.preparar(nombre, textos, dominio_df, a, b, act_modelo, E, carpeta, log)
            listo = json.loads((carpeta / nombre / "listo.json").read_text(encoding="utf-8"))
            assert listo["filas_train"] == len(a) and listo["filas_validacion"] == len(b), f"la caché de {nombre} no corresponde a estas filas"
    del textos, dominio_df
    te = te[["subject_id", "y"]]
    y_all = np.concatenate([tr.y.values, te.y.values]).astype(int)
    comun_all = np.concatenate([comun_tr, comun_te])
    y_tr = y_all[:n_tr]
    log.info("caché de conteos de la semana 7 verificada para las 9 particiones (filas de entrenamiento y de evaluación)")

    evf = EvaluadorPorFold(carpeta, y_all, comun_all, P, S, MODELO)
    ev7 = hp.Evaluador(carpeta, part, y_all, comun_all, P, S, MODELO, log)     # caché de resultados de la semana 7
    n_folds = H["n_folds_busqueda"]

    # ---------- referencia: la configuración actual en los 3 folds, fold por fold (fuera del presupuesto) y control frente a la semana 7
    ref = resumen_folds([evf(n, a, b, ACT) for n, (a, b) in part["busqueda"].items()])
    ref7 = [f["pr_auc_comun"] for f in R7["referencia_actual_busqueda"]["folds"]]
    control_ref = {"pr_auc_semana7": [round(x, 6) for x in ref7], "pr_auc_esta_corrida": [round(f["pr_auc_comun"], 6) for f in ref["folds"]]}
    control_ref["dif_maxima"] = round(float(np.max(np.abs(np.array(ref7) - [f["pr_auc_comun"] for f in ref["folds"]]))), 9)
    control_ref["reproduce"] = bool(control_ref["dif_maxima"] <= 1e-6)
    log.info(f"referencia (actual) en {n_folds} folds: PR-AUC {ref['pr_auc_media']:.4f} ± {ref['pr_auc_std']:.4f} · {ref['segundos']:.0f} s · "
             f"control frente a la semana 7: {control_ref}" + (" · no aplica en --prueba" if prueba else ""))

    # ---------- búsquedas aleatoria y bayesiana con pruning y el mismo presupuesto
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    filas, vistos, seg_metodo, estados = [], {}, {}, {}
    for metodo, sampler in (("aleatoria", optuna.samplers.RandomSampler(seed=S)),
                            ("bayesiana", optuna.samplers.TPESampler(seed=S, n_startup_trials=Hp["ensayos_iniciales_tpe"],
                                                                     multivariate=H["tpe_multivariado"]))):
        t_m = time.time()

        def objetivo(trial, metodo=metodo):
            prm = hp.espacio(trial, E)
            kc = clave_prm(prm)
            reutilizado = vistos.get(kc, "")
            folds, estado, media = [], "completo", float("nan")
            for i, (nombre, (a, b)) in enumerate(part["busqueda"].items(), 1):
                folds.append(evf(nombre, a, b, prm))
                media = float(np.mean([x["pr_auc_comun"] for x in folds]))
                trial.report(media, step=i)
                if i < n_folds:
                    if trial.should_prune():
                        estado = "cortado por pruning"
                    elif sum(x["segundos"] for x in folds) > Hp["max_segundos_por_ensayo"]:
                        estado = "cortado por tiempo"
                    if estado != "completo":
                        break
            vistos.setdefault(kc, f"{metodo} #{trial.number + 1}")
            prs = [x["pr_auc_comun"] for x in folds]
            fila = {"metodo": metodo, "ensayo": trial.number + 1, **prm, "estado": estado, "folds_evaluados": len(folds),
                    "pr_auc_media": round(media, 6), "pr_auc_std": round(float(np.std(prs, ddof=1)), 6) if len(prs) > 1 else None,
                    **{f"pr_auc_fold{i}": round(x["pr_auc_comun"], 6) for i, x in enumerate(folds, 1)},
                    "roc_auc_media": round(float(np.mean([x["roc_auc_comun"] for x in folds])), 6),
                    "n_atributos": folds[0]["n_atributos"], "n_iter_max": max(x["n_iter"] for x in folds),
                    "convergio": all(x["convergio"] for x in folds), "segundos": round(sum(x["segundos"] for x in folds), 1),
                    "segundos_calculados_ahora": round(sum(x["segundos"] for x in folds if x["calculado_ahora"]), 1),
                    "reutilizado_de": reutilizado}
            filas.append(fila)
            pd.DataFrame(filas).to_csv(salida / "hp_pruning_ensayos.csv", index=False)
            log.info(f"{metodo} #{trial.number + 1:02d} · {hp.texto_prm(prm)} · {estado} tras {len(folds)} fold(s) · PR-AUC {media:.4f} · "
                     f"{fila['segundos']:.0f} s · iteraciones ≤ {fila['n_iter_max']}"
                     + (f" (igual a {reutilizado}: se reutiliza)" if reutilizado else ""))
            if estado != "completo":
                raise optuna.TrialPruned()
            return media

        estudio = optuna.create_study(direction="maximize", sampler=sampler, study_name=metodo,
                                      pruner=optuna.pruners.MedianPruner(n_startup_trials=pr["n_startup_trials"],
                                                                         n_warmup_steps=pr["n_warmup_steps"],
                                                                         interval_steps=pr["interval_steps"]))
        estudio.optimize(objetivo, n_trials=Hp["n_ensayos"], timeout=Hp["max_minutos_por_busqueda"] * 60)
        seg_metodo[metodo] = round(time.time() - t_m, 1)
        estados[metodo] = {s.name: sum(t.state == s for t in estudio.trials) for s in (optuna.trial.TrialState.COMPLETE,
                                                                                        optuna.trial.TrialState.PRUNED)}
        log.info(f"búsqueda {metodo}: {len(estudio.trials)} ensayos · {estados[metodo]} · {seg_metodo[metodo]:.0f} s de reloj")
    ens = pd.DataFrame(filas)
    comp = ens[ens.estado == "completo"]
    mejores = {m: g.sort_values(["pr_auc_media", "ensayo"], ascending=[False, True]).iloc[0] for m, g in comp.groupby("metodo", sort=False)}
    cand_m = max(mejores, key=lambda m: (mejores[m].pr_auc_media, m == "aleatoria"))
    CAND = prm_de(mejores[cand_m])
    log.info("mejor por método (ensayos completos): " + " · ".join(f"{m} #{r.ensayo} PR-AUC {r.pr_auc_media:.4f}" for m, r in mejores.items())
             + f" · candidata: la de la {cand_m} ({hp.texto_prm(CAND)})")

    # ---------- efecto del pruning sobre los mismos ensayos: los 15 primeros de la aleatoria frente a la semana 7
    efecto = efecto_mismos_ensayos(s7, ens, n_folds)
    log.info(f"efecto del pruning en los {efecto['n_ensayos']} primeros ensayos de la aleatoria (mismas configuraciones que la semana 7): "
             f"{efecto}" + (" · no aplica en --prueba" if prueba else ""))
    # ahorro estimado en las dos búsquedas: cada fold no evaluado de un ensayo cortado se estima con el tiempo medio de sus folds evaluados
    unicos = ens[ens.reutilizado_de == ""]
    faltan = (n_folds - unicos.folds_evaluados) * unicos.segundos / unicos.folds_evaluados
    ahorro = {"configuraciones_evaluadas": int(len(unicos)), "folds_evaluados": int(unicos.folds_evaluados.sum()),
              "folds_sin_pruning": int(n_folds * len(unicos)), "segundos_calculados": round(float(unicos.segundos.sum()), 1),
              "segundos_ahorrados_estimados": round(float(faltan.sum()), 1)}
    log.info(f"ahorro estimado del pruning en las dos búsquedas: {ahorro}")

    # ---------- confirmación con los 5 folds del proyecto y regla (caché de resultados de la semana 7)
    t0 = time.time()
    conf = {"actual": ev7("confirmacion", ACT), "candidata": ev7("confirmacion", CAND)}
    seg_conf = round(time.time() - t0, 1)
    pa = np.array([f["pr_auc_comun"] for f in conf["actual"]["folds"]])
    pc = np.array([f["pr_auc_comun"] for f in conf["candidata"]["folds"]])
    regla = hp.regla_de_decision(pc - pa, REGLA)
    cand_igual_s7 = clave_prm(CAND) == clave_prm(R7["candidata"]["parametros"])
    log.info(f"confirmación (5 folds): actual {pa.mean():.4f} ± {pa.std(ddof=1):.4f} · candidata {pc.mean():.4f} ± {pc.std(ddof=1):.4f} · "
             f"regla: {regla}" + (" · la candidata es la misma de la semana 7" if cand_igual_s7 else ""))
    GAN = CAND if regla["adopta_candidata"] else ACT

    # ---------- test: solo si la regla adopta la candidata
    test = None
    if regla["adopta_candidata"]:
        test = {}
        _, b_te = part["test"]["test"]
        yt, ct, gt = y_all[b_te], comun_all[b_te], te.subject_id.values
        for nombre, prm in (("actual", ACT), ("candidata", CAND)):
            r = ev7("test", prm, guardar_p=True)
            p = r["folds"][0]["p"]
            test[nombre] = {"comun": {**{k: round(v, 4) for k, v in metricas(yt[ct], p[ct], prev).items()},
                                      "pr_auc_ic95_pacientes": ic_pacientes(yt[ct], p[ct], gt[ct], H["n_bootstrap"], S),
                                      "brier": round(float(brier_score_loss(yt[ct], p[ct])), 4)},
                            "completa": {**{k: round(v, 4) for k, v in metricas(yt, p, prev).items()},
                                         "pr_auc_ic95_pacientes": ic_pacientes(yt, p, gt, H["n_bootstrap"], S)}}
            log.info(f"TEST {nombre}: {test[nombre]}")
    else:
        log.info("la regla no adopta la candidata: se mantiene la configuración actual y no se mide el test")

    # ---------- respaldo: las mejores configuraciones completas de las semanas 7 y 8, distintas de la ganadora
    pool = armar_pool(s7, comp)
    respaldo_pool = ordenar_respaldo(pool, GAN, ACT, ref)
    conf_gan = ev7("confirmacion", GAN)
    pg = np.array([f["pr_auc_comun"] for f in conf_gan["folds"]])
    respaldos = []
    for j, r in enumerate(respaldo_pool[:Hp["n_respaldo"]], 1):
        c5 = ev7("confirmacion", r["prm"])
        pb = np.array([f["pr_auc_comun"] for f in c5["folds"]])
        respaldos.append({"rol": f"respaldo_{j}", **{k: v for k, v in r.items() if k != "prm"}, "parametros": r["prm"],
                          "confirmacion_5_folds": {"pr_auc_media": float(pb.mean()), "pr_auc_std": float(pb.std(ddof=1)),
                                                   "pr_auc_por_fold": [round(float(x), 6) for x in pb],
                                                   "dif_con_la_ganadora_por_fold": [round(float(x), 6) for x in pb - pg],
                                                   "folds_mejor_que_la_ganadora": int((pb > pg).sum()),
                                                   "roc_auc_media": float(c5["roc_auc_media"]), "segundos": float(c5["segundos"])}})
        log.info(f"respaldo {j}: {r['origen']} · {hp.texto_prm(r['prm'])} · PR-AUC 3 folds {r['pr_auc_media']:.4f} · 5 folds {pb.mean():.4f} "
                 f"(ganadora {pg.mean():.4f})")

    # ---------- importancia de los hiperparámetros (ensayos completos de las cuatro búsquedas, sin repetir configuraciones)
    imp = importancia(pool, E, Hp["importancia_min_sobre_cuantil"])
    log.info(f"importancia de los hiperparámetros: {imp}")

    # ---------- curva de aprendizaje de la ganadora (5 folds del proyecto)
    t0 = time.time()
    part_curva = dict(list(part["confirmacion"].items())[:2]) if prueba else part["confirmacion"]
    curva = curva_aprendizaje(tr, y_tr, comun_tr, part_curva, carpeta, GAN, P, S, MODELO, Hp["curva_fracciones"], X_cols, log)
    seg_curva = round(time.time() - t0, 1)
    curva.to_csv(salida / "hp_pruning_curva.csv", index=False)
    v1 = curva[curva.fraccion == 1].sort_values("fold").pr_auc_validacion.values
    control_curva = {"pr_auc_validacion_fraccion_1": [round(float(x), 6) for x in v1],
                     "pr_auc_confirmacion": [round(float(x), 6) for x in pg[:len(v1)]],
                     "dif_maxima": round(float(np.max(np.abs(v1 - pg[:len(v1)]))), 9)}
    control_curva["reproduce"] = bool(control_curva["dif_maxima"] <= 1e-6)
    log.info(f"control de la curva (fracción 1 frente a la confirmación de la ganadora): {control_curva}")

    # ---------- artefactos: modelos de la ganadora y de los respaldos, reentrenados con todo train (filas de D2)
    t0 = time.time()
    roles = {"ganadora": GAN, **{r["rol"]: r["parametros"] for r in respaldos}}
    modelos = guardar_modelos(roles, tr, y_tr, usar, X_cols, carpeta, P, S, MODELO, destino_modelos, log)
    seg_modelos = round(time.time() - t0, 1)

    # ---------- tabla top-k: configuraciones completas de las semanas 7 y 8 y la referencia
    topk = tabla_topk(pool, ACT, ref, GAN, CAND, respaldos, regla["adopta_candidata"])
    topk.to_csv(salida / "hp_pruning_topk.csv", index=False)

    # ---------- registro único de ensayos (lámina 16)
    runs = hpo_runs(s7, ens, S, H, Hp)
    runs.to_csv((salida if prueba else LOGS) / "hpo_runs.csv", index=False)

    # ---------- resultados
    sin_p = lambda r: {**{k: v for k, v in r.items() if k != "folds"},
                       "folds": [{k: v for k, v in f.items() if k != "p"} for f in r["folds"]]}
    duracion = round(time.time() - t_inicio, 1)
    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S, "modelo": MODELO,
         "variante": H["variante"], "prueba": prueba, "parametros": Hp, "espacio": E, "regla_semana7": REGLA,
         "optuna": optuna.__version__, "sklearn": sklearn.__version__,
         "protocolo": (f"búsqueda: StratifiedGroupKFold({n_folds}, shuffle=True, random_state={S}) por subject_id en train, evaluada fold "
                       f"por fold con MedianPruner; confirmación: StratifiedGroupKFold({H['n_folds_confirmacion']}, shuffle=True, "
                       f"random_state={S}) (los folds de las semanas 5 a 7); entrenan las filas de {H['variante']}; métrica: PR-AUC "
                       "media en las epicrisis de validación sin etiqueta dudosa; test solo si la regla adopta la candidata"),
         "prevalencia_real_con_epicrisis": prev, "configuracion_actual": ACT,
         "referencia_actual_busqueda": ref, "control_referencia_semana7": control_ref,
         "estados_por_busqueda": estados, "segundos_reloj_por_busqueda": seg_metodo,
         "mejor_por_metodo": {m: {"ensayo": int(r.ensayo), "parametros": prm_de(r), "pr_auc_media": float(r.pr_auc_media),
                                  "pr_auc_std": float(r.pr_auc_std), "segundos": float(r.segundos)} for m, r in mejores.items()},
         "candidata": {"metodo": cand_m, "parametros": CAND, "igual_a_la_candidata_de_la_semana_7": cand_igual_s7},
         "efecto_pruning_mismos_ensayos": efecto, "ahorro_pruning": ahorro,
         "confirmacion_5_folds": {k: sin_p(v) for k, v in conf.items()}, "regla": {**REGLA, **regla},
         "decision": "se adopta la candidata" if regla["adopta_candidata"] else "se mantiene la configuración actual",
         "ganadora": GAN, "test": test, "respaldo": respaldos, "importancia": imp,
         "curva_aprendizaje": {"fracciones": Hp["curva_fracciones"], "control": control_curva, "segundos": seg_curva},
         "modelos": modelos, "segundos_modelos": seg_modelos,
         "tiempos_segundos": {"busqueda_aleatoria_reloj": seg_metodo["aleatoria"], "busqueda_bayesiana_reloj": seg_metodo["bayesiana"],
                              "referencia": ref["segundos"], "confirmacion": seg_conf, "curva": seg_curva, "modelos": seg_modelos,
                              "esta_corrida": duracion},
         "folds_de_corridas_previas": len(evf.previas), "evaluaciones_semana7_reutilizadas": len(ev7.previas)}
    guardar_json(R, salida / "hp_pruning_resultados.json")
    figuras_hp_pruning.dibujar(R, ens, s7, curva, salida / "figuras")
    escribir_metrics(R, ens, topk, salida if prueba else LOGS)
    log.info(f"Salidas: metrics_hp_pruning.txt · hpo_runs.csv · hp_pruning_*.csv · hp_pruning_resultados.json · figuras "
             f"fig_hp_pruning_*.png · modelos en {destino_modelos} · duración {duracion:.0f} s")


def escribir_metrics(R, ens, topk, carpeta):
    f = lambda x: f"{x:.4f}"
    Hp, ref = R["parametros"], R["referencia_actual_busqueda"]
    L = [f"# metrics_hp_pruning.txt · {R['generado']} · modelo {R['modelo']} con {R['variante']} · semilla {R['semilla']} · "
         f"Optuna {R['optuna']} · scikit-learn {R['sklearn']}",
         f"protocolo: {R['protocolo']}",
         f"presupuesto: {Hp['n_ensayos']} ensayos por método · {Hp['ensayos_iniciales_tpe']} al azar antes del TPE · pruner {Hp['pruner']} · "
         f"máximo {Hp['max_segundos_por_ensayo']} s por ensayo y {Hp['max_minutos_por_busqueda']} min por búsqueda",
         f"configuración actual: {hp.texto_prm(R['configuracion_actual'])}",
         f"referencia (actual, 3 folds): PR-AUC {f(ref['pr_auc_media'])} ± {f(ref['pr_auc_std'])} · control frente a la semana 7: "
         f"{R['control_referencia_semana7']}", "",
         "ENSAYOS (uno por línea): método · n.º · parámetros · estado · folds · PR-AUC media de los folds evaluados · iteraciones · tiempo"]
    for r in ens.itertuples():
        L.append(f"{r.metodo:<9} #{r.ensayo:02d} · {hp.texto_prm(prm_de(r._asdict()))} · {r.estado} · {r.folds_evaluados} fold(s) · "
                 f"PR-AUC {f(r.pr_auc_media)} · iteraciones ≤ {r.n_iter_max} · {r.segundos:.0f} s"
                 + (f" · igual a {r.reutilizado_de} (reutilizado)" if r.reutilizado_de else ""))
    L += ["", f"estados por búsqueda: {R['estados_por_busqueda']}",
          f"efecto del pruning sobre los mismos ensayos (aleatoria, semanas 7 y 8): {R['efecto_pruning_mismos_ensayos']}",
          f"ahorro estimado en las dos búsquedas: {R['ahorro_pruning']}", "", "MEJOR POR MÉTODO (ensayos completos)"]
    for m, r in R["mejor_por_metodo"].items():
        L.append(f"  {m}: ensayo #{r['ensayo']} · {hp.texto_prm(r['parametros'])} · PR-AUC {f(r['pr_auc_media'])} ± {f(r['pr_auc_std'])}")
    c, g = R["confirmacion_5_folds"], R["regla"]
    L += [f"  candidata: la de la búsqueda {R['candidata']['metodo']} (igual a la de la semana 7: "
          f"{R['candidata']['igual_a_la_candidata_de_la_semana_7']})", "",
          "CONFIRMACIÓN (5 folds del proyecto; PR-AUC en las epicrisis sin etiqueta dudosa)"]
    for k in ("actual", "candidata"):
        L.append(f"  {k:<9} " + " · ".join(f(x["pr_auc_comun"]) for x in c[k]["folds"]) + f" · media {f(c[k]['pr_auc_media'])} ± "
                 f"{f(c[k]['pr_auc_std'])}")
    L += [f"  diferencias (candidata − actual): {g['diferencias_por_fold']} · folds mejores {g['folds_mejores']}/{g['n_folds']} · "
          f"media {g['dif_media']:+.4f} · desv. {g['dif_std']:.4f}",
          f"  regla: criterio de folds {g['criterio_folds']} · criterio de desviación {g['criterio_desviacion']} → {R['decision']}",
          f"  test: {R['test'] if R['test'] else 'no se midió (la regla no adoptó la candidata)'}", "", "RESPALDO"]
    for r in R["respaldo"]:
        cf = r["confirmacion_5_folds"]
        L.append(f"  {r['rol']}: {r['origen']} · {hp.texto_prm(r['parametros'])} · PR-AUC 3 folds {f(r['pr_auc_media'])} · 5 folds "
                 f"{f(cf['pr_auc_media'])} ± {f(cf['pr_auc_std'])} · mejor que la ganadora en {cf['folds_mejor_que_la_ganadora']} de 5 folds")
    L += ["", f"importancia de los hiperparámetros: {R['importancia']}", "",
          f"curva de aprendizaje: control {R['curva_aprendizaje']['control']} (detalle en reportes/hp_pruning_curva.csv)", "",
          "MODELOS GUARDADOS (fuera de git)"]
    for rol, m in R["modelos"].items():
        L.append(f"  {rol}: {m['archivo']} · {m['megabytes']} MB · sha256 {m['sha256']} · control {m['control_mismas_probabilidades_que_la_cache']}")
    L += ["", "TOP 10 (configuraciones completas de las semanas 7 y 8 y la referencia)"]
    for r in topk.head(10).itertuples():
        L.append(f"  {r.puesto:>2}. {r.origen} · {r.parametros} · PR-AUC {f(r.pr_auc_media)} ± {f(r.pr_auc_std)} · {r.segundos:.0f} s"
                 + (f" · {r.rol}" if r.rol else ""))
    L += ["", f"tiempos (s): {R['tiempos_segundos']}"]
    (carpeta / "metrics_hp_pruning.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
