# -*- coding: utf-8 -*-
"""
FEATURE ENGINEERING (Sprint 2, semana 5) — bloques de atributos y configuraciones del experimento.

Diseño de ablación de la guía de la semana 4 (A, A+B, A+C, A+B+C): cada configuración cambia una cosa
respecto del baseline. Todos los bloques son transformadores de scikit-learn dentro de un Pipeline:
se ajustan solo con los datos de entrenamiento de cada fold y se aplican a validación o test (sin fuga).

  A  Texto: TF-IDF de palabras sueltas (1-gramas). Es el baseline de la semana 3, sin cambios.
  B  Transformación del texto: TF-IDF de bigramas de palabras («pressure ulcer», «in stent»).
  C  Dominio: variables de la hospitalización (src/variables_dominio.py) más el largo de la nota.
       - nota: largo de la epicrisis (log1p, winsorización p1-p99 de train, escalado y marca de nota atípica);
       - paciente, servicio, hospital, atención: ver variables_dominio.GRUPOS.
       Transformaciones, todas ajustadas en train: conteos con cola larga -> log1p + winsorización p1-p99 +
       escalado; otras numéricas -> imputación por la mediana + escalado; categóricas -> one-hot con las
       categorías de menos de `min_frecuencia` casos agrupadas («infrecuentes») y el faltante como categoría;
       binarias -> imputación con 0. Las numéricas llevan además una columna que marca el faltante
       (p. ej., horas en emergencia falta en quien no pasó por emergencia).
     No entran el número de diagnósticos, la severidad APR-DRG ni los procedimientos CIE: salen de la
     codificación de facturación (decisión del EDA de la semana 4).
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from variables_dominio import CONTEO, columnas

CONFIGURACIONES = {
    "A": ["A"],
    "A+B": ["A", "B"],
    "A+C": ["A", "C"],
    "A+B+C": ["A", "B", "C"],
}
DESCRIPCION = {
    "A": "Baseline: TF-IDF de palabras (1-gramas)",
    "A+B": "A + bigramas de palabras",
    "A+C": "A + dominio (paciente, servicio, hospitalización, atención y largo de la nota)",
    "A+B+C": "A + bigramas + dominio",
}


class Winsor(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles (p_inf, p_sup) calculados en train."""

    def __init__(self, p_inf=1, p_sup=99):
        self.p_inf, self.p_sup = p_inf, p_sup

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_ = np.nanpercentile(X, [self.p_inf, self.p_sup], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lim_[0], self.lim_[1])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class LargoNota(BaseEstimator, TransformerMixin):
    """Largo de la epicrisis: log1p -> winsorización (p_inf, p_sup de train) -> z-score; más una marca de outlier."""

    def __init__(self, p_inf=1, p_sup=99):
        self.p_inf, self.p_sup = p_inf, p_sup

    def _log(self, X):
        return np.log1p(np.asarray([len(t) for t in np.ravel(X)], dtype=float))

    def fit(self, X, y=None):
        v = self._log(X)
        self.lim_ = np.percentile(v, [self.p_inf, self.p_sup])
        w = np.clip(v, *self.lim_)
        self.media_, self.desv_ = w.mean(), w.std() or 1.0
        return self

    def transform(self, X):
        v = self._log(X)
        z = (np.clip(v, *self.lim_) - self.media_) / self.desv_
        flag = (v > self.lim_[1]).astype(float)
        return sparse.csr_matrix(np.column_stack([z, flag]))

    def get_feature_names_out(self, input_features=None):
        return np.array(["largo_log_winsor_z", "largo_atipico"])


def bloque(nombre, p):
    if nombre == "A":
        return ("A_palabras", TfidfVectorizer(max_features=p["A"]["max_features"], min_df=p["A"]["min_df"],
                                              sublinear_tf=True, strip_accents="unicode", lowercase=True), "text")
    if nombre == "B":
        return ("B_bigramas", TfidfVectorizer(ngram_range=(2, 2), max_features=p["B"]["max_features"], min_df=p["B"]["min_df"],
                                              sublinear_tf=True, strip_accents="unicode", lowercase=True), "text")
    if nombre == "C":
        return ("C_dominio", dominio(p["C"], p["C"]["grupos"]), columnas_dominio(p["C"]["grupos"]))
    raise ValueError(nombre)


def columnas_dominio(grupos):
    num, cat, b = columnas([g for g in grupos if g != "nota"])
    return (["text"] if "nota" in grupos else []) + num + cat + b


def dominio(c, grupos):
    """Transformador del bloque C para los grupos pedidos (todo se ajusta en train)."""
    num, cat, b = columnas([g for g in grupos if g != "nota"])
    conteo = [v for v in num if v in CONTEO]
    otras = [v for v in num if v not in CONTEO]
    w0, w1 = c["winsor"]
    partes = []
    if "nota" in grupos:
        partes.append(("largo", LargoNota(w0, w1), "text"))
    if conteo:
        partes.append(("conteo", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                               Winsor(w0, w1), StandardScaler()), conteo))
    if otras:
        partes.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), otras))
    if cat:
        partes.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Sin dato"),
                                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=c["min_frecuencia"])), cat))
    if b:
        partes.append(("bin", SimpleImputer(strategy="constant", fill_value=0), b))
    return ColumnTransformer(partes, sparse_threshold=1.0)


def pipeline(config, p, semilla):
    """Pipeline completo de una configuración: bloques de atributos + regresión logística del baseline."""
    bloques = ColumnTransformer([bloque(b, p) for b in CONFIGURACIONES[config]], sparse_threshold=1.0)
    lg = p["logistica"]
    return Pipeline([("atributos", bloques),
                     ("clf", LogisticRegression(C=lg["C"], class_weight="balanced", max_iter=2000,
                                                solver="liblinear", random_state=semilla))])
