# -*- coding: utf-8 -*-
"""
CATÁLOGO del Anexo N.° 02 y parámetros de la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01 (edición 2021).

- Eventos: data/catalogos/anexo02_catalogo_231.csv (versionado en el repositorio), con el nombre oficial tomado del PDF.
- Impactos (C, D, E, F e impacto impreso): la tabla no está en el repositorio; se lee de un CSV local indicado en
  PROTOTIPO_IMPACTOS_CSV. Sin ese CSV, la interfaz pide valorar los cuatro impactos de cada evento con la escala del
  num. 19.11 b, y esa valoración queda rotulada como del usuario.
"""
import hashlib
from functools import lru_cache

import pandas as pd

from .rutas import CATALOGOS, ruta_impactos

# num. 19.11 b (p. 20): escala de los cuatro indicadores de riesgo
ESCALA = {"Muy alto": 9, "Alto": 6, "Medio": 3, "Bajo": 1, "Nulo": 0}
# num. 19.11 b i a iv (p. 20) y c i a iv (p. 21): indicadores y ponderaciones; letras del Anexo N.° 03 (p. 39)
DIMENSIONES = {
    "C": "Prolongación de estancia hospitalaria",
    "D": "Complicaciones intrahospitalarias",
    "E": "Sobrecostos asociados a la no calidad",
    "F": "Insatisfacción del paciente/Imagen institucional",
}
PESOS = {"C": 0.40, "D": 0.20, "E": 0.15, "F": 0.25}
# num. 19.6 (p. 20): procesos a los que pertenece el ERSP
PROCESOS = {
    "Asistenciales": ["Ambulatorio", "Extramural", "Hospitalización", "Emergencia", "Quirúrgica", "Apoyo al Diagnóstico"],
    "No asistenciales": ["Ingeniería Hospitalaria", "Logística", "Informática", "Servicios Médicos de Apoyo", "Otro"],
}
LISTA_PROCESOS = PROCESOS["Asistenciales"] + PROCESOS["No asistenciales"]


@lru_cache(maxsize=1)
def eventos():
    """Los 231 eventos del Anexo N.° 02: índice id_ev, columnas nombre_oficial y nat (naturaleza)."""
    c = pd.read_csv(CATALOGOS / "anexo02_catalogo_231.csv", usecols=["id_ev", "nombre_oficial", "nat"])
    if len(c) != 231 or not c.id_ev.is_unique:
        raise ValueError("El catálogo del Anexo N.° 02 debe tener 231 eventos con identificador único.")
    c["nombre_oficial"] = c.nombre_oficial.str.split().str.join(" ")
    return c.set_index("id_ev").sort_index()


def nombre(id_ev):
    return eventos().loc[int(id_ev), "nombre_oficial"]


def naturaleza(id_ev):
    return eventos().loc[int(id_ev), "nat"]


def etiqueta(id_ev):
    e = eventos().loc[int(id_ev)]
    return f"{int(id_ev)} · {e.nombre_oficial} ({e.nat.capitalize()})"


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _leer_impactos(ruta_texto, _marca_tiempo):
    t = pd.read_csv(ruta_texto)
    requeridas = {"id_ev", "estancia", "comp", "sobrecosto", "insat", "impacto"}
    falta = requeridas - set(t.columns)
    if falta:
        raise ValueError(f"El CSV de impactos no tiene las columnas {sorted(falta)}.")
    t = t.rename(columns={"estancia": "C", "comp": "D", "sobrecosto": "E", "insat": "F", "impacto": "impreso"})
    t = t[["id_ev", "C", "D", "E", "F", "impreso"]].copy()
    t["id_ev"] = t.id_ev.astype(int)
    if not t.id_ev.is_unique:
        raise ValueError("El CSV de impactos repite identificadores de evento.")
    fuera = set(t.id_ev) - set(eventos().index)
    if fuera:
        raise ValueError(f"El CSV de impactos trae {len(fuera)} identificadores que no están en el catálogo del Anexo N.° 02.")
    t["fuera_de_escala"] = ~t[["C", "D", "E", "F"]].isin(list(ESCALA.values())).all(axis=1)
    return t.set_index("id_ev").sort_index()


def impactos(ruta=None):
    """Tabla de impactos del Anexo N.° 02 desde el CSV local, o (None, {...}) si no hay CSV configurado."""
    ruta = ruta or ruta_impactos()
    if not ruta or not ruta.exists():
        return None, {"fuente": None}
    t = _leer_impactos(str(ruta), ruta.stat().st_mtime)
    info = {"fuente": "Anexo N.° 02 (tabla local)", "archivo": ruta.name, "sha256": sha256(ruta),
            "eventos": int(len(t)), "fuera_de_escala": int(t.fuera_de_escala.sum())}
    return t, info
