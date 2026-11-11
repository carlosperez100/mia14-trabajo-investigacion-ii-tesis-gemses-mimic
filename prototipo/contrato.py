# -*- coding: utf-8 -*-
"""
CONTRATO DE ENTRADA Y SALIDA del detector (versión 1.0).

Entrada, una fila por epicrisis:
  id_epicrisis  identificador que asigna el usuario: letras, dígitos, guion, guion bajo o punto, hasta 64 caracteres.
                No debe ser un identificador de MIMIC-IV.
  texto         la epicrisis en inglés, de 100 caracteres o más. El corpus de entrenamiento descartó las notas más cortas
                (config/config.yaml, preprocesado.largo_min_texto).
  variables     las 18 variables de la hospitalización del bloque C (src/variables_dominio.py), todas opcionales. El
                faltante se imputa igual que en el entrenamiento (src/features.py): mediana y marca de faltante en las
                numéricas, “Sin dato” en las categóricas y 0 en las binarias. Las numéricas deben ser finitas y no
                negativas, y las binarias, 0 o 1. Las categorías son las de MIMIC-IV; una categoría que el modelo no vio
                va a la categoría “infrecuente” si el codificador la formó en el entrenamiento y, si no, queda en ceros.
  codigos_cie   códigos CIE del alta (opcional), con o sin punto; version_cie 9 o 10 (por defecto, 10).
Una fila que no cumple el contrato se rechaza con su motivo, sin detener el resto del lote. Si su identificador no es
válido, se la nombra por su número de fila, para no repetir en pantalla un valor que podría contener texto.

Salida, una fila por epicrisis: la probabilidad, la marca (probabilidad de 0.5 o más, el corte de la detección del
plan de tesis), los códigos de evento encontrados, la cola a la que pasa y la huella SHA-256 del modelo que la calculó.
La salida no repite el texto de la epicrisis.
"""
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

VERSION = "1.0"
CORTE = 0.5          # detección = probabilidad de 0.5 o más (plan de tesis, Tabla 1, definición de términos)
LARGO_MIN = 100      # config/config.yaml, preprocesado.largo_min_texto

# Mismo orden que columnas_dominio() de src/features.py (numéricas, categóricas y binarias de los grupos
# paciente, servicio, hospital y atención). tests/test_contrato.py verifica que coinciden.
NUMERICAS = ["edad", "horas_emergencia", "n_servicios", "estancia_dias", "n_traslados", "n_unidades", "dias_uci",
             "n_medicamentos", "n_cultivos"]
CATEGORICAS = ["sexo", "tipo_ingreso", "procedencia", "seguro", "servicio", "servicio_alta", "destino_alta"]
BINARIAS = ["servicio_quirurgico", "uci"]
VARIABLES = NUMERICAS + CATEGORICAS + BINARIAS

# Nombres en castellano, tomados del docstring de src/variables_dominio.py
ETIQUETAS = {
    "edad": "Edad (años)", "horas_emergencia": "Horas en emergencia", "n_servicios": "N.° de servicios distintos",
    "estancia_dias": "Estancia (días)", "n_traslados": "N.° de traslados", "n_unidades": "N.° de unidades distintas",
    "dias_uci": "Días en UCI", "n_medicamentos": "N.° de medicamentos distintos prescritos",
    "n_cultivos": "N.° de muestras para cultivo microbiológico", "sexo": "Sexo", "tipo_ingreso": "Tipo de ingreso",
    "procedencia": "Procedencia", "seguro": "Seguro", "servicio": "Servicio al ingreso", "servicio_alta": "Servicio al alta",
    "destino_alta": "Destino al alta", "servicio_quirurgico": "Pasó por un servicio quirúrgico", "uci": "Pasó por UCI",
}

COLAS = {
    "sin_codigo": "Posible evento sin código",
    "con_codigo": "Marcada, con código de evento",
    "no_marcada": "No marcada",
}

_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


@dataclass
class Epicrisis:
    id_epicrisis: str
    texto: str
    variables: dict = field(default_factory=dict)
    codigos_cie: list = field(default_factory=list)
    version_cie: int = 10


@dataclass
class Deteccion:
    id_epicrisis: str
    probabilidad: float
    marca: bool
    sin_codigo: bool
    cola: str
    codigos_evento: list
    corte: float
    modelo_sha256: str
    modelo_tipo: str
    version_contrato: str = VERSION
    fecha_hora: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def como_dict(self):
        return asdict(self)


def vacio(v):
    """None, NaN o texto en blanco."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()


def leer_codigos(valor):
    """Separa una cadena de códigos (“T81.4XXA; I10”) en una lista. Acepta también una lista."""
    if vacio(valor):
        return []
    if isinstance(valor, (list, tuple)):
        return [str(c).strip() for c in valor if not vacio(c)]
    return [c for c in re.split(r"[;,\s]+", str(valor).strip()) if c]


def version_valida(v):
    """9 o 10 (también “9”, “10” o “10.0”); vacío vale 10. Devuelve None si no es una versión admitida."""
    if vacio(v):
        return 10
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return int(x) if x in (9.0, 10.0) else None


def id_valido(v):
    """El identificador cumple el contrato: de 1 a 64 letras, dígitos, guiones, guiones bajos o puntos."""
    return bool(_ID.match(str(v or "")))


def etiqueta_segura(ep, i):
    """Nombre con que se informa una fila rechazada: su identificador si es válido o, si no, su número de fila."""
    return ep.id_epicrisis if id_valido(ep.id_epicrisis) else f"fila {i + 1}"


def validar(ep):
    """Devuelve la lista de errores de una epicrisis (vacía si cumple el contrato)."""
    errores = []
    if not id_valido(ep.id_epicrisis):
        errores.append("id_epicrisis: use de 1 a 64 caracteres entre letras, dígitos, guion, guion bajo o punto.")
    if len((ep.texto or "").strip()) < LARGO_MIN:
        errores.append(f"texto: la epicrisis debe tener {LARGO_MIN} caracteres o más, como en el entrenamiento.")
    for nombre, valor in (ep.variables or {}).items():
        if nombre not in VARIABLES:
            errores.append(f"{nombre}: no es una de las 18 variables del contrato.")
            continue
        if vacio(valor):
            continue
        if nombre in NUMERICAS or nombre in BINARIAS:
            try:
                x = float(valor)
            except (TypeError, ValueError):
                errores.append(f"{nombre}: debe ser un número.")
                continue
            if math.isnan(x):
                continue
            if math.isinf(x):
                errores.append(f"{nombre}: debe ser un número finito.")
                continue
            if x < 0:
                errores.append(f"{nombre}: no admite valores negativos.")
            if nombre in BINARIAS and x not in (0.0, 1.0):
                errores.append(f"{nombre}: debe ser 0 o 1.")
    if version_valida(ep.version_cie) is None:
        errores.append("version_cie: debe ser 9 o 10.")
    return errores


def a_tabla(epicrisis):
    """Arma la tabla que recibe el pipeline: columna “text” más las 18 variables, con NaN en los faltantes."""
    filas = []
    for ep in epicrisis:
        fila = {"text": ep.texto}
        v = ep.variables or {}
        for n in NUMERICAS + BINARIAS:
            fila[n] = np.nan if vacio(v.get(n)) else float(v[n])
        for n in CATEGORICAS:
            fila[n] = np.nan if vacio(v.get(n)) else str(v[n]).strip()
        filas.append(fila)
    t = pd.DataFrame(filas, columns=["text"] + VARIABLES)
    t[NUMERICAS + BINARIAS] = t[NUMERICAS + BINARIAS].astype(float)
    t[CATEGORICAS] = t[CATEGORICAS].astype(object)
    return t


def desde_fila(fila):
    """Epicrisis a partir de una fila de un CSV de entrada (columnas del contrato; las ausentes quedan vacías).
    Una version_cie no admitida se conserva tal cual, para que validar() rechace la fila sin detener el lote."""
    version = fila.get("version_cie")
    return Epicrisis(
        id_epicrisis=str(fila.get("id_epicrisis", "")).strip(),
        texto=str(fila.get("texto", "") if not vacio(fila.get("texto")) else ""),
        variables={n: fila.get(n) for n in VARIABLES if n in fila},
        codigos_cie=leer_codigos(fila.get("codigos_cie")),
        version_cie=version_valida(version) or version,
    )


def tabla_salida(detecciones):
    """Salida del contrato, una fila por epicrisis y sin el texto; codigos_evento lista solo los códigos de evento."""
    filas = []
    for d in detecciones:
        f = d.como_dict()
        f["codigos_evento"] = "; ".join(c["codigo"] for c in d.codigos_evento if c.get("es_evento"))
        filas.append(f)
    return pd.DataFrame(filas, columns=list(Deteccion.__dataclass_fields__))


def cola(marca, sin_codigo):
    if not marca:
        return COLAS["no_marcada"]
    return COLAS["sin_codigo"] if sin_codigo else COLAS["con_codigo"]
