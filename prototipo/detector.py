# -*- coding: utf-8 -*-
"""
DETECTOR: carga el modelo, verifica su huella y calcula la probabilidad de evento adverso de cada epicrisis.

Orden de carga: el modelo local entrenado con MIMIC-IV (PROTOTIPO_MODELO) y, si no está configurado, el modelo de
demostración (entrenado con epicrisis inventadas). Un .joblib ejecuta código al abrirse, así que el detector solo lo
abre si su SHA-256 coincide con todas las huellas que exige su origen:
  - modelo de demostración de la carpeta local: la huella de su ficha, que se escribe al entrenarlo en esta máquina;
  - modelo local (PROTOTIPO_MODELO): la de su ficha y la fijada en el .env (PROTOTIPO_MODELO_SHA256), sin la cual no
    se abre, porque quien pueda cambiar el modelo también puede cambiar la ficha que está a su lado;
  - cualquier otra ruta (por ejemplo, --modelo del lote): la de su ficha y la huella esperada que indica el usuario.
La huella se calcula sobre los mismos bytes que luego se abren, sin volver a leer el archivo.

El detector solo marca. No muestra términos, coeficientes ni resaltados del texto, y la decisión es del revisor.
"""
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import codigos, contrato
from .rutas import SRC, ruta_modelo_demo, ruta_modelo_local, sha256_esperado

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))          # el pipeline guardado usa las clases de src/features.py


class ErrorModelo(Exception):
    pass


@dataclass
class Detector:
    pipe: object
    sha256: str
    ficha: dict
    ruta: Path
    avisos: list = field(default_factory=list)

    @property
    def tipo(self):
        return self.ficha.get("tipo", "desconocido")

    @property
    def es_demo(self):
        return self.tipo == "demostracion"

    def probabilidades(self, tabla):
        return self.pipe.predict_proba(tabla[["text"] + contrato.VARIABLES])[:, 1]

    def categorias(self):
        """Categorías que el modelo vio en su entrenamiento, por variable (sin “Sin dato”)."""
        try:
            dom = self.pipe.named_steps["atributos"].named_transformers_["C_dominio"]
            cols = next(t[2] for t in dom.transformers_ if t[0] == "cat")
            enc = dom.named_transformers_["cat"].named_steps["onehotencoder"]
            return {c: [str(x) for x in cats if str(x) != "Sin dato"] for c, cats in zip(cols, enc.categories_)}
        except (AttributeError, KeyError, StopIteration):
            return {}


def ruta_activa():
    return ruta_modelo_local() or ruta_modelo_demo()


def _misma_ruta(a, b):
    """Compara dos rutas ya resueltas, para que la misma ruta escrita de otra forma se reconozca."""
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def cargar(ruta=None, huella_esperada=None):
    """Abre el modelo solo si su huella SHA-256 coincide con todas las que exige su origen (ver el docstring del módulo)."""
    ruta = Path(ruta) if ruta else ruta_activa()
    if not ruta.exists():
        raise ErrorModelo(f"No se encontró el modelo {ruta.name}. Para la demostración, ejecutar: python -m prototipo.modelo_demo")
    datos = ruta.read_bytes()                          # una sola lectura: lo que se verifica es lo que se abre
    huella = hashlib.sha256(datos).hexdigest()
    ruta_ficha = ruta.with_suffix(".ficha.json")
    ficha = json.loads(ruta_ficha.read_text(encoding="utf-8")) if ruta_ficha.exists() else {}
    if not ficha.get("sha256"):
        raise ErrorModelo("El modelo no tiene ficha con su huella registrada: no se abre.")
    referencias = [ficha["sha256"]]
    if _misma_ruta(ruta, ruta_modelo_local()):
        if not sha256_esperado():
            raise ErrorModelo("Falta PROTOTIPO_MODELO_SHA256 en el .env: el modelo local solo se abre con su huella fijada.")
        referencias.append(sha256_esperado())
    elif not _misma_ruta(ruta, ruta_modelo_demo()) and not huella_esperada:
        raise ErrorModelo("Un modelo de otra ruta solo se abre si se indica su huella SHA-256 esperada (--sha256).")
    if huella_esperada:
        referencias.append(str(huella_esperada).strip().lower())
    if any(r != huella for r in referencias):
        raise ErrorModelo("La huella SHA-256 del modelo no coincide con la registrada: no se abre.")
    if ficha.get("columnas") and ficha["columnas"] != ["text"] + contrato.VARIABLES:
        raise ErrorModelo("Las columnas del modelo no coinciden con el contrato de entrada.")
    avisos = []
    version = ficha.get("entorno", {}).get("scikit_learn")
    if version and version != sklearn.__version__:
        avisos.append(f"El modelo se entrenó con scikit-learn {version} y el entorno tiene {sklearn.__version__}.")
    pipe = joblib.load(io.BytesIO(datos))              # joblib usa pickle: solo llega aquí con las huellas verificadas
    return Detector(pipe, huella, ficha, ruta, avisos)


def detectar(detector, epicrisis):
    """Devuelve (detecciones, errores). Las epicrisis que no cumplen el contrato no se procesan."""
    validas, errores = [], {}
    for i, ep in enumerate(epicrisis):
        e = contrato.validar(ep)
        if e:
            errores[contrato.etiqueta_segura(ep, i)] = e
        else:
            validas.append(ep)
    if not validas:
        return [], errores
    p = detector.probabilidades(contrato.a_tabla(validas))
    salida = []
    for ep, prob in zip(validas, p):
        sin_codigo, anotados = codigos.revisar(ep.codigos_cie, ep.version_cie)
        marca = bool(prob >= contrato.CORTE)
        salida.append(contrato.Deteccion(
            id_epicrisis=ep.id_epicrisis, probabilidad=round(float(prob), 4), marca=marca, sin_codigo=sin_codigo,
            cola=contrato.cola(marca, sin_codigo), codigos_evento=anotados, corte=contrato.CORTE,
            modelo_sha256=detector.sha256, modelo_tipo=detector.tipo))
    return salida, errores


def categorias_no_vistas(detector, epicrisis):
    """Avisa de categorías que el modelo no vio: van a la categoría “infrecuente” si el codificador la formó en el
    entrenamiento y, si no, se codifican como ceros (OneHotEncoder con handle_unknown="infrequent_if_exist")."""
    vistas = detector.categorias()
    avisos = []
    for ep in epicrisis:
        for n in contrato.CATEGORICAS:
            v = (ep.variables or {}).get(n)
            if not contrato.vacio(v) and vistas.get(n) and str(v).strip() not in vistas[n]:
                avisos.append(f"{ep.id_epicrisis}: “{v}” en {contrato.ETIQUETAS[n]} no estaba en el entrenamiento.")
    return avisos


def resumen_probabilidades(p):
    p = np.asarray(p, dtype=float)
    return {"n": int(p.size), "marcadas": int((p >= contrato.CORTE).sum())}
