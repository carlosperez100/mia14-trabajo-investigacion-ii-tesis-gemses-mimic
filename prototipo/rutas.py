# -*- coding: utf-8 -*-
"""
Rutas y variables de entorno del prototipo.

Nada de lo que el prototipo genera entra al repositorio: la base SQLite, los modelos y los registros viven en la
carpeta PROTOTIPO_DATOS, que por defecto es ~/.mia14_prototipo (fuera del repositorio). Las variables se leen del
.env de la raíz, el mismo del pipeline (ver .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
CATALOGOS = RAIZ / "data" / "catalogos"
CONFIG = RAIZ / "config" / "config.yaml"
DEMO = Path(__file__).resolve().parent / "datos_demo"

load_dotenv(RAIZ / ".env", override=False)


def _ruta_env(nombre):
    valor = os.environ.get(nombre, "").strip()
    return Path(valor).expanduser() if valor else None


def dir_datos():
    """Carpeta local del prototipo (se crea si no existe)."""
    d = _ruta_env("PROTOTIPO_DATOS") or Path.home() / ".mia14_prototipo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ruta_base():
    return dir_datos() / "prototipo.sqlite"


def ruta_modelo_demo():
    return dir_datos() / "modelo_demo.joblib"


def ruta_modelo_local():
    """Modelo entrenado con MIMIC-IV en la máquina del investigador (PROTOTIPO_MODELO). Nunca va al repositorio."""
    return _ruta_env("PROTOTIPO_MODELO")


def sha256_esperado():
    """Huella que el investigador fija en el .env para el modelo local. Es obligatoria: sin ella, el modelo local no se abre."""
    v = os.environ.get("PROTOTIPO_MODELO_SHA256", "").strip().lower()
    return v or None


def ruta_impactos():
    """CSV local con los impactos del Anexo N.° 02 (PROTOTIPO_IMPACTOS_CSV); la tabla no está en el repositorio."""
    return _ruta_env("PROTOTIPO_IMPACTOS_CSV")
