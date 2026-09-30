# -*- coding: utf-8 -*-
"""Utilidades comunes: configuración, rutas, logging y hash de archivos."""
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
RAW, INTERIM, PROCESSED = DATA / "raw", DATA / "interim", DATA / "processed"
REPORTES = RAIZ / "reportes"
LOGS = RAIZ / "logs"


def cargar_config():
    """Lee config/config.yaml y expande ${VAR} con las variables del .env."""
    load_dotenv(RAIZ / ".env")
    texto = (RAIZ / "config" / "config.yaml").read_text(encoding="utf-8")

    def sustituir(m):
        v = os.environ.get(m.group(1))
        if v is None:
            raise KeyError(f"Falta la variable {m.group(1)} en .env (ver .env.example)")
        return v.replace("\\", "/")

    return yaml.safe_load(re.sub(r"\$\{(\w+)\}", sustituir, texto))


def logger(etapa):
    """Logger a consola y a logs/<etapa>_<fecha>.log."""
    LOGS.mkdir(exist_ok=True)
    archivo = LOGS / f"{etapa}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = logging.getLogger(etapa)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(archivo, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info(f"log -> {archivo.relative_to(RAIZ)}")
    return log


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while trozo := f.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def sha256_cacheado(ruta):
    """Hash con caché por (tamaño, fecha de modificación): la ingesta es idempotente
    y no vuelve a leer 1 GB si el archivo no cambió."""
    ruta = Path(ruta)
    cache_f = INTERIM / "_cache_hash.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    st = ruta.stat()
    clave = f"{ruta.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
    if clave not in cache:
        cache[clave] = sha256(ruta)
        INTERIM.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[clave]


def guardar_json(obj, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
