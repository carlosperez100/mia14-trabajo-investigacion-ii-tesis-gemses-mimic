# -*- coding: utf-8 -*-
"""
ETAPA 1 — INGESTA (idempotente)

1. Verifica que existan las fuentes crudas (MIMIC-IV hosp y MIMIC-IV-Note) y las
   registra con SHA-256, tamaño y fecha. No se copian: DUA PhysioNet y 1.1 GB.
2. Copia a data/interim/ las tres tablas de etiquetas de la tesis (Fases 3 v2, 8 y 7)
   solo si cambió su hash.
3. Extrae la tabla de época CIE (hadm_id -> icd_version) a data/interim/epoca.parquet.
4. Escribe la bitácora de datos: data/bitacora_datos.csv (fuente, fecha, hash, tamaño).

Uso:  python src/ingesta.py
"""
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import INTERIM, RAIZ, cargar_config, guardar_json, logger, sha256_cacheado

ETAPA = "ingesta"


def registrar(nombre, ruta, rol, log):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"{nombre}: no existe {ruta}. Revisa .env")
    st = ruta.stat()
    h = sha256_cacheado(ruta)
    log.info(f"{nombre:<18} {st.st_size/1e6:>9.1f} MB  sha256 {h[:16]}…")
    return {"nombre": nombre, "rol": rol, "archivo": ruta.name,
            "fecha_modificacion": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "tamano_bytes": st.st_size, "sha256": h,
            "registrado": datetime.now().isoformat(timespec="seconds")}


def main():
    log = logger(ETAPA)
    cfg = cargar_config()["ingesta"]
    INTERIM.mkdir(parents=True, exist_ok=True)
    filas = []

    log.info("Fuentes crudas (se registran, no se copian)")
    for nombre, ruta in cfg["crudas"].items():
        filas.append(registrar(nombre, ruta, "cruda_externa_DUA", log))

    log.info("Etiquetas de la tesis -> data/interim/")
    for nombre, ruta in cfg["etiquetas"].items():
        fila = registrar(nombre, ruta, "etiqueta_copiada", log)
        destino = INTERIM / f"{nombre}.csv"
        if destino.exists() and sha256_cacheado(destino) == fila["sha256"]:
            log.info(f"  {destino.name}: sin cambios, no se copia")
        else:
            shutil.copy2(ruta, destino)
            log.info(f"  {destino.name}: copiado")
        filas.append(fila)

    destino = INTERIM / "epoca.parquet"
    if destino.exists():
        log.info("epoca.parquet ya existe: se reutiliza")
    else:
        log.info("Extrayendo época CIE por hospitalización ...")
        d = pd.read_csv(cfg["crudas"]["diagnoses_icd"], usecols=["hadm_id", "icd_version"])
        epoca = d.groupby("hadm_id")["icd_version"].max().rename("epoca").reset_index()
        epoca.to_parquet(destino, index=False)
        log.info(f"  {len(epoca):,} hospitalizaciones · era 9 {(epoca.epoca==9).mean():.1%} "
                 f"· era 10 {(epoca.epoca==10).mean():.1%}")
    filas.append(registrar("epoca", destino, "derivado_interim", log))

    bit = pd.DataFrame(filas)
    bit.to_csv(RAIZ / "data" / "bitacora_datos.csv", index=False, encoding="utf-8")
    guardar_json({"etapa": ETAPA, "fuentes": filas}, INTERIM / "ingesta_manifiesto.json")
    log.info(f"Bitácora -> data/bitacora_datos.csv ({len(bit)} fuentes)")


if __name__ == "__main__":
    main()
