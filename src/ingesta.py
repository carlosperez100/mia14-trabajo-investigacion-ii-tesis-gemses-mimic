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
import json
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

    # resumen del dataset (registros, pacientes, hospitalizaciones) para documentarlo en el README
    resumen_f = INTERIM / "resumen_fuentes.json"
    clave = {f["nombre"]: f["sha256"] for f in filas if f["rol"] == "cruda_externa_DUA"}
    previo = json.loads(resumen_f.read_text(encoding="utf-8")) if resumen_f.exists() else {}
    if previo.get("sha256") == clave:
        resumen = previo
        log.info("Resumen de fuentes sin cambios: se reutiliza")
    else:
        log.info("Contando registros de las fuentes crudas ...")
        n = 0; pac = set(); hadm = set()
        for ch in pd.read_csv(cfg["crudas"]["discharge"], usecols=["note_id", "subject_id", "hadm_id"],
                              chunksize=100000):
            n += len(ch); pac.update(ch.subject_id); hadm.update(ch.hadm_id)
        d = pd.read_csv(cfg["crudas"]["diagnoses_icd"], usecols=["subject_id", "hadm_id", "icd_version"])
        resumen = {"sha256": clave,
                   "discharge": {"epicrisis": n, "pacientes": len(pac), "hospitalizaciones": len(hadm)},
                   "diagnoses_icd": {"filas": int(len(d)), "pacientes": int(d.subject_id.nunique()),
                                     "hospitalizaciones": int(d.hadm_id.nunique()),
                                     "por_version_cie": {str(k): int(v) for k, v in d.icd_version.value_counts().items()}}}
        resumen_f.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    log.info(f"discharge: {resumen['discharge']} · diagnoses_icd: {resumen['diagnoses_icd']}")
    guardar_json({"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"),
                  "fuentes": [{k: f[k] for k in ("nombre", "archivo", "fecha_modificacion", "tamano_bytes", "sha256")}
                              for f in filas],
                  "resumen": {k: v for k, v in resumen.items() if k != "sha256"}},
                 RAIZ / "reportes" / "ingesta_resumen.json")

    bit = pd.DataFrame(filas)
    bit.to_csv(RAIZ / "data" / "bitacora_datos.csv", index=False, encoding="utf-8")
    guardar_json({"etapa": ETAPA, "fuentes": filas}, INTERIM / "ingesta_manifiesto.json")
    log.info(f"Bitácora -> data/bitacora_datos.csv ({len(bit)} fuentes)")


if __name__ == "__main__":
    main()
