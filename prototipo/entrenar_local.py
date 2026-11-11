# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO LOCAL del modelo adoptado (configuración actual), para correr en la máquina del investigador.

Configuración: la de src/features.py con los parámetros de feature_engineering de config/config.yaml, que es la
configuración actual registrada en he1_estratos.configuracion: bloques A+B+C, regresión logística con C 1.0 y
class_weight balanced, A y B con 20 000 términos y min_df 3, sublinear_tf. Filas: las que conserva la variante D2.

  --filas todo   las epicrisis D2 de todo el corpus. Según el plan de tesis, el modelo que marcó las detecciones de la
                 HE2 y la HE3 se reentrenó así (“las 61 319 epicrisis que la variante D2 conserva en todo el corpus”).
  --filas train  solo las filas D2 de train: el modelo con las métricas de prueba conocidas.

Lee tres archivos de la carpeta data/ del repositorio de trabajo (todos fuera de git): processed/dataset.parquet,
interim/dominio_hadm.parquet e interim/variantes_datos.parquet. Escribe el .joblib y su ficha (.ficha.json) en la
carpeta local del prototipo, nunca en el repositorio, y deja en la ficha solo conteos y huellas, sin texto.

Uso:  python -m prototipo.entrenar_local --datos "<repositorio de trabajo>/data" --filas todo
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from . import contrato
from .catalogo import sha256
from .rutas import CONFIG, RAIZ, SRC, dir_datos

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parametros():
    """Parámetros de feature_engineering, verificados contra he1_estratos.configuracion (la configuración actual)."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, actual = cfg["feature_engineering"], cfg["he1_estratos"]["configuracion"]
    comprobaciones = {     # con if y no con assert, para que no se omitan al ejecutar con python -O
        "modelo A+B+C y variante D2": cfg["hiperparametros"]["modelo"] == "A+B+C" and cfg["hiperparametros"]["variante"] == "D2",
        "C de la regresión logística": p["logistica"]["C"] == actual["C"],
        "bloque A": (p["A"]["max_features"], p["A"]["min_df"]) == (actual["A_max_features"], actual["A_min_df"]),
        "bloque B": (p["B"]["max_features"], p["B"]["min_df"]) == (actual["B_max_features"], actual["B_min_df"]),
        "pesos balanceados y sublinear_tf": actual["class_weight"] == "balanced" and actual["sublinear_tf"] is True,
    }
    fallan = [k for k, ok in comprobaciones.items() if not ok]
    if fallan:
        raise ValueError("config/config.yaml no coincide con la configuración actual (he1_estratos) en: " + ", ".join(fallan))
    return p, int(cfg["semilla"])


def construir_pipeline():
    import features                      # src/features.py: una sola fuente para el pipeline
    p, semilla = parametros()
    return features.pipeline("A+B+C", p, semilla), p, semilla


def _commit():
    try:
        # “-dirty” al final indica que el código tenía cambios sin commit al entrenar
        return subprocess.run(["git", "-C", str(RAIZ), "describe", "--always", "--dirty", "--abbrev=7"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def entrenar(X, y, salida, tipo, procedencia):
    """Ajusta el pipeline, guarda el .joblib y su ficha, y devuelve la ficha (con la huella SHA-256)."""
    t0 = time.time()
    pipe, p, semilla = construir_pipeline()
    X = X[["text"] + contrato.VARIABLES]
    pipe.fit(X, np.asarray(y, dtype=int))
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, salida, compress=3)
    ficha = {
        "tipo": tipo,
        "sha256": sha256(salida),
        "archivo": salida.name,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {"bloques": "A+B+C", "variante": "D2", "C": p["logistica"]["C"], "class_weight": "balanced",
                          "A": p["A"], "B": p["B"], "C_dominio": p["C"], "sublinear_tf": True, "semilla": semilla},
        "corte": contrato.CORTE,
        "columnas": ["text"] + contrato.VARIABLES,
        "version_contrato": contrato.VERSION,
        "filas": int(len(X)),
        "positivas": int(np.sum(y)),
        "procedencia": procedencia,
        "codigo": _commit(),
        "entorno": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                    "pandas": pd.__version__},
        "segundos": round(time.time() - t0, 1),
    }
    salida.with_suffix(".ficha.json").write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    return ficha


def cargar_corpus(datos):
    """Une el corpus con las variables de dominio y las marcas de las variantes, como el pipeline del curso."""
    datos = Path(datos)
    rutas = {"dataset": datos / "processed" / "dataset.parquet", "dominio": datos / "interim" / "dominio_hadm.parquet",
             "variantes": datos / "interim" / "variantes_datos.parquet"}
    for nombre, r in rutas.items():
        if not r.exists():
            raise FileNotFoundError(f"Falta {nombre}: se esperaba en {r}")
    df = pd.read_parquet(rutas["dataset"])
    dom = pd.read_parquet(rutas["dominio"])
    mar = pd.read_parquet(rutas["variantes"])
    if not set(df.hadm_id) <= set(dom.hadm_id) or not set(df.hadm_id) <= set(mar.hadm_id):
        raise ValueError("Las cachés de dominio o de variantes no cubren todas las hospitalizaciones del corpus.")
    df = df.merge(dom, on="hadm_id", how="left", validate="m:1").merge(mar, on="hadm_id", how="left", validate="m:1")
    huellas = {n: sha256(r) for n, r in rutas.items()}
    return df, huellas


def main(argv=None):
    ap = argparse.ArgumentParser(description="Entrena el modelo adoptado (configuración actual) en la máquina local.")
    ap.add_argument("--datos", required=True, help="carpeta data/ del repositorio de trabajo")
    ap.add_argument("--filas", choices=["todo", "train"], default="todo")
    ap.add_argument("--salida", default=None, help="ruta del .joblib (por defecto, en la carpeta local del prototipo)")
    a = ap.parse_args(argv)
    import variantes_datos as vd         # src/variantes_datos.py: la misma regla de D2

    df, huellas = cargar_corpus(a.datos)
    base = df if a.filas == "todo" else df[df.split == "train"]
    usar = vd.filas_de_entrenamiento(base, "D2").values
    X, y = base[usar].reset_index(drop=True), base.y.values[usar]
    salida = Path(a.salida) if a.salida else dir_datos() / f"modelo_local_{a.filas}.joblib"
    print(f"Entrenando con {len(X):,} epicrisis ({int(y.sum()):,} positivas), filas “{a.filas}” de D2 ...", flush=True)
    ficha = entrenar(X, y, salida, "local", {"filas": a.filas, "huellas_datos": huellas})
    print(f"Listo en {ficha['segundos']} s.\n  modelo: {salida}\n  SHA-256: {ficha['sha256']}\n"
          f"Para usarlo, poner en el .env:\n  PROTOTIPO_MODELO={salida}\n  PROTOTIPO_MODELO_SHA256={ficha['sha256']}")


if __name__ == "__main__":
    main()
