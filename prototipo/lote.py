# -*- coding: utf-8 -*-
"""
DETECCIÓN POR LOTE desde la línea de comandos.

    python -m prototipo.lote entrada.csv salida.csv [--guardar] [--reemplazar] [--modelo RUTA --sha256 HUELLA]

La entrada sigue el contrato (prototipo/contrato.py): id_epicrisis, texto, las 18 variables opcionales, codigos_cie y
version_cie. La salida tiene una fila por epicrisis válida y no repite el texto. Las que no cumplen el contrato se
listan en la consola con su identificador (o su número de fila, si el identificador no es válido) y el motivo, sin
detener el lote; la consola no muestra texto clínico. Un identificador válido repetido sí detiene el lote, porque no se
sabría a qué epicrisis corresponde cada detección.

--guardar     registra las epicrisis y sus detecciones en la base local (origen “lote”) para revisarlas en la interfaz.
--reemplazar  con --guardar, reemplaza las epicrisis que ya existan y reinicia su revisión (queda en la bitácora).
--modelo      un .joblib con su ficha; por defecto, el modelo activo (local o de demostración).
--sha256      huella SHA-256 esperada del modelo de --modelo; sin ella, un modelo de otra ruta no se abre.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import almacen, contrato, detector


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m prototipo.lote", description="Detección por lote; la salida no lleva texto.")
    ap.add_argument("entrada", type=Path, help="CSV con las columnas del contrato")
    ap.add_argument("salida", type=Path, help="CSV de salida (probabilidad, marca, cola, códigos de evento, huella)")
    ap.add_argument("--guardar", action="store_true", help="registra las epicrisis y sus detecciones en la base local")
    ap.add_argument("--reemplazar", action="store_true", help="con --guardar, reemplaza las epicrisis que ya existan")
    ap.add_argument("--modelo", type=Path, help="modelo .joblib con su ficha (por defecto, el activo)")
    ap.add_argument("--sha256", help="huella SHA-256 esperada del modelo indicado con --modelo")
    a = ap.parse_args(argv)
    if a.salida.resolve() == a.entrada.resolve():
        ap.error("la salida no puede sobrescribir la entrada")
    try:
        det = detector.cargar(a.modelo, a.sha256)
    except detector.ErrorModelo as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    df = pd.read_csv(a.entrada, dtype=str, keep_default_na=False)
    falta = {"id_epicrisis", "texto"} - set(df.columns)
    if falta:
        print(f"Error: faltan las columnas {sorted(falta)}.", file=sys.stderr)
        return 2
    eps = [contrato.desde_fila(r) for r in df.to_dict("records")]
    validos = [e.id_epicrisis for e in eps if contrato.id_valido(e.id_epicrisis)]
    repetidos = sorted({i for i in validos if validos.count(i) > 1})
    if repetidos:                  # no se sabría a qué epicrisis corresponde cada detección
        print(f"Error: el CSV repite identificadores de epicrisis: {', '.join(repetidos[:5])}"
              + (" …" if len(repetidos) > 5 else "") + ".", file=sys.stderr)
        return 2

    dets, errores = detector.detectar(det, eps)
    contrato.tabla_salida(dets).to_csv(a.salida, index=False, encoding="utf-8-sig")
    omitidas = []
    if a.guardar:
        with almacen.conexion() as con:
            omitidas = almacen.guardar_lote(con, eps, dets, "lote", reemplazar=a.reemplazar)

    if det.es_demo:
        print("DEMOSTRACIÓN: modelo entrenado con epicrisis inventadas, sin valor clínico.")
    print(f"Modelo: {det.tipo} · SHA-256 {det.sha256[:16]}…")
    print(f"Filas: {len(eps)} · procesadas: {len(dets)} · marcadas: {sum(d.marca for d in dets)} · "
          f"posible evento sin código: {sum(d.marca and d.sin_codigo for d in dets)} · "
          f"no cumplen el contrato: {len(errores)}")
    for id_epicrisis, motivos in errores.items():
        print(f"  {id_epicrisis}: {' '.join(motivos)}")
    if a.guardar:
        print(f"Guardadas en la base local: {len(dets) - len(omitidas)}"
              + (f" · ya existían y no se reemplazaron: {', '.join(omitidas)}" if omitidas else ""))
    print(f"Salida: {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
