# -*- coding: utf-8 -*-
"""
VARIANTE D3 (29-10-2026) — perfil de las hospitalizaciones liberadas frente a las del corpus de la época CIE-10.

Describe, sin entrenar ningún modelo, cómo son las liberadas (src/negativas_liberadas.py) frente a las negativas y a las
positivas de la época CIE-10 del corpus. Los tres grupos van sin etiqueta dudosa, como en D2 y D3, e incluyen train y
test, como el EDA. Variables: largo de la epicrisis en caracteres (el mismo que usa el bloque C), estancia en días,
edad, número de medicamentos distintos y paso por UCI. Los conteos menores de 10 se escriben «<10», como en la tabla
por evento del EDA (src/eda_variables.py).

Salida: reportes/d3_perfil_liberadas.json
Uso:  python src/perfil_liberadas.py   (después de src/negativas_liberadas.py)
"""
from datetime import datetime

import pandas as pd

import variables_dominio
import variantes_datos as vd
from utils import INTERIM, PROCESSED, REPORTES, cargar_config, guardar_json, logger

ETAPA = "perfil_liberadas"
MEDIANAS = {"largo_epicrisis_caracteres": "largo", "estancia_dias": "estancia_dias", "edad": "edad",
            "medicamentos_distintos": "n_medicamentos"}


def celda(n):
    return "<10" if 0 < n < 10 else int(n)


def perfil(d):
    r = {"hospitalizaciones": celda(len(d))}
    r.update({k: {"mediana": round(float(d[c].median()), 2), "p25": round(float(d[c].quantile(0.25)), 2),
                  "p75": round(float(d[c].quantile(0.75)), 2), "sin_dato": celda(d[c].isna().sum())} for k, c in MEDIANAS.items()})
    r["proporcion_con_uci"] = round(float(d.uci.mean()), 4)
    return r


def main():
    log = logger(ETAPA)
    S = cargar_config()["semilla"]
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    lib = pd.read_parquet(INTERIM / "negativas_liberadas.parquet")
    sin_dudosa = lambda d: d[~(d.negativa_dudosa | d.positiva_dudosa)]
    grupos = {"liberadas": sin_dudosa(lib),
              "negativas_epoca10_corpus": sin_dudosa(df[(df.y == 0) & (df.epoca == 10)]),
              "positivas_epoca10_corpus": sin_dudosa(df[(df.y == 1) & (df.epoca == 10)])}
    R = {"etapa": ETAPA, "generado": datetime.now().isoformat(timespec="seconds"), "semilla": S,
         "nota": "medianas y percentiles 25 y 75; grupos sin etiqueta dudosa, train y test juntos",
         "grupos": {}}
    for g, d in grupos.items():
        d = d.assign(largo=d.text.str.len())
        R["grupos"][g] = perfil(d)
        p = R["grupos"][g]
        log.info(f"{g}: {p['hospitalizaciones']:,} · largo {p['largo_epicrisis_caracteres']['mediana']:,.0f} · estancia "
                 f"{p['estancia_dias']['mediana']} días · edad {p['edad']['mediana']} · medicamentos {p['medicamentos_distintos']['mediana']} · "
                 f"UCI {p['proporcion_con_uci']:.4f}")
    guardar_json(R, REPORTES / "d3_perfil_liberadas.json")
    log.info("Salida: reportes/d3_perfil_liberadas.json")


if __name__ == "__main__":
    main()
