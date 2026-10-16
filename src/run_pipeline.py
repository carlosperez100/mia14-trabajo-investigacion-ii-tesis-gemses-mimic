# -*- coding: utf-8 -*-
"""
Orquestador del pipeline: ingesta -> preprocesado -> eda -> baseline -> features.

Uso:
  python src/run_pipeline.py                      # todo
  python src/run_pipeline.py --desde preprocesado
  python src/run_pipeline.py --hasta ingesta
  python src/run_pipeline.py --desde features         # solo los experimentos de feature engineering
  python src/run_pipeline.py --sin-referencia     # baseline sin el modelo de la Fase 9
"""
import argparse
import sys
import time

import baseline
import eda
import experimentos_fe
import ingesta
import preprocesado

ETAPAS = ["ingesta", "preprocesado", "eda", "baseline", "features"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", choices=ETAPAS, default=ETAPAS[0])
    ap.add_argument("--hasta", choices=ETAPAS, default=ETAPAS[-1])
    ap.add_argument("--sin-referencia", action="store_true")
    a = ap.parse_args()
    elegidas = ETAPAS[ETAPAS.index(a.desde):ETAPAS.index(a.hasta) + 1]
    for etapa in elegidas:
        t0 = time.time()
        print(f"\n===== {etapa.upper()} =====", flush=True)
        if etapa == "ingesta":
            ingesta.main()
        elif etapa == "preprocesado":
            preprocesado.main()
        elif etapa == "eda":
            eda.main()
        elif etapa == "features":
            experimentos_fe.main()
        else:
            sys.argv = [sys.argv[0]] + (["--sin-referencia"] if a.sin_referencia else [])
            baseline.main()
        print(f"===== {etapa} OK en {time.time()-t0:.0f} s =====", flush=True)


if __name__ == "__main__":
    main()
