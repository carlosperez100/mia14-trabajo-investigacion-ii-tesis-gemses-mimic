# -*- coding: utf-8 -*-
"""
REPORTE DEL BASELINE — escribe logs/metrics_baseline.txt y la figura de matrices de confusión
a partir de reportes/baseline_resultados.json (no reentrena nada).

Lo llama baseline.py al terminar; también se puede correr solo:
    python src/reporte_baseline.py
"""
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decimal import ROUND_HALF_UP, Decimal

from utils import LOGS, RAIZ, REPORTES, cargar_config


def r3(x):
    """Redondeo a 3 decimales «hacia arriba en el medio» (0.7775 -> 0.778), igual que en el README."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

NOMBRES = {"dummy_mayoritaria": "Dummy (más frecuente)", "dummy_estratificada": "Dummy (estratificado)",
           "logistica_tfidf": "Regresión logística TF-IDF", "referencia_fase9": "Referencia tesis (LinearSVC)"}


def main():
    r = json.loads((REPORTES / "baseline_resultados.json").read_text(encoding="utf-8"))
    pre = json.loads((REPORTES / "preprocesado_resumen.json").read_text(encoding="utf-8"))
    cfg = cargar_config()
    LOGS.mkdir(exist_ok=True)

    L = []
    L.append(f"# metrics_baseline.txt · generado {datetime.now():%Y-%m-%d %H:%M:%S} desde reportes/baseline_resultados.json")
    L.append("tarea: detección binaria de evento adverso en epicrisis (etiqueta A1 por CIE)")
    L.append(f"semilla: {r['semilla']} · split: GroupShuffleSplit por paciente, test_size {cfg['preprocesado']['test_size']}"
             f" · sha256 splits.csv: {pre['sha256_splits'][:16]}…")
    L.append(f"n_train: {r['n_train']} · n_test: {r['n_test']} · positivas en test: "
             f"{pre['conteos']['particion']['positivas_test']} · prevalencia real: {r['prevalencia_real']}")
    L.append(f"hiperparámetros: {json.dumps(cfg['baseline'], ensure_ascii=False)}")
    L.append("")
    L.append(f"{'modelo':<30}{'F1(+)':>8}{'Recall':>8}{'Precis':>8}{'PR-AUC':>8}{'ROC-AUC':>9}{'Especif':>9}{'VPP real':>10}")
    for k, m in r["modelos"].items():
        L.append(f"{NOMBRES.get(k, k):<30}{m['f1']:>8.4f}{m['sensibilidad']:>8.4f}{m['precision']:>8.4f}"
                 f"{m['pr_auc']:>8.4f}{m['auc']:>9.4f}{m['especificidad']:>9.4f}{m['vpp_prevalencia_real']:>10.4f}")
    L.append("")
    L.append("IC 95 % (bootstrap, 200 remuestreos):")
    for k, m in r["modelos"].items():
        L.append(f"  {NOMBRES.get(k, k):<30} recall {m['sensibilidad_ic95']} · ROC-AUC {m['auc_ic95']}")
    L.append("")
    L.append("matrices de confusión (test):")
    for k, m in r["modelos"].items():
        c = m["matriz"]
        L.append(f"  {NOMBRES.get(k, k):<30} VN {c['vn']:>5}  FP {c['fp']:>5}  FN {c['fn']:>5}  VP {c['vp']:>5}")
    v = r.get("verificacion_reproducibilidad")
    if v:
        L.append("")
        L.append(f"verificación de reproducibilidad (Fase 9): {'REPRODUCE' if v['reproduce'] else 'NO REPRODUCE'} · "
                 f"diferencias {v['diferencia_abs']} · tolerancia {v['tolerancia']}")
    (LOGS / "metrics_baseline.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    # figura: matrices de confusión normalizadas por fila
    mods = list(r["modelos"].items())
    fig, axs = plt.subplots(1, len(mods), figsize=(3.6 * len(mods), 3.6))
    for ax, (k, m) in zip(np.atleast_1d(axs), mods):
        c = m["matriz"]
        M = np.array([[c["vn"], c["fp"]], [c["fn"], c["vp"]]])
        P = M / M.sum(axis=1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}\n({P[i, j]:.0%})", ha="center", va="center",
                        color="white" if P[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred. 0", "pred. 1"]); ax.set_yticks([0, 1], ["real 0", "real 1"])
        ax.set_title(f"{NOMBRES.get(k, k)}\nF1(+) {r3(m['f1'])} · PR-AUC {r3(m['pr_auc'])}", fontsize=9)
    fig.suptitle("Matrices de confusión en test (partición por paciente)", fontsize=10)
    fig.tight_layout()
    (REPORTES / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTES / "figuras" / "fig_matrices_confusion.png", dpi=160)
    print(f"-> {(LOGS / 'metrics_baseline.txt').relative_to(RAIZ)} · reportes/figuras/fig_matrices_confusion.png")


if __name__ == "__main__":
    main()
