# -*- coding: utf-8 -*-
"""
Figuras de la semana 6 (variantes de los datos y diagnósticos). Cada figura lleva al pie qué es el eje x, qué es
el eje y y qué muestra, con las cifras de la corrida.

Las llama src/experimentos_datos.py al terminar. También se pueden redibujar sin volver a entrenar, desde los
resultados guardados:  python src/figuras_datos.py
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import variantes_datos as vd
from utils import REPORTES

FIG = REPORTES / "figuras"
COLOR = {"D0": "#9FB8CF", "D1": "#1F4E79", "D2": "#9B2226"}
NOMBRE = {"texto: palabras": "palabras del texto", "texto: bigramas": "bigramas del texto", "largo de la nota": "largo de la nota",
          "estancia_dias": "estancia (días)", "servicio": "servicio al ingreso", "servicio_alta": "servicio al alta",
          "destino_alta": "destino al alta", "n_cultivos": "n.º de cultivos", "n_medicamentos": "n.º de medicamentos",
          "tipo_ingreso": "tipo de ingreso", "procedencia": "procedencia", "horas_emergencia": "horas en emergencia",
          "n_unidades": "n.º de unidades", "dias_uci": "días en UCI", "n_servicios": "n.º de servicios", "n_traslados": "n.º de traslados",
          "servicio_quirurgico": "servicio quirúrgico", "edad": "edad", "sexo": "sexo", "seguro": "seguro", "uci": "paso por UCI"}
ALTO_LINEA = 0.165  # pulgadas por línea del pie (8.4 pt con interlineado 1.35)


def con_pie(ancho, alto_graficos, parrafos, columnas, ncols=1):
    """Crea la figura dejando abajo el espacio justo para el pie."""
    lineas = [l for p in parrafos for l in textwrap.wrap(p, columnas)]
    alto_pie = ALTO_LINEA * (len(lineas) + len(parrafos) - 1) + 0.3
    fig, ax = plt.subplots(1, ncols, figsize=(ancho, alto_graficos + alto_pie))
    return fig, ax, "\n\n".join("\n".join(textwrap.wrap(p, columnas)) for p in parrafos), alto_pie / (alto_graficos + alto_pie)


def cerrar(fig, texto, fraccion, nombre):
    fig.tight_layout(rect=(0, fraccion, 1, 1))
    fig.text(0.012, 0.012, texto, ha="left", va="bottom", fontsize=8.4, color="#222", linespacing=1.35)
    fig.savefig(FIG / nombre, dpi=130)
    plt.close(fig)


def mil(n):
    return f"{int(n):,}".replace(",", " ")


def veces(n, total=5):
    return f"en los {total} folds" if n == total else f"en {n} de {total} folds"


def dibujar(R, folds_df):
    FIG.mkdir(parents=True, exist_ok=True)
    el = R["elegida_por_cv"]

    # 1 · PR-AUC por fold y diferencia con D0
    F = folds_df.pivot(index="fold", columns="variante", values="pr_auc_comun")[list(vd.VARIANTES)]
    D = F[["D1", "D2"]].sub(F["D0"], axis=0)
    m, arriba = F.mean(), int((D[el] > 0).sum())
    izq = (f"Gráfico izquierdo. Eje x: fold de la validación cruzada (1 a {len(F)}). Eje y: PR-AUC de cada variante, medida sobre las "
           f"mismas filas de validación. {el} queda por encima de D0 {veces(arriba, len(F))} (media {m[el]:.4f} frente a {m['D0']:.4f}); "
           f"D1 casi coincide con D0 (media {m['D1']:.4f}).")
    der = (f"Gráfico derecho. Eje x: fold. Eje y: PR-AUC de la variante menos la de D0 en el mismo fold; sobre cero, la variante "
           f"mejora. Las barras de D2 van de {D['D2'].min():+.4f} a {D['D2'].max():+.4f} y las de D1 de {D['D1'].min():+.4f} a "
           f"{D['D1'].max():+.4f}: quitar solo las negativas dudosas casi no cambia el resultado; quitar además las positivas dudosas, sí.")
    fig, ax, pie, fr = con_pie(12.5, 4.4, [izq, der], 175, 2)
    for v in F:
        ax[0].plot(F.index, F[v], "o-", color=COLOR[v], label=f"{v}: {vd.VARIANTES[v]}")
    ax[0].set(xlabel="fold", ylabel="PR-AUC (mismas filas de validación)", xticks=list(F.index))
    lo, hi = F.min().min(), F.max().max()
    ax[0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    ax[0].set_title("PR-AUC de cada fold por variante de datos", fontsize=10); ax[0].legend(fontsize=8, loc="upper center")
    xx = np.arange(len(D))
    ax[1].bar(xx - 0.2, D["D1"], 0.4, color=COLOR["D1"], label="D1 − D0"); ax[1].bar(xx + 0.2, D["D2"], 0.4, color=COLOR["D2"], label="D2 − D0")
    ax[1].axhline(0, c="gray", lw=0.8); ax[1].set_xticks(xx, D.index)
    ax[1].set(xlabel="fold", ylabel="diferencia de PR-AUC con D0"); ax[1].set_title("Diferencia con los datos actuales en el mismo fold", fontsize=10)
    ax[1].legend(fontsize=8)
    cerrar(fig, pie, fr, "fig_datos_folds.png")

    # 2 · curva de aprendizaje y calibración
    c = pd.DataFrame(R["curva_de_aprendizaje"]["puntos"])
    brecha, sube = c.pr_auc_train.iloc[-1] - c.pr_auc_validacion.iloc[-1], c.pr_auc_validacion.iloc[-1] - c.pr_auc_validacion.iloc[-2]
    izq = (f"Gráfico izquierdo. Eje x: filas con que se entrena en el fold 1 (del {c.fraccion.iloc[0] * 100:.0f} % al {c.fraccion.iloc[-1] * 100:.0f} %, "
           f"de {mil(c.filas.iloc[0])} a {mil(c.filas.iloc[-1])}). Eje y: PR-AUC. Con más filas, la de train baja de {c.pr_auc_train.iloc[0]:.4f} "
           f"a {c.pr_auc_train.iloc[-1]:.4f} y la de validación sube de {c.pr_auc_validacion.iloc[0]:.4f} a {c.pr_auc_validacion.iloc[-1]:.4f}: "
           f"el modelo se pega menos a los casos de entrenamiento y generaliza mejor. Al final quedan {brecha:.3f} entre las dos curvas "
           f"({'una brecha grande, señal de varianza' if brecha > 0.08 else 'una brecha moderada'}) y la validación "
           + (f"todavía sube (+{sube:.4f} en el último tramo), así que más datos aún ayudarían." if sube > 0.002
              else f"ya casi no sube (+{sube:.4f} en el último tramo), así que más filas del mismo tipo ayudarían poco."))
    cal = R["calibracion"]
    dist = {v: np.abs(np.array(k["frecuencia_real"]) - np.array(k["prob_media"])) for v, k in cal.items()}
    vmax = max(dist, key=lambda v: dist[v].max())
    i = int(np.argmin(np.abs(np.array(cal[el]["prob_media"]) - 0.35)))
    der = (f"Gráfico derecho. Eje x: probabilidad de evento que da el modelo, agrupada en tramos de 0.1. Eje y: proporción de esas "
           f"notas que de verdad tenían evento. Los puntos de {' y '.join(cal)} quedan sobre la diagonal (la mayor distancia es "
           f"{dist[vmax].max():.3f}, en {vmax}): por ejemplo, donde {el} da en promedio {cal[el]['prob_media'][i]:.2f}, tuvo evento el "
           f"{cal[el]['frecuencia_real'][i] * 100:.0f} % de las notas. Brier: " + ", ".join(f"{k['brier']:.4f} con {v}" for v, k in cal.items())
           + " (más bajo es mejor). Esto vale para la proporción de eventos del corpus de estudio; para la prevalencia real hay que recalibrar.")
    fig, ax, pie, fr = con_pie(12.5, 4.4, [izq, der], 175, 2)
    ax[0].plot(c.filas, c.pr_auc_train, "o-", color="#1F4E79", label="train")
    ax[0].plot(c.filas, c.pr_auc_validacion, "s-", color="#9B2226", label="validación")
    ax[0].set(xlabel="filas de entrenamiento", ylabel="PR-AUC"); ax[0].set_title(f"Curva de aprendizaje ({el}, fold 1)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot([0, 1], [0, 1], ls="--", c="gray", label="calibración perfecta")
    for v, k in cal.items():
        ax[1].plot(k["prob_media"], k["frecuencia_real"], "o-", color=COLOR[v], label=f"{v} (Brier {k['brier']:.3f})")
    ax[1].set(xlabel="probabilidad que da el modelo", ylabel="proporción real de eventos", xlim=(0, 1), ylim=(0, 1))
    ax[1].set_title("Calibración (predicciones fuera de fold)", fontsize=10); ax[1].legend(fontsize=8)
    cerrar(fig, pie, fr, "fig_datos_diagnosticos.png")

    # 3 · importancia por permutación
    imp, est = R["importancia"][el], R["importancia_veces_en_top5"]
    I = pd.DataFrame(imp).T.sort_values("media", ascending=True).tail(12)
    orden = list(I.index[::-1])
    hosp = [k for k in orden if not k.startswith("texto") and k != "largo de la nota"][:2]
    fijos = next((n for n, k in enumerate(orden[:5]) if est.get(k, 0) != 5), 5)
    txt = (f"Eje x: cuánto baja la PR-AUC de validación al barajar los valores de ese grupo entre las notas (media de 5 folds; la línea "
           f"negra es la desviación entre folds). Eje y: los {len(I)} grupos de atributos con mayor caída. Las palabras del texto son lo "
           f"que más pesa: si se barajan, la PR-AUC baja {imp['texto: palabras']['media']:.4f}. Les siguen los bigramas "
           f"({imp['texto: bigramas']['media']:.4f}) y, entre las variables de la hospitalización, "
           + " y ".join(f"{NOMBRE.get(k, k).replace(' (días)', ' en días')} ({imp[k]['media']:.4f})" for k in hosp)
           + ". Al lado de cada barra, en cuántos de los 5 folds el grupo quedó entre los cinco primeros: "
           + ("los cinco primeros, en todos" if fijos == 5 else f"los {fijos} primeros, en todos; {NOMBRE.get(orden[4], orden[4])}, en {est.get(orden[4], 0)}")
           + ", así que el orden es estable entre folds.")
    fig, ax, pie, fr = con_pie(9.5, 5.0, [txt], 125)
    ax.barh([NOMBRE.get(k, k) for k in I.index], I.media, xerr=I["std"], color="#1F4E79", capsize=3)
    for j, k in enumerate(I.index):
        ax.text(I.media.iloc[j] + I["std"].iloc[j] + 0.0005, j, f"top 5 en {est.get(k, 0)}/5", va="center", fontsize=8)
    ax.set(xlabel="caída de la PR-AUC al barajar el grupo (media de 5 folds)")
    ax.set_title(f"Importancia por permutación ({el})", fontsize=10)
    cerrar(fig, pie, fr, "fig_datos_importancia.png")


if __name__ == "__main__":
    dibujar(json.loads((REPORTES / "datos_resultados.json").read_text(encoding="utf-8")), pd.read_csv(REPORTES / "datos_folds.csv"))
    print("figuras redibujadas en", FIG)
