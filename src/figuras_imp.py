# -*- coding: utf-8 -*-
"""
Figuras de la importancia de variables (semana 7). Cada figura lleva al pie qué es el eje x, qué es el eje y y qué muestra,
con las cifras de la corrida, y cierra con lo que eso implica.

Las llama src/importancia_variables.py al terminar. También se redibujan sin volver a entrenar: python src/figuras_imp.py
El cuaderno notebooks/05_sprint3_semana7_importancia_variables.ipynb usa las mismas funciones.
Solo usa términos que ya pasaron el filtro de privacidad (reportes/importancia_terminos.csv).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figuras_hp import cerrar, con_pie
from utils import REPORTES

AZUL, ROJO, GRIS, VERDE = "#1F4E79", "#9B2226", "#8A8A8A", "#4F7942"
NOMBRE = {"largo de la nota": "largo de la nota", "estancia_dias": "estancia (días)", "servicio": "servicio al ingreso",
          "servicio_alta": "servicio al alta", "destino_alta": "destino al alta", "n_cultivos": "n.º de cultivos",
          "n_medicamentos": "n.º de medicamentos", "tipo_ingreso": "tipo de ingreso", "procedencia": "procedencia",
          "horas_emergencia": "horas en emergencia", "n_unidades": "n.º de unidades", "dias_uci": "días en UCI",
          "n_servicios": "n.º de servicios", "n_traslados": "n.º de traslados", "servicio_quirurgico": "servicio quirúrgico",
          "edad": "edad", "sexo": "sexo", "seguro": "seguro", "uci": "paso por UCI"}


def fig_bloques(R):
    B, C = R["importancia_bloques"], R["comparacion"]
    nombres = list(B)
    a, b, c, ab = (B[k] for k in nombres)
    suma = a["media"] + b["media"]
    orden = sorted(nombres[:3], key=lambda k: -B[k]["media"])
    corto = {nombres[0]: "A, de las palabras", nombres[1]: "B, de los bigramas", nombres[2]: "C, del contexto"}
    txt1 = (f"Gráfico izquierdo. Eje x: caída de la PR-AUC de validación al permutar el bloque entre las epicrisis del fold "
            f"({R['parametros']['n_repeticiones']} permutaciones por fold; la barra es la media de 5 folds, la línea negra su desviación "
            f"estándar y los puntos, cada fold). Eje y: bloque de atributos. El bloque de mayor peso es {corto[orden[0]]} "
            f"({B[orden[0]]['media']:.4f} ± {B[orden[0]]['std']:.4f}), seguido de {corto[orden[1]]} ({B[orden[1]]['media']:.4f} ± "
            f"{B[orden[1]]['std']:.4f}) y de {corto[orden[2]]} ({B[orden[2]]['media']:.4f} ± {B[orden[2]]['std']:.4f}). Al permutar A y B juntos la caída es {ab['media']:.4f}, "
            + ("mayor" if ab["media"] > suma else "menor") + f" que la suma de las dos por separado ({suma:.4f}). "
            + "Esto implica que la mayor parte de la señal está en el texto. A criterio del autor, que la caída conjunta "
            + ("supere a la suma es coherente con que palabras y bigramas compartan información, aunque no lo demuestra."
               if ab["media"] > suma else "no supere a la suma no permite afirmar que palabras y bigramas compartan información."))
    d6 = max(abs(a["media"] - C["permutacion_semana6"]["A"]), abs(b["media"] - C["permutacion_semana6"]["B"]))
    menor = C["ablacion_semana5"]["B"] < b["media"] and C["ablacion_semana5"]["C"] < c["media"]
    txt2 = ("Gráfico derecho. Eje x: bloque. Eje y: caída de la PR-AUC según tres mediciones, que no son directamente comparables: la "
            "permutación de esta semana, la de la semana 6 (3 permutaciones por variable, sin el bloque C completo) y la ablación de la "
            "semana 5 (se quita el bloque y se reentrena, con los datos D0 y todas las filas de validación). "
            f"La permutación de la semana 6 da {C['permutacion_semana6']['A']:.4f} para A y {C['permutacion_semana6']['B']:.4f} para B, y la "
            f"ablación da {C['ablacion_semana5']['B']:.4f} para B y {C['ablacion_semana5']['C']:.4f} para C. Esto implica que las dos "
            "permutaciones " + ("coinciden" if d6 < 0.01 else "difieren") + (
            ", y que la ablación da valores mucho menores, lo cual es coherente con que, al reentrenar sin un bloque, los otros "
            "recuperan buena parte de su información." if menor else ", y que la ablación no da valores menores que la permutación en todos los bloques."))
    fig, pie, fr = con_pie(12.5, 4.2, [txt1, txt2], 175)
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    yy = np.arange(len(nombres))[::-1]
    a1.barh(yy, [B[k]["media"] for k in nombres], xerr=[B[k]["std"] for k in nombres], color=[AZUL, AZUL, VERDE, GRIS], capsize=3)
    for y_, k in zip(yy, nombres):
        a1.scatter(B[k]["por_fold"], np.full(len(B[k]["por_fold"]), y_), s=10, color="k", zorder=3)
        a1.text(B[k]["media"] + B[k]["std"] + 0.003, y_, f"{B[k]['media']:.4f}", va="center", fontsize=8)
    a1.set_yticks(yy, nombres); a1.set_xlim(0, max(B[k]["media"] for k in nombres) * 1.22)
    a1.set(xlabel="caída de la PR-AUC al permutar el bloque"); a1.set_title("Importancia por bloque (5 folds)", fontsize=10)
    a1.grid(alpha=0.3, axis="x")
    x = np.arange(3)
    est = [a["media"], b["media"], c["media"]]
    s6 = [C["permutacion_semana6"]["A"], C["permutacion_semana6"]["B"], np.nan]
    ab5 = [np.nan, C["ablacion_semana5"]["B"], C["ablacion_semana5"]["C"]]
    a2.bar(x - 0.27, est, 0.26, color=AZUL, label="permutación, esta semana (D2)")
    a2.bar(x, s6, 0.26, color="#9FB8CF", label="permutación, semana 6 (D2)")
    a2.bar(x + 0.27, ab5, 0.26, color=ROJO, label="ablación, semana 5 (D0)")
    for xx, v in zip(x - 0.27, est):
        a2.text(xx, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    a2.set_xticks(x, ["A · palabras", "B · bigramas", "C · contexto"])
    a2.set(ylabel="caída de la PR-AUC", ylim=(0, max(est) * 1.5)); a2.set_title("Comparación con las semanas 5 y 6", fontsize=10)
    a2.legend(fontsize=7.5, loc="upper center"); a2.grid(alpha=0.3, axis="y")
    return cerrar(fig, pie, fr)


def fig_contexto(R):
    V = R["importancia_contexto"]
    s6 = R["comparacion"]["permutacion_semana6"]["variables_de_C"]
    k = list(V)[:12][::-1]
    top = list(V)[:3]
    casi_cero = sum(1 for v in V.values() if v["media"] < 0.0005)
    suma_c = sum(v["media"] for v in V.values())
    bloque_c = R["importancia_bloques"]["C · contexto"]["media"]
    txt = (f"Eje x: caída de la PR-AUC de validación al permutar la variable de contexto (sus columnas one-hot y su marca de faltante se "
           f"permutan juntas; media de 5 folds y desviación estándar entre folds). Eje y: las {len(k)} variables de C con mayor caída; "
           f"los rombos son la permutación de la semana 6. Las tres primeras son "
           + ", ".join(f"{NOMBRE.get(t, t)} ({V[t]['media']:.4f})" for t in top)
           + f", y {casi_cero} de las {len(V)} variables quedan por debajo de 0.0005. La suma de las caídas de todas las variables por "
           f"separado es {suma_c:.4f}, " + ("menor" if suma_c < bloque_c else "mayor") + f" que la del bloque C completo ({bloque_c:.4f}). "
           "Esto implica que, dentro del contexto, el modelo se apoya en pocas variables, y la permutación no dice que esas variables "
           "causen el evento. " + ("A criterio del autor, que el bloque completo caiga más que la suma es coherente con que las variables "
           "compartan información, aunque no lo demuestra." if suma_c < bloque_c else ""))
    fig, pie, fr = con_pie(9.5, 4.6, [txt], 125)
    ax = fig.subplots()
    yy = np.arange(len(k))
    ax.barh(yy, [V[t]["media"] for t in k], xerr=[V[t]["std"] for t in k], color=VERDE, capsize=3, label="esta semana (30 permutaciones)")
    ax.scatter([s6.get(t, np.nan) for t in k], yy, marker="D", s=22, color=ROJO, zorder=3, label="semana 6 (3 permutaciones)")
    ax.set_yticks(yy, [NOMBRE.get(t, t) for t in k])
    ax.set(xlabel="caída de la PR-AUC al permutar la variable (media de 5 folds)")
    ax.set_title("Importancia de las variables de contexto (bloque C)", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return cerrar(fig, pie, fr)


def fig_terminos(R, T):
    E, top = R["estabilidad_coeficientes"], R["parametros"]["top_terminos"]
    pos = {blq: E[blq]["positivo"]["en_los_5_folds"] for blq in ("A", "B")}
    neg = {blq: E[blq]["negativo"]["en_los_5_folds"] for blq in ("A", "B")}
    txt = (f"Eje x: coeficiente de la regresión logística del término (media de 5 folds y desviación estándar entre folds); a la "
           f"derecha, en cuántos folds el término estuvo entre los {top} de mayor coeficiente de ese signo. Eje y: los términos de mayor "
           f"coeficiente positivo (empujan hacia evento) y negativo (empujan hacia sin evento) que pasaron el filtro de privacidad. "
           f"Entre folds, la correlación de rangos de los coeficientes es {E['A']['spearman_coeficientes']['media']:.2f} en las palabras y "
           f"{E['B']['spearman_coeficientes']['media']:.2f} en los bigramas. De los {top} términos de mayor coeficiente positivo se repiten "
           f"en los 5 folds {pos['A']} en las palabras y {pos['B']} en los bigramas, frente a {neg['A']} y {neg['B']} de los {top} de "
           f"coeficiente negativo. Esto implica que "
           + ("los términos asociados al evento se repiten de un fold a otro, mientras que los asociados a la ausencia de evento cambian"
              if min(pos.values()) > 2 * max(neg.values()) else "la estabilidad de los extremos es parecida en los dos signos")
           + "; además, cada coeficiente depende de la escala TF-IDF y de los términos con que se correlaciona, de modo que se lee como "
           "asociación del modelo y no como efecto de la palabra.")
    fig, pie, fr = con_pie(13, 8.2, [txt], 185)
    axs = fig.subplots(2, 2)
    for i, blq in enumerate(("A", "B")):
        for j, (signo, col) in enumerate((("positivo", ROJO), ("negativo", AZUL))):
            ax = axs[i, j]
            t = T[(T.bloque == blq) & (T.signo == signo)].sort_values("puesto", ascending=False)
            yy = np.arange(len(t))
            ax.barh(yy, t.coef_media, xerr=t.coef_std, color=col, capsize=2)
            ax.set_yticks(yy, t.termino, fontsize=7.5)
            lim = max(abs(t.coef_media).max() + t.coef_std.max(), 0.1) * 1.3
            ax.set_xlim(0, lim) if signo == "positivo" else ax.set_xlim(-lim, 0)
            for y_, r in zip(yy, t.itertuples()):
                ax.text(lim * 0.98 if signo == "positivo" else -lim * 0.98, y_, f"{r.folds_en_top}/5", va="center",
                        ha="right" if signo == "positivo" else "left", fontsize=7, color="#444")
            ax.set_title(f"{'A · palabras' if blq == 'A' else 'B · bigramas'}: coeficiente {signo}", fontsize=9.5, loc="left")
            ax.grid(alpha=0.3, axis="x")
            if i == 1:
                ax.set_xlabel("coeficiente (media de 5 folds)")
    return cerrar(fig, pie, fr)


def dibujar(R, T, carpeta=None):
    carpeta = carpeta or REPORTES / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, f in (("fig_imp_bloques.png", lambda: fig_bloques(R)), ("fig_imp_contexto.png", lambda: fig_contexto(R)),
                      ("fig_imp_terminos.png", lambda: fig_terminos(R, T))):
        fig = f()
        fig.savefig(carpeta / nombre, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    dibujar(json.loads((REPORTES / "importancia_resultados.json").read_text(encoding="utf-8")),
            pd.read_csv(REPORTES / "importancia_terminos.csv", keep_default_na=False))
    print("figuras redibujadas en", REPORTES / "figuras")
