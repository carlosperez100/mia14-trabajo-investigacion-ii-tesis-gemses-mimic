# -*- coding: utf-8 -*-
"""
Figuras de la semana 7 (búsqueda de hiperparámetros). Cada figura lleva al pie qué es el eje x, qué es el eje y y qué
muestra, con las cifras de la corrida, y cierra con lo que eso implica.

Las llama src/experimentos_hp.py al terminar. También se pueden redibujar sin volver a entrenar, desde los resultados
guardados:  python src/figuras_hp.py
El cuaderno notebooks/04_sprint3_semana7_hiperparametros.ipynb usa las mismas funciones para mostrarlas.
"""
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import REPORTES

COLOR = {"aleatoria": "#1F4E79", "bayesiana": "#9B2226", "actual": "#8A8A8A", "candidata": "#4F7942"}
MARCA = {"aleatoria": "o", "bayesiana": "^"}
NOMBRE = {"A_max_features": "A: máximo de palabras", "A_min_df": "A: min_df (palabras)",
          "B_max_features": "B: máximo de bigramas", "B_min_df": "B: min_df (bigramas)",
          "class_weight": "peso de las clases", "sublinear_tf": "frecuencia sublineal"}
ALTO_LINEA = 0.165  # pulgadas por línea del pie (8.4 pt con interlineado 1.35)
TRAMOS_C = [0, 0.01, 0.1, 0.3, 3, 100]   # tramos de C para resumir su efecto (también en el cuaderno 04)


def con_pie(ancho, alto_graficos, parrafos, columnas, **kw):
    lineas = [l for p in parrafos for l in textwrap.wrap(p, columnas, break_on_hyphens=False)]
    alto_pie = ALTO_LINEA * (len(lineas) + len(parrafos) - 1) + 0.3
    fig = plt.figure(figsize=(ancho, alto_graficos + alto_pie), **kw)
    return fig, "\n\n".join("\n".join(textwrap.wrap(p, columnas, break_on_hyphens=False)) for p in parrafos), alto_pie / (alto_graficos + alto_pie)


def cerrar(fig, texto, fraccion):
    fig.tight_layout(rect=(0, fraccion, 1, 1))
    fig.text(0.012, 0.012, texto, ha="left", va="bottom", fontsize=8.4, color="#222", linespacing=1.35)
    return fig


def unicos(ens):
    """Un punto por configuración evaluada (los ensayos reutilizados no se repiten)."""
    return ens[ens.reutilizado_de.fillna("") == ""]


def fig_busqueda(R, ens):
    ref = R["referencia_actual_busqueda"]["pr_auc_media"]
    k0, N = R["parametros"]["ensayos_iniciales_tpe"], R["parametros"]["n_ensayos"]
    acum = {m: g.sort_values("ensayo").pr_auc_media.cummax().values for m, g in ens.groupby("metodo", sort=False)}
    mej = R["mejor_por_metodo"]
    post = {m: g[g.ensayo > k0].pr_auc_media for m, g in ens.groupby("metodo", sort=False)}
    sobre = {m: int((g.pr_auc_media > ref).sum()) for m, g in ens.groupby("metodo", sort=False)}
    ganador = max(mej, key=lambda m: mej[m]["pr_auc_media"])
    otro = [m for m in mej if m != ganador][0]
    dif_ref = mej[ganador]["pr_auc_media"] - ref
    izq = (f"Gráfico izquierdo. Eje x: número de ensayo de cada método, del 1 al {N}. Eje y: la mejor PR-AUC media de 3 folds obtenida "
           f"hasta ese ensayo. Los dos métodos comparten los {k0} primeros ensayos, sorteados al azar con la misma semilla; desde el ensayo "
           f"{k0 + 1} la bayesiana elige cada configuración a partir de los resultados anteriores. La línea gris es la configuración actual "
           f"medida en los mismos 3 folds ({ref:.4f}). La {ganador} llega a {mej[ganador]['pr_auc_media']:.4f} en el ensayo "
           f"{mej[ganador]['ensayo']} y la {otro} a {mej[otro]['pr_auc_media']:.4f} en el ensayo {mej[otro]['ensayo']}, de modo que la "
           f"mejor queda {abs(dif_ref):.4f} {'por encima' if dif_ref > 0 else 'por debajo'} de la actual. "
           + (f"Esto implica que, con este presupuesto, la búsqueda encuentra configuraciones apenas por encima de la actual en los 3 folds, "
              "y que la diferencia debe confirmarse con los 5 folds antes de tomarla como mejora." if dif_ref > 0 else
              "Esto implica que, con este presupuesto, ninguna de las dos búsquedas supera a la configuración actual en los 3 folds."))
    der = (f"Gráfico derecho. Eje x: número de ensayo. Eje y: PR-AUC media de 3 folds de cada ensayo (un punto por ensayo; la escala se "
           f"corta abajo para que se lean las diferencias, y los ensayos que quedan por debajo se marcan sobre el borde). Después de los "
           f"ensayos iniciales, la mediana de la bayesiana es {post['bayesiana'].median():.4f} y la de la aleatoria "
           f"{post['aleatoria'].median():.4f}, y "
           + ("ninguno de los ensayos de los dos métodos supera a la actual. " if not any(sobre.values()) else
              f"superan a la actual {sobre['bayesiana']} de {N} ensayos de la bayesiana y {sobre['aleatoria']} de {N} de la aleatoria. ")
           + ("Esto implica que, en esta corrida y con una sola semilla, la bayesiana concentró sus ensayos en la zona del espacio donde "
              "la PR-AUC es más alta, mientras que la aleatoria repartió el presupuesto por todo el espacio."
              if post["bayesiana"].median() > post["aleatoria"].median() else
              "Esto implica que, en esta corrida y con una sola semilla, la bayesiana no concentró sus ensayos en una zona mejor que la "
              "aleatoria."))
    fig, pie, fr = con_pie(12.5, 4.4, [izq, der], 175)
    a1, a2 = fig.subplots(1, 2)
    for m, v in acum.items():
        a1.step(np.arange(1, len(v) + 1), v, where="post", color=COLOR[m], lw=3.2 if m == "aleatoria" else 1.8, label=f"búsqueda {m}")
    a1.axhline(ref, color=COLOR["actual"], ls="--", lw=1, label="configuración actual")
    a1.axvline(k0 + 0.5, color="#BBB", ls=":", lw=1)
    y0, y1 = a1.get_ylim()
    a1.text(k0 + 0.6, y0 + 0.55 * (y1 - y0), "desde aquí la bayesiana\naprende de los ensayos previos", fontsize=7.5, color="#666", va="center")
    a1.set(xlabel="n.º de ensayo", ylabel="mejor PR-AUC media acumulada (3 folds)", xticks=range(1, N + 1, 2 if N > 12 else 1))
    a1.set_title("Mejor resultado acumulado por método", fontsize=10); a1.legend(fontsize=8, loc="lower right"); a1.grid(alpha=0.3)
    altos = ens.pr_auc_media[ens.pr_auc_media >= ens.pr_auc_media.quantile(0.25)]
    piso = min(ref, altos.min()) - 0.25 * (ens.pr_auc_media.max() - min(ref, altos.min()) + 1e-4)
    for m, g in ens.groupby("metodo", sort=False):
        x = g.ensayo + (-0.12 if m == "aleatoria" else 0.12)
        a2.scatter(x, g.pr_auc_media.clip(lower=piso), color=COLOR[m], marker=MARCA[m], s=28, label=f"búsqueda {m}", zorder=3)
    a2.axhline(ref, color=COLOR["actual"], ls="--", lw=1, label="configuración actual")
    a2.set_ylim(piso - 0.0005, ens.pr_auc_media.max() + 0.25 * (ens.pr_auc_media.max() - piso))
    a2.set(xlabel="n.º de ensayo", ylabel="PR-AUC media de cada ensayo (3 folds)", xticks=range(1, N + 1, 2 if N > 12 else 1))
    a2.set_title("Resultado de cada ensayo", fontsize=10); a2.legend(fontsize=8, loc="lower right"); a2.grid(alpha=0.3)
    return cerrar(fig, pie, fr)


def fig_parametros(R, ens):
    u = unicos(ens).copy()
    ref = R["referencia_actual_busqueda"]["pr_auc_media"]
    top = u.nlargest(5, "pr_auc_media")
    chico = u.nsmallest(1, "C").iloc[0]
    med = lambda col: u.groupby(col).pr_auc_media.median()
    cw, sl = med("class_weight"), med("sublinear_tf")
    rango = {c: med(c) for c in ("A_max_features", "A_min_df", "B_max_features", "B_min_df")}
    amp = {c: float(v.max() - v.min()) for c, v in rango.items()}
    tramo_c = u.groupby(pd.cut(u.C, TRAMOS_C), observed=True).pr_auc_media.median()
    amp_c = float(tramo_c.max() - tramo_c.min())
    amp_otros = max([*amp.values(), float(cw.max() - cw.min()), float(sl.max() - sl.min())])
    mejor_tramo = tramo_c.idxmax()
    txt = (f"Gráfico a). Eje x: valor de C, el inverso de la regularización de la regresión logística, en escala logarítmica (a mayor C, "
           f"menos se penalizan los coeficientes). Eje y: PR-AUC media de 3 folds. Cada punto es una configuración evaluada ({len(u)} en "
           f"total, sin repetir las compartidas), con color según el peso de las clases. Los cinco mejores ensayos tienen C entre "
           f"{top.C.min():.3g} y {top.C.max():.3g}, y el de menor C ({chico.C:.3g}) obtiene {chico.pr_auc_media:.4f}. Gráficos b) a g). "
           f"Eje x: valores de cada hiperparámetro. Eje y: PR-AUC media de 3 folds (puntos) y su mediana (raya). La mediana con "
           f"class_weight balanced es {cw.get('balanced', np.nan):.4f} y sin pesos {cw.get('ninguno', np.nan):.4f}; con sublinear_tf "
           f"{sl.get(True, np.nan):.4f} y sin él {sl.get(False, np.nan):.4f}. Entre los valores de max_features y min_df de A y B, las "
           f"medianas se separan como máximo {max(amp.values()):.4f}, frente a {amp_c:.4f} entre los tramos de C (la mediana más alta está "
           f"con C entre {mejor_tramo.left:g} y {mejor_tramo.right:g}). La lectura es marginal, porque cada ensayo cambia varios "
           f"hiperparámetros a la vez. "
           + (f"Esto implica que C es el hiperparámetro que más mueve la PR-AUC dentro del rango probado, y que las diferencias entre los "
              "valores de los demás no llegan a la mitad de las que produce C."
              if amp_c > 2 * amp_otros else
              "Esto implica que, dentro del rango probado, C no domina con claridad a los demás hiperparámetros."))
    fig, pie, fr = con_pie(12.5, 6.4, [txt], 175)
    gs = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gs[0, :2])
    for c, col in (("balanced", COLOR["aleatoria"]), ("ninguno", COLOR["bayesiana"])):
        for m in MARCA:
            g = u[(u.class_weight == c) & (u.metodo == m)]
            a.scatter(g.C, g.pr_auc_media, color=col, marker=MARCA[m], s=30, alpha=0.85,
                      label=f"class_weight {c} · {m}")
    a.axhline(ref, color=COLOR["actual"], ls="--", lw=1, label="actual (C = 1, balanced)")
    a.set_xscale("log"); a.set(xlabel="C (escala logarítmica)", ylabel="PR-AUC media (3 folds)")
    a.set_title("a) Efecto de C", loc="left", fontsize=10); a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3)
    lim = (u.pr_auc_media.min() - 0.002, u.pr_auc_media.max() + 0.002)
    for i, (col, pos) in enumerate(zip(["class_weight", "sublinear_tf", "A_max_features", "A_min_df", "B_max_features", "B_min_df"],
                                       [gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]])):
        ax = fig.add_subplot(pos)
        vals = R["parametros"]["espacio"][col]
        for j, v in enumerate(vals):
            y = u.loc[u[col] == v, "pr_auc_media"]
            ax.scatter(np.full(len(y), j) + np.linspace(-0.15, 0.15, len(y)) if len(y) > 1 else np.full(len(y), j), y,
                       s=14, color=COLOR["aleatoria"], alpha=0.6)
            if len(y):
                ax.plot([j - 0.25, j + 0.25], [y.median()] * 2, color="k", lw=1.6)
        ax.set_xticks(range(len(vals)), [f"{v:,}".replace(",", " ") if isinstance(v, int) and not isinstance(v, bool) else str(v) for v in vals],
                      fontsize=7.5)
        ax.set_ylim(*lim); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"{'bcdefg'[i]}) {NOMBRE[col]}", loc="left", fontsize=8.5)
        if i in (0, 2):
            ax.set_ylabel("PR-AUC media (3 folds)")
    return cerrar(fig, pie, fr)


def fig_folds(R):
    c, g = R["confirmacion_5_folds"], R["regla"]
    pa = np.array([f["pr_auc_comun"] for f in c["actual"]["folds"]])
    pc = np.array([f["pr_auc_comun"] for f in c["candidata"]["folds"]])
    d = pc - pa
    k = np.arange(1, len(d) + 1)
    izq = (f"Gráfico izquierdo. Eje x: fold de la validación cruzada del proyecto (cada una de las {len(d)} partes de train; el modelo se "
           f"entrena con {len(d) - 1} y se evalúa con la que queda), del 1 al {len(d)}. Eje y: PR-AUC en las epicrisis de validación sin "
           f"etiqueta dudosa. La configuración actual promedia {pa.mean():.4f} y la candidata, que salió de la búsqueda "
           f"{R['candidata']['metodo']}, {pc.mean():.4f}. Esto implica que buena parte de la variación entre folds es común a las dos "
           "configuraciones, por lo cual la comparación pertinente es la diferencia dentro de cada fold.")
    der = (f"Gráfico derecho. Eje x: fold, del 1 al {len(d)}. Eje y: PR-AUC de la candidata menos la de la actual en el mismo fold; sobre "
           f"cero, la candidata es mejor. La línea continua es la diferencia media ({g['dif_media']:+.4f}) y la franja, una desviación "
           f"estándar de las diferencias ({g['dif_std']:.4f}). La candidata supera a la actual en {g['folds_mejores']} de {g['n_folds']} "
           f"folds; la regla prerregistrada pedía al menos {g['folds_mejores_minimo']} y una diferencia media mayor que una desviación "
           f"estándar, y {'se cumplen las dos condiciones' if g['adopta_candidata'] else 'no se cumple ' + ('ninguna de las dos condiciones' if not (g['criterio_folds'] or g['criterio_desviacion']) else ('la de los folds' if not g['criterio_folds'] else 'la de la desviación'))}. "
           + ("Esto implica que la configuración ajustada reemplaza a la actual." if g["adopta_candidata"] else
              "Esto implica que se mantiene la configuración actual, dado que la búsqueda no encontró en el espacio explorado una "
              "configuración que la mejore."))
    fig, pie, fr = con_pie(12.5, 4.2, [izq, der], 175)
    a1, a2 = fig.subplots(1, 2)
    a1.plot(k, pa, "o-", color=COLOR["actual"], label="configuración actual")
    a1.plot(k, pc, "s-", color=COLOR["candidata"], label=f"candidata (búsqueda {R['candidata']['metodo']})")
    a1.set(xlabel="fold (parte de la validación cruzada)", ylabel="PR-AUC (sin etiqueta dudosa)", xticks=list(k))
    lo, hi = min(pa.min(), pc.min()), max(pa.max(), pc.max())
    a1.set_ylim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
    a1.set_title("PR-AUC de cada fold", fontsize=10); a1.legend(fontsize=8, loc="upper center"); a1.grid(alpha=0.3)
    a2.axhspan(g["dif_media"] - g["dif_std"], g["dif_media"] + g["dif_std"], color="#E4E4E4", zorder=1,
               label=f"± 1 desviación estándar ({g['dif_std']:.4f})")
    a2.bar(k, d, 0.55, color=[COLOR["candidata"] if x > 0 else COLOR["bayesiana"] for x in d], zorder=3)
    a2.axhline(0, c="gray", lw=0.8, zorder=4)
    a2.axhline(g["dif_media"], color="k", lw=1.2, zorder=4, label=f"diferencia media ({g['dif_media']:+.4f})")
    a2.set(xlabel="fold (parte de la validación cruzada)", ylabel="PR-AUC candidata − actual", xticks=list(k))
    lo = min(d.min(), g["dif_media"] - g["dif_std"], 0); hi = max(d.max(), g["dif_media"] + g["dif_std"], 0)
    a2.set_ylim(lo - 0.45 * (hi - lo), hi + 0.45 * (hi - lo))
    a2.set_title("Diferencia con la configuración actual en el mismo fold", fontsize=10)
    a2.legend(fontsize=8, loc="lower right" if g["dif_media"] > 0 else "lower left"); a2.grid(alpha=0.3, axis="y")
    return cerrar(fig, pie, fr)


def dibujar(R, ens, carpeta=None):
    carpeta = carpeta or REPORTES / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, f in (("fig_hp_busqueda.png", lambda: fig_busqueda(R, ens)), ("fig_hp_parametros.png", lambda: fig_parametros(R, ens)),
                      ("fig_hp_folds.png", lambda: fig_folds(R))):
        fig = f()
        fig.savefig(carpeta / nombre, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    dibujar(json.loads((REPORTES / "hp_resultados.json").read_text(encoding="utf-8")),
            pd.read_csv(REPORTES / "hp_ensayos.csv", keep_default_na=False))
    print("figuras redibujadas en", REPORTES / "figuras")
