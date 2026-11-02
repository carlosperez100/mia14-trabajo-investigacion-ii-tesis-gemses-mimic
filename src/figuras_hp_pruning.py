# -*- coding: utf-8 -*-
"""
Figuras de la semana 8 (búsqueda con pruning). Igual que en la semana 7, cada figura lleva al pie qué es el eje x, qué es el eje y y
qué muestra, con las cifras de la corrida, y cierra con lo que eso implica.

Las llama src/experimentos_hp_pruning.py al terminar. También se pueden redibujar sin volver a entrenar:
    python src/figuras_hp_pruning.py
Colores: los dos primeros de la paleta categórica validada para daltonismo (azul para la aleatoria y naranja para la bayesiana) y gris
para la configuración actual. La semana 7 va con línea discontinua y la semana 8 con línea continua.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from figuras_hp import cerrar, con_pie
from utils import REPORTES

COLOR = {"aleatoria": "#2a78d6", "bayesiana": "#eb6834", "actual": "#8A8A8A"}
MARCA = {"aleatoria": "o", "bayesiana": "^"}
NOMBRE = {"C": "C (regularización)", "A_max_features": "A: máximo de palabras", "A_min_df": "A: min_df (palabras)",
          "B_max_features": "B: máximo de bigramas", "B_min_df": "B: min_df (bigramas)", "class_weight": "peso de las clases",
          "sublinear_tf": "frecuencia sublineal"}
mil = lambda x: f"{int(round(x)):,}".replace(",", " ")


def acumulado_completos(g):
    """Mejor PR-AUC acumulada contando solo los ensayos completos (un ensayo cortado no tiene la media de los 3 folds)."""
    v = g.sort_values("ensayo").apply(lambda r: r.pr_auc_media if r.get("estado", "completo") == "completo" else np.nan, axis=1)
    return v.cummax().values


def fig_evolucion(R, ens, s7):
    ref = R["referencia_actual_busqueda"]["pr_auc_media"]
    N, k0 = R["parametros"]["n_ensayos"], R["parametros"]["ensayos_iniciales_tpe"]
    mej = R["mejor_por_metodo"]
    m7 = {m: g.pr_auc_media.max() for m, g in s7.groupby("metodo", sort=False)}
    cortes = {m: int((g.estado != "completo").sum()) for m, g in ens.groupby("metodo", sort=False)}
    por_tiempo = int((ens.estado == "cortado por tiempo").sum())
    gan = max(mej, key=lambda m: mej[m]["pr_auc_media"])
    dif = mej[gan]["pr_auc_media"] - ref
    sobre = int(((ens.estado == "completo") & (ens.pr_auc_media > ref)).sum())
    izq = (f"Gráfico izquierdo. Eje x: número de ensayo de cada búsqueda ({int((s7.metodo == 'aleatoria').sum())} por método en la "
           f"semana 7 y {N} en la semana 8). Eje y: la "
           f"mejor PR-AUC media de 3 folds obtenida hasta ese ensayo, contando solo los ensayos completos. Las líneas discontinuas son las "
           f"búsquedas de la semana 7, sin pruning, y las continuas las de la semana 8, con pruning. La línea gris es la configuración actual "
           f"en los mismos 3 folds ({ref:.4f}). En la semana 8, la {gan} llega a {mej[gan]['pr_auc_media']:.4f} en el ensayo "
           f"{mej[gan]['ensayo']}, {abs(dif):.4f} {'por encima' if dif > 0 else 'por debajo'} de la actual, frente a "
           f"{max(m7.values()):.4f} como mejor resultado de la semana 7. "
           + ("Esto implica que, con el doble de ensayos, la búsqueda tampoco encuentra una configuración que supere a la actual en los "
              "3 folds." if dif <= 0 else
              "Esto implica que la búsqueda encuentra una configuración por encima de la actual en los 3 folds, y que la diferencia debe "
              "confirmarse con los 5 folds antes de tomarla como mejora."))
    der = (f"Gráfico derecho. Eje x: número de ensayo de la semana 8. Eje y: PR-AUC media de los folds que llegó a evaluar cada ensayo. "
           f"Los puntos llenos son ensayos completos (3 folds) y los vacíos, ensayos cortados por el pruning, que se dibujan con el valor "
           f"de los folds que alcanzaron a evaluar. Se cortaron {cortes.get('aleatoria', 0)} de {N} ensayos de la aleatoria y "
           f"{cortes.get('bayesiana', 0)} de {N} de la bayesiana"
           + (f" ({por_tiempo} de ellos por tiempo y el resto por el pruning)" if por_tiempo else ", todos por el pruning")
           + f", y {sobre} ensayos completos superan a la actual. Desde el ensayo "
           f"{k0 + 1}, la bayesiana elige cada configuración a partir de los resultados anteriores. "
           "Esto implica que los ensayos cortados por el pruning son los que ya iban por debajo de la mediana en el primer o en el "
           "segundo fold, de modo que el presupuesto se concentra en las configuraciones que compiten con la actual.")
    fig, pie, fr = con_pie(12.5, 4.4, [izq, der], 175)
    a1, a2 = fig.subplots(1, 2)
    for m in ("aleatoria", "bayesiana"):
        v7 = acumulado_completos(s7[s7.metodo == m])
        v8 = acumulado_completos(ens[ens.metodo == m])
        a1.step(np.arange(1, len(v7) + 1), v7, where="post", color=COLOR[m], lw=1.6, ls="--", label=f"{m}, semana 7 (sin pruning)")
        a1.step(np.arange(1, len(v8) + 1), v8, where="post", color=COLOR[m], lw=2.2, label=f"{m}, semana 8 (con pruning)")
    a1.axhline(ref, color=COLOR["actual"], ls=":", lw=1.2, label="configuración actual")
    a1.set(xlabel="n.º de ensayo", ylabel="mejor PR-AUC media acumulada (3 folds)", xticks=range(1, N + 1, 2 if N > 12 else 1))
    a1.set_title("Mejor resultado acumulado por búsqueda", fontsize=10); a1.legend(fontsize=7.5, loc="lower right"); a1.grid(alpha=0.3)
    altos = ens.pr_auc_media[ens.pr_auc_media >= ens.pr_auc_media.quantile(0.25)]
    piso = min(ref, altos.min()) - 0.25 * (ens.pr_auc_media.max() - min(ref, altos.min()) + 1e-4)
    for m, g in ens.groupby("metodo", sort=False):
        x = g.ensayo + (-0.12 if m == "aleatoria" else 0.12)
        c = g.estado == "completo"
        a2.scatter(x[c], g.pr_auc_media[c].clip(lower=piso), color=COLOR[m], marker=MARCA[m], s=30, label=f"{m}: completo", zorder=3)
        a2.scatter(x[~c], g.pr_auc_media[~c].clip(lower=piso), facecolors="white", edgecolors=COLOR[m], marker=MARCA[m], s=30,
                   linewidths=1.3, label=f"{m}: cortado", zorder=3)
    a2.axhline(ref, color=COLOR["actual"], ls=":", lw=1.2, label="configuración actual")
    a2.set_ylim(piso - 0.0005, ens.pr_auc_media.max() + 0.25 * (ens.pr_auc_media.max() - piso))
    a2.set(xlabel="n.º de ensayo (semana 8)", ylabel="PR-AUC media de los folds evaluados", xticks=range(1, N + 1, 2 if N > 12 else 1))
    a2.set_title("Resultado de cada ensayo de la semana 8", fontsize=10); a2.legend(fontsize=7.5, loc="lower right", ncol=2)
    a2.grid(alpha=0.3)
    return cerrar(fig, pie, fr)


def fig_costo(R, ens, s7):
    E, A = R["efecto_pruning_mismos_ensayos"], R["ahorro_pruning"]
    N = R["parametros"]["n_ensayos"]
    calc = lambda g: np.cumsum(np.where(g.sort_values("ensayo").reutilizado_de.fillna("") == "", g.sort_values("ensayo").segundos, 0)) / 60
    tot = {(s, m): float(calc(d[d.metodo == m])[-1]) for s, d in ((7, s7), (8, ens)) for m in ("aleatoria", "bayesiana")}
    izq = (f"Gráfico izquierdo. Eje x: número de ensayo. Eje y: minutos de cómputo acumulados de cada búsqueda (los ensayos que repiten una "
           f"configuración ya evaluada no suman tiempo). En la semana 7, sin pruning, las dos búsquedas de 15 ensayos sumaron "
           f"{tot[(7, 'aleatoria')]:.1f} y {tot[(7, 'bayesiana')]:.1f} minutos; en la semana 8, con pruning y {N} ensayos, "
           f"{tot[(8, 'aleatoria')]:.1f} y {tot[(8, 'bayesiana')]:.1f} minutos. "
           + (f"Los 15 primeros ensayos de la aleatoria son las mismas configuraciones en las dos semanas; el pruning cortó {E['cortados']} "
              f"de ellos, se evaluaron {E['folds_evaluados']} de {E['folds_sin_pruning']} folds, y tomaron {E['segundos_semana8'] / 60:.1f} "
              f"minutos frente a {E['segundos_semana7'] / 60:.1f}. " if E["mismas_configuraciones_que_la_semana_7"] else "")
           + "Los minutos dependen también de la carga del equipo en cada corrida, mientras que el número de folds evaluados no.")
    der = (f"Gráfico derecho. Eje y: número de ensayos de cada búsqueda de la semana 8 según su estado (completo, cortado después del "
           f"primer fold o del segundo, o cortado por tiempo). En las dos búsquedas se evaluaron {A['folds_evaluados']} de los "
           f"{A['folds_sin_pruning']} folds que habrían requerido sin pruning las {A['configuraciones_evaluadas']} configuraciones distintas, "
           f"y se estima que el pruning ahorró unos {A['segundos_ahorrados_estimados'] / 60:.0f} minutos (cada fold no evaluado se estima "
           f"con el tiempo medio de los folds evaluados del mismo ensayo). "
           "Esto implica que el pruning reduce el costo de la búsqueda sin cambiar el diseño, porque solo deja de evaluar los folds de los "
           "ensayos que ya iban por debajo de la mediana"
           + (", y que el límite por ensayo cortó además los que pasaban del tiempo máximo." if (ens.estado == "cortado por tiempo").any()
              else "."))
    fig, pie, fr = con_pie(12.5, 4.2, [izq, der], 175)
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    for s, d, ls, lw in ((7, s7, "--", 1.6), (8, ens, "-", 2.2)):
        for m in ("aleatoria", "bayesiana"):
            y = calc(d[d.metodo == m])
            a1.plot(np.arange(1, len(y) + 1), y, color=COLOR[m], ls=ls, lw=lw,
                    label=f"{m}, semana {s} ({'con' if s == 8 else 'sin'} pruning)")
    a1.set(xlabel="n.º de ensayo", ylabel="minutos de cómputo acumulados", xticks=range(1, N + 1, 2 if N > 12 else 1))
    a1.set_title("Costo acumulado de cada búsqueda", fontsize=10); a1.legend(fontsize=7.5, loc="upper left"); a1.grid(alpha=0.3)
    cats = [("completo", "completo"), ("cortado por pruning", "cortado tras 1 fold"), ("cortado por pruning", "cortado tras 2 folds"),
            ("cortado por tiempo", "cortado por tiempo")]
    base = np.zeros(2)
    tonos = ["#2a2a2a", "#8A8A8A", "#BDBDBD", "#E4E4E4"]
    metodos = ["aleatoria", "bayesiana"]
    for (estado, etiqueta), tono in zip(cats, tonos):
        if etiqueta == "cortado tras 1 fold":
            n = [int(((ens.metodo == m) & (ens.estado == estado) & (ens.folds_evaluados == 1)).sum()) for m in metodos]
        elif etiqueta == "cortado tras 2 folds":
            n = [int(((ens.metodo == m) & (ens.estado == estado) & (ens.folds_evaluados == 2)).sum()) for m in metodos]
        else:
            n = [int(((ens.metodo == m) & (ens.estado == estado)).sum()) for m in metodos]
        if sum(n) == 0:
            continue
        a2.bar(metodos, n, 0.55, bottom=base, color=tono, edgecolor="white", linewidth=2, label=etiqueta)
        for j, v in enumerate(n):
            if v:
                a2.text(j, base[j] + v / 2, str(v), ha="center", va="center", fontsize=8,
                        color="white" if tono in ("#2a2a2a", "#8A8A8A") else "#222")
        base += n
    a2.set(ylabel="n.º de ensayos (semana 8)", ylim=(0, N * 1.12))
    a2.yaxis.set_major_locator(MaxNLocator(integer=True))
    a2.set_title("Estado de los ensayos de la semana 8", fontsize=10); a2.legend(fontsize=7.5, loc="upper right"); a2.grid(alpha=0.3, axis="y")
    return cerrar(fig, pie, fr)


def fig_importancia(R):
    I = R["importancia"]
    imp = pd.Series(I["importancias"]).sort_values()
    primero, segundo = imp.index[-1], imp.index[-2]
    txt = (f"Eje x: importancia relativa de cada hiperparámetro (suman 1), calculada con el evaluador por defecto de Optuna "
           f"{R['optuna']} (PED-ANOVA) sobre las {I['n_configuraciones']} configuraciones completas y distintas de las cuatro búsquedas. "
           f"Eje y: hiperparámetro. El evaluador compara la distribución de cada hiperparámetro en el {I['target_quantile'] * 100:.1f} % "
           f"de mejores ensayos ({I['ensayos_sobre_el_cuantil']} configuraciones) con su distribución en todos los ensayos. "
           f"{NOMBRE[primero]} obtiene {imp[primero]:.2f} y {NOMBRE[segundo]} {imp[segundo]:.2f}. La medida es relativa a los rangos "
           "probados y, con este número de ensayos, se toma como orden de magnitud. "
           + (f"Esto implica que {NOMBRE[primero]} es el hiperparámetro que más separa a los mejores ensayos del resto, lo cual concuerda "
              "con la semana 7, donde las medianas de la PR-AUC cambiaban sobre todo con C."
              if primero == "C" else
              f"Esto implica que, en estos ensayos, {NOMBRE[primero]} es el hiperparámetro que más separa a los mejores ensayos del resto."))
    fig, pie, fr = con_pie(9.5, 3.6, [txt], 130)
    a = fig.subplots()
    a.barh([NOMBRE[k] for k in imp.index], imp.values, color=COLOR["aleatoria"], height=0.6, edgecolor="white", linewidth=2)
    for j, v in enumerate(imp.values):
        a.text(v + 0.01, j, f"{v:.2f}", va="center", fontsize=8, color="#222")
    a.set(xlabel="importancia relativa (suma 1)", xlim=(0, max(imp.values) * 1.18))
    a.set_title("Importancia de los hiperparámetros (PED-ANOVA, Optuna)", fontsize=10); a.grid(alpha=0.3, axis="x")
    return cerrar(fig, pie, fr)


def fig_curva(R, curva):
    g = curva.groupby("fraccion").agg(n=("filas_entrenamiento", "mean"), tr=("pr_auc_entrenamiento", "mean"),
                                      tr_sd=("pr_auc_entrenamiento", "std"), va=("pr_auc_validacion", "mean"),
                                      va_sd=("pr_auc_validacion", "std")).reset_index()
    a, z = g.iloc[0], g.iloc[-1]
    pen = g.iloc[-2] if len(g) > 1 else a
    brecha = z.tr - z.va
    sube = z.va - pen.va
    txt = (f"Eje x: número de filas con que se entrena el modelo, es decir, una fracción ({', '.join(f'{x:g}' for x in g.fraccion)}) de las "
           f"filas de entrenamiento de cada uno de los {curva.fold.nunique()} folds del proyecto. Eje y: PR-AUC media de los folds, en las "
           f"mismas filas de entrenamiento (naranja) y en las epicrisis de validación sin etiqueta dudosa (azul); la franja es ± 1 "
           f"desviación estándar entre folds. Con {mil(a.n)} filas, la PR-AUC es {a.tr:.4f} en entrenamiento y {a.va:.4f} en validación; "
           f"con {mil(z.n)} filas, {z.tr:.4f} y {z.va:.4f}, una brecha de {brecha:.4f}. Entre {mil(pen.n)} y {mil(z.n)} filas, la de "
           f"validación sube {sube:+.4f}. "
           + ("Esto implica que el modelo ajusta en entrenamiento bastante más de lo que generaliza, lo esperable en una regresión con "
              "decenas de miles de términos, y que la validación todavía mejora al sumar filas, de modo que más datos etiquetados "
              "ayudarían más que seguir ajustando hiperparámetros."
              if sube > 0.002 else
              "Esto implica que el modelo ajusta en entrenamiento bastante más de lo que generaliza, lo esperable en una regresión con "
              "decenas de miles de términos, y que la validación casi no cambia al sumar filas, por lo que más datos del mismo tipo "
              "aportarían poco."))
    fig, pie, fr = con_pie(9.5, 4.0, [txt], 130)
    ax = fig.subplots()
    for col, sd, color, etiqueta in (("tr", "tr_sd", "#eb6834", "entrenamiento (las mismas filas)"),
                                     ("va", "va_sd", "#2a78d6", "validación (sin etiqueta dudosa)")):
        ax.plot(g.n, g[col], "o-", color=color, lw=2, ms=6, label=etiqueta)
        ax.fill_between(g.n, g[col] - g[sd].fillna(0), g[col] + g[sd].fillna(0), color=color, alpha=0.15, linewidth=0)
    ax.annotate("", xy=(z.n, z.va), xytext=(z.n, z.tr), arrowprops=dict(arrowstyle="<->", color="#52514e", lw=1))
    ax.text(z.n, (z.tr + z.va) / 2, f"  brecha {brecha:.3f}", va="center", ha="left", fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.set_xticks(g.n, [mil(x) for x in g.n], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(g.n.min() * 0.8, g.n.max() * 1.6)
    ax.set(xlabel="filas de entrenamiento (escala logarítmica)", ylabel="PR-AUC media de los folds")
    ax.set_title("Curva de aprendizaje de la configuración ganadora", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    return cerrar(fig, pie, fr)


def dibujar(R, ens, s7, curva, carpeta=None):
    carpeta = carpeta or REPORTES / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, f in (("fig_hp_pruning_evolucion.png", lambda: fig_evolucion(R, ens, s7)),
                      ("fig_hp_pruning_costo.png", lambda: fig_costo(R, ens, s7)),
                      ("fig_hp_pruning_importancia.png", lambda: fig_importancia(R)),
                      ("fig_hp_pruning_curva.png", lambda: fig_curva(R, curva))):
        fig = f()
        fig.savefig(carpeta / nombre, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    leer = lambda n: pd.read_csv(REPORTES / n, keep_default_na=False)
    ens = pd.read_csv(REPORTES / "hp_pruning_ensayos.csv", keep_default_na=False, na_values=[""])
    ens["reutilizado_de"] = ens.reutilizado_de.fillna("")
    dibujar(json.loads((REPORTES / "hp_pruning_resultados.json").read_text(encoding="utf-8")), ens, leer("hp_ensayos.csv"),
            pd.read_csv(REPORTES / "hp_pruning_curva.csv"))
    print("figuras redibujadas en", REPORTES / "figuras")
