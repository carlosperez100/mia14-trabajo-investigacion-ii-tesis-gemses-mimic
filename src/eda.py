# -*- coding: utf-8 -*-
"""
ETAPA 3 — EDA (análisis exploratorio de datos) reproducible y con log automático.

Entrada : data/processed/dataset.parquet y splits.csv (salida de src/preprocesado.py).
Salidas : logs/eda_<fecha>.log            log automático (hora, tamaño, pasos, resultados)
          logs/metrics_eda.txt            resumen legible de calidad, balance, riesgos y decisiones
          reportes/eda_resumen.json       todas las cifras del EDA (fuente del README y del notebook)
          reportes/figuras/fig_eda_*.png  histogramas, balance/época y matriz de correlación

Regla de datos: MIMIC-IV está bajo DUA de PhysioNet. Este script solo escribe AGREGADOS;
nunca imprime ni guarda texto clínico.

Secciones (cada una vigila un riesgo concreto, como en notebooks/01_EDA_sprint1.ipynb):
  0 calidad: nulos, tipos, rangos, duplicados, outliers
  1 estadísticas descriptivas (largo, palabras, notas por paciente, naturalezas)
  2 volumen y balance de la variable objetivo por partición
  3 época de codificación CIE-9/CIE-10 por clase (confusor / drift)
  4 largo de la nota por clase (atajo)
  5 códigos CIE escritos en el texto (fuga directa de la etiqueta)
  6 pacientes compartidos entre train y test (fuga por paciente)
  7 naturaleza del evento entre las positivas (clases raras para la etapa 2)
  8 correlaciones de Spearman con la variable objetivo en la ÚLTIMA fila
  9 riesgos y decisiones accionables para el siguiente sprint
  + figura del flujo del EDA (reportes/figuras/fig_eda_flujo.png) con el resultado de cada paso

Uso:  python src/eda.py
"""
import re
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import LOGS, PROCESSED, RAIZ, REPORTES, cargar_config, guardar_json, logger, sha256

ETAPA = "eda"
FIG = REPORTES / "figuras"
PAT_CIE10 = re.compile(r"\b[A-TV-Z]\d{2}\.\d{1,4}\b")
PAT_CIE9 = re.compile(r"\b(?:E|V)?\d{3}\.\d{1,2}\b")

DECISIONES = [
    {"n": 1, "decision": "Adoptar la PR-AUC como métrica central y reportar F1 (+) y Recall a su lado",
     "paso": "Métrica",
     "evidencia": "el dummy que siempre dice «sí» obtiene F1 (+) = 0.698 porque el test tiene 53.7 % de positivas; "
                  "el piso de la PR-AUC es la proporción de positivas (0.537)",
     "control": "tabla de métricas con PR-AUC e IC 95 % en logs/metrics_baseline.txt, comparada contra el piso"},
    {"n": 2, "decision": "Separar la evaluación por deciles de largo y añadir el largo como variable de control",
     "paso": "Split / features",
     "evidencia": "el largo solo separa las clases (AUC del largo, ver sección 4) y la proporción de positivas "
                  "crece del decil más corto al más largo: riesgo de atajo",
     "control": "PR-AUC dentro de cada decil; si cae mucho frente a la global, el modelo se apoya en el largo"},
    {"n": 3, "decision": "Enmascarar los patrones de código CIE en el texto antes de vectorizar",
     "paso": "Preprocesado",
     "evidencia": "códigos con formato CIE-10 aparecen algo más en las positivas que en las negativas (sección 5): "
                  "fuga posible, aunque pequeña",
     "control": "diferencia de PR-AUC con y sin máscara en la misma partición"},
]


def describir(serie):
    d = serie.describe(percentiles=[.25, .5, .75])
    return {k: round(float(d[k]), 2) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}


NOMBRE_NAT = {"Medicacion": "Medicación", "Infeccion nosocomial": "Infección nosocomial",
              "Dispositivo medico": "Dispositivo médico", "Sistema/Organizacion": "Sistema/Organización"}


def _mil(n):
    """Miles separados con espacio que no se parte entre renglones (70 000)."""
    return f"{int(round(n)):,}".replace(",", " ")


def _r1(x):
    """Redondeo convencional a un decimal (53.65 -> 53.7)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pie(fig, texto, ancho, alto):
    """Explicación corta al pie de la figura, para que se entienda aunque se abra sola."""
    import textwrap
    fig.tight_layout(rect=(0, alto, 1, 1))
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(texto, ancho)), ha="left", va="bottom",
             fontsize=8.4, color="#333", linespacing=1.35)


def figura_flujo(R, ruta):
    """Diagrama del flujo del EDA: los diez pasos en orden, con el resultado de cada uno y su estado.
    Todas las cifras salen del diccionario de resultados R (nada se escribe a mano)."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def r1(x):  # redondeo convencional a un decimal (53.65 -> 53.7)
        from decimal import Decimal, ROUND_HALF_UP
        return str(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

    def mil(n):
        return f"{int(round(n)):,}".replace(",", "\u00a0")

    ESTADO = {"dato": ("#EAF2F8", "#1F4E79", "Dato"),
              "ok": ("#E8F5EE", "#1E7B4F", "Controlado"),
              "atencion": ("#FFF3E6", "#B35C00", "Atención")}
    nat, cy = R["naturaleza"], R["correlaciones_spearman"]["con_y"]
    BONITO = {"Medicacion": "Medicación", "Infeccion nosocomial": "Infección nosocomial",
              "Dispositivo medico": "Dispositivo médico", "Sistema/Organizacion": "Sistema/Organización"}
    dec, menor = R["largo"]["pct_positivas_por_decil"], R["naturaleza"]["clases_menores_de_200"]
    if menor:
        res_nat = (f"{BONITO.get(menor[0], menor[0])}: {mil(nat['conteo'][menor[0]])} notas, no evaluable · "
                   f"{nat['pct_multietiqueta']:.1f} % multietiqueta")
    else:
        res_nat = f"todas las clases con 200 notas o más · {nat['pct_multietiqueta']:.1f} % multietiqueta"
    pasos = [
        ("Datos de entrada", "Corpus de modelado del preprocesado, con partición por paciente",
         f"{mil(R['entrada']['notas'])} epicrisis · {mil(R['entrada']['pacientes'])} pacientes · semilla {R['semilla']}", "dato"),
        ("Calidad", "Nulos, tipos, duplicados y valores atípicos",
         f"{R['calidad']['nulos']['columnas_sin_labels']} nulos · {R['calidad']['duplicados']['note_id']} duplicados · "
         f"{mil(R['calidad']['outliers_largo']['n'])} atípicos ({R['calidad']['outliers_largo']['pct']:.1f} %) conservados", "ok"),
        ("Descriptivas", "Largo, palabras y notas por paciente",
         f"mediana de {mil(R['descriptivas']['largo_caracteres']['50%'])} caracteres y {mil(R['descriptivas']['palabras']['50%'])} "
         f"palabras · {R['descriptivas']['notas_por_paciente']['mean']} notas por paciente", "dato"),
        ("Balance de la clase", "Proporción de notas con evento adverso",
         f"{r1(R['balance']['train']['pct_positivas'])} % en train · {r1(R['balance']['test']['pct_positivas'])} % en test · "
         f"prevalencia real {100 * R['balance']['prevalencia_real_mimic']:.1f} %", "dato"),
        ("Época CIE-9 / CIE-10", "¿La época de codificación separa las clases? (drift)",
         f"{R['epoca']['pct_por_clase']['positiva']['CIE-9']:.1f} % CIE-9 en ambas clases · "
         f"AUC de la época = {R['epoca']['auc_epoca_sola']:.3f}", "ok"),
        ("Largo de la nota", "¿El largo predice el evento por sí solo? (atajo)",
         f"AUC del largo = {R['largo']['auc_largo_solo']:.3f} · positivas del decil 1 al 10: {dec[0]:.1f} % → {dec[-1]:.1f} %",
         "atencion"),
        ("Códigos CIE en el texto", "¿La nota trae escrita la respuesta? (fuga)",
         f"CIE-10 en {R['fuga_codigos']['pct_con_cie10']['positiva']:.2f} % de positivas vs "
         f"{R['fuga_codigos']['pct_con_cie10']['negativa']:.2f} % de negativas", "atencion"),
        ("Pacientes", "¿Un mismo paciente en train y en test? (fuga)",
         f"{R['pacientes']['compartidos_train_test']} pacientes compartidos", "ok"),
        ("Naturaleza del evento", "Clases disponibles para la etapa 2", res_nat, "atencion"),
        ("Correlaciones", "Spearman entre variables de contexto y la etiqueta",
         f"largo {cy['largo (caracteres)']:.3f} · notas del paciente {cy['notas del paciente']:.3f} · "
         f"época {abs(cy['época CIE']):.3f}", "atencion"),
    ]
    fuentes = ["del paso 4", "de los pasos 6 y 10", "del paso 7"]

    fig, ax = plt.subplots(figsize=(16, 10.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 67.5); ax.axis("off")
    ax.text(50, 66.6, "Flujo del análisis exploratorio (EDA) y resultado de cada paso", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1F4E79")
    AZUL = "#1F4E79"

    # banda superior: dónde está el EDA dentro del pipeline
    etapas = ["Ingesta", "Preprocesado", "EDA (este análisis)", "Baseline"]
    for i, e in enumerate(etapas):
        x = 14 + i * 19
        actual = i == 2
        ax.add_patch(FancyBboxPatch((x, 59), 15, 4.6, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=AZUL if actual else "white", ec=AZUL, lw=1.6))
        ax.text(x + 7.5, 61.3, e, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if actual else AZUL)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 15.4, 61.3), (x + 18.6, 61.3), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))
    ax.text(1.5, 61.3, "Pipeline:", ha="left", va="center", fontsize=10, color="#444", fontweight="bold")
    # leyenda de estados
    for k, (clave, (fc, ec, nombre)) in enumerate(ESTADO.items()):
        x = 66 + k * 11
        ax.add_patch(FancyBboxPatch((x, 55.4), 2.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.3", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 2.9, 56.2, nombre, ha="left", va="center", fontsize=8.5, color="#333")

    # diez pasos en dos filas
    W, H, G = 17.6, 19.0, 2.0
    filas_y = [35.0, 12.5]
    cajas = []
    for n, (titulo, que, resultado, estado) in enumerate(pasos):
        fila, col = divmod(n, 5)
        x, y = 1.5 + col * (W + G), filas_y[fila]
        fc, ec, nombre = ESTADO[estado]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.6))
        ax.text(x + 0.9, y + H - 1.3, f"{n + 1} · {titulo}", ha="left", va="top", fontsize=9.6, fontweight="bold", color=ec)
        ax.text(x + 0.9, y + H - 4.4, "\n".join(textwrap.wrap(que, 34)), ha="left", va="top", fontsize=7.6,
                style="italic", color="#444", linespacing=1.3)
        ax.text(x + 0.9, y + H - 9.0, "\n".join(textwrap.wrap(resultado, 27)), ha="left", va="top", fontsize=8.2,
                color="#111", fontweight="bold", linespacing=1.35)
        ax.add_patch(FancyBboxPatch((x + 0.9, y + 0.8), 6.4, 1.9, boxstyle="round,pad=0.12,rounding_size=0.5", fc=ec, ec=ec))
        ax.text(x + 4.1, y + 1.75, nombre, ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
        cajas.append((x, y))
    for n in range(9):
        if n == 4:  # de la fila 1 a la fila 2
            x5, y5 = cajas[4]; x6, y6 = cajas[5]
            ax.plot([x5 + W / 2, x5 + W / 2, x6 + W / 2], [y5 - 0.4, 33.4, 33.4], color=AZUL, lw=1.5)
            ax.add_patch(FancyArrowPatch((x6 + W / 2, 33.4), (x6 + W / 2, y6 + H + 0.4), arrowstyle="-|>",
                                         mutation_scale=13, color=AZUL, lw=1.5))
        elif n != 4:
            (xa, ya), (xb, yb) = cajas[n], cajas[n + 1]
            if ya == yb:
                ax.add_patch(FancyArrowPatch((xa + W + 0.35, ya + H / 2), (xb - 0.35, yb + H / 2), arrowstyle="-|>",
                                             mutation_scale=13, color=AZUL, lw=1.5))

    # decisiones
    ax.add_patch(FancyBboxPatch((1.5, 0.6), 96.0, 9.2, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=AZUL, lw=1.8))
    ax.text(3.0, 8.4, "Decisiones para el Sprint 2", ha="left", va="center", fontsize=10.5, fontweight="bold", color=AZUL)
    for k, d in enumerate(R["decisiones"]):
        x = 3.0 + k * 31.6
        ax.text(x, 5.6, f"D{d['n']} · " + "\n".join(textwrap.wrap(d["decision"], 52)) + f"\n(sale {fuentes[k]})",
                ha="left", va="top", fontsize=8.4, color="#111", linespacing=1.35)
    xc = cajas[7][0] + W / 2
    ax.add_patch(FancyArrowPatch((xc, cajas[7][1] - 0.4), (xc, 10.2), arrowstyle="-|>", mutation_scale=13, color=AZUL, lw=1.5))

    ax.text(1.5, -1.3, "Cómo leer: cada caja es un paso del análisis; en cursiva, la pregunta que responde, y en negrita, el "
            "resultado de esta corrida. Las flechas marcan el orden, y cada decisión de abajo indica de qué paso sale.",
            ha="left", va="top", fontsize=9.5, color="#333")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    log = logger(ETAPA)
    cfg = cargar_config()
    semilla = cfg["semilla"]
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"etapa": ETAPA, "fecha": datetime.now().isoformat(timespec="seconds"), "semilla": semilla,
         "entrada": {"dataset": "data/processed/dataset.parquet", "splits": "data/processed/splits.csv"}}

    # ---------- carga
    ruta = PROCESSED / "dataset.parquet"
    df = pd.read_parquet(ruta)
    R["entrada"]["tamano_bytes"] = ruta.stat().st_size
    R["entrada"]["sha256_splits"] = sha256(PROCESSED / "splits.csv")
    R["entrada"]["notas"] = int(len(df))
    R["entrada"]["pacientes"] = int(df.subject_id.nunique())
    df["largo"] = df.text.str.len()
    df["palabras"] = df.text.str.count(r"\S+")
    df["notas_paciente"] = df.groupby("subject_id").note_id.transform("count")
    df["n_naturalezas"] = df.labels.apply(lambda l: len(l) if l is not None else 0)
    log.info(f"carga: {len(df):,} notas · {df.subject_id.nunique():,} pacientes · "
             f"{df.hadm_id.nunique():,} hospitalizaciones · {ruta.stat().st_size/1e6:.0f} MB · "
             f"semilla {semilla} · sha256 splits {R['entrada']['sha256_splits'][:16]}…")

    # ---------- 0. calidad
    cols = ["note_id", "subject_id", "hadm_id", "text", "y", "epoca", "split", "labels"]
    calidad = {c: {"tipo": str(df[c].dtype), "nulos": int(df[c].isna().sum()),
                   "unicos": int(df[c].nunique()) if c != "labels" else None} for c in cols}
    q1, q3 = df.largo.quantile([.25, .75])
    lim = float(q3 + 1.5 * (q3 - q1))
    R["calidad"] = {
        "columnas": calidad,
        "duplicados": {"note_id": int(df.note_id.duplicated().sum()), "hadm_id": int(df.hadm_id.duplicated().sum()),
                       "texto_identico": int(df.text.duplicated().sum())},
        "faltantes": {"texto_vacio": int((df.largo == 0).sum()),
                      "positivas_sin_naturaleza": int(((df.y == 1) & (df.n_naturalezas == 0)).sum()),
                      "negativas_con_naturaleza": int(((df.y == 0) & (df.n_naturalezas > 0)).sum())},
        "rango_largo": [int(df.largo.min()), int(df.largo.max())],
        "outliers_largo": {"limite_q3_1_5iqr": round(lim), "n": int((df.largo > lim).sum()),
                           "pct": round(100 * float((df.largo > lim).mean()), 2), "accion": "se conservan: son notas reales"},
    }
    nulos_sin_labels = sum(v["nulos"] for c, v in calidad.items() if c != "labels")
    R["calidad"]["nulos"] = {"columnas_sin_labels": nulos_sin_labels,
                             "labels_none": calidad["labels"]["nulos"],
                             "nota": "labels es None en las negativas por construcción (no tienen naturaleza): no es un faltante"}
    log.info(f"0. calidad: nulos {nulos_sin_labels} (labels None en {calidad['labels']['nulos']} negativas, esperado) · "
             f"duplicados {R['calidad']['duplicados']} · "
             f"faltantes {R['calidad']['faltantes']} · largo {R['calidad']['rango_largo']} · "
             f"outliers {R['calidad']['outliers_largo']['n']} ({R['calidad']['outliers_largo']['pct']} %)")

    # ---------- 1. estadísticas descriptivas
    R["descriptivas"] = {"largo_caracteres": describir(df.largo), "palabras": describir(df.palabras),
                         "notas_por_paciente": describir(df.groupby("subject_id").size()),
                         "naturalezas_por_positiva": describir(df.loc[df.y == 1, "n_naturalezas"])}
    for k, v in R["descriptivas"].items():
        log.info(f"1. {k}: media {v['mean']} · mediana {v['50%']} · desv {v['std']} · min {v['min']} · max {v['max']}")

    # ---------- 2. volumen y balance
    t = pd.crosstab(df.split, df.y, margins=True)
    t.columns = ["negativa", "positiva", "total"]
    t["pct_positivas"] = (100 * t.positiva / t.total).round(2)
    R["balance"] = {idx: {k: (float(v) if k == "pct_positivas" else int(v)) for k, v in fila.items()}
                    for idx, fila in t.to_dict(orient="index").items()}
    R["balance"]["prevalencia_real_mimic"] = 0.2012
    log.info(f"2. balance: " + " · ".join(f"{k}: {v['total']:,} notas, {v['pct_positivas']} % positivas"
                                            for k, v in R["balance"].items() if isinstance(v, dict)))

    # ---------- 3. época por clase (confusor / drift)
    ep = pd.crosstab(df.y, df.epoca, normalize="index").mul(100).round(2)
    auc_epoca = float(roc_auc_score(df.y, df.epoca))
    R["epoca"] = {"pct_por_clase": {("positiva" if y == 1 else "negativa"): {f"CIE-{int(c)}": float(ep.loc[y, c]) for c in ep.columns}
                                    for y in ep.index},
                  "auc_epoca_sola": round(auc_epoca, 3),
                  "pct_positivas_por_epoca": {f"CIE-{int(e)}": round(100 * float(df.loc[df.epoca == e, "y"].mean()), 2)
                                              for e in sorted(df.epoca.unique())}}
    log.info(f"3. época: {R['epoca']['pct_por_clase']} · AUC de la época sola {auc_epoca:.3f} (≈0.5 = controlado)")

    # ---------- 4. largo por clase (atajo)
    auc_largo = float(roc_auc_score(df.y, df.largo))
    dec = pd.qcut(df.largo, 10, labels=False)
    pct_decil = (df.groupby(dec).y.mean() * 100).round(1).tolist()
    R["largo"] = {"mediana_por_clase": {"negativa": int(df.loc[df.y == 0, "largo"].median()),
                                        "positiva": int(df.loc[df.y == 1, "largo"].median())},
                  "auc_largo_solo": round(auc_largo, 3), "pct_positivas_por_decil": pct_decil}
    log.info(f"4. largo: medianas {R['largo']['mediana_por_clase']} · AUC del largo solo {auc_largo:.3f} · "
             f"positivas por decil {pct_decil[0]} % → {pct_decil[-1]} %")

    # ---------- 5. códigos CIE en el texto (fuga directa)
    cie10 = df.text.str.contains(PAT_CIE10)
    cie9 = df.text.str.contains(PAT_CIE9)
    R["fuga_codigos"] = {"pct_con_cie10": {"negativa": round(100 * float(cie10[df.y == 0].mean()), 2),
                                           "positiva": round(100 * float(cie10[df.y == 1].mean()), 2)},
                         "pct_con_patron_cie9": {"negativa": round(100 * float(cie9[df.y == 0].mean()), 2),
                                                 "positiva": round(100 * float(cie9[df.y == 1].mean()), 2)},
                         "nota": "el patrón tipo CIE-9 (123.45) también captura decimales comunes; importa la diferencia entre clases"}
    log.info(f"5. códigos en texto: CIE-10 {R['fuga_codigos']['pct_con_cie10']} · patrón CIE-9 {R['fuga_codigos']['pct_con_patron_cie9']}")

    # ---------- 6. fuga por paciente
    comp = set(df.loc[df.split == "train", "subject_id"]) & set(df.loc[df.split == "test", "subject_id"])
    R["pacientes"] = {"compartidos_train_test": len(comp), "notas_por_paciente_media": R["descriptivas"]["notas_por_paciente"]["mean"]}
    log.info(f"6. pacientes compartidos train/test: {len(comp)} · notas por paciente {R['pacientes']['notas_por_paciente_media']}")

    # ---------- 7. naturaleza del evento (etapa 2)
    pos = df[df.y == 1]
    nat = pos.labels.explode().value_counts()
    R["naturaleza"] = {"conteo": {str(k): int(v) for k, v in nat.items()},
                       "pct_sobre_positivas": {str(k): round(100 * float(v) / len(pos), 2) for k, v in nat.items()},
                       "pct_multietiqueta": round(100 * float((pos.n_naturalezas > 1).mean()), 2),
                       "clases_menores_de_200": [str(k) for k, v in nat.items() if v < 200]}
    log.info(f"7. naturalezas: {R['naturaleza']['conteo']} · multietiqueta {R['naturaleza']['pct_multietiqueta']} % · "
             f"clases < 200 notas: {R['naturaleza']['clases_menores_de_200']}")

    # ---------- 8. correlaciones (variable objetivo en la ÚLTIMA fila)
    num = pd.DataFrame({"época CIE": df.epoca, "largo (caracteres)": df.largo, "log10 largo": np.log10(df.largo),
                        "palabras": df.palabras, "notas del paciente": df.notas_paciente, "y (evento)": df.y})
    corr = num.corr(method="spearman").round(3)
    R["correlaciones_spearman"] = {"variables": list(corr.columns), "matriz": corr.values.tolist(),
                                   "con_y": {c: float(corr.loc["y (evento)", c]) for c in corr.columns if c != "y (evento)"}}
    log.info(f"8. Spearman con y: {R['correlaciones_spearman']['con_y']}")

    # ---------- figuras
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
    for y, nombre in [(0, "negativa"), (1, "positiva")]:
        ax[0].hist(np.log10(df.loc[df.y == y, "largo"]), bins=60, alpha=.55, density=True, label=nombre)
    ax[0].set(xlabel="log10(caracteres)", ylabel="densidad", title=f"Largo de la epicrisis (AUC del largo solo = {auc_largo:.3f})")
    ax[0].legend()
    ax[1].bar(range(1, 11), pct_decil, color="#1F4E79")
    ax[1].axhline(100 * df.y.mean(), ls="--", c="gray", label=f"promedio {100*df.y.mean():.1f} %")
    ax[1].set(xlabel="decil de largo (1 = más corto)", ylabel="% de notas con evento", title="Positivas por decil de largo")
    ax[1].legend()
    med = R["largo"]["mediana_por_clase"]
    pie(fig, "Cómo leer: a la izquierda, la distribución del largo de las epicrisis en escala logarítmica; en azul las notas sin "
             "evento y en naranja las notas con evento. A la derecha, el porcentaje de notas con evento en cada decil de largo, con "
             "el promedio en línea punteada. Lectura: las notas con evento son más largas (mediana de "
             f"{_mil(med['positiva'])} frente a {_mil(med['negativa'])} caracteres; AUC del largo solo = {auc_largo:.3f}) y el "
             f"porcentaje de positivas sube de {pct_decil[0]} % en el decil más corto a {pct_decil[-1]} % en el más largo: riesgo "
             "moderado de que el modelo use el largo como atajo.", 175, 0.17)
    fig.savefig(FIG / "fig_eda_largo.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    ax[0, 0].hist(df.palabras, bins=60, color="#1F4E79"); ax[0, 0].set(title="Palabras por epicrisis", xlabel="palabras", ylabel="notas")
    npp = df.groupby("subject_id").size()
    ax[0, 1].bar(*np.unique(np.clip(npp, 1, 6), return_counts=True), color="#1F4E79")
    ax[0, 1].set(title="Notas por paciente (6 = 6 o más)", xlabel="notas", ylabel="pacientes")
    bal = pd.crosstab(df.split, df.y, normalize="index").mul(100)
    bal.plot(kind="bar", stacked=True, ax=ax[1, 0], color=["#9FB8CF", "#9B2226"], rot=0)
    ax[1, 0].set(title="Balance de la variable objetivo por partición", ylabel="%"); ax[1, 0].legend(["negativa", "positiva"])
    ax[1, 1].barh([NOMBRE_NAT.get(k, k) for k in R["naturaleza"]["conteo"].keys()][::-1], list(R["naturaleza"]["conteo"].values())[::-1], color="#1E7B4F")
    ax[1, 1].set(title="Naturaleza del evento (positivas)", xlabel="notas")
    menor = R["naturaleza"]["clases_menores_de_200"]
    txt_menor = (f"{NOMBRE_NAT.get(menor[0], menor[0])} ({_mil(R['naturaleza']['conteo'][menor[0]])} notas) es demasiado pequeña para evaluarse."
                 if menor else "todas las naturalezas tienen 200 notas o más.")
    pie(fig, "Cómo leer: arriba a la izquierda, cuántas palabras tiene cada epicrisis (mediana de "
             f"{_mil(R['descriptivas']['palabras']['50%'])}). Arriba a la derecha, cuántas notas aporta cada paciente: el "
             f"{100 * float((npp == 1).mean()):.0f} % aporta una sola. Abajo a la izquierda, la proporción de notas con y sin evento "
             f"en train y en test ({_r1(R['balance']['train']['pct_positivas'])} % y {_r1(R['balance']['test']['pct_positivas'])} % "
             "de positivas): la partición por paciente conserva el mismo balance. Abajo a la derecha, cuántas notas tiene cada "
             f"naturaleza del evento; {txt_menor}", 175, 0.09)
    fig.savefig(FIG / "fig_eda_distribuciones.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.9))
    epc = pd.crosstab(df.y, df.epoca, normalize="index").mul(100)
    epc.index = ["negativa", "positiva"]; epc.columns = [f"CIE-{int(c)}" for c in epc.columns]
    epc.plot(kind="bar", stacked=True, ax=ax[0], rot=0, color=["#9FB8CF", "#1F4E79"])
    ax[0].set(title=f"Época de codificación por clase (AUC época = {auc_epoca:.3f})", ylabel="%")
    ppe = R["epoca"]["pct_positivas_por_epoca"]
    ax[1].bar(list(ppe.keys()), list(ppe.values()), color="#9B2226")
    ax[1].set(title="% de positivas por época (drift de codificación)", ylabel="% positivas")
    pc = R["epoca"]["pct_por_clase"]["positiva"]
    pie(fig, "Cómo leer: a la izquierda, qué parte de cada clase se codificó en CIE-9 y qué parte en CIE-10; las dos barras son "
             f"iguales ({pc['CIE-9']:.1f} % y {pc['CIE-10']:.1f} %). A la derecha, el porcentaje de notas con evento dentro de cada "
             f"época ({ppe['CIE-9']} % en CIE-9 y {ppe['CIE-10']} % en CIE-10). Lectura: la época de codificación no separa las clases "
             f"(AUC de la época sola = {auc_epoca:.3f}), así que el modelo no puede aprender la época en lugar del evento.", 160, 0.15)
    fig.savefig(FIG / "fig_eda_balance_epoca.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right"); ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{abs(corr.iloc[i, j]) if corr.iloc[i, j] == 0 else corr.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Correlación de Spearman\n(variable objetivo en la última fila)", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.85)
    cy = R["correlaciones_spearman"]["con_y"]
    pie(fig, "Cómo leer: cada celda es la correlación de Spearman entre dos variables, de -1 a 1 (rojo, positiva; azul, negativa; "
             "blanco, sin relación). La última fila es la variable objetivo: su relación con el largo "
             f"({cy['largo (caracteres)']:.3f}), las palabras ({cy['palabras']:.3f}) y las notas del paciente "
             f"({cy['notas del paciente']:.3f}) es débil, y con la época es nula ({abs(cy['época CIE']):.3f}). El bloque rojo del "
             "centro solo indica que largo, logaritmo del largo y palabras miden lo mismo.", 100, 0.15)
    fig.savefig(FIG / "fig_eda_correlaciones.png", dpi=130); plt.close(fig)
    log.info("figuras: fig_eda_largo.png, fig_eda_distribuciones.png, fig_eda_balance_epoca.png, fig_eda_correlaciones.png")

    # ---------- 9. riesgos y decisiones
    R["riesgos"] = {
        "desbalance": f"corpus de modelado con {R['balance']['All']['pct_positivas']} % de positivas frente a una prevalencia real de 20.12 %: "
                      "las métricas se reportan también a prevalencia real (VPP real)",
        "leakage_etiqueta": f"códigos CIE-10 en el texto: {R['fuga_codigos']['pct_con_cie10']['positiva']} % de positivas vs "
                            f"{R['fuga_codigos']['pct_con_cie10']['negativa']} % de negativas",
        "leakage_paciente": f"{len(comp)} pacientes compartidos entre train y test",
        "drift": f"época CIE-9/CIE-10 emparejada por clase; AUC de la época sola {auc_epoca:.3f}",
        "atajo_largo": f"AUC del largo solo {auc_largo:.3f}; positivas por decil {pct_decil[0]} % → {pct_decil[-1]} %",
        "clases_raras_etapa2": R["naturaleza"]["clases_menores_de_200"],
        "sesgo_etiqueta": "la etiqueta es débil: códigos CIE de facturación, no una lectura clínica de la nota",
    }
    R["decisiones"] = DECISIONES
    figura_flujo(R, FIG / "fig_eda_flujo.png")
    log.info("figura del flujo del EDA: fig_eda_flujo.png")
    R["duracion_s"] = round(time.time() - t0, 1)
    guardar_json(R, REPORTES / "eda_resumen.json")

    # ---------- metrics_eda.txt (resumen legible, estilo metrics_baseline.txt)
    lineas = [f"# metrics_eda.txt · generado {datetime.now():%Y-%m-%d %H:%M:%S} por src/eda.py (fuente: reportes/eda_resumen.json)",
              f"entrada: {len(df):,} notas · {df.subject_id.nunique():,} pacientes · {ruta.stat().st_size/1e6:.0f} MB · "
              f"semilla {semilla} · split por paciente (sha256 splits.csv {R['entrada']['sha256_splits'][:16]}…)", "",
              f"calidad: nulos {R['calidad']['nulos']['columnas_sin_labels']} en las columnas de datos "
              f"(labels None en {R['calidad']['nulos']['labels_none']} negativas: esperado, no tienen naturaleza) · duplicados note_id/hadm_id/texto "
              f"{R['calidad']['duplicados']['note_id']}/{R['calidad']['duplicados']['hadm_id']}/{R['calidad']['duplicados']['texto_identico']} · "
              f"largo {R['calidad']['rango_largo'][0]}-{R['calidad']['rango_largo'][1]} car. · outliers {R['calidad']['outliers_largo']['n']} "
              f"({R['calidad']['outliers_largo']['pct']} %, se conservan)", "",
              "descriptivas (media / mediana / desv):"]
    for k, v in R["descriptivas"].items():
        lineas.append(f"  {k:<26} {v['mean']:>10} / {v['50%']:>10} / {v['std']:>10}")
    lineas += ["", "balance de la variable objetivo:"]
    for k in ["train", "test", "All"]:
        v = R["balance"][k]
        lineas.append(f"  {k:<6} notas {v['total']:>7,} · positivas {v['pct_positivas']:>6} %")
    lineas += [f"  prevalencia real en MIMIC-IV: 20.12 %", "",
               f"época por clase: {R['epoca']['pct_por_clase']} · AUC época sola {auc_epoca:.3f}",
               f"largo por clase: medianas {R['largo']['mediana_por_clase']} · AUC largo solo {auc_largo:.3f} · "
               f"positivas por decil {pct_decil}",
               f"códigos CIE-10 en texto: {R['fuga_codigos']['pct_con_cie10']} · patrón CIE-9: {R['fuga_codigos']['pct_con_patron_cie9']}",
               f"pacientes compartidos train/test: {len(comp)}",
               f"naturalezas (positivas): {R['naturaleza']['conteo']} · multietiqueta {R['naturaleza']['pct_multietiqueta']} %",
               f"Spearman con y: {R['correlaciones_spearman']['con_y']}", "",
               "riesgos: " + " | ".join(f"{k}: {v}" for k, v in R["riesgos"].items()), "",
               "decisiones accionables para el siguiente sprint:"]
    for d in DECISIONES:
        lineas.append(f"  {d['n']}. {d['decision']} [{d['paso']}] — porque {d['evidencia']} — control: {d['control']}")
    lineas += ["", f"duración: {R['duracion_s']} s"]
    (LOGS / "metrics_eda.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    log.info(f"9. decisiones: {len(DECISIONES)} · resumen -> reportes/eda_resumen.json · logs/metrics_eda.txt · {R['duracion_s']} s")


if __name__ == "__main__":
    main()
