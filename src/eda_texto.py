# -*- coding: utf-8 -*-
"""
EDA del contenido del texto de las epicrisis (lo llama src/eda.py).

  T1 · Secciones: qué encabezados habituales trae cada nota y si su presencia cambia entre clases
       (equivale a los «datos faltantes por atributo», pero dentro del texto).
  T2 · Vocabulario: cuántos términos frecuentes hay y cuáles distinguen a las notas con evento
       de las notas sin evento. Se calcula SOLO con train.

Regla de datos: solo se escriben AGREGADOS. De cada término se guarda el porcentaje de notas en
que aparece, y solo si está en al menos el 1 % de las notas de train; nunca se guarda texto clínico.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

ROJO, GRIS = "#9B2226", "#9FB8CF"
MIN_DF = 0.01
N_TOP = 15
SECCIONES = ["Chief Complaint:", "Major Surgical or Invasive Procedure:", "History of Present Illness:", "Past Medical History:",
             "Social History:", "Family History:", "Physical Exam:", "Pertinent Results:", "Brief Hospital Course:",
             "Medications on Admission:", "Discharge Medications:", "Discharge Disposition:", "Discharge Diagnosis:",
             "Discharge Condition:", "Discharge Instructions:", "Followup Instructions:"]


def _mil(n):
    return f"{int(round(n)):,}".replace(",", " ")   # espacio que no se parte entre renglones


def _pct(x):
    return round(100 * float(x), 6)


def analizar_texto(df, R, log, FIG, pie):
    T = R["texto"] = {}
    neg, pos = df.y == 0, df.y == 1

    # T1. secciones
    presente = {s: df.text.str.contains(s, case=False, regex=False) for s in SECCIONES}
    por_seccion = {s.rstrip(":"): {"pct": _pct(p.mean()), "pct_negativas": _pct(p[neg].mean()), "pct_positivas": _pct(p[pos].mean())}
                   for s, p in presente.items()}
    n_sec = sum(p.astype(int) for p in presente.values())
    S = T["secciones"] = {"por_seccion": por_seccion,
                          "presentes_en_mas_del_95_pct": int(sum(v["pct"] > 95 for v in por_seccion.values())),
                          "mediana_por_nota": float(n_sec.median()),
                          "pct_notas_con_todas": _pct((n_sec == len(SECCIONES)).mean()),
                          "pct_notas_con_menos_de_12": _pct((n_sec < 12).mean()),
                          "auc_numero_de_secciones": round(float(roc_auc_score(df.y, n_sec)), 4)}
    menos = min(por_seccion, key=lambda k: por_seccion[k]["pct"])
    dif = max(por_seccion, key=lambda k: abs(por_seccion[k]["pct_positivas"] - por_seccion[k]["pct_negativas"]))
    log.info(f"T1. secciones: {S['presentes_en_mas_del_95_pct']} de {len(SECCIONES)} en más del 95 % de las notas · con todas "
             f"{S['pct_notas_con_todas']:.2f} % · la menos frecuente: {menos} {por_seccion[menos]['pct']:.2f} % · mayor diferencia entre "
             f"clases: {dif} ({por_seccion[dif]['pct_positivas']:.2f} % positivas, {por_seccion[dif]['pct_negativas']:.2f} % negativas) · "
             f"AUC del número de secciones {S['auc_numero_de_secciones']:.3f}")

    # T2. vocabulario (solo train). Los términos se ordenan por el logaritmo de la razón de odds dividido entre su error estándar
    # (puntuación z): así un término raro no queda arriba solo por ser raro.
    tr = df[df.split == "train"]
    vec = CountVectorizer(binary=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", stop_words="english", min_df=MIN_DF)
    X = vec.fit_transform(tr.text)
    terminos = np.array(vec.get_feature_names_out())
    yp, yn = (tr.y == 1).values, (tr.y == 0).values
    n_p, n_n = int(yp.sum()), int(yn.sum())
    fp, fn = np.asarray(X[yp].sum(axis=0)).ravel(), np.asarray(X[yn].sum(axis=0)).ravel()
    lor = np.log((fp + 0.5) / (n_p - fp + 0.5)) - np.log((fn + 0.5) / (n_n - fn + 0.5))
    z = lor / np.sqrt(1 / (fp + 0.5) + 1 / (n_p - fp + 0.5) + 1 / (fn + 0.5) + 1 / (n_n - fn + 0.5))
    orden = np.argsort(z)

    def fila(i):
        return {"termino": str(terminos[i]), "pct_positivas": round(100 * fp[i] / n_p, 2), "pct_negativas": round(100 * fn[i] / n_n, 2),
                "log_odds": round(float(lor[i]), 3), "z": round(float(z[i]), 1)}

    Vc = T["vocabulario"] = {"notas_train": int(len(tr)), "terminos": int(len(terminos)), "frecuencia_minima_pct": 100 * MIN_DF,
                             "mas_asociados_al_evento": [fila(i) for i in orden[::-1][:N_TOP]],
                             "mas_asociados_a_sin_evento": [fila(i) for i in orden[:N_TOP]],
                             "nota": "términos de 3 letras o más presentes en al menos el 1 % de las notas de train; se ordenan por la "
                                     "puntuación z del logaritmo de la razón de odds de aparecer en una nota con evento frente a una sin evento"}
    log.info(f"T2. vocabulario (train, {_mil(len(tr))} notas): {_mil(len(terminos))} términos en al menos el 1 % de las notas · más "
             "asociados al evento: " + ", ".join(t["termino"] for t in Vc["mas_asociados_al_evento"]) + " · más asociados a sin "
             "evento: " + ", ".join(t["termino"] for t in Vc["mas_asociados_a_sin_evento"]))

    # ---------------- figura
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    nombres = list(por_seccion.keys())[::-1]
    yy = np.arange(len(nombres)); h = 0.38
    ax[0].barh(yy + h / 2, [por_seccion[k]["pct_negativas"] for k in nombres], h, color=GRIS)
    ax[0].barh(yy - h / 2, [por_seccion[k]["pct_positivas"] for k in nombres], h, color=ROJO)
    ax[0].set_yticks(yy, nombres, fontsize=8); ax[0].set(xlabel="% de epicrisis que traen la sección", xlim=(0, 100))
    ax[0].set_title("Secciones de la epicrisis por clase\n(gris: sin evento · rojo: con evento)", fontsize=10)
    for a, clave, titulo in [(ax[1], "mas_asociados_al_evento", "Términos que más distinguen\na las notas con evento"),
                             (ax[2], "mas_asociados_a_sin_evento", "Términos que más distinguen\na las notas sin evento")]:
        t = Vc[clave][::-1]
        y2 = np.arange(len(t))
        a.barh(y2 + h / 2, [x["pct_negativas"] for x in t], h, color=GRIS, label="sin evento")
        a.barh(y2 - h / 2, [x["pct_positivas"] for x in t], h, color=ROJO, label="con evento")
        a.set_yticks(y2, [x["termino"] for x in t], fontsize=8.5); a.set_xlabel("% de epicrisis de train que usan el término")
        a.set_xlim(0, 100); a.set_title(titulo, fontsize=10); a.legend(fontsize=8, loc="lower right")
    p1, n1 = Vc["mas_asociados_al_evento"][0], Vc["mas_asociados_a_sin_evento"][0]
    pie(fig, f"Cómo leer: a la izquierda, el porcentaje de epicrisis que trae cada uno de los {len(SECCIONES)} encabezados habituales de la "
             "nota de alta, separado por clase. Al centro y a la derecha, los términos que más distinguen a las notas con evento y a "
             f"las notas sin evento, entre los {_mil(len(terminos))} términos que aparecen en al menos el 1 % de las notas de train; las "
             "barras dicen en qué porcentaje de las notas de cada clase aparece el término. Lectura: "
             f"{S['presentes_en_mas_del_95_pct']} de las {len(SECCIONES)} secciones están en más del 95 % de las notas y el "
             f"{S['pct_notas_con_todas']:.1f} % de las notas trae las {len(SECCIONES)}; la que más falta es «{menos}» "
             f"({por_seccion[menos]['pct']:.1f} %) y la que más cambia entre clases es «{dif}» ({por_seccion[dif]['pct_positivas']:.1f} % "
             f"con evento y {por_seccion[dif]['pct_negativas']:.1f} % sin evento). En el vocabulario, «{p1['termino']}» aparece en el "
             f"{p1['pct_positivas']:.1f} % de las notas con evento y en el {p1['pct_negativas']:.1f} % de las notas sin evento; "
             f"«{n1['termino']}», en el {n1['pct_positivas']:.1f} % y el {n1['pct_negativas']:.1f} %.", 215, 0.2)
    fig.savefig(FIG / "fig_eda_texto.png", dpi=130); plt.close(fig)
    log.info("figura: fig_eda_texto.png")
