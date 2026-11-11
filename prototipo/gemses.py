# -*- coding: utf-8 -*-
"""
MATRIZ DE PRIORIZACIÓN de los ERSP según la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01 (edición 2021): num. 19.9 a 19.13
(pp. 20 y 21) y Anexo N.° 03 (p. 39). Cada paso cita su numeral.

  19.9   agrupa los ERSP Notificados según su Id. de Evento (aquí, por proceso e Id. de Evento; ver el punto 3).
  19.10  cada 15 días hace el corte y prioriza los ERSP notificados en ese periodo.
  19.11  a) índice de frecuencia: ERSP con el mismo Id. de Evento en el proceso “entre” todos los ERSP del último
            semestre en el mismo proceso;  b) riesgo: cuatro indicadores valorados 9, 6, 3, 1 o 0;
         c) ponderaciones 0.40, 0.20, 0.15 y 0.25;  d) impacto = suma ponderada;  e) valor de riesgo = índice de
            frecuencia × impacto;  f) índice de gestión = valor de riesgo “entre” la suma de los valores de riesgo.
  Anexo 03  J = I × 100.
  19.12  niveles por percentil: I (Verde) P25, Servicios; II (Amarillo) P50, Servicio y/o Departamentos;
         III (Rojo) P75, Director de la IPRESS.
  19.13  comité según el percentil y el nivel de la IPRESS.

Puntos que la Directiva no resuelve y cómo se resuelven aquí (rotulados):
  1. Criterio del autor. El num. 19.11 a y f dice “entre” y el Anexo 03 imprime “B= (A+Σa)” e “I= H + Σh”. Se aplica
     el artículo: B = A / ΣA e I = H / ΣH. La diferencia se declara.
  2. Derivación. El num. 19.11 a pone las dos sumatorias en una sola oración que termina “en el último semestre, en el
     mismo proceso donde se identificó el ERSP”; se aplica esa ventana a las dos. Criterio del autor: el semestre son
     los 6 meses que terminan en la fecha de corte, y el periodo de corte, los 15 días que terminan en ella (fecha de
     corte incluida).
  3. Derivación. Por el num. 19.10 (“de los ERSP Notificados en dicho periodo”), las filas de la matriz son los grupos
     (proceso, Id. de Evento) con al menos un ERSP en el periodo de corte; A y ΣA cuentan todo el semestre.
  4. Criterio del autor. El num. 19.12 no dice cómo se asignan los valores que caen entre percentiles:
     J > P75 → Nivel III; P50 < J ≤ P75 → Nivel II; J ≤ P50 → Nivel I. Percentiles con interpolación lineal (numpy).
     Con pocos grupos se avisa, sin inventar un corte absoluto.
  5. Criterio del autor. ΣH y los percentiles se calculan sobre las filas de la matriz que se muestra: todos los
     procesos o uno solo, si se filtra.
  6. Nombres. El num. 19.12 dice “Servicio y/o Departamentos” y “Director de la IPRESS”; el Anexo 03, “Departamentos” y
     “Director del establecimiento”. Se usan los del artículo. El Anexo 03 llama “Nivel de Riesgo” a H, que el num.
     19.11 e llama “Valor de Riesgo”.
  7. Derivación. El num. 19.11 d define el impacto como “la sumatoria de todas las ponderaciones obtenidas”, así que
     G es la suma ponderada de C, D, E y F, fórmula que el Anexo 03 imprime como “G=(cx0.4)+(dx0.20)+(ex0.15)+(fx0.25)”.
     El “Impacto” impreso en el Anexo N.° 02 no coincide con esa suma en 230
     de los 231 eventos (p. ej., el 101: 0, 3, 3 y 6 dan 2.55 y el impreso es 2.25); solo se usa en el análisis de
     sensibilidad. Criterio del autor: el evento con valores fuera de la escala 9, 6, 3, 1 y 0 se calcula con la suma
     ponderada de esos valores tal como están impresos, y se avisa.
  8. Criterio del autor (el mismo del plan de tesis). Los num. 19.9 y 19.10 hablan de los ERSP “Notificados”; la
     matriz se aplica a los ERSP que confirma la revisión, aunque no se hayan notificado.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PESOS = {"C": 0.40, "D": 0.20, "E": 0.15, "F": 0.25}      # num. 19.11 c
ESCALA = (9, 6, 3, 1, 0)                                  # num. 19.11 b
DIAS_CORTE = 15                                           # num. 19.10
MESES_SEMESTRE = 6                                        # num. 19.11 a, “último semestre”
MIN_GRUPOS_AVISO = 4                                      # criterio del autor: por debajo, se avisa

RESPONSABLE = {"I": "Servicios", "II": "Servicio y/o Departamentos", "III": "Director de la IPRESS"}   # num. 19.12
COLOR = {"I": "Verde", "II": "Amarillo", "III": "Rojo"}                                                # num. 19.12
NIVELES_IPRESS = {"I": "IPRESS de I Nivel", "II": "IPRESS de II Nivel", "III": "IPRESS de III Nivel",
                  "OPN": "Órgano Prestador Nacional"}
COMITE_LOCAL = "Comité Local de Gestión de la Calidad"
COMITE_SERVICIOS = "Comité de Gestión de la Calidad de los Servicios de Salud"
COMITE_OPN = "Comité de Gestión de la Calidad de los Órganos Prestadores Nacionales"

# Encabezados del Anexo N.° 03 (p. 39) para la exportación
ENCABEZADOS_ANEXO03 = {
    "id_ev": "Id. Evento", "evento": "Eventos o fallas de los procesos", "proceso": "Proceso",
    "A": "A. Frecuencia", "B": "B. Índice de frecuencia",
    "C": "C. Prolongación de estancia hospitalaria", "D": "D. Complicaciones intrahospitalarias. Infecciosas y no infecciosas",
    "E": "E. Sobrecostos asociados a la no calidad", "F": "F. Insatisfacción del paciente/imagen institucional",
    "G": "G. Impacto", "H": "H. Nivel de Riesgo", "I": "I. Índice de gestión", "J": "J. Nivel de Gestión (I × 100)",
    "nivel": "Nivel de gestión", "color": "Color", "responsable": "Responsable (num. 19.12)", "comite": "Comité (num. 19.13)",
}


def comite(nivel, nivel_ipress):
    """num. 19.13. a) ERSP con percentil P25 y P50 (Niveles I y II); b) ERSP con percentil P75 (Nivel III)."""
    if nivel in ("I", "II"):
        return COMITE_LOCAL if nivel_ipress == "I" else COMITE_SERVICIOS
    if nivel == "III":
        return COMITE_OPN if nivel_ipress == "OPN" else COMITE_LOCAL
    return ""


def impacto(c, d, e, f):
    """num. 19.11 d: suma de los cuatro valores de riesgo ponderados según el num. 19.11 c."""
    return PESOS["C"] * c + PESOS["D"] * d + PESOS["E"] * e + PESOS["F"] * f


def ventanas(fecha_corte):
    """(inicio del periodo de corte, inicio del semestre, fecha de corte). Los inicios quedan fuera; el corte, dentro."""
    fin = pd.Timestamp(fecha_corte).normalize()
    return fin - pd.Timedelta(days=DIAS_CORTE), fin - pd.DateOffset(months=MESES_SEMESTRE), fin


def nivel_por_percentil(j, p50, p75):
    """Regla A (criterio del autor): J > P75 → III; P50 < J ≤ P75 → II; J ≤ P50 → I."""
    if j > p75:
        return "III"
    if j > p50:
        return "II"
    return "I"


@dataclass
class Matriz:
    tabla: pd.DataFrame
    percentiles: dict
    avisos: list = field(default_factory=list)
    periodo: tuple = ()
    semestre: tuple = ()


COLUMNAS = ["proceso", "id_ev", "n_periodo", "A", "total_proceso", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "nivel", "color", "responsable", "comite", "fuera_de_escala"]


def calcular(eventos, impactos, fecha_corte, nivel_ipress="III", proceso=None, usar_impreso=False):
    """
    eventos   una fila por ERSP confirmado, con id_ev, proceso y fecha (fecha de registro).
    impactos  tabla indexada por id_ev con C, D, E y F (y “impreso”, el impacto impreso del Anexo N.° 02).
    usar_impreso  análisis de sensibilidad: G = impacto impreso en lugar de la suma ponderada del num. 19.11 d.
    """
    ini_p, ini_s, fin = ventanas(fecha_corte)
    avisos = []
    ev = eventos[["id_ev", "proceso", "fecha"]].copy()
    ev["id_ev"] = ev.id_ev.astype(int)
    ev["fecha"] = pd.to_datetime(ev.fecha).dt.normalize()
    if proceso:
        ev = ev[ev.proceso == proceso]
    semestre = ev[(ev.fecha > ini_s) & (ev.fecha <= fin)]                       # num. 19.11 a
    periodo = semestre[semestre.fecha > ini_p]                                   # num. 19.10
    base = Matriz(pd.DataFrame(columns=COLUMNAS), {}, avisos, (ini_p, fin), (ini_s, fin))
    if periodo.empty:
        avisos.append("No hay ERSP registrados en los 15 días que terminan en la fecha de corte.")
        return base

    m = periodo.groupby(["proceso", "id_ev"]).size().rename("n_periodo").reset_index()     # num. 19.9
    m = m.join(semestre.groupby(["proceso", "id_ev"]).size().rename("A"), on=["proceso", "id_ev"])
    m = m.join(semestre.groupby("proceso").size().rename("total_proceso"), on="proceso")
    m["B"] = m.A / m.total_proceso                                                         # num. 19.11 a

    imp = impactos if impactos is not None else pd.DataFrame(columns=["C", "D", "E", "F"])
    if usar_impreso and "impreso" not in imp.columns:
        raise ValueError("La sensibilidad necesita el impacto impreso del Anexo N.° 02 (columna “impreso”).")
    cols = [c for c in ("C", "D", "E", "F", "impreso") if c in imp.columns]
    m = m.join(imp[cols].astype(float), on="id_ev")
    m["G"] = m.impreso if usar_impreso else impacto(m.C, m.D, m.E, m.F)                     # num. 19.11 d
    m["fuera_de_escala"] = ~m[["C", "D", "E", "F"]].isin(ESCALA).all(axis=1) & m[["C", "D", "E", "F"]].notna().all(axis=1)
    sin_valor = m.G.isna()
    if sin_valor.any():
        avisos.append(f"{int(sin_valor.sum())} grupo(s) sin impacto valorado: no entran al cálculo de I, J ni de los percentiles.")
    if m.fuera_de_escala.any():
        avisos.append(f"{int(m.fuera_de_escala.sum())} grupo(s) con valores del Anexo N.° 02 fuera de la escala 9, 6, 3, 1 y 0 "
                      "del num. 19.11 b; su impacto se calcula con la suma ponderada de esos valores, tal como están "
                      "impresos.")

    ok = m[~sin_valor].copy()
    ok["H"] = ok.B * ok.G                                                                   # num. 19.11 e
    suma_h = float(ok.H.sum())
    percentiles = {}
    if len(ok) and suma_h > 0:
        ok["I"] = ok.H / suma_h                                                             # num. 19.11 f
        ok["J"] = 100 * ok.I                                                                # Anexo 03
        percentiles = dict(zip(("P25", "P50", "P75"), (float(x) for x in np.percentile(ok.J, [25, 50, 75]))))
        ok["nivel"] = [nivel_por_percentil(j, percentiles["P50"], percentiles["P75"]) for j in ok.J]   # num. 19.12
        ok["color"] = ok.nivel.map(COLOR)
        ok["responsable"] = ok.nivel.map(RESPONSABLE)
        ok["comite"] = [comite(n, nivel_ipress) for n in ok.nivel]                           # num. 19.13
        if len(ok) < MIN_GRUPOS_AVISO:
            avisos.append(f"La matriz tiene {len(ok)} grupo(s): con tan pocos valores los percentiles casi no separan los "
                          "niveles. La Directiva no fija un mínimo de grupos.")
    elif len(ok):
        avisos.append("La suma de los valores de riesgo (H) es cero: el índice de gestión no se puede calcular.")
    partes = [x for x in (ok, m[sin_valor]) if len(x)]
    tabla = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0].reset_index(drop=True)
    for c in COLUMNAS:
        if c not in tabla:
            tabla[c] = np.nan
    tabla = tabla.sort_values(["J", "proceso", "id_ev"], ascending=[False, True, True], na_position="last")
    return Matriz(tabla[COLUMNAS + ([c for c in ("impreso",) if c in tabla])].reset_index(drop=True),
                  percentiles, avisos, (ini_p, fin), (ini_s, fin))


def sensibilidad(eventos, impactos, fecha_corte, nivel_ipress="III", proceso=None):
    """Compara el nivel con el impacto del num. 19.11 d frente al impacto impreso en el Anexo N.° 02."""
    if impactos is None or "impreso" not in impactos.columns:
        return None
    a = calcular(eventos, impactos, fecha_corte, nivel_ipress, proceso, usar_impreso=False).tabla
    b = calcular(eventos, impactos, fecha_corte, nivel_ipress, proceso, usar_impreso=True).tabla
    if a.empty:
        return a
    c = a[["proceso", "id_ev", "G", "J", "nivel"]].merge(
        b[["proceso", "id_ev", "G", "J", "nivel"]], on=["proceso", "id_ev"], suffixes=("_19_11", "_impreso"))
    c["cambia_nivel"] = c.nivel_19_11.ne(c.nivel_impreso) & c[["nivel_19_11", "nivel_impreso"]].notna().all(axis=1)
    return c


def anexo03(tabla, nombres=None):
    """Tabla con los encabezados del Anexo N.° 03, lista para exportar."""
    t = tabla.copy()
    t.insert(2, "evento", t.id_ev.map(nombres) if nombres is not None else "")
    cols = [c for c in ENCABEZADOS_ANEXO03 if c in t.columns]
    return t[cols].rename(columns=ENCABEZADOS_ANEXO03)
