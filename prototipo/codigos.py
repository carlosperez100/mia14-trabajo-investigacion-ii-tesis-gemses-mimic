# -*- coding: utf-8 -*-
"""
CÓDIGOS DE EVENTO: decide si la epicrisis ya tiene un código CIE de evento adverso o pasa a la cola de
“posible evento sin código”.

Premisas (citadas):
  - Plan de tesis, población de la HE2 y la HE3: las epicrisis sin código no tienen “filas en las tablas de la etiqueta
    ni códigos de las familias de causa explícita o de efecto adverso de medicamento”.
  - src/eda_variables.py: PREF_CAUSAL (T80 a T88, Y63, Y65, Y83, Y84, W00 a W19, L89), _efecto_adverso (T36 a T50 con un
    5 en el sexto carácter) y FAM_CIE9 (las cinco familias CIE-9).
  - data/catalogos/anexo02_equivalencia_cie10_oficial.csv: la equivalencia CIE-10 con el Anexo N.° 02, validada código
    por código el 29-10-2026.

Derivación (regla del prototipo, que aproxima pero no reproduce la población de la HE2 y la HE3, porque las tablas de
la etiqueta se arman por hospitalización con el pipeline de la tesis):
  En CIE-10, un código es de evento si empieza con un código de la equivalencia validado con evento, solo con
  naturaleza o sin equivalente, o si cae en PREF_CAUSAL o en _efecto_adverso. En CIE-9 cuentan las cinco familias.
Criterio del autor: los códigos validados como “no es evento adverso” no cuentan, salvo que caigan en una familia,
aunque en la tesis son del estrato A2 de la tabla de la etiqueta. Por eso la cola “sin código” es más amplia que la
población del plan: por ejemplo, N17.9 no saca a la epicrisis de la cola.
"""
import re
from functools import lru_cache

import pandas as pd

from .rutas import CATALOGOS

# Copias exactas de src/eda_variables.py (tests/test_codigos.py verifica que no se separen)
PREF_CAUSAL = tuple([f"T8{i}" for i in range(9)] + ["Y63", "Y65", "Y83", "Y84"] + [f"W{i:02d}" for i in range(20)] + ["L89"])
FAM_CIE9 = {"complicaciones_atencion": r"^99[6-9]", "incidentes_asistenciales": r"^E87[0-9]",
            "farmacos_uso_terapeutico": r"^E9[34][0-9]", "caidas": r"^E88[0-8]", "ulcera_presion": r"^707[0-9]"}

NO_EA = "no es evento adverso"


def normalizar(codigo):
    """Mayúsculas, sin punto ni espacios: “t81.4xxa” -> “T814XXA” (MIMIC-IV guarda los códigos sin punto)."""
    return re.sub(r"[^A-Z0-9]", "", str(codigo).upper())


def efecto_adverso(c):
    """Efecto adverso de un medicamento en CIE-10-CM: T36 a T50 con un 5 en el sexto carácter (T38.0X5A)."""
    return "T36" <= c[:3] <= "T50" and len(c) >= 6 and c[5] == "5"


@lru_cache(maxsize=1)
def tabla():
    t = pd.read_csv(CATALOGOS / "anexo02_equivalencia_cie10_oficial.csv", dtype=str)
    t["norm"] = t.codigo_cie10.map(normalizar)
    return t[["codigo_cie10", "norm", "clasificacion", "estado_validado", "id_ev_validado", "evento_validado",
              "naturaleza_validada"]]


def clasificar(codigo, version=10):
    """Clasifica un código. Devuelve un dict con es_evento, motivo y, si la equivalencia lo da, el evento sugerido."""
    c = normalizar(codigo)
    out = {"codigo": str(codigo).strip(), "es_evento": False, "motivo": "", "id_ev_sugerido": None, "evento_sugerido": None}
    if not c:
        return out
    if int(version) == 9:
        for familia, rx in FAM_CIE9.items():
            if re.match(rx, c):
                out.update(es_evento=True, motivo=f"familia CIE-9 “{familia}”")
                return out
        return out
    t = tabla()
    filas = t[[c.startswith(n) for n in t.norm]]
    validas = filas[filas.estado_validado != NO_EA]
    motivos = []
    if len(validas):
        estados = sorted(set(validas.estado_validado))
        motivos.append("equivalencia con el Anexo N.° 02 (" + ", ".join(estados) + ")")
        con_evento = validas[validas.estado_validado == "con evento"].drop_duplicates("id_ev_validado")
        if len(con_evento) == 1:
            out.update(id_ev_sugerido=int(float(con_evento.id_ev_validado.iloc[0])),
                       evento_sugerido=con_evento.evento_validado.iloc[0])
    if c.startswith(PREF_CAUSAL):
        motivos.append("familia de causa explícita")
    if efecto_adverso(c):
        motivos.append("efecto adverso de medicamento")
    if motivos:
        out.update(es_evento=True, motivo="; ".join(motivos))
    elif len(filas):
        out["motivo"] = "en la equivalencia, validado como “no es evento adverso”"
    return out


def revisar(codigos, version=10):
    """Clasifica una lista de códigos. Devuelve (sin_codigo, lista de los códigos de evento y de los descartados)."""
    resultados = [clasificar(c, version) for c in codigos]
    de_evento = [r for r in resultados if r["es_evento"]]
    anotados = [r for r in resultados if r["es_evento"] or r["motivo"]]
    return len(de_evento) == 0, anotados
