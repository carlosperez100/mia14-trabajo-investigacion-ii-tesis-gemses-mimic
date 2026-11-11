# -*- coding: utf-8 -*-
"""
REVISIÓN de cada posible evento por una persona, con los criterios del manual de anotación v2 de la tesis
(08_anexos/CODEBOOK_v2_MANUAL_ANOTACION.md), y tipificación según la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01.

Del manual v2 se toman:
  §1.2   las tres condiciones del evento adverso: D1 daño, D2 asociación con la atención y D3 no intencional;
  §1.3   la escala de nexo C1 a C4 (solo se anota C3 o C4) y el deslinde con la enfermedad de base (D2b), que va primero;
  §1.4   el umbral de gravedad E o mayor de la escala NCC MERP/IHI;
  §3     el eje ERSP (evento adverso o evento centinela, con los criterios EC-1 a EC-3);
  §4.1   la regla del ancla: sin cita literal de la epicrisis no hay evento (comparación normalizada);
  §4.5   la confianza definida por criterio;
  §4.8   las reglas de bloqueo.
De la Directiva: el Id. de Evento del Anexo N.° 02 (num. 19.5), el proceso (num. 19.6) y el servicio (num. 19.7).

Simplificaciones del prototipo respecto del manual (criterio del autor; se listan en prototipo/README.md, sección 7):
  - el ancla se pega en un campo y se verifica contra el texto, en lugar de capturarse por selección con sus posiciones;
  - para EC-2 y EC-3 se piden las cuatro casillas del art. 5.4 y una sola cita, la del desenlace, no una por casilla, y
    no se verifica que esa cita contenga la frase de vínculo de EC-1 o el desenlace de EC-2;
  - la vía (A o B) sale del nexo elegido; cada evento lleva un solo Id. de Evento (num. 19.5), no varias naturalezas;
  - no se registran el origen del daño ni su marcador, ni los minutos y las secciones revisadas al cerrar el caso;
  - no hay bitácoras B-1, B-2 ni B-3: el caso sin eventos anotables se cierra con su motivo (sin daño ≥ E, C1, C2 o
    D3), que queda en la bitácora general sin texto, y el C2 no pasa a adjudicación;
  - la adjudicación de los dudosos y de los centinelas la registra cualquier revisor, con su nombre en la bitácora.
El evento centinela pasa siempre a adjudicación, como pide el manual (§5, pasos P5 y P6).
"""
import re
import unicodedata

from . import catalogo

GRAVEDADES = {   # §1.4.2
    "E": "E · Daño temporal que requirió intervención",
    "F": "F · Daño temporal que motivó UCI, traslado, reintervención o retraso del alta",
    "G": "G · Daño permanente",
    "H": "H · Intervención necesaria para mantener la vida",
    "I": "I · Muerte atribuida al evento por frase de vínculo (EC-1)",
    "A-D": "A–D o daño menor que E · Incidente sin daño (A–D) o evento adverso infraumbral: no crea fila",
}
D2B = {          # §1.3.1
    "b": "(b) El texto no ofrece explicación de enfermedad de base: pasar al nexo",
    "a": "(a) El texto atribuye el daño a la enfermedad de base o a una condición presente al ingreso: C1, no se anota",
    "c": "(c) El texto sostiene ambas causas, explicación competidora (R-11): C2, no se anota",
}
NEXOS = {        # §1.3
    "C4": "C4 · Atribución explícita (Vía A, término de la lista cerrada)",
    "C3": "C3 · Vínculo documentado (Vía B, tres condiciones)",
}
VIA_B = ["Intervención nombrable", "Temporalidad documentada", "Acción confirmatoria"]   # §1.3.3
ERSP = {"EA": "Evento adverso (art. 5.3)", "EC": "Evento centinela (art. 5.4)"}         # §3.1
CRITERIOS_EC = {"EC-1": "EC-1 · Muerte con frase de vínculo",                           # §3.2.1
                "EC-2": "EC-2 · Daño físico con los cuatro elementos del art. 5.4",
                "EC-3": "EC-3 · Daño psicológico con los cuatro elementos del art. 5.4 (reservado: requiere consulta previa al "
                         "adjudicador, §3.2.5)"}
ELEMENTOS_54 = ["Severo", "Permanente", "De novo (no estaba presente antes)",              # §3.2.2
                "Requiere tratamiento permanente o cambio en el estilo de vida"]
MOTIVOS_DUDOSO = ["redactado", "temporalidad", "atribución", "foco", "desenlace", "otro"]  # §4.2
MIN_PALABRAS_CITA = 3            # criterio del autor: evita anclas triviales (“the”, “pain”)
# Cierre sin eventos anotables (criterio del autor): el motivo reemplaza a las bitácoras B-1 a B-3 del manual
MOTIVOS_SIN_EVENTO = {
    "sin_dano": "No hay daño de gravedad E o mayor (incidente sin daño, A–D, o evento adverso infraumbral, con daño "
                "menor que E; el manual los lleva a B-1 y B-2)",
    "C1": "El daño se atribuye a la enfermedad de base o a una condición presente al ingreso (C1)",
    "C2": "Explicación competidora o vínculo posible (C2, R-11; el manual lo lleva a B-3 y a adjudicación)",
    "D3": "El daño era el fin buscado o parte del plan documentado (D3, R-08)",
}

_COMILLAS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})


def normalizar(s):
    """§4.1: colapsa espacios y saltos de línea, ignora mayúsculas y comillas tipográficas."""
    s = unicodedata.normalize("NFKC", str(s or "")).translate(_COMILLAS)
    return re.sub(r"\s+", " ", s).strip().lower()


def cita_en_texto(cita, texto):
    c = normalizar(cita)
    return len(c.split()) >= MIN_PALABRAS_CITA and c in normalizar(texto)


def estado_inicial(ev):
    """“dudoso” (pasa a adjudicación) si el revisor lo marcó o si es un evento centinela (§5, P5 y P6); si no, “confirmado”."""
    return "dudoso" if ev.get("dudoso") or ev.get("ersp") == "EC" else "confirmado"


def confianza(nexo, dudoso):
    """§4.5: Alta si Vía A (C4); Media si Vía B (C3); Baja si quedó alguna duda."""
    if dudoso:
        return "Baja"
    return "Alta" if nexo == "C4" else "Media"


def validar(ev, texto):
    """Reglas de bloqueo (§4.8 y la Directiva). Devuelve la lista de razones por las que el evento no se registra."""
    errores = []
    if not cita_en_texto(ev.get("cita_ancla"), texto):
        errores.append(f"La cita ancla debe copiarse literal de la epicrisis (§4.1) y, a criterio del autor, tener "
                       f"{MIN_PALABRAS_CITA} palabras o más.")
    g = ev.get("gravedad")
    if g not in ("E", "F", "G", "H", "I"):
        errores.append("Gravedad menor que E: el incidente sin daño (A–D) o el evento adverso infraumbral no crea fila "
                       "(§4.8, regla 4). Si no hay otro evento, el caso se cierra sin eventos con el motivo “No hay daño "
                       "de gravedad E o mayor”.")
    if ev.get("d2b") == "a":
        errores.append("El texto atribuye el daño a la enfermedad de base: nexo C1, no se anota (§1.3.1).")
    elif ev.get("d2b") == "c":
        errores.append("Explicación competidora (R-11): nexo C2, no se anota (§1.3.1 y §4.8, regla 5). Si no hay otro evento, "
                       "el caso se cierra sin eventos con el motivo “Explicación competidora o vínculo posible”.")
    elif ev.get("d2b") != "b":
        errores.append("Falta el deslinde con la enfermedad de base (D2b), que se resuelve primero (§1.3.1).")
    nexo = ev.get("nexo")
    if nexo not in ("C3", "C4"):
        errores.append("El nexo debe ser C3 o C4 (§1.3).")
    elif nexo == "C3":
        via_b = ev.get("via_b") or []
        if len(via_b) < 3 or not all(via_b):
            errores.append("Con C3 deben cumplirse las tres condiciones de la Vía B; si falta una, es C2 y no se anota (§1.3.3).")
    if ev.get("intencional"):
        errores.append("El daño era el fin buscado o parte del plan: no hay evento (D3, R-08).")
    ersp = ev.get("ersp")
    if ersp not in ERSP:
        errores.append("Falta la categoría ERSP: evento adverso o evento centinela (§3.1).")
    if ersp == "EC":
        crit = ev.get("criterio_ec")
        if crit not in CRITERIOS_EC:
            errores.append("Un evento centinela necesita su criterio EC-1, EC-2 o EC-3 (§4.8, regla 2).")
        if not cita_en_texto(ev.get("cita_desenlace"), texto):
            errores.append("Un evento centinela necesita la cita literal del desenlace (§4.8, regla 2).")
        if crit in ("EC-2", "EC-3") and not (len(ev.get("elementos_54") or []) == 4 and all(ev["elementos_54"])):
            errores.append("EC-2 y EC-3 exigen los cuatro elementos del art. 5.4 (§4.8, regla 3).")
    if g == "I" and not (ersp == "EC" and ev.get("criterio_ec") == "EC-1"):
        errores.append("La gravedad I exige un evento centinela EC-1 con frase de vínculo; la muerte no atribuida no da I (R-10).")
    try:
        catalogo.nombre(ev.get("id_ev"))
    except (KeyError, TypeError, ValueError):
        errores.append("Falta el Id. de Evento del Anexo N.° 02 (num. 19.5).")
    if ev.get("proceso") not in catalogo.LISTA_PROCESOS:
        errores.append("Falta el proceso al que pertenece el ERSP (num. 19.6).")
    if not str(ev.get("servicio") or "").strip():
        errores.append("Falta el departamento o servicio (num. 19.7).")
    if ev.get("dudoso") and ev.get("motivo_dudoso") not in MOTIVOS_DUDOSO:
        errores.append("Un evento dudoso necesita su motivo (§4.2).")
    return errores
