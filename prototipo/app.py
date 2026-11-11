# -*- coding: utf-8 -*-
"""
INTERFAZ LOCAL del prototipo (Streamlit): 1) detección, 2) revisión y 3) priorización con la Matriz de Priorización
GEMSES, según la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01.

Ejecutar desde la raíz del repositorio:   streamlit run prototipo/app.py
Atiende solo en 127.0.0.1 (prototipo/.streamlit/config.toml, que Streamlit lee desde la carpeta del script) y se
detiene si la dirección de escucha es otra, salvo en el contenedor. No muestra términos, coeficientes ni resaltados del
modelo.
"""
import html
import io
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from prototipo import almacen, catalogo, contrato, detector, gemses, modelo_demo, revision  # noqa: E402
from prototipo.rutas import DEMO  # noqa: E402

st.set_page_config(page_title="Prototipo · eventos adversos y Matriz GEMSES", page_icon="🩺", layout="wide")

# Privacidad: fuera del contenedor (que publica el puerto solo en 127.0.0.1), la aplicación solo atiende en esta máquina.
if os.environ.get("PROTOTIPO_CONTENEDOR") != "1" and st.get_option("server.address") not in ("127.0.0.1", "localhost", "::1"):
    st.error("Por privacidad, el prototipo solo atiende en esta máquina (127.0.0.1), como fija prototipo/.streamlit/config.toml. "
             "Se detuvo porque la dirección de escucha es otra.")
    st.stop()

# La traducción automática del navegador altera el DOM de Streamlit y rompe la página; además, el manual v2 prohíbe
# usar la traducción para decidir o citar. Se marca la página como no traducible.
components.html("""<script>
try { const d = window.parent.document; d.documentElement.setAttribute('translate', 'no');
      d.documentElement.classList.add('notranslate');
      if (!d.querySelector('meta[name="google"]')) { const m = d.createElement('meta'); m.name = 'google';
        m.content = 'notranslate'; d.head.appendChild(m); } } catch (e) {}
</script>""", height=0)

st.markdown("""<style>
.epicrisis {white-space: pre-wrap; font-family: Consolas, 'Courier New', monospace; font-size: 0.86rem; line-height: 1.45;}
.banda-demo {background: #9B2226; color: #fff; padding: 0.45rem 0.8rem; border-radius: 6px; font-weight: 600;}
</style>""", unsafe_allow_html=True)

COLOR_NIVEL = {"I": "#d4edda", "II": "#fff3cd", "III": "#f5c6cb"}
CLAVES_EVENTO = {"ev_cita": "", "ev_grav": "E", "ev_d2b": "b", "ev_nexo": "C4", "ev_vb0": False, "ev_vb1": False,
                 "ev_vb2": False, "ev_d3": "No", "ev_ersp": "EA", "ev_crit": "", "ev_cdes": "", "ev_e0": False,
                 "ev_e1": False, "ev_e2": False, "ev_e3": False, "ev_proc": "Hospitalización", "ev_serv": "",
                 "ev_dudoso": False, "ev_motivo": "", "ev_nota": ""}

st.session_state.setdefault("revisor", "Revisor 1")
for k, v in CLAVES_EVENTO.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------- recursos
@st.cache_resource(show_spinner="Cargando el modelo…")
def _detector(ruta, marca_tiempo):      # la fecha de modificación entra a la clave: un modelo reentrenado se recarga
    return detector.cargar(ruta)


def obtener_detector():
    ruta = detector.ruta_activa()
    if not ruta.exists():
        return None, f"No se encontró el modelo ({ruta.name})."
    try:
        return _detector(str(ruta), ruta.stat().st_mtime), None
    except detector.ErrorModelo as e:
        return None, str(e)


@st.cache_data
def ejemplos():
    return pd.read_csv(DEMO / "epicrisis_demo.csv", dtype=str, keep_default_na=False)


def avisar(clave, tipo, texto):
    st.session_state[clave] = (tipo, texto)


def mostrar_aviso(clave):
    if clave in st.session_state:
        tipo, texto = st.session_state.pop(clave)
        getattr(st, tipo)(texto)


# ---------------------------------------------------------------- acciones (callbacks)
def _entrenar_demo():
    modelo_demo.main()
    _detector.clear()
    avisar("aviso_lateral", "success", "Modelo de demostración entrenado.")


def _cargar_epicrisis_demo():
    det, _ = obtener_detector()
    eps = {e.id_epicrisis: e for e in (contrato.desde_fila(r) for r in ejemplos().to_dict("records"))}
    dets, errores = detector.detectar(det, list(eps.values()))
    with almacen.conexion() as con:
        almacen.guardar_lote(con, list(eps.values()), dets, "demo", reemplazar=True, accion="demo_epicrisis_cargadas")
    if errores:
        avisar("aviso_lateral", "error", "No cumplen el contrato: " + ", ".join(errores))
    else:
        avisar("aviso_lateral", "success", f"{len(dets)} epicrisis inventadas procesadas por el modelo; sus revisiones "
                                           "anteriores se reiniciaron.")


def _cargar_ersp_demo():
    tabla = modelo_demo.ersp_demo(date.today())
    with almacen.conexion() as con:
        almacen.cargar_ersp_demo(con, tabla)
    avisar("aviso_lateral", "success", f"{len(tabla)} ERSP inventados cargados en los últimos seis meses; los cargados antes "
                                       "se reemplazaron.")


def _borrar_demo():
    with almacen.conexion() as con:
        n_e, n_v = almacen.borrar_demo(con)
    avisar("aviso_lateral", "info", f"Datos de demostración borrados: {n_e} epicrisis y {n_v} ERSP.")


def _prefill_ejemplo():
    sel = st.session_state.get("ejemplo_sel")
    if not sel:
        return
    fila = ejemplos().set_index("id_epicrisis").loc[sel]
    det, _ = obtener_detector()
    cats = det.categorias() if det else {}
    st.session_state.update(f_id=sel, f_texto=fila.texto, f_codigos=fila.codigos_cie, f_version=int(fila.version_cie or 10),
                            f_reemplazar=False)         # reemplazar borra la revisión: solo si el revisor lo marca
    for n in contrato.NUMERICAS:
        st.session_state[f"f_{n}"] = fila.get(n, "")
    for n in contrato.CATEGORICAS:
        v = fila.get(n, "")
        st.session_state[f"f_{n}"] = v if v in cats.get(n, []) else ""
    for n in contrato.BINARIAS:
        st.session_state[f"f_{n}"] = fila.get(n, "")


def _detectar_una():
    det, _ = obtener_detector()
    s = st.session_state
    variables = {n: s.get(f"f_{n}", "") for n in contrato.VARIABLES}
    ep = contrato.Epicrisis(s.get("f_id", "").strip(), s.get("f_texto", ""), {k: v for k, v in variables.items() if v != ""},
                            contrato.leer_codigos(s.get("f_codigos", "")), int(s.get("f_version", 10)))
    dets, errores = detector.detectar(det, [ep])
    if errores:
        s["res_una"] = {"errores": sum(errores.values(), [])}
        return
    res = {"deteccion": dets[0].como_dict(), "avisos": detector.categorias_no_vistas(det, [ep])}
    with almacen.conexion() as con:
        if almacen.existe(con, ep.id_epicrisis) and not s.get("f_reemplazar"):
            res["info"] = (f"La epicrisis {ep.id_epicrisis} ya estaba registrada: la detección se muestra, pero no se guardó. "
                           "Para guardarla de nuevo y reiniciar su revisión, se marca “Reemplazar”.")
            s["res_una"] = res
            return
        ej = ejemplos().set_index("id_epicrisis")          # “demo” solo si es un ejemplo inventado sin cambios
        es_ejemplo = ep.id_epicrisis in ej.index and ep.texto == ej.loc[ep.id_epicrisis, "texto"]
        almacen.guardar_epicrisis(con, ep, "demo" if es_ejemplo else "usuario", reemplazar=True)
        almacen.guardar_deteccion(con, dets[0])
    s["res_una"] = res


def _reset_evento():
    for k, v in CLAVES_EVENTO.items():
        st.session_state[k] = v
    st.session_state.pop("ev_idev", None)
    st.session_state["ev_fecha"] = date.today()


def _registrar_evento(id_epicrisis, texto, origen):
    s = st.session_state
    ev = {"id_epicrisis": id_epicrisis, "cita_ancla": s.ev_cita, "gravedad": s.ev_grav, "d2b": s.ev_d2b, "nexo": s.ev_nexo,
          "via_b": [s.ev_vb0, s.ev_vb1, s.ev_vb2], "intencional": s.ev_d3 == "Sí", "ersp": s.ev_ersp,
          "criterio_ec": s.ev_crit or None, "cita_desenlace": s.ev_cdes, "elementos_54": [s.ev_e0, s.ev_e1, s.ev_e2, s.ev_e3],
          "id_ev": s.get("ev_idev"), "proceso": s.ev_proc, "servicio": s.ev_serv.strip(), "dudoso": s.ev_dudoso,
          "motivo_dudoso": s.ev_motivo or None}
    errores = revision.validar(ev, texto)
    if errores:
        s["ev_resultado"] = ("error", errores)
        return
    ev.update(fecha=s.ev_fecha.isoformat(), via="A" if ev["nexo"] == "C4" else "B",
              confianza=revision.confianza(ev["nexo"], ev["dudoso"]), nota=s.ev_nota.strip() or None,
              revisor=s.revisor.strip() or "Revisor", origen=origen, estado=revision.estado_inicial(ev))
    if ev["ersp"] != "EC":
        ev.update(criterio_ec=None, cita_desenlace=None)
    with almacen.conexion() as con:
        id_evento = almacen.guardar_evento(con, ev)
    if ev["ersp"] == "EC":
        nota = " (evento centinela: pasa a adjudicación)."
    else:
        nota = " (dudoso: pasa a adjudicación)." if ev["dudoso"] else "."
    s["ev_resultado"] = ("ok", f"Evento {id_evento} registrado: {catalogo.etiqueta(ev['id_ev'])}{nota}")
    _reset_evento()


def _estado_evento(id_evento, estado):
    with almacen.conexion() as con:
        almacen.cambiar_estado(con, id_evento, estado, st.session_state.revisor)


def _cerrar(id_epicrisis, resultado):
    motivo = st.session_state.get("rev_motivo") or None if resultado == "sin_eventos" else None
    with almacen.conexion() as con:
        almacen.cerrar(con, id_epicrisis, resultado, st.session_state.revisor, motivo)
    st.session_state["rev_confirma"] = False
    st.session_state["rev_motivo"] = ""


def _reabrir(id_epicrisis):
    with almacen.conexion() as con:
        almacen.reabrir(con, id_epicrisis, st.session_state.revisor)


def _guardar_impactos(tabla):
    with almacen.conexion() as con:
        for id_ev, f in tabla.iterrows():
            if f[["C", "D", "E", "F"]].notna().all():
                almacen.guardar_impacto(con, id_ev, f.C, f.D, f.E, f.F)


# ---------------------------------------------------------------- barra lateral
def barra_lateral(det, error):
    with st.sidebar:
        st.markdown("### Prototipo del sistema")
        st.caption("Detección → revisión → Matriz de Priorización")
        if det is None:
            st.error(error)
            st.button("Entrenar el modelo de demostración", on_click=_entrenar_demo, type="primary", width="stretch")
        else:
            if det.es_demo:
                st.markdown('<div class="banda-demo">DEMOSTRACIÓN · modelo entrenado con epicrisis inventadas, sin valor clínico</div>',
                            unsafe_allow_html=True)
            else:
                st.success("Modelo local: configuración actual (D2, A+B+C, corte 0.5)")
            st.caption(f"Huella SHA-256 del modelo: `{det.sha256[:16]}…`")
            for a in det.avisos:
                st.warning(a)
        st.text_input("Revisor", key="revisor", help="Nombre o código de quien revisa; queda en la bitácora.")
        st.divider()
        st.markdown("**Datos de demostración**")
        st.button("Cargar o reiniciar las 10 epicrisis inventadas", on_click=_cargar_epicrisis_demo, disabled=det is None,
                  width="stretch")
        st.button("Cargar o reiniciar los ERSP inventados del semestre", on_click=_cargar_ersp_demo, width="stretch")
        st.button("Borrar los datos de demostración", on_click=_borrar_demo, width="stretch")
        mostrar_aviso("aviso_lateral")


# ---------------------------------------------------------------- 1. detección
def tab_deteccion(det):
    st.markdown("#### 1 · Detección")
    st.caption("El modelo calcula la probabilidad de evento adverso de la epicrisis y la marca con el corte de 0.5. "
               "Si la epicrisis marcada no tiene ningún código de evento, pasa a la cola “Posible evento sin código”. "
               "El modelo solo marca; la decisión es del revisor.")
    if det is None:
        st.info("Primero se necesita un modelo (ver la barra lateral).")
        return
    modo = st.radio("Entrada", ["Una epicrisis", "Lote (CSV)"], horizontal=True, key="modo_entrada")
    if modo == "Una epicrisis":
        formulario_una(det)
    else:
        formulario_lote(det)
    st.markdown("##### Epicrisis procesadas")
    tabla_detecciones()


def formulario_una(det):
    cats = det.categorias()
    for n in contrato.VARIABLES:
        st.session_state.setdefault(f"f_{n}", "")
    st.selectbox("Cargar una epicrisis inventada de ejemplo", [""] + ejemplos().id_epicrisis.tolist(), key="ejemplo_sel",
                 on_change=_prefill_ejemplo)
    with st.form("f_detectar"):
        c1, c2 = st.columns([1, 2])
        c1.text_input("Identificador", key="f_id", help="Letras, dígitos, guion, guion bajo o punto. No se usan identificadores de MIMIC-IV.")
        c1.text_input("Códigos CIE del alta, separados por “;”", key="f_codigos")
        c1.radio("Versión CIE", [10, 9], horizontal=True, key="f_version")
        c1.checkbox("Reemplazar si el identificador ya existe", key="f_reemplazar")
        c2.text_area("Epicrisis (en inglés)", height=300, key="f_texto")
        with st.expander("Variables de la hospitalización (opcionales; el faltante se imputa como en el entrenamiento)"):
            cols = st.columns(3)
            for i, n in enumerate(contrato.NUMERICAS):
                cols[i % 3].text_input(contrato.ETIQUETAS[n], key=f"f_{n}")
            for i, n in enumerate(contrato.CATEGORICAS):
                cols[i % 3].selectbox(contrato.ETIQUETAS[n], [""] + cats.get(n, []), key=f"f_{n}")
            for i, n in enumerate(contrato.BINARIAS):
                cols[i % 3].selectbox(contrato.ETIQUETAS[n], ["", "0", "1"], key=f"f_{n}")
        st.form_submit_button("Detectar", type="primary", on_click=_detectar_una)
    res = st.session_state.pop("res_una", None)
    if res and res.get("errores"):
        st.error("La epicrisis no cumple el contrato de entrada:\n\n" + "\n".join(f"- {e}" for e in res["errores"]))
    elif res:
        if res.get("info"):
            st.info(res["info"])
        d = res["deteccion"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad", f"{d['probabilidad']:.3f}")
        c2.metric("Marca (corte 0.5)", "Sí" if d["marca"] else "No")
        c3.metric("Cola", d["cola"])
        codigos_evento = [c for c in d["codigos_evento"] if c["es_evento"]]
        if codigos_evento:
            st.write("Códigos de evento encontrados: " + "; ".join(f"{c['codigo']} ({c['motivo']})" for c in codigos_evento))
        else:
            st.write("La epicrisis no tiene ningún código de evento.")
        for a in res.get("avisos", []):
            st.warning(a)


def _procesar_lote():
    s = st.session_state
    det, _ = obtener_detector()
    df = pd.read_csv(s.f_lote, dtype=str, keep_default_na=False)
    falta = {"id_epicrisis", "texto"} - set(df.columns)
    if falta:
        s["res_lote"] = {"error": f"Faltan las columnas {sorted(falta)}."}
        return
    eps = [contrato.desde_fila(r) for r in df.to_dict("records")]
    if len({e.id_epicrisis for e in eps}) < len(eps):
        s["res_lote"] = {"error": "El CSV repite identificadores de epicrisis."}
        return
    dets, errores = detector.detectar(det, eps)
    with almacen.conexion() as con:
        omitidas = almacen.guardar_lote(con, eps, dets, "lote", reemplazar=s.f_lote_reemplazar)
    s["res_lote"] = {"dets": dets, "errores": errores, "omitidas": omitidas,
                     "csv": contrato.tabla_salida(dets).to_csv(index=False).encode("utf-8-sig")}


def formulario_lote(det):
    st.caption("Columnas: id_epicrisis, texto, las 18 variables (opcionales), codigos_cie (separados por “;”) y version_cie. "
               "La salida no repite el texto. Ejemplo: prototipo/datos_demo/epicrisis_demo.csv.")
    st.file_uploader("CSV de entrada", type=["csv"], key="f_lote", on_change=lambda: st.session_state.pop("res_lote", None))
    st.checkbox("Reemplazar las epicrisis que ya existan", key="f_lote_reemplazar")
    st.button("Procesar el lote", type="primary", on_click=_procesar_lote, disabled=st.session_state.get("f_lote") is None)
    res = st.session_state.get("res_lote")
    if not res:
        return
    if res.get("error"):
        st.error(res["error"])
        return
    dets = res["dets"]
    st.success(f"{len(dets)} epicrisis procesadas; {sum(d.marca for d in dets)} marcadas; "
               f"{sum(d.marca and d.sin_codigo for d in dets)} pasan a la cola “Posible evento sin código”.")
    if res["errores"]:
        st.error("No cumplen el contrato: " + "; ".join(f"{k}: {' '.join(v)}" for k, v in res["errores"].items()))
    if res["omitidas"]:
        st.warning(f"Ya existían y no se reemplazaron: {', '.join(res['omitidas'])}.")
    st.download_button("Descargar el resultado del lote (CSV, sin texto)", res["csv"], "resultado_lote.csv", "text/csv",
                       on_click="ignore")


def tabla_detecciones():
    with almacen.conexion() as con:
        t = almacen.detecciones(con)
    if t.empty:
        st.info("Aún no hay epicrisis procesadas.")
        return
    t["codigos"] = t.codigos_evento.map(lambda l: "; ".join(c["codigo"] for c in l if c["es_evento"]) or "ninguno")
    t["revision"] = t.cierre.map({"con_eventos": "Cerrada con eventos", "sin_eventos": "Cerrada sin eventos"}).fillna("Pendiente")
    vista = t[["id_epicrisis", "probabilidad", "cola", "codigos", "n_eventos", "revision", "modelo_tipo", "fecha_hora"]]
    st.dataframe(vista, hide_index=True, width="stretch", column_config={
        "id_epicrisis": "Epicrisis", "probabilidad": st.column_config.NumberColumn("Probabilidad", format="%.3f"),
        "cola": "Cola", "codigos": "Códigos de evento", "n_eventos": "Eventos registrados", "revision": "Revisión",
        "modelo_tipo": "Modelo", "fecha_hora": "Fecha y hora"})
    salida = t.drop(columns=["codigos_evento", "codigos", "revision", "motivo_cierre"])
    st.download_button("Descargar las detecciones (CSV, sin texto)", salida.to_csv(index=False).encode("utf-8-sig"),
                       "detecciones.csv", "text/csv", on_click="ignore")


# ---------------------------------------------------------------- 2. revisión
def tab_revision():
    st.markdown("#### 2 · Revisión")
    st.caption("El revisor decide con los tres criterios del manual de anotación v2: daño de gravedad E o mayor (D1), "
               "asociación con la atención con nexo C3 o C4, resolviendo antes el deslinde con la enfermedad de base (D2), "
               "y carácter no intencional (D3). Luego tipifica el evento con el Anexo N.° 02 (num. 19.5), el proceso "
               "(num. 19.6) y el servicio (num. 19.7). Una epicrisis puede tener varios eventos.")
    with almacen.conexion() as con:
        t = almacen.detecciones(con)
    colas = st.multiselect("Colas", list(contrato.COLAS.values()), default=[contrato.COLAS["sin_codigo"]], key="rev_colas")
    pendientes = st.toggle("Solo casos sin cerrar", value=True, key="rev_pend")
    if not t.empty:
        t = t[t.cola.isin(colas)]
        if pendientes:
            t = t[t.cierre.isna()]
    if t.empty:
        st.info("No hay epicrisis en las colas elegidas.")
        mostrar_dudosos()
        return
    prob = dict(zip(t.id_epicrisis, t.probabilidad))
    if st.session_state.get("rev_sel") not in prob:
        st.session_state["rev_sel"] = t.id_epicrisis.iloc[0]
    sel = st.selectbox("Epicrisis", list(prob), format_func=lambda i: f"{i} · probabilidad {prob[i]:.3f}", key="rev_sel",
                       on_change=_reset_evento)
    with almacen.conexion() as con:
        ep = almacen.epicrisis(con, sel)
        evs = almacen.eventos(con, ("confirmado", "dudoso"), sel)
    fila = t[t.id_epicrisis == sel].iloc[0]
    izq, der = st.columns([1.05, 1])
    with izq:
        st.markdown(f"**{sel}** · probabilidad {fila.probabilidad:.3f} · {fila.cola}")
        with st.container(height=560, border=True):
            st.markdown(f'<div class="epicrisis notranslate" translate="no">{html.escape(ep["texto"])}</div>', unsafe_allow_html=True)
        with st.expander("Datos de la hospitalización y códigos (solo lectura)"):
            v = {contrato.ETIQUETAS[k]: x for k, x in ep["variables"].items() if not contrato.vacio(x)}
            st.write(v or "Sin variables registradas.")
            st.write("Códigos CIE: " + ("; ".join(ep["codigos_cie"]) or "ninguno"))
            for c in fila.codigos_evento:
                st.caption(f"{c['codigo']}: {c['motivo']}" + (f" · sugiere {catalogo.etiqueta(c['id_ev_sugerido'])}" if c.get("id_ev_sugerido") else ""))
    cerrado = isinstance(fila.cierre, str)
    with der:
        if cerrado:
            st.info("El caso está cerrado: para registrar o anular eventos, primero se reabre (abajo).")
        else:
            formulario_evento(ep, fila)
        eventos_del_caso(evs, cerrado)
        cierre_del_caso(sel, fila, evs)
    mostrar_dudosos()


def formulario_evento(ep, fila):
    sugeridos = [c["id_ev_sugerido"] for c in fila.codigos_evento if c.get("id_ev_sugerido")]
    ids = catalogo.eventos().sort_values(["nat", "nombre_oficial"]).index.tolist()
    if st.session_state.get("ev_idev") not in ids:          # sin sugerencia, queda vacío: el revisor debe elegirlo
        st.session_state["ev_idev"] = sugeridos[0] if sugeridos else None
    st.session_state.setdefault("ev_fecha", date.today())
    with st.form("f_evento"):
        st.markdown("**Registrar un evento**")
        st.text_area("Cita ancla: frase literal de la epicrisis que contiene el daño (§4.1)", key="ev_cita", height=90)
        st.selectbox("D1 · Gravedad (escala NCC MERP/IHI; se anota E o mayor)", list(revision.GRAVEDADES),
                     format_func=revision.GRAVEDADES.get, key="ev_grav")
        st.radio("D2b · Deslinde con la enfermedad de base (se resuelve primero)", list(revision.D2B),
                 format_func=revision.D2B.get, key="ev_d2b")
        st.radio("D2a · Nexo con la atención", list(revision.NEXOS), format_func=revision.NEXOS.get, key="ev_nexo", horizontal=True)
        st.caption("Si el nexo es C3, se marcan las tres condiciones de la Vía B:")
        c = st.columns(3)
        for i, nombre in enumerate(revision.VIA_B):
            c[i].checkbox(nombre, key=f"ev_vb{i}")
        st.radio("D3 · ¿El daño era el fin buscado o parte constitutiva del plan documentado?", ["No", "Sí"], key="ev_d3", horizontal=True)
        st.radio("Categoría ERSP", list(revision.ERSP), format_func=revision.ERSP.get, key="ev_ersp", horizontal=True)
        with st.expander("Solo si es evento centinela"):
            st.caption("El evento centinela pasa siempre a adjudicación (manual, §5).")
            st.selectbox("Criterio", [""] + list(revision.CRITERIOS_EC), format_func=lambda k: revision.CRITERIOS_EC.get(k, "Sin elegir"), key="ev_crit")
            st.text_area("Cita literal del desenlace", key="ev_cdes", height=70)
            st.caption("Para EC-2 y EC-3, los cuatro elementos del art. 5.4:")
            c = st.columns(2)
            for i, nombre in enumerate(revision.ELEMENTOS_54):
                c[i % 2].checkbox(nombre, key=f"ev_e{i}")
        st.selectbox("Id. de Evento del Anexo N.° 02 (num. 19.5)", ids, format_func=catalogo.etiqueta, key="ev_idev",
                     index=None, placeholder="Evento del Anexo N.° 02 (se puede escribir el número o el nombre)")
        c1, c2 = st.columns(2)
        c1.selectbox("Proceso (num. 19.6)", catalogo.LISTA_PROCESOS, key="ev_proc",
                     help="Asistenciales: " + ", ".join(catalogo.PROCESOS["Asistenciales"]) + ". No asistenciales: "
                          + ", ".join(catalogo.PROCESOS["No asistenciales"]) + ".")
        c2.text_input("Departamento y/o servicio (num. 19.7)", key="ev_serv")
        c1.date_input("Fecha de registro del ERSP", key="ev_fecha", format="DD/MM/YYYY")
        c2.checkbox("Dudoso: pasa a adjudicación", key="ev_dudoso")
        c2.selectbox("Motivo de la duda", [""] + revision.MOTIVOS_DUDOSO, key="ev_motivo")
        st.text_input("Nota (opcional)", key="ev_nota")
        st.form_submit_button("Registrar el evento", type="primary", on_click=_registrar_evento,
                              args=(ep["id_epicrisis"], ep["texto"], ep["origen"]))
    res = st.session_state.pop("ev_resultado", None)
    if res and res[0] == "error":
        st.error("El evento no se registró:\n\n" + "\n".join(f"- {e}" for e in res[1]))
    elif res:
        st.success(res[1])


def eventos_del_caso(evs, cerrado=False):
    if evs.empty:
        return
    st.markdown("**Eventos registrados en esta epicrisis**")
    for f in evs.itertuples():
        c1, c2 = st.columns([5, 1])
        c1.write(f"{f.id}. {catalogo.etiqueta(f.id_ev)} · gravedad {f.gravedad} · {f.nexo} · {f.ersp} · {f.proceso} · "
                 f"{f.servicio} · confianza {f.confianza} · {f.estado}")
        c2.button("Anular", key=f"anular_{f.id}", on_click=_estado_evento, args=(f.id, "descartado"), disabled=cerrado)


def cierre_del_caso(sel, fila, evs):
    st.markdown("**Cierre del caso**")
    if isinstance(fila.cierre, str):
        motivo = revision.MOTIVOS_SIN_EVENTO.get(fila.motivo_cierre) if isinstance(fila.motivo_cierre, str) else None
        st.info(f"Caso cerrado: {'con eventos' if fila.cierre == 'con_eventos' else 'sin eventos'}"
                + (f". Motivo: {motivo}." if motivo else "."))
        st.button("Reabrir el caso", on_click=_reabrir, args=(sel,))
        return
    c1, c2 = st.columns(2)
    c1.button("Cerrar con eventos", on_click=_cerrar, args=(sel, "con_eventos"), disabled=evs.empty, width="stretch")
    c2.selectbox("Motivo del cierre sin eventos", [""] + list(revision.MOTIVOS_SIN_EVENTO), key="rev_motivo",
                 format_func=lambda k: revision.MOTIVOS_SIN_EVENTO.get(k, "Sin elegir"))
    confirma = c2.checkbox("Leí la epicrisis completa y no hay ningún evento anotable", key="rev_confirma")
    c2.button("Cerrar sin eventos", on_click=_cerrar, args=(sel, "sin_eventos"), width="stretch",
              disabled=not confirma or not evs.empty or not st.session_state.get("rev_motivo"))


def mostrar_dudosos():
    with almacen.conexion() as con:
        d = almacen.eventos(con, ("dudoso",))
    if d.empty:
        return
    st.markdown("##### Eventos por adjudicar (dudosos y centinelas)")
    st.caption("La adjudicación queda en la bitácora con el nombre del revisor.")
    for f in d.itertuples():
        c1, c2, c3 = st.columns([5, 1, 1])
        motivo = f.motivo_dudoso or ("evento centinela" if f.ersp == "EC" else "sin motivo")
        c1.write(f"{f.id}. {f.id_epicrisis} · {catalogo.etiqueta(f.id_ev)} · motivo: {motivo}")
        c2.button("Confirmar", key=f"conf_{f.id}", on_click=_estado_evento, args=(f.id, "confirmado"))
        c3.button("Descartar", key=f"desc_{f.id}", on_click=_estado_evento, args=(f.id, "descartado"))


# ---------------------------------------------------------------- 3. priorización
def tab_priorizacion():
    st.markdown("#### 3 · Priorización con la Matriz de Priorización GEMSES (Directiva N.° 7-OGCyH-ESSALUD-2020)")
    st.caption("Agrupa los ERSP confirmados por proceso e Id. de Evento (num. 19.9 y 19.11 a), hace el corte de 15 días "
               "(num. 19.10), calcula la Matriz según el num. 19.11 y el Anexo N.° 03, y asigna el nivel de gestión "
               "(num. 19.12) y el comité (num. 19.13).")
    c1, c2, c3 = st.columns(3)
    fecha = c1.date_input("Fecha de corte", value=date.today(), key="m_fecha", format="DD/MM/YYYY")
    nivel = c2.selectbox("Nivel de la IPRESS", list(gemses.NIVELES_IPRESS), index=2, format_func=gemses.NIVELES_IPRESS.get, key="m_nivel")
    proceso = c3.selectbox("Proceso", ["Todos"] + catalogo.LISTA_PROCESOS, key="m_proceso")
    with almacen.conexion() as con:
        ev = almacen.eventos(con)
        imp_usuario = almacen.impactos_usuario(con)
    if ev.empty:
        st.info("Aún no hay ERSP confirmados. Para ver la Matriz con datos inventados, se usa el botón “Cargar o reiniciar los "
                "ERSP inventados del semestre”.")
        return
    try:
        imp_anexo, info = catalogo.impactos()
    except ValueError as e:
        imp_anexo, info = None, {"fuente": None}
        st.error(f"El CSV de impactos no es válido: {e}")
    if imp_anexo is not None:
        impactos = imp_anexo
        st.success(f"Impactos C, D, E y F: {info['fuente']} · {info['eventos']} eventos · SHA-256 {info['sha256'][:12]}…"
                   + (f" · {info['fuera_de_escala']} evento(s) con valores fuera de la escala del num. 19.11 b" if info["fuera_de_escala"] else ""))
    else:
        impactos = imp_usuario
        editor_impactos(ev, fecha, imp_usuario)
    m = gemses.calcular(ev, impactos, fecha, nivel, None if proceso == "Todos" else proceso)
    ini_p, fin = m.periodo
    ini_s, _ = m.semestre
    st.caption(f"Periodo de corte: del {(ini_p + pd.Timedelta(days=1)):%d/%m/%Y} al {fin:%d/%m/%Y} · semestre: del "
               f"{(ini_s + pd.Timedelta(days=1)):%d/%m/%Y} al {fin:%d/%m/%Y}")
    for a in m.avisos:
        st.warning(a)
    if m.tabla.empty:
        return
    if m.percentiles:
        c = st.columns(4)
        c[0].metric("Grupos con impacto", int(m.tabla.J.notna().sum()))
        for i, k in enumerate(("P25", "P50", "P75"), start=1):
            c[i].metric(f"{k} de J", f"{m.percentiles[k]:.2f}")
    nombres = catalogo.eventos().nombre_oficial
    vista = gemses.anexo03(m.tabla, nombres)
    estilo = vista.style.apply(lambda f: [f"background-color: {COLOR_NIVEL.get(f['Nivel de gestión'], '')}"] * len(f), axis=1)
    e = gemses.ENCABEZADOS_ANEXO03
    formatos = {e["A"]: "%d", e["B"]: "%.4f", e["C"]: "%d", e["D"]: "%d", e["E"]: "%d", e["F"]: "%d", e["G"]: "%.2f",
                e["H"]: "%.4f", e["I"]: "%.4f", e["J"]: "%.2f"}
    st.dataframe(estilo, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format=f) for c, f in formatos.items()})
    st.caption("Regla de niveles, a criterio del autor: J > P75 → Nivel III; P50 < J ≤ P75 → Nivel II; J ≤ P50 → Nivel I.")
    c1, c2 = st.columns(2)
    c1.download_button("Descargar la Matriz (CSV)", vista.to_csv(index=False).encode("utf-8-sig"), "matriz_anexo03.csv",
                       "text/csv", on_click="ignore")
    xlsx = io.BytesIO()
    vista.to_excel(xlsx, index=False, sheet_name="Matriz Anexo 03")
    c2.download_button("Descargar la Matriz (Excel)", xlsx.getvalue(), "matriz_anexo03.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
    if imp_anexo is not None:
        with st.expander("Análisis de sensibilidad: impacto impreso en el Anexo N.° 02"):
            s = gemses.sensibilidad(ev, imp_anexo, fecha, nivel, None if proceso == "Todos" else proceso)
            st.caption("Recalcula la Matriz con el “Impacto” impreso en el Anexo N.° 02 en lugar de la suma ponderada del num. 19.11 d.")
            n = st.column_config.NumberColumn
            st.dataframe(s, hide_index=True, width="stretch", column_config={
                "proceso": "Proceso", "id_ev": "Id. Evento", "G_19_11": n("G (num. 19.11 d)", format="%.2f"),
                "J_19_11": n("J (num. 19.11 d)", format="%.2f"), "nivel_19_11": "Nivel (num. 19.11 d)",
                "G_impreso": n("G impreso", format="%.2f"), "J_impreso": n("J con el impreso", format="%.2f"),
                "nivel_impreso": "Nivel con el impreso", "cambia_nivel": st.column_config.CheckboxColumn("Cambia de nivel")})
            st.write(f"Grupos que cambian de nivel: {int(s.cambia_nivel.sum())} de {len(s)}.")
    with st.expander("Criterios aplicados donde la Directiva no resuelve o no coincide con sus anexos"):
        st.markdown(CRITERIOS)


def editor_impactos(ev, fecha, imp_usuario):
    st.warning("No hay tabla de impactos del Anexo N.° 02 configurada (PROTOTIPO_IMPACTOS_CSV). Valore los cuatro impactos de "
               "cada evento con la escala del num. 19.11 b (9, 6, 3, 1, 0); la valoración queda registrada como del usuario.")
    _, ini_s, fin = gemses.ventanas(fecha)
    fechas = pd.to_datetime(ev.fecha)
    ids = sorted(set(ev.loc[(fechas > ini_s) & (fechas <= fin), "id_ev"].astype(int)))
    base = pd.DataFrame(index=pd.Index(ids, name="id_ev"), columns=["C", "D", "E", "F"], dtype="Int64")
    if imp_usuario is not None:
        base.update(imp_usuario)
    base.insert(0, "evento", [catalogo.etiqueta(i) for i in ids])
    opciones = [9, 6, 3, 1, 0]
    editado = st.data_editor(base, key="editor_impactos", disabled=["evento"], width="stretch", column_config={
        k: st.column_config.SelectboxColumn(f"{k}. {catalogo.DIMENSIONES[k]}", options=opciones) for k in "CDEF"})
    st.button("Guardar la valoración", on_click=_guardar_impactos, args=(editado.drop(columns="evento"),))


CRITERIOS = """
1) A criterio del autor, se aplica el texto del num. 19.11 a y f, que dice “entre”, y no la fórmula impresa en el Anexo N.° 03
   (“B= (A+Σa)” e “I= H + Σh”). Es decir, B = A / ΣA e I = H / ΣH.
2) El num. 19.11 a pone las dos sumatorias en una oración que termina “en el último semestre, en el mismo proceso donde se
   identificó el ERSP”, de lo cual se deriva que esa ventana se aplica a las dos. A criterio del autor, el semestre son los
   6 meses que terminan en la fecha de corte y el periodo de corte, los 15 días que terminan en ella.
3) Del num. 19.10 (“de los ERSP Notificados en dicho periodo”) se deriva que las filas son los grupos de proceso e Id. de
   Evento con al menos un ERSP en el periodo de corte, mientras que A y ΣA cuentan todo el semestre.
4) El num. 19.12 no dice cómo se asignan los valores que caen entre percentiles. A criterio del autor, se usa la regla
   J > P75 → Nivel III, P50 < J ≤ P75 → Nivel II y J ≤ P50 → Nivel I, con percentiles por interpolación lineal. Con pocos
   grupos, la aplicación avisa.
5) A criterio del autor, ΣH y los percentiles se calculan sobre las filas que se muestran, sean de todos los procesos o de uno.
6) Se usan los nombres del num. 19.12 (“Servicio y/o Departamentos” y “Director de la IPRESS”), aunque el Anexo N.° 03 dice
   “Departamentos” y “Director del establecimiento” y llama “Nivel de Riesgo” al valor de riesgo del num. 19.11 e.
7) Del num. 19.11 d, que define el impacto como “la sumatoria de todas las ponderaciones obtenidas”, se deriva que G es la
   suma ponderada de C, D, E y F, fórmula que el Anexo N.° 03 imprime como “G=(cx0.4)+(dx0.20)+(ex0.15)+(fx0.25)”. El
   “Impacto” impreso en el Anexo N.° 02 no coincide con esa suma en 230 de los 231 eventos,
   por lo que solo se usa en el análisis de sensibilidad. A criterio del autor, el evento con valores fuera de la escala
   9, 6, 3, 1 y 0 se calcula con la suma ponderada de esos valores, tal como están impresos.
8) A criterio del autor, al igual que en el plan de tesis, la Matriz se aplica a los ERSP que confirma la revisión, aunque los
   num. 19.9 y 19.10 hablan de los ERSP “Notificados”.
"""


# ---------------------------------------------------------------- 4. acerca del sistema
def tab_acerca(det):
    st.markdown("#### Acerca del prototipo")
    st.markdown("""
El prototipo lee la epicrisis, marca las que probablemente tienen un evento adverso sin código, permite que una persona
confirme y tipifique cada evento, y prioriza los eventos confirmados con la Matriz de Priorización GEMSES, según la Directiva.

Gobernanza del modelo:
1) Validación clínica: prevista en la revisión ciega de la tesis (HE2 y HE3), que evalúa el modelo adoptado; el modelo de
   demostración no tiene validación clínica.
2) Privacidad: la aplicación corre solo en esta máquina, el modelo entrenado con MIMIC-IV no se publica y la bitácora no guarda texto.
3) Supervisión humana: ningún evento entra a la Matriz sin la decisión del revisor, salvo los ERSP inventados de la demostración.
4) Auditoría: la bitácora registra cada acción con su fecha, las decisiones de revisión con el revisor, cada detección con la
   huella del modelo y cada reemplazo de una epicrisis con el número de eventos y de cierres que se borraron.

El prototipo no muestra términos, coeficientes ni resaltados del modelo, no se conecta a la historia clínica y no notifica a nadie.
""")
    if det is not None:
        with st.expander("Ficha del modelo"):
            st.json({k: v for k, v in det.ficha.items() if k != "procedencia"} | {"sha256_calculado": det.sha256})
    with st.expander("Bitácora (últimas 100 acciones)"):
        with almacen.conexion() as con:
            st.dataframe(almacen.bitacora(con, 100), hide_index=True, width="stretch")


# ---------------------------------------------------------------- página
det, error = obtener_detector()
barra_lateral(det, error)
st.markdown("### Detección y priorización de eventos adversos en epicrisis")
if det is not None and det.es_demo:
    st.markdown('<div class="banda-demo">DEMOSTRACIÓN: el modelo activo se entrenó con epicrisis inventadas y no tiene valor clínico.</div>',
                unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["1 · Detección", "2 · Revisión", "3 · Priorización", "Acerca del sistema"])
with t1:
    tab_deteccion(det)
with t2:
    tab_revision()
with t3:
    tab_priorizacion()
with t4:
    tab_acerca(det)
