# -*- coding: utf-8 -*-
"""
ALMACÉN LOCAL (SQLite) del prototipo: epicrisis, detecciones, eventos confirmados, cierres de caso, impactos valorados
por el usuario y bitácora de acciones.

La base vive en la carpeta local del prototipo (fuera del repositorio). La bitácora registra quién hizo qué y cuándo,
con identificadores y conteos, pero nunca el texto de una epicrisis ni las citas.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .rutas import ruta_base

ESQUEMA = """
CREATE TABLE IF NOT EXISTS epicrisis (
    id_epicrisis TEXT PRIMARY KEY,
    texto        TEXT NOT NULL,
    variables    TEXT NOT NULL,
    codigos_cie  TEXT NOT NULL,
    version_cie  INTEGER NOT NULL,
    origen       TEXT NOT NULL,
    creado       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detecciones (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    id_epicrisis     TEXT NOT NULL REFERENCES epicrisis(id_epicrisis) ON DELETE CASCADE,
    probabilidad     REAL NOT NULL,
    marca            INTEGER NOT NULL,
    sin_codigo       INTEGER NOT NULL,
    cola             TEXT NOT NULL,
    codigos_evento   TEXT NOT NULL,
    corte            REAL NOT NULL,
    modelo_sha256    TEXT NOT NULL,
    modelo_tipo      TEXT NOT NULL,
    version_contrato TEXT NOT NULL,
    fecha_hora       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eventos (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    id_epicrisis   TEXT REFERENCES epicrisis(id_epicrisis) ON DELETE CASCADE,
    id_ev          INTEGER NOT NULL,
    proceso        TEXT NOT NULL,
    servicio       TEXT NOT NULL,
    fecha          TEXT NOT NULL,
    cita_ancla     TEXT,
    gravedad       TEXT,
    nexo           TEXT,
    via            TEXT,
    ersp           TEXT,
    criterio_ec    TEXT,
    cita_desenlace TEXT,
    dudoso         INTEGER NOT NULL DEFAULT 0,
    motivo_dudoso  TEXT,
    confianza      TEXT,
    nota           TEXT,
    estado         TEXT NOT NULL,
    revisor        TEXT NOT NULL,
    origen         TEXT NOT NULL,
    creado         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cierres (
    id_epicrisis TEXT PRIMARY KEY REFERENCES epicrisis(id_epicrisis) ON DELETE CASCADE,
    resultado    TEXT NOT NULL,
    revisor      TEXT NOT NULL,
    fecha        TEXT NOT NULL,
    motivo       TEXT
);
CREATE TABLE IF NOT EXISTS impactos_usuario (
    id_ev INTEGER PRIMARY KEY,
    C INTEGER NOT NULL, D INTEGER NOT NULL, E INTEGER NOT NULL, F INTEGER NOT NULL,
    fecha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bitacora (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha        TEXT NOT NULL,
    accion       TEXT NOT NULL,
    id_epicrisis TEXT,
    detalle      TEXT
);
"""


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def conexion(ruta=None):
    con = sqlite3.connect(ruta or ruta_base())
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ESQUEMA)
    if "motivo" not in {f["name"] for f in con.execute("PRAGMA table_info(cierres)")}:   # bases creadas antes del motivo
        con.execute("ALTER TABLE cierres ADD COLUMN motivo TEXT")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def registrar(con, accion, id_epicrisis=None, detalle=None):
    """Bitácora: acción, identificador y un detalle corto sin texto clínico."""
    con.execute("INSERT INTO bitacora (fecha, accion, id_epicrisis, detalle) VALUES (?, ?, ?, ?)",
                (_ahora(), accion, id_epicrisis, detalle))


# ---------------------------------------------------------------- epicrisis y detecciones
def existe(con, id_epicrisis):
    return con.execute("SELECT 1 FROM epicrisis WHERE id_epicrisis = ?", (id_epicrisis,)).fetchone() is not None


def guardar_epicrisis(con, ep, origen, reemplazar=False):
    """Guarda la epicrisis. Si ya existe y se pide reemplazarla, se borran en cascada sus detecciones, sus eventos y su
    cierre, y la bitácora registra cuántos eventos y cierres se borraron (sin texto)."""
    if existe(con, ep.id_epicrisis):
        if not reemplazar:
            raise ValueError(f"Ya existe una epicrisis con el identificador {ep.id_epicrisis}.")
        n_ev = con.execute("SELECT COUNT(*) FROM eventos WHERE id_epicrisis = ?", (ep.id_epicrisis,)).fetchone()[0]
        n_c = con.execute("SELECT COUNT(*) FROM cierres WHERE id_epicrisis = ?", (ep.id_epicrisis,)).fetchone()[0]
        con.execute("DELETE FROM epicrisis WHERE id_epicrisis = ?", (ep.id_epicrisis,))
        registrar(con, "epicrisis_reemplazada", ep.id_epicrisis, f"eventos_borrados={n_ev}; cierres_borrados={n_c}")
    con.execute("INSERT INTO epicrisis VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ep.id_epicrisis, ep.texto, json.dumps(ep.variables, default=str), json.dumps(ep.codigos_cie),
                 int(ep.version_cie), origen, _ahora()))


def guardar_deteccion(con, d):
    con.execute("""INSERT INTO detecciones (id_epicrisis, probabilidad, marca, sin_codigo, cola, codigos_evento, corte,
                   modelo_sha256, modelo_tipo, version_contrato, fecha_hora) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (d.id_epicrisis, d.probabilidad, int(d.marca), int(d.sin_codigo), d.cola,
                 json.dumps(d.codigos_evento, ensure_ascii=False), d.corte, d.modelo_sha256, d.modelo_tipo,
                 d.version_contrato, d.fecha_hora))
    registrar(con, "deteccion", d.id_epicrisis, f"p={d.probabilidad:.4f}; cola={d.cola}; modelo={d.modelo_sha256[:12]}")


def guardar_lote(con, epicrisis, detecciones, origen, reemplazar=False, accion="lote_procesado"):
    """Guarda cada epicrisis válida con su detección. Devuelve los identificadores omitidos porque ya existían."""
    por_id = {e.id_epicrisis: e for e in epicrisis}
    omitidas = []
    for d in detecciones:
        if existe(con, d.id_epicrisis) and not reemplazar:
            omitidas.append(d.id_epicrisis)
            continue
        guardar_epicrisis(con, por_id[d.id_epicrisis], origen, reemplazar=True)
        guardar_deteccion(con, d)
    registrar(con, accion, None, f"filas={len(epicrisis)}; validas={len(detecciones)}; omitidas={len(omitidas)}")
    return omitidas


def detecciones(con):
    """Última detección de cada epicrisis, con su estado de revisión (sin el texto)."""
    q = """SELECT d.*, e.origen, c.resultado AS cierre, c.motivo AS motivo_cierre,
                  (SELECT COUNT(*) FROM eventos v WHERE v.id_epicrisis = d.id_epicrisis AND v.estado != 'descartado') AS n_eventos
           FROM detecciones d JOIN epicrisis e USING (id_epicrisis)
           LEFT JOIN cierres c USING (id_epicrisis)
           WHERE d.id = (SELECT MAX(id) FROM detecciones x WHERE x.id_epicrisis = d.id_epicrisis)
           ORDER BY d.probabilidad DESC"""
    t = pd.read_sql_query(q, con)
    t["codigos_evento"] = t.codigos_evento.map(json.loads)
    return t


def epicrisis(con, id_epicrisis):
    f = con.execute("SELECT * FROM epicrisis WHERE id_epicrisis = ?", (id_epicrisis,)).fetchone()
    if f is None:
        return None
    d = dict(f)
    d["variables"], d["codigos_cie"] = json.loads(d["variables"]), json.loads(d["codigos_cie"])
    return d


# ---------------------------------------------------------------- eventos y cierres
CAMPOS_EVENTO = ["id_epicrisis", "id_ev", "proceso", "servicio", "fecha", "cita_ancla", "gravedad", "nexo", "via", "ersp",
                 "criterio_ec", "cita_desenlace", "dudoso", "motivo_dudoso", "confianza", "nota", "estado", "revisor", "origen"]


def guardar_evento(con, ev):
    fila = {k: ev.get(k) for k in CAMPOS_EVENTO}
    fila["dudoso"] = int(bool(fila["dudoso"]))
    fila["estado"] = fila["estado"] or ("dudoso" if fila["dudoso"] else "confirmado")
    cur = con.execute(f"INSERT INTO eventos ({', '.join(CAMPOS_EVENTO)}, creado) VALUES ({', '.join('?' * len(CAMPOS_EVENTO))}, ?)",
                      [fila[k] for k in CAMPOS_EVENTO] + [_ahora()])
    registrar(con, "evento_registrado", fila["id_epicrisis"],
              f"evento={cur.lastrowid}; id_ev={fila['id_ev']}; estado={fila['estado']}; revisor={fila['revisor']}")
    return cur.lastrowid


def eventos(con, estados=("confirmado",), id_epicrisis=None):
    q = f"SELECT * FROM eventos WHERE estado IN ({', '.join('?' * len(estados))})"
    p = list(estados)
    if id_epicrisis:
        q += " AND id_epicrisis = ?"
        p.append(id_epicrisis)
    return pd.read_sql_query(q + " ORDER BY fecha, id", con, params=p)


def cambiar_estado(con, id_evento, estado, revisor):
    con.execute("UPDATE eventos SET estado = ? WHERE id = ?", (estado, int(id_evento)))
    f = con.execute("SELECT id_epicrisis FROM eventos WHERE id = ?", (int(id_evento),)).fetchone()
    registrar(con, f"evento_{estado}", f["id_epicrisis"] if f else None, f"evento={id_evento}; revisor={revisor}")


def cerrar(con, id_epicrisis, resultado, revisor, motivo=None):
    """Cierra el caso. El cierre sin eventos lleva su motivo (revision.MOTIVOS_SIN_EVENTO), que queda en la bitácora."""
    con.execute("INSERT OR REPLACE INTO cierres (id_epicrisis, resultado, revisor, fecha, motivo) VALUES (?, ?, ?, ?, ?)",
                (id_epicrisis, resultado, revisor, _ahora(), motivo))
    registrar(con, "caso_cerrado", id_epicrisis, f"resultado={resultado}; " + (f"motivo={motivo}; " if motivo else "")
              + f"revisor={revisor}")


def reabrir(con, id_epicrisis, revisor):
    con.execute("DELETE FROM cierres WHERE id_epicrisis = ?", (id_epicrisis,))
    registrar(con, "caso_reabierto", id_epicrisis, f"revisor={revisor}")


# ---------------------------------------------------------------- impactos valorados por el usuario
def impactos_usuario(con):
    t = pd.read_sql_query("SELECT id_ev, C, D, E, F FROM impactos_usuario", con)
    return t.set_index("id_ev") if len(t) else None


def guardar_impacto(con, id_ev, c, d, e, f):
    con.execute("INSERT OR REPLACE INTO impactos_usuario VALUES (?, ?, ?, ?, ?, ?)",
                (int(id_ev), int(c), int(d), int(e), int(f), _ahora()))
    registrar(con, "impacto_valorado", None, f"id_ev={id_ev}; C={c}; D={d}; E={e}; F={f}")


# ---------------------------------------------------------------- bitácora y demostración
def bitacora(con, limite=200):
    return pd.read_sql_query("SELECT fecha, accion, id_epicrisis, detalle FROM bitacora ORDER BY id DESC LIMIT ?",
                             con, params=(limite,))


def cargar_ersp_demo(con, tabla):
    """ERSP inventados para la Matriz (sin epicrisis asociada). Reemplaza los que se hayan cargado antes, para no
    duplicar la frecuencia A."""
    previos = con.execute("DELETE FROM eventos WHERE origen = 'demo' AND id_epicrisis IS NULL").rowcount
    for f in tabla.itertuples(index=False):
        con.execute(f"INSERT INTO eventos ({', '.join(CAMPOS_EVENTO)}, creado) VALUES ({', '.join('?' * len(CAMPOS_EVENTO))}, ?)",
                    [None, int(f.id_ev), f.proceso, f.servicio, pd.Timestamp(f.fecha).date().isoformat(), None, None, None,
                     None, "EA", None, None, 0, None, None, "ERSP inventado para la demostración", "confirmado",
                     "demostración", "demo", _ahora()])
    registrar(con, "demo_ersp_cargados", None, f"n={len(tabla)}; reemplazados={previos}")


def borrar_demo(con):
    n_e = con.execute("SELECT COUNT(*) FROM epicrisis WHERE origen = 'demo'").fetchone()[0]
    n_v = con.execute("SELECT COUNT(*) FROM eventos WHERE origen = 'demo'").fetchone()[0]
    con.execute("DELETE FROM eventos WHERE origen = 'demo'")
    con.execute("DELETE FROM epicrisis WHERE origen = 'demo'")
    registrar(con, "demo_borrada", None, f"epicrisis={n_e}; ersp={n_v}")
    return n_e, n_v
