# -*- coding: utf-8 -*-
"""Reglas de bloqueo de la revisión (manual v2, §4.8) y almacén SQLite, con una epicrisis inventada."""
import sqlite3

import pandas as pd
import pytest

from prototipo import almacen, revision
from prototipo.contrato import Deteccion, Epicrisis

TEXTO = ("Invented discharge summary. On postoperative day 2 she had an unwitnessed fall from bed and an x-ray "
         "showed a displaced left distal radius fracture,\n   which was reduced and splinted.")


def _evento(**cambios):
    ev = {"cita_ancla": "unwitnessed fall from bed and an x-ray showed a displaced left distal radius fracture",
          "gravedad": "F", "d2b": "b", "nexo": "C4", "via_b": [False, False, False], "intencional": False, "ersp": "EA",
          "criterio_ec": None, "cita_desenlace": "", "elementos_54": [False] * 4, "id_ev": 202, "proceso": "Hospitalización",
          "servicio": "Traumatología", "dudoso": False, "motivo_dudoso": None}
    ev.update(cambios)
    return ev


def test_evento_valido():
    assert revision.validar(_evento(), TEXTO) == []


def test_cita_normalizada_y_literal():
    assert revision.cita_en_texto("Displaced left distal radius fracture, which was reduced", TEXTO)   # salto de línea
    assert not revision.cita_en_texto("fracture", TEXTO)                                              # muy corta
    assert revision.validar(_evento(cita_ancla="a fall that never happened in the text"), TEXTO)


def test_bloqueos_del_manual():
    assert revision.validar(_evento(gravedad="A-D"), TEXTO)                    # incidente
    assert revision.validar(_evento(d2b="c"), TEXTO)                           # R-11, C2
    assert revision.validar(_evento(d2b="a"), TEXTO)                           # C1
    assert revision.validar(_evento(nexo="C3", via_b=[True, True, False]), TEXTO)
    assert revision.validar(_evento(nexo="C3", via_b=[True, True, True]), TEXTO) == []
    assert revision.validar(_evento(intencional=True), TEXTO)                  # R-08
    assert revision.validar(_evento(gravedad="I"), TEXTO)                      # R-10
    assert revision.validar(_evento(ersp="EC", criterio_ec="EC-2", cita_desenlace="displaced left distal radius fracture",
                                    elementos_54=[True, True, True, False]), TEXTO)
    assert revision.validar(_evento(id_ev=999999), TEXTO)                      # fuera del Anexo N.° 02
    assert revision.validar(_evento(proceso="Pabellón"), TEXTO)                # fuera del num. 19.6
    assert revision.validar(_evento(dudoso=True), TEXTO)                       # sin motivo


def test_confianza_por_criterio():
    assert revision.confianza("C4", False) == "Alta"
    assert revision.confianza("C3", False) == "Media"
    assert revision.confianza("C4", True) == "Baja"


def test_almacen_flujo_completo(tmp_path):
    ruta = tmp_path / "prueba.sqlite"
    with almacen.conexion(ruta) as con:
        almacen.guardar_epicrisis(con, Epicrisis("DEMO-X", TEXTO, {"edad": 70}, ["I10"]), "demo")
        almacen.guardar_deteccion(con, Deteccion("DEMO-X", 0.91, True, True, "Posible evento sin código", [], 0.5, "ab" * 32, "demostracion"))
        ev = _evento()
        ev.update(id_epicrisis="DEMO-X", fecha="2026-11-10", via="A", confianza="Alta", revisor="R1", origen="demo", estado=None)
        id_evento = almacen.guardar_evento(con, ev)
        almacen.cerrar(con, "DEMO-X", "con_eventos", "R1")
        d = almacen.detecciones(con)
        assert d.loc[0, "n_eventos"] == 1 and d.loc[0, "cierre"] == "con_eventos"
        assert len(almacen.eventos(con)) == 1
        almacen.cambiar_estado(con, id_evento, "descartado", "R1")
        assert almacen.eventos(con).empty
        almacen.cargar_ersp_demo(con, pd.DataFrame({"id_ev": [202], "proceso": ["Emergencia"], "servicio": ["Emergencia"],
                                                    "fecha": ["2026-11-01"]}))
        assert len(almacen.eventos(con)) == 1
        b = almacen.bitacora(con)
        assert not b.detalle.str.contains("fall from bed").any()                # la bitácora no guarda texto
        assert almacen.borrar_demo(con) == (1, 2)                               # el evento descartado también era de demo
        assert almacen.detecciones(con).empty


def test_estado_inicial_centinela_va_a_adjudicacion():
    assert revision.estado_inicial({"ersp": "EA", "dudoso": False}) == "confirmado"
    assert revision.estado_inicial({"ersp": "EA", "dudoso": True}) == "dudoso"
    assert revision.estado_inicial({"ersp": "EC", "dudoso": False}) == "dudoso"            # manual, §5, P5 y P6


def test_reemplazo_y_cierre_con_motivo_quedan_en_la_bitacora(tmp_path):
    with almacen.conexion(tmp_path / "b.sqlite") as con:
        ep = Epicrisis("DEMO-Y", TEXTO)
        almacen.guardar_epicrisis(con, ep, "demo")
        ev = _evento()
        ev.update(id_epicrisis="DEMO-Y", fecha="2026-11-10", via="A", confianza="Alta", revisor="R1", origen="demo", estado=None)
        almacen.guardar_evento(con, ev)
        almacen.cerrar(con, "DEMO-Y", "con_eventos", "R1")
        with pytest.raises(ValueError):
            almacen.guardar_epicrisis(con, ep, "demo")                                  # sin reemplazar, no se pisa
        almacen.guardar_epicrisis(con, ep, "demo", reemplazar=True)
        b = almacen.bitacora(con, 1).iloc[0]
        assert (b.accion, b.detalle) == ("epicrisis_reemplazada", "eventos_borrados=1; cierres_borrados=1")
        assert almacen.eventos(con, ("confirmado", "dudoso", "descartado"), "DEMO-Y").empty
        almacen.cerrar(con, "DEMO-Y", "sin_eventos", "R2", motivo="C2")
        assert con.execute("SELECT motivo FROM cierres WHERE id_epicrisis = 'DEMO-Y'").fetchone()[0] == "C2"
        assert almacen.bitacora(con, 1).detalle.iloc[0] == "resultado=sin_eventos; motivo=C2; revisor=R2"


def test_ersp_inventados_no_se_duplican(tmp_path):
    t = pd.DataFrame({"id_ev": [202, 601], "proceso": ["Hospitalización"] * 2, "servicio": ["Medicina Interna"] * 2,
                      "fecha": ["2026-11-01", "2026-11-02"]})
    with almacen.conexion(tmp_path / "c.sqlite") as con:
        almacen.cargar_ersp_demo(con, t)
        almacen.cargar_ersp_demo(con, t)
        assert len(almacen.eventos(con)) == 2
        assert almacen.bitacora(con, 1).detalle.iloc[0] == "n=2; reemplazados=2"


def test_base_anterior_recibe_la_columna_motivo(tmp_path):
    ruta = tmp_path / "anterior.sqlite"
    con = sqlite3.connect(ruta)
    con.execute("CREATE TABLE cierres (id_epicrisis TEXT PRIMARY KEY, resultado TEXT NOT NULL, revisor TEXT NOT NULL, "
                "fecha TEXT NOT NULL)")
    con.commit()
    con.close()
    with almacen.conexion(ruta) as con:
        assert "motivo" in {f["name"] for f in con.execute("PRAGMA table_info(cierres)")}
