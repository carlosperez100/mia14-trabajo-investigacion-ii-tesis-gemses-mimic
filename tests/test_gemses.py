# -*- coding: utf-8 -*-
"""
Matriz de Priorización contra un ejemplo calculado a mano (fracciones exactas).

Fecha de corte 13-11-2026. Periodo de corte: (29-10, 13-11]. Semestre: (13-05, 13-11].
Hospitalización, en el semestre: evento 101 ×3 (dos en el periodo), 202 ×1 (en el periodo) y 303 ×4 (ninguno en el
periodo; uno cae justo el 29-10, que queda fuera del periodo). Total del proceso: 8.
Emergencia: 101 ×2 (uno en el periodo) y 404 ×1 (en el periodo). Total del proceso: 3.
Fuera de toda ventana: 101 el 13-05 (inicio del semestre, excluido) y 101 el 14-11 (después del corte).

Impactos de prueba: 101 (9, 6, 3, 1) → G = 3.6 + 1.2 + 0.45 + 0.25 = 5.5
                    202 (0, 3, 3, 6) → G = 0 + 0.6 + 0.45 + 1.5 = 2.55
                    404 (3, 1, 0, 0) → G = 1.2 + 0.2 = 1.4
B: H-101 = 3/8, H-202 = 1/8, E-101 = 2/3, E-404 = 1/3.
H: 33/16, 51/160, 11/3, 7/15  →  en 480-avos: 990, 153, 1760, 224;  ΣH = 3127/480.
I = n/3127 con n = 990, 153, 1760, 224;  J = 100·I.
J ordenados (×3127/100): 153, 224, 990, 1760.
P25 = 153 + 0.75·71 = 206.25;  P50 = (224 + 990)/2 = 607;  P75 = 990 + 0.25·770 = 1182.5  (en las mismas unidades).
Regla A: E-101 (1760 > 1182.5) → III;  H-101 (607 < 990 ≤ 1182.5) → II;  E-404 (224 ≤ 607) → I;  H-202 → I.
"""
import pandas as pd
import pytest

from prototipo import gemses

ESCALA_J = 100 / 3127


def _eventos():
    filas = [
        (101, "Hospitalización", "2026-06-10"), (101, "Hospitalización", "2026-11-01"), (101, "Hospitalización", "2026-11-13"),
        (202, "Hospitalización", "2026-11-05"),
        (303, "Hospitalización", "2026-07-01"), (303, "Hospitalización", "2026-08-15"), (303, "Hospitalización", "2026-09-20"),
        (303, "Hospitalización", "2026-10-29"),
        (101, "Emergencia", "2026-09-01"), (101, "Emergencia", "2026-11-10"),
        (404, "Emergencia", "2026-11-12"),
        (101, "Hospitalización", "2026-05-13"),   # inicio del semestre: excluido
        (101, "Hospitalización", "2026-11-14"),   # después del corte: excluido
    ]
    return pd.DataFrame(filas, columns=["id_ev", "proceso", "fecha"])


def _impactos():
    return pd.DataFrame({"id_ev": [101, 202, 303, 404], "C": [9, 0, 6, 3], "D": [6, 3, 6, 1], "E": [3, 3, 6, 0],
                         "F": [1, 6, 6, 0], "impreso": [5.0, 2.25, 6.0, 1.4]}).set_index("id_ev")


def _fila(t, proceso, id_ev):
    return t[(t.proceso == proceso) & (t.id_ev == id_ev)].iloc[0]


def test_matriz_contra_calculo_a_mano():
    m = gemses.calcular(_eventos(), _impactos(), "2026-11-13", nivel_ipress="III")
    t = m.tabla
    assert len(t) == 4                                            # 303 no tiene ERSP en el periodo de corte
    esperado = {("Hospitalización", 101): (2, 3, 8, 3 / 8, 5.5, 990, "II"),
                ("Hospitalización", 202): (1, 1, 8, 1 / 8, 2.55, 153, "I"),
                ("Emergencia", 101): (1, 2, 3, 2 / 3, 5.5, 1760, "III"),
                ("Emergencia", 404): (1, 1, 3, 1 / 3, 1.4, 224, "I")}
    for (proceso, id_ev), (n_per, a, total, b, g, n, nivel) in esperado.items():
        f = _fila(t, proceso, id_ev)
        assert f.n_periodo == n_per and f.A == a and f.total_proceso == total
        assert f.B == pytest.approx(b)
        assert f.G == pytest.approx(g)
        assert f.J == pytest.approx(n * ESCALA_J)
        assert f.nivel == nivel
    assert t.I.sum() == pytest.approx(1.0)
    assert m.percentiles["P25"] == pytest.approx(206.25 * ESCALA_J)
    assert m.percentiles["P50"] == pytest.approx(607 * ESCALA_J)
    assert m.percentiles["P75"] == pytest.approx(1182.5 * ESCALA_J)
    assert list(t.nivel) == ["III", "II", "I", "I"]              # ordenada por J descendente


def test_responsables_y_comites():
    t = gemses.calcular(_eventos(), _impactos(), "2026-11-13", nivel_ipress="III").tabla
    f = _fila(t, "Emergencia", 101)
    assert f.color == "Rojo" and f.responsable == "Director de la IPRESS"
    assert f.comite == "Comité Local de Gestión de la Calidad"                       # 19.13 b, IPRESS de III Nivel
    f = _fila(t, "Hospitalización", 101)
    assert f.color == "Amarillo" and f.responsable == "Servicio y/o Departamentos"
    assert f.comite == "Comité de Gestión de la Calidad de los Servicios de Salud"   # 19.13 a, II y III Nivel
    assert gemses.comite("I", "I") == "Comité Local de Gestión de la Calidad"        # 19.13 a, I Nivel
    assert gemses.comite("III", "OPN") == "Comité de Gestión de la Calidad de los Órganos Prestadores Nacionales"


def test_filtro_por_proceso_recalcula_I_y_percentiles():
    m = gemses.calcular(_eventos(), _impactos(), "2026-11-13", proceso="Emergencia")
    assert set(m.tabla.proceso) == {"Emergencia"}
    assert m.tabla.I.sum() == pytest.approx(1.0)
    assert any("grupo(s)" in a for a in m.avisos)                 # dos grupos: aviso de pocos grupos


def test_sin_impacto_no_entra_al_calculo():
    imp = _impactos().drop(index=404)
    m = gemses.calcular(_eventos(), imp, "2026-11-13")
    f = _fila(m.tabla, "Emergencia", 404)
    assert pd.isna(f.G) and pd.isna(f.J)
    assert m.tabla.I.sum() == pytest.approx(1.0)
    assert any("sin impacto" in a for a in m.avisos)
    s = gemses.sensibilidad(_eventos(), imp, "2026-11-13")
    assert not s.loc[(s.proceso == "Emergencia") & (s.id_ev == 404), "cambia_nivel"].iloc[0]   # sin nivel en ambos


def test_periodo_vacio():
    m = gemses.calcular(_eventos(), _impactos(), "2026-12-31")
    assert m.tabla.empty and m.avisos


def test_sensibilidad_con_impacto_impreso():
    s = gemses.sensibilidad(_eventos(), _impactos(), "2026-11-13")
    assert len(s) == 4
    f = s[(s.proceso == "Hospitalización") & (s.id_ev == 101)].iloc[0]
    assert f.G_19_11 == pytest.approx(5.5) and f.G_impreso == pytest.approx(5.0)


def test_valores_fuera_de_escala_se_avisan():
    imp = _impactos()
    imp.loc[101, "C"] = 4                                         # como el evento 6024 del Anexo N.° 02
    m = gemses.calcular(_eventos(), imp, "2026-11-13")
    assert any("fuera de la escala" in a and "suma ponderada" in a for a in m.avisos)
    assert _fila(m.tabla, "Hospitalización", 101).G == pytest.approx(3.5)     # 4·0.40 + 6·0.20 + 3·0.15 + 1·0.25


def test_ersp_inventados_cubren_los_16_grupos():
    from prototipo import modelo_demo
    t = modelo_demo.ersp_demo("2026-11-13")
    assert len(t) == 48 and t.groupby(["proceso", "id_ev"]).ngroups == 16
    dias = (pd.Timestamp("2026-11-13") - pd.to_datetime(t.fecha)).dt.days
    assert dias.between(0, 177).all()                             # todos dentro del semestre
    ids = sorted({g[0] for g in modelo_demo.ERSP_DEMO})
    imp = pd.DataFrame({"C": 9, "D": 6, "E": 3, "F": 1}, index=pd.Index(ids, name="id_ev"))
    assert len(gemses.calcular(t, imp, "2026-11-13").tabla) == 8   # los 8 grupos de índice par, en el periodo de corte


def test_exportacion_anexo03():
    t = gemses.calcular(_eventos(), _impactos(), "2026-11-13").tabla
    x = gemses.anexo03(t, {101: "uno", 202: "dos", 404: "cuatro"})
    assert list(x.columns[:4]) == ["Id. Evento", "Eventos o fallas de los procesos", "Proceso", "A. Frecuencia"]
    assert "J. Nivel de Gestión (I × 100)" in x.columns
    assert x["Eventos o fallas de los procesos"].iloc[0] == "uno"          # E-101 encabeza la matriz
