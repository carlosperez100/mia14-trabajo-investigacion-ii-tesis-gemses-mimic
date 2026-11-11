# -*- coding: utf-8 -*-
"""El contrato usa las mismas 18 variables que el bloque C del pipeline y valida la entrada."""
import math

import numpy as np

from prototipo import contrato
from prototipo.contrato import Epicrisis

TEXTO = "Patient admitted for community-acquired pneumonia, treated with antibiotics and discharged home. " * 2


def test_variables_iguales_a_las_del_bloque_C():
    import features
    import yaml
    from prototipo.rutas import CONFIG
    grupos = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["feature_engineering"]["C"]["grupos"]
    assert features.columnas_dominio(grupos) == ["text"] + contrato.VARIABLES
    assert len(contrato.VARIABLES) == 18


def test_validar():
    assert contrato.validar(Epicrisis("demo-01", TEXTO)) == []
    errores = contrato.validar(Epicrisis("con espacio", "corto", {"edad": -3, "uci": 2, "otra": 1}, version_cie=11))
    assert len(errores) == 6


def test_version_y_valores_no_finitos_se_rechazan_por_fila():
    assert [contrato.version_valida(v) for v in (None, "", "9", "10.0", 10)] == [10, 10, 9, 10, 10]
    assert contrato.version_valida("CIE10") is None and contrato.version_valida(11) is None
    ep = contrato.desde_fila({"id_epicrisis": "x2", "texto": TEXTO, "version_cie": "CIE10", "edad": "inf"})
    errores = contrato.validar(ep)
    assert len(errores) == 2 and any(e.startswith("version_cie") for e in errores) and any("finito" in e for e in errores)
    assert contrato.etiqueta_segura(Epicrisis("con espacio", TEXTO), 4) == "fila 5"     # nunca se repite un id inválido
    assert contrato.etiqueta_segura(Epicrisis("ok-1", TEXTO), 4) == "ok-1"


def test_a_tabla_faltantes_como_nan():
    t = contrato.a_tabla([Epicrisis("a", TEXTO, {"edad": "70", "sexo": "Mujer"}), Epicrisis("b", TEXTO)])
    assert list(t.columns) == ["text"] + contrato.VARIABLES
    assert t.edad.tolist()[0] == 70.0 and math.isnan(t.edad.tolist()[1])
    assert t.sexo.tolist()[0] == "Mujer" and t.sexo.tolist()[1] is np.nan
    assert t.uci.dtype == float


def test_desde_fila_y_codigos():
    ep = contrato.desde_fila({"id_epicrisis": "x1", "texto": TEXTO, "codigos_cie": "T81.4XXA; I10", "version_cie": "10",
                              "edad": 50})
    assert ep.codigos_cie == ["T81.4XXA", "I10"] and ep.variables == {"edad": 50}


def test_cola():
    assert contrato.cola(True, True) == "Posible evento sin código"
    assert contrato.cola(True, False) == "Marcada, con código de evento"
    assert contrato.cola(False, True) == "No marcada"
