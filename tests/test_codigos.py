# -*- coding: utf-8 -*-
"""Regla de “sin código”: familias copiadas de src/eda_variables.py y equivalencia validada con el Anexo N.° 02."""
import pandas as pd

from prototipo import codigos


def test_familias_identicas_a_las_del_pipeline():
    import eda_variables
    assert codigos.PREF_CAUSAL == eda_variables.PREF_CAUSAL
    assert codigos.FAM_CIE9 == eda_variables.FAM_CIE9
    muestra = pd.Series(["T380X5A", "T50905A", "T360X1A", "T5105XA", "T3695XA", "T45", "I10", "T42"])
    esperado = eda_variables._efecto_adverso(muestra).tolist()
    assert [codigos.efecto_adverso(c) for c in muestra] == esperado


def test_normalizar():
    assert codigos.normalizar(" t81.4xxa ") == "T814XXA"


def test_codigos_de_evento_cie10():
    assert codigos.clasificar("T81.4XXA")["es_evento"]                 # familia T81 y equivalencia
    assert codigos.clasificar("T38.0X5A")["es_evento"]                 # efecto adverso de medicamento
    assert codigos.clasificar("W19.XXXA")["es_evento"]                 # caída, familia de causa explícita
    assert not codigos.clasificar("I10")["es_evento"]


def test_evento_sugerido_por_la_equivalencia():
    r = codigos.clasificar("T81.3")                                    # dehiscencia: equivalencia “con evento” 803
    assert r["es_evento"] and r["id_ev_sugerido"] == 803


def test_no_es_evento_adverso_no_saca_de_la_cola():
    r = codigos.clasificar("F68.1")                                    # validado como “no es evento adverso”
    assert not r["es_evento"] and "no es evento adverso" in r["motivo"]
    sin_codigo, anotados = codigos.revisar(["F68.1", "I10"])
    assert sin_codigo and len(anotados) == 1


def test_cie9_por_familias():
    assert codigos.clasificar("998.59", version=9)["es_evento"]        # complicaciones de la atención
    assert codigos.clasificar("E8781", version=9)["es_evento"]
    assert not codigos.clasificar("4019", version=9)["es_evento"]


def test_revisar_lista():
    sin_codigo, anotados = codigos.revisar(["I10", "T81.4XXA"])
    assert not sin_codigo and anotados[0]["codigo"] == "T81.4XXA"
    assert codigos.revisar([])[0]
