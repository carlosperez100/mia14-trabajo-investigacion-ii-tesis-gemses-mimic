# -*- coding: utf-8 -*-
"""De punta a punta con un modelo de demostración pequeño: entrenar, verificar la huella, detectar por lote y guardar."""
import json
import shutil

import pandas as pd
import pytest

from prototipo import almacen, contrato, detector, lote, modelo_demo
from prototipo.entrenar_local import entrenar
from prototipo.rutas import DEMO


@pytest.fixture(scope="module")
def modelo(tmp_path_factory):
    d = modelo_demo.generar_epicrisis(n=120, semilla=7)
    ruta = tmp_path_factory.mktemp("modelo") / "modelo_demo.joblib"
    entrenar(d.drop(columns="y"), d.y.values, ruta, "demostracion", {"datos": "prueba"})
    return ruta


def _huella(ruta):
    return json.loads(ruta.with_suffix(".ficha.json").read_text(encoding="utf-8"))["sha256"]


def _copiar(modelo, destino):
    shutil.copy(modelo, destino)
    shutil.copy(modelo.with_suffix(".ficha.json"), destino.with_suffix(".ficha.json"))
    return destino


def _demo():
    return pd.read_csv(DEMO / "epicrisis_demo.csv", dtype=str, keep_default_na=False)


def test_huella_alterada_no_abre(modelo, tmp_path):
    copia = tmp_path / modelo.name
    shutil.copy(modelo, copia)
    with pytest.raises(detector.ErrorModelo, match="ficha"):            # sin ficha no se abre
        detector.cargar(copia, _huella(modelo))
    ficha = json.loads(modelo.with_suffix(".ficha.json").read_text(encoding="utf-8"))
    ficha["sha256"] = "0" * 64
    copia.with_suffix(".ficha.json").write_text(json.dumps(ficha), encoding="utf-8")
    with pytest.raises(detector.ErrorModelo, match="no coincide"):      # con la ficha alterada, tampoco
        detector.cargar(copia, _huella(modelo))


def test_otra_ruta_exige_su_huella(modelo):
    with pytest.raises(detector.ErrorModelo, match="--sha256"):
        detector.cargar(modelo)
    with pytest.raises(detector.ErrorModelo, match="no coincide"):
        detector.cargar(modelo, "0" * 64)
    det = detector.cargar(modelo, _huella(modelo).upper())              # se abre desde los bytes ya verificados
    assert det.es_demo and det.sha256 == _huella(modelo)
    p = det.probabilidades(contrato.a_tabla([contrato.desde_fila(r) for r in _demo().to_dict("records")]))
    assert len(p) == 10 and ((p >= 0) & (p <= 1)).all()


def test_modelo_de_demostracion_de_la_carpeta_local(modelo, tmp_path, monkeypatch):
    monkeypatch.setenv("PROTOTIPO_DATOS", str(tmp_path))
    _copiar(modelo, tmp_path / "modelo_demo.joblib")
    det = detector.cargar()                                              # la ruta activa es la del modelo de demostración
    assert det.es_demo and det.sha256 == _huella(modelo)


def test_modelo_local_exige_la_huella_del_env(modelo, tmp_path, monkeypatch):
    local = _copiar(modelo, tmp_path / "modelo_local_todo.joblib")
    monkeypatch.setenv("PROTOTIPO_MODELO", str(local))
    with pytest.raises(detector.ErrorModelo, match="PROTOTIPO_MODELO_SHA256"):
        detector.cargar()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(detector.ErrorModelo, match="PROTOTIPO_MODELO_SHA256"):    # la misma ruta, escrita de otra forma
        detector.cargar(local.name, _huella(modelo))
    monkeypatch.setenv("PROTOTIPO_MODELO_SHA256", "0" * 64)
    with pytest.raises(detector.ErrorModelo, match="no coincide"):
        detector.cargar()
    monkeypatch.setenv("PROTOTIPO_MODELO_SHA256", _huella(modelo))
    assert detector.cargar().sha256 == _huella(modelo)


def test_lote_sin_texto_y_guardado(modelo, tmp_path, monkeypatch):
    monkeypatch.setenv("PROTOTIPO_DATOS", str(tmp_path))
    salida = tmp_path / "salida.csv"
    args = [str(DEMO / "epicrisis_demo.csv"), str(salida), "--guardar", "--modelo", str(modelo), "--sha256", _huella(modelo)]
    assert lote.main(args) == 0
    s = pd.read_csv(salida)
    assert len(s) == 10 and list(s.columns) == list(contrato.Deteccion.__dataclass_fields__)
    texto = pd.read_csv(DEMO / "epicrisis_demo.csv").texto
    assert not s.astype(str).apply(lambda col: col.isin(texto)).any().any()      # la salida no repite el texto
    with almacen.conexion() as con:
        assert len(almacen.detecciones(con)) == 10
    assert lote.main(args) == 0                        # sin --reemplazar, las ya guardadas se omiten
    with almacen.conexion() as con:
        assert almacen.bitacora(con, 1).detalle.iloc[0].endswith("omitidas=10")


def test_lote_sigue_con_las_filas_que_no_cumplen(modelo, tmp_path, capsys):
    d = _demo()
    d.loc[0, "version_cie"] = "CIE10"
    d.loc[1, "edad"] = "inf"
    d.loc[2, "id_epicrisis"] = "id con espacios"
    d.loc[[3, 4], "id_epicrisis"] = ""                  # dos vacíos: se rechazan por fila, no como repetidos
    entrada, salida = tmp_path / "entrada.csv", tmp_path / "s.csv"
    d.to_csv(entrada, index=False)
    assert lote.main([str(entrada), str(salida), "--modelo", str(modelo), "--sha256", _huella(modelo)]) == 0
    s = pd.read_csv(salida)
    assert len(s) == 5 and not s.id_epicrisis.isin(d.id_epicrisis[:5]).any()
    consola = capsys.readouterr().out
    assert "no cumplen el contrato: 5" in consola and "id con espacios" not in consola
    assert all(f"fila {i}:" in consola for i in (3, 4, 5))
    assert "version_cie" in consola and "finito" in consola


def test_lote_sin_huella_no_abre_otro_modelo(modelo, tmp_path, capsys):
    assert lote.main([str(DEMO / "epicrisis_demo.csv"), str(tmp_path / "s.csv"), "--modelo", str(modelo)]) == 2
    assert "--sha256" in capsys.readouterr().err


def test_lote_rechaza_ids_repetidos(modelo, tmp_path, capsys):
    entrada = tmp_path / "entrada.csv"
    pd.read_csv(DEMO / "epicrisis_demo.csv").assign(id_epicrisis="DEMO-01").to_csv(entrada, index=False)
    assert lote.main([str(entrada), str(tmp_path / "s.csv"), "--modelo", str(modelo), "--sha256", _huella(modelo)]) == 2
    assert "repite" in capsys.readouterr().err
