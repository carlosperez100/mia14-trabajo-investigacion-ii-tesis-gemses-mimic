# -*- coding: utf-8 -*-
"""Pone la raíz del repositorio y src/ en sys.path, y aísla cada prueba de la configuración local (.env)."""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
for p in (RAIZ, RAIZ / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def entorno_aislado(tmp_path, monkeypatch):
    """Cada prueba usa su propia carpeta local y no ve el modelo ni los impactos que fije el .env de la máquina."""
    monkeypatch.setenv("PROTOTIPO_DATOS", str(tmp_path / "datos_locales"))
    for nombre in ("PROTOTIPO_MODELO", "PROTOTIPO_MODELO_SHA256", "PROTOTIPO_IMPACTOS_CSV"):
        monkeypatch.delenv(nombre, raising=False)
