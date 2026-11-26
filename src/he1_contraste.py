# -*- coding: utf-8 -*-
"""
CONTRASTE DE LA HE1: aporte del contenido textual de la epicrisis frente a las variables de contexto.

Protocolo que aplica, sin cambios (config/config.yaml):
  - he1_estratos, prerregistro del 05-11-2026 (commit fda6828): modelo adoptado A+B+C entrenado con las filas de entrenamiento
    de la variante D2; comparador C, el modelo de contexto sin texto, entrenado con las mismas filas; configuración actual;
    partición de prueba; estratos de servicio al ingreso, grupo de edad y decil de largo, con sus mínimos; sensibilidad con
    todas las epicrisis y la etiqueta original, y con la configuración candidata.
  - he1_enmienda_1, Enmienda 1 (registrada el 12-11-2026 en el commit ad116a5; redacción vigente del 13-11-2026 en el commit
    977a1ce, sin cambio de criterios ni de valores): criterio global; IC 95 % por percentiles con 2 000
    remuestreos de pacientes pareados; estabilidad con las semillas 43 a 47; deciles y mínimos contados en cada análisis; el
    modelo de contexto de la candidata con la regresión logística de la candidata.
  - Enmiendas 2 a 7: no cambian la HE1 (sección “Lo que no cambia” de cada una).
  - he1_enmienda_8, Enmienda 8 (docs/enmienda_8_prerregistro.md): los detalles de ejecución que los textos anteriores no fijan
    y lo que se hace si algo falla. El script lee esa sección y se detiene si no describe lo que este script calcula.

La HE1 se cumple si, en las epicrisis de prueba sin etiqueta dudosa (12 321 en la partición del proyecto), se cumplen a la vez:
  1) la diferencia global de ROC-AUC (A+B+C menos C) es de al menos 0.05;
  2) el límite inferior de su IC 95 % por paciente es mayor que cero;
  3) ese límite inferior es mayor que cero en cada una de las cinco particiones por paciente adicionales (semillas 43 a 47), en
     las que ambos modelos se reentrenan, evaluado en las epicrisis de prueba sin etiqueta dudosa de cada partición.
Como sensibilidad, se informa además el criterio por estrato del prerregistro: el límite inferior es mayor que cero en todos
los estratos evaluables.

Detalles de ejecución (Enmienda 8, sección 2.3, con su numeración):
  1) Cada partición adicional repite el paso 6 de src/preprocesado.py (GroupShuffleSplit por paciente, test_size de
     preprocesado.test_size) con la semilla de la partición, sobre el mismo corpus y en el mismo orden. Como se sortean
     pacientes, el número de epicrisis de prueba cambia entre particiones, y se informa en cada una. Con la semilla 42 se
     comprueba antes que se reproduce la partición guardada.
  2) Solo cambia la partición. La regresión logística y el remuestreo usan la semilla del proyecto (42) en todos los análisis,
     y cada análisis empieza su propio generador con esa semilla.
  3) El modelo de contexto usa la regresión logística del modelo adoptado en src/features.py (liblinear, max_iter 2000), con la
     C y los pesos de cada configuración.
  4) Se remuestrean pacientes (subject_id) con reposición, tantos como pacientes tiene el conjunto analizado; en un estrato, los
     pacientes del estrato. Un remuestreo que queda con una sola clase se reemplaza por otro hasta completar los registrados,
     con un tope del doble de sorteos; si no se completan, el IC no se calcula. Se informa cuántos se reemplazaron.
  5) Un servicio faltante, o una edad faltante o fuera de los cortes, forma el estrato “Sin dato”, que se informa con los mismos
     mínimos y no entra en el criterio por estrato.
  6) La candidata repite las tres condiciones, con sus seis particiones, y los estratos y la etiqueta original en la partición
     del proyecto (semilla 42), igual que la actual. No decide la HE1.
  7) La condición 1 se evalúa con la diferencia sin redondear.
  8) De un estrato no evaluable se publican el número de epicrisis y el motivo, sin conteos por clase. Los no evaluables con
     menos de 50 epicrisis (MIN_CELDA de src/eda_variables.py) se agrupan en “Otros no evaluables”, del que se publica su total,
     como hace el análisis exploratorio con epicrisis_en_categorias_no_publicadas.
  9) La actual se calcula completa, y sus resultados se escriben (he1_resultados_actual.json y he1.json) antes de empezar la
     candidata.

Controles que detienen el script antes de entrenar o evaluar el modelo de contexto (Enmienda 8, sección 2.4):
  - los archivos del corpus y las cachés de data/interim existen y las cachés cubren todas las hospitalizaciones del corpus
    (no se recalculan desde MIMIC; si una cambia al leer el corpus, se detiene); sus huellas SHA-256 son las que registra
    he1_enmienda_8.huellas_de_los_datos; la huella de data/processed/splits.csv es la de reportes/preprocesado_resumen.json, y
    la partición del corpus es la de ese archivo;
  - la partición con la semilla 42 reproduce la guardada, con 55 915 epicrisis de entrenamiento, 48 998 filas de D2, 14 085 de
    prueba y 12 321 de prueba sin etiqueta dudosa, y ninguna partición tiene pacientes en sus dos partes;
  - la configuración actual del código (experimentos_hp.actual) es la registrada (he1_estratos.configuracion), la candidata
    sale de he1_estratos.sensibilidad.configuracion_candidata, y he1_enmienda_8 describe lo que este script calcula, con la
    versión de este script y sin claves que el script no conozca; en el contraste real, además, la Enmienda 8 está registrada
    (fecha AAAA-MM-DD, sin la palabra BORRADOR en su texto ni en su sección de config.yaml) y la huella SHA-256 de este script
    es la que registra;
  - A+B+C reproduce en la prueba, con las configuraciones que se calculan, la ROC-AUC y la PR-AUC ya publicadas en
    reportes/hp_pruning_resultados.json, sin etiqueta dudosa y con todas las epicrisis, con una tolerancia de 0.0001 frente a sus
    cuatro decimales. Esos valores ya se conocían, y el del modelo de contexto no se calcula hasta pasar este control;
  - el script, la Enmienda 8 y los dos archivos de referencia de reportes/ están en git; src/, config/, docs/ y esos dos archivos
    no tienen cambios sin commit, el remoto origin es el repositorio de GitHub del proyecto (REPOSITORIO, también en
    he1_enmienda_8.repositorio_de_github) y el commit está en una de sus ramas, comprobado con git ls-remote (el código queda
    publicado antes de que exista ningún resultado);
  - en el contraste real, desde el commit que registra la Enmienda 8 (el primero cuya sección he1_enmienda_8 trae la fecha de
    registro) no cambiaron los módulos de src/ que carga el script, la Enmienda 8, los archivos de referencia ni las secciones
    de config.yaml que usa el cálculo;
  - no existe ningún resultado ni la marca de inicio, ningún registro anterior dice que escribió la marca y, en el contraste
    real, después de traer el historial de GitHub (git fetch), ningún archivo de la marca o de resultados y ningún registro con
    la marca figuran en el historial de git de ninguna rama: la prueba se mide una sola vez. En el contraste real, además, el
    registro de todo intento anterior está en git sin cambios, de modo que la causa de un ALTO queda escrita en el repositorio
    antes de otro intento. La marca (he1_inicio.json) se
    escribe al pasar los controles, antes del primer modelo de contexto, completa y solo si no existe (dos ejecuciones
    simultáneas no la escriben las dos), y guarda el commit, el commit de registro de la Enmienda 8, el remoto, la versión, el
    nombre de su registro, las huellas de los datos, del código y de la configuración registrada, y las versiones.
Si un control falla, no se calcula nada, y la causa se registra por escrito en el repositorio antes de otro intento.

Corte después de la marca de inicio (Enmienda 8, sección 2.4): --repetir-tras-corte REGISTRO permite una sola repetición. Un
corte es un registro que, después de la marca, no tiene línea de cierre (por ejemplo, un corte de energía) o que termina con una
interrupción (KeyboardInterrupt), una falta de memoria (MemoryError) o un error del sistema de archivos (OSError, como falta de
espacio o un archivo bloqueado; cuenta cualquier OSError, porque el script no distingue su causa, y si viene del código la única
repetición vuelve a fallar). Un registro que, después de la marca, termina con un error inesperado o con un ALTO no se
repite: el error del código se corrige con una nueva enmienda, que informa las cifras que el intento alcanzó a escribir junto a
las del script corregido y dice qué se hace con la marca existente. La repetición exige
que el registro sea el que guarda la marca y el único que la escribió, en la carpeta y en el historial de git; que en el
contraste real el registro y lo que el intento escribió estén en git sin cambios; y que el commit de registro de la Enmienda 8,
el remoto, las huellas del código, de la configuración registrada y de los datos, y las versiones, sean las de la marca. Debe reproducir
las cifras que el intento cortado escribió en su registro. Si el corte fue en la candidata, comprueba que los resultados
escritos de la actual coinciden con su registro y calcula solo la candidata. Escribe he1_repeticion.json al pasar los controles.

Limitaciones que se declaran: en las semillas 43 a 47, la prueba incluye epicrisis que en la semilla 42 sirvieron para elegir
A+B+C, D2 y la candidata, de modo que la condición 3 mide estabilidad y no una validación independiente; los IC por estrato no
se ajustan por multiplicidad, porque son sensibilidad.

Salidas, solo agregadas (ninguna fila de paciente, ninguna predicción y ningún modelo): reportes/he1_resultados_actual.json y
reportes/he1.json (con la clave he1_se_cumple que lee el programa del análisis de la revisión ciega), escritos al terminar la
actual; reportes/he1_resultados.json y logs/metrics_he1.txt, al terminar la candidata; reportes/he1_inicio.json. Se registran
las versiones de Python y de las librerías y las huellas de los datos y del código.

Uso:
  python src/he1_contraste.py --datos-inventados CARPETA [--remuestreos N]   # prueba del flujo con un corpus inventado;
                                                                               # no lee MIMIC ni escribe en el repositorio
  python src/he1_contraste.py --contraste                                    # el contraste real, una sola vez
  python src/he1_contraste.py --contraste --repetir-tras-corte logs/he1_contraste_AAAAMMDD_HHMMSS_MICROSEGUNDOS.log
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ETAPA = "he1_contraste"
VERSION = "1.4 (25-11-2026)"
HIPER = ("C", "A_max_features", "A_min_df", "B_max_features", "B_min_df", "class_weight", "sublinear_tf")
CORTES_EDAD = [0, 44, 64, 79, 200]                    # src/eda_variables.py, l. 235
ETIQUETAS_EDAD = ["18 a 44", "45 a 64", "65 a 79", "80 o más"]
SIN_DATO = "Sin dato"
OTROS = "Otros no evaluables"
METRICAS = ("roc_auc", "pr_auc")
TOLERANCIA = 0.0001
DOC_E8 = "docs/enmienda_8_prerregistro.md"
REFERENCIAS = ("reportes/hp_pruning_resultados.json", "reportes/preprocesado_resumen.json")
RASTREADOS = ("src/he1_contraste.py", DOC_E8, *REFERENCIAS)
SECCIONES = ("semilla", "preprocesado", "feature_engineering", "he1_estratos", "he1_enmienda_1", "he1_enmienda_8")
NOMBRES = {"inicio": "he1_inicio.json", "actual": "he1_resultados_actual.json", "he1": "he1.json",
           "resultados": "he1_resultados.json", "repeticion": "he1_repeticion.json", "texto": "metrics_he1.txt"}
INTENTO = tuple(f"reportes/{NOMBRES[k]}" for k in ("inicio", "actual", "he1", "resultados", "repeticion")) + ("logs/metrics_he1.txt",)
FINALES = ("reportes/he1_resultados.json", "reportes/he1_repeticion.json", "logs/metrics_he1.txt")
GUARDA_LA_MARCA = ("commit", "commit_de_registro", "remoto", "version", "registro", "huellas_de_los_datos",
                   "huellas_del_codigo", "versiones")
CORTES = (KeyboardInterrupt, MemoryError, OSError)    # interrupción, falta de memoria, falta de espacio o archivo bloqueado
MARCA = "marca de inicio escrita"
ACTUAL_ESCRITA = "resultados de la actual escritos"
FIRMA = re.compile(r"\| INFO \| (actual|candidata), semilla (\d+): (ROC-AUC A\+B\+C .*?\])")
REPOSITORIO = "carlosperez100/mia14-trabajo-investigacion-ii-tesis-gemses-mimic"      # el repositorio de GitHub del proyecto
REMOTO_GITHUB = re.compile(r"(https://github\.com/|git@github\.com:|ssh://git@github\.com/)(?P<repo>[\w.-]+/[\w.-]+?)(\.git)?/?")
ARCHIVOS_DE_DATOS = ("dataset", "particion", "dominio", "marcas")
LIBRERIAS = ("scikit-learn", "numpy", "pandas", "scipy", "pyarrow", "PyYAML")
PUBLICA_EVALUABLE = ["numero_de_epicrisis", "con_evento", "sin_evento", "pacientes", "cifras"]
EXIGE_LA_REPETICION = ["registro_y_archivos_del_intento_en_github", "unico_registro_con_la_marca", "mismo_commit_de_registro",
                       "mismo_remoto", "mismas_huellas_de_codigo_datos_y_versiones", "reproducir_las_cifras_del_registro"]
# Claves de he1_enmienda_8: el script se detiene si la sección trae una que no aplica (Enmienda 8, sección 2.4, numeral 1)
CLAVES_E8 = {"registrada", "version_del_script", "sha256_del_script", "huellas_de_los_datos", "repositorio_de_github", "origen",
             "aplicacion_de_lo_registrado", "particiones_adicionales", "semilla_de_la_regresion_y_del_remuestreo",
             "regresion_del_modelo_de_contexto", "remuestreo_con_una_sola_clase", "estrato_sin_dato", "candidata", "condicion_1",
             "publicacion_de_los_estratos", "orden_del_calculo", "control_de_reproduccion", "si_falla_un_control",
             "marca_de_inicio", "registro_de_cada_intento", "corte_despues_de_la_marca_de_inicio",
             "error_del_codigo_despues_de_la_marca", "sello", "sin_cambios_desde_el_registro", "congelamiento"}
CLAVES_ANIDADAS = {
    "huellas_de_los_datos": ARCHIVOS_DE_DATOS,
    "aplicacion_de_lo_registrado": ("particiones", "estratos_y_etiqueta_original"),
    "regresion_del_modelo_de_contexto": ("solver", "max_iter", "C_y_pesos"),
    "remuestreo_con_una_sola_clase": ("regla", "tope_de_sorteos", "si_no_se_completan"),
    "estrato_sin_dato": ("forma", "criterio_por_estrato"),
    "publicacion_de_los_estratos": ("de_un_evaluable_se_publica", "de_un_no_evaluable_se_publica", "agrupar_si_tienen_menos_de",
                                    "del_grupo_se_publica"),
    "control_de_reproduccion": ("metricas", "tolerancia"),
    "marca_de_inicio": ("archivo", "creacion", "guarda"),
    "corte_despues_de_la_marca_de_inicio": ("repeticiones", "archivo_de_la_repeticion", "que_es_un_corte", "no_se_repite_tras",
                                            "exige", "si_el_corte_fue_en_la_candidata", "si_la_repeticion_no_termina"),
    "corte_despues_de_la_marca_de_inicio.si_la_repeticion_no_termina": ("se_declara", "sin_resultados_de_la_actual",
                                                                        "sin_la_candidata"),
    "congelamiento": ("protocolo", "precisa")}

# Controles del contraste real. Los tamaños salen de logs/experimentos_hp_20261028_081924.log (l. 8) y del plan (Anexo 2,
# pasos 6 y 7), y las métricas de reportes/hp_pruning_resultados.json (test, comun y completa). Con datos inventados no se
# usan, salvo que una prueba los fije.
TAMANOS = {"entrenamiento": 55915, "filas_d2": 48998, "prueba": 14085, "prueba_sin_dudosa": 12321}
REPRODUCIR = {"actual": {"sin_dudosa": {"roc_auc": 0.8706, "pr_auc": 0.8671}, "original": {"roc_auc": 0.8398, "pr_auc": 0.858}},
              "candidata": {"sin_dudosa": {"roc_auc": 0.8699, "pr_auc": 0.867}, "original": {"roc_auc": 0.8397, "pr_auc": 0.8581}}}
TAMANOS_INVENTADO = None
REPRODUCIR_INVENTADO = None


class Alto(Exception):
    """Detiene el script sin escribir resultados."""


def valores_publicados(reportes):
    """ROC-AUC y PR-AUC de A+B+C en la prueba, ya publicadas en reportes/hp_pruning_resultados.json (clave test)."""
    t = json.loads((reportes / "hp_pruning_resultados.json").read_text(encoding="utf-8"))["test"]
    leidos = {c: {a: {m: t[c][k][m] for m in METRICAS} for a, k in (("sin_dudosa", "comun"), ("original", "completa"))}
              for c in ("actual", "candidata")}
    if leidos != REPRODUCIR:
        raise Alto(f"los valores publicados {leidos} no son los que cita este script {REPRODUCIR}.")
    return leidos


def fuera_de_tolerancia(obtenido, publicado, tolerancia):
    """Métricas cuyo valor obtenido, sin redondear, se aleja del publicado (cuatro decimales) más que la tolerancia."""
    return {f"{a}.{m}": round(abs(obtenido[a][m] - publicado[a][m]), 6) for a in publicado for m in METRICAS
            if not abs(obtenido[a][m] - publicado[a][m]) <= tolerancia}


# ------------------------------------------------------------------------------------------------ huellas
def huella_normalizada(archivo):
    """SHA-256 del archivo con los finales de línea normalizados (git usa CRLF en el árbol y LF en el repositorio)."""
    return hashlib.sha256(Path(archivo).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def huellas_del_codigo(raiz, cfg):
    """Huellas de los módulos de src/ cargados por el script, de la Enmienda 8, de los archivos de referencia y de las secciones
    de config.yaml que usa el cálculo. La marca de inicio las guarda y la repetición tras un corte exige las mismas. Antes se
    cargan los módulos del cálculo, para que la lista no dependa del momento en que se calculan las huellas."""
    import eda_variables, experimentos_hp, features, utils, variables_dominio, variantes_datos   # noqa: F401, E401
    raiz = Path(raiz).resolve()
    src = raiz / "src"
    modulos = sorted({Path(m.__file__).resolve().relative_to(raiz).as_posix() for m in list(sys.modules.values())
                      if getattr(m, "__file__", None) and Path(m.__file__).resolve().parent == src})
    archivos = {f: huella_normalizada(raiz / f) for f in [*modulos, DOC_E8, *REFERENCIAS]}
    config = {s: hashlib.sha256(json.dumps(cfg[s], sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
              for s in SECCIONES}
    return {"archivos": archivos, "config": config}


def huella_del_corpus(df):
    """Huella de un corpus inventado (en el contraste real se usan las huellas de los archivos)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def plano(d, prefijo=""):
    if not isinstance(d, dict):
        return {prefijo: d}
    out = {}
    for k, v in d.items():
        out.update(plano(v, f"{prefijo}.{k}" if prefijo else str(k)))
    return out


def diferencias(antes, ahora):
    """Claves cuyo valor no coincide entre la marca de inicio y la repetición."""
    a, b = plano(antes or {}), plano(ahora or {})
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def sin_nan(o):
    """NaN no es JSON válido: un IC que no se calculó se escribe null."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: sin_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sin_nan(v) for v in o]
    return o


def escribir(archivo, texto):
    """Escritura atómica: un corte deja el archivo completo o no lo deja."""
    archivo = Path(archivo)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.with_name(archivo.name + ".tmp")
    temporal.write_text(texto, encoding="utf-8")
    os.replace(temporal, archivo)


def guardar(archivo, contenido):
    escribir(archivo, json.dumps(sin_nan(contenido), indent=2, ensure_ascii=False, allow_nan=False))


def crear_exclusivo(archivo, contenido):
    """Escribe la marca de inicio (o la de la repetición) completa y solo si no existe. Se escribe primero en un archivo temporal
    propio, que se guarda en el disco (fsync), y después se enlaza con su nombre (os.link), que falla si el nombre ya existe. Así,
    dos ejecuciones simultáneas no pueden escribirla las dos, y la marca no queda a medio escribir. El temporal se borra al final;
    si no se puede borrar (por ejemplo, porque otro programa lo tiene abierto), queda sin efecto y la marca sigue válida."""
    archivo = Path(archivo)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.with_name(f"{archivo.name}.{os.getpid()}.tmp")
    texto = json.dumps(sin_nan(contenido), indent=2, ensure_ascii=False, allow_nan=False)
    try:
        with open(temporal, "w", encoding="utf-8") as fh:
            fh.write(texto)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(temporal, archivo)
    except FileExistsError:
        raise Alto(f"{archivo.name} ya existe: otra ejecución la escribió antes que esta. La prueba se mide una sola vez.")
    finally:
        try:
            temporal.unlink(missing_ok=True)
        except OSError:
            pass


def fecha_valida(valor):
    """Fecha de registro con el formato AAAA-MM-DD (YAML puede leerla como fecha o como texto)."""
    s = str(valor)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def sin_credenciales(texto):
    """Tapa el usuario y la clave de una dirección (https://usuario:clave@...), para que no queden en el registro, que se sube a
    GitHub."""
    return re.sub(r"//[^/@\s]+@", "//***@", str(texto))


def es_github(url):
    """El remoto es el repositorio de GitHub del proyecto (REPOSITORIO), por https, ssh o git@. No basta con que la dirección
    contenga «github.com», ni con que sea otro repositorio de GitHub."""
    m = REMOTO_GITHUB.fullmatch(str(url).strip())
    return m is not None and m.group("repo").lower() == REPOSITORIO.lower()


def datos_registrados(huellas, registradas):
    """Contraste real: las huellas SHA-256 de los cuatro archivos de datos son las que registra la Enmienda 8."""
    distintas = [k for k in ARCHIVOS_DE_DATOS if huellas.get(k) != registradas.get(k)]
    if distintas or set(huellas) != set(ARCHIVOS_DE_DATOS):
        raise Alto(f"las huellas de los datos {distintas or sorted(huellas)} no son las que registra la Enmienda 8 "
                   "(he1_enmienda_8.huellas_de_los_datos): no se calcula con otros datos.")


def versiones_de_librerias():
    from importlib.metadata import PackageNotFoundError, version
    v = {"python": platform.python_version()}
    for lib in LIBRERIAS:
        try:
            v[lib] = version(lib)
        except PackageNotFoundError:
            v[lib] = "no instalada"
    return v


# ------------------------------------------------------------------------------------------------ registro y protocolo
def registro(archivo):
    log = logging.getLogger(ETAPA)
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    archivo.parent.mkdir(parents=True, exist_ok=True)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(archivo, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def protocolo(cfg):
    """Lee lo registrado y comprueba que el script sabe calcularlo tal como está escrito."""
    import eda_variables
    H0, E1 = cfg["he1_estratos"], cfg["he1_enmienda_1"]
    S = cfg["semilla"]
    if (H0["modelo"], H0["variante"], H0["comparador"], H0["particion"], H0["etiqueta"]) != ("A+B+C", "D2", "C", "test", "sin_dudosa"):
        raise Alto("he1_estratos no describe el modelo, la variante, el comparador, la partición o la etiqueta que este script calcula.")
    if E1["criterio"] != "global" or E1["analisis_principal"] != "sin_dudosa":
        raise Alto("he1_enmienda_1 no fija el criterio global en las epicrisis sin etiqueta dudosa.")
    ic = E1["ic95"]
    if ic["metodo"] != "percentiles" or ic["pareado"] is not True:
        raise Alto("el IC registrado no es el de percentiles con remuestreo pareado que este script calcula.")
    if E1["estabilidad"]["criterio"] != "limite_inferior_ic95_mayor_que_cero_en_todas":
        raise Alto("la condición de estabilidad registrada no es la que este script evalúa.")
    if E1["deciles_largo"] != "por_analisis" or E1["minimos_por_analisis"] is not True:
        raise Alto("los deciles o los mínimos registrados no se cuentan por análisis.")
    if E1["bloque_c_con_candidata"] != "regresion_de_la_candidata":
        raise Alto("el modelo de contexto de la candidata registrado no es el que este script entrena.")
    if H0["estratos"]["grupo_edad"] != ETIQUETAS_EDAD:
        raise Alto("los grupos de edad registrados no son los de src/eda_variables.py.")
    if H0["estratos"]["servicio_ingreso"] != "servicio" or int(H0["estratos"]["decil_largo"]) != 10 \
            or H0["sensibilidad"]["todas_las_epicrisis"] is not True:
        raise Alto("los estratos o la sensibilidad con la etiqueta original registrados no son los que este script calcula.")
    if S != 42 or [int(s) for s in E1["estabilidad"]["semillas"]] != [43, 44, 45, 46, 47]:
        raise Alto("la semilla del proyecto o las de las particiones adicionales no son las registradas.")

    E8 = cfg.get("he1_enmienda_8")
    if not E8:
        raise Alto("config.yaml no tiene la sección he1_enmienda_8 con los detalles de ejecución de la HE1.")
    otras = {}
    for ruta, permitidas in (("", CLAVES_E8), *CLAVES_ANIDADAS.items()):
        d = E8
        for parte in ruta.split(".") if ruta else []:
            d = d.get(parte) if isinstance(d, dict) else None
        if isinstance(d, dict) and set(d) - set(permitidas):
            otras[ruta or "he1_enmienda_8"] = sorted(set(d) - set(permitidas))
    if otras:
        raise Alto(f"he1_enmienda_8 trae claves que este script no aplica: {otras}.")
    try:
        apl, reg = E8["aplicacion_de_lo_registrado"], E8["regresion_del_modelo_de_contexto"]
        rem, sd, pub = E8["remuestreo_con_una_sola_clase"], E8["estrato_sin_dato"], E8["publicacion_de_los_estratos"]
        ctl, mar, cor = E8["control_de_reproduccion"], E8["marca_de_inicio"], E8["corte_despues_de_la_marca_de_inicio"]
        datos = {k: str(E8["huellas_de_los_datos"][k]) for k in ARCHIVOS_DE_DATOS}
        revisar = {
            "version_del_script": str(E8["version_del_script"]) == VERSION,
            "aplicacion_de_lo_registrado": (apl["particiones"], apl["estratos_y_etiqueta_original"])
                                           == ("filas_d2_de_su_entrenamiento_y_su_prueba_sin_etiqueta_dudosa",
                                               "solo_en_la_particion_del_proyecto"),
            "particiones_adicionales": E8["particiones_adicionales"] == "paso_6_de_preprocesado_con_la_semilla_de_cada_particion",
            "semilla_de_la_regresion_y_del_remuestreo": int(E8["semilla_de_la_regresion_y_del_remuestreo"]) == S,
            "regresion_del_modelo_de_contexto": (reg["solver"], int(reg["max_iter"]), reg["C_y_pesos"])
                                                == ("liblinear", 2000, "los_de_cada_configuracion"),
            "huellas_de_los_datos": all(re.fullmatch(r"[0-9a-f]{64}", v) for v in datos.values()),
            "repositorio_de_github": str(E8["repositorio_de_github"]) == REPOSITORIO,
            "origen": isinstance(E8["origen"], str) and E8["origen"].strip() not in ("", "None"),
            "remuestreo_con_una_sola_clase": (rem["regla"], rem["tope_de_sorteos"], rem["si_no_se_completan"])
                                             == ("se_reemplaza_por_otro", "el_doble_de_los_remuestreos_registrados",
                                                 "el_ic_no_se_calcula_y_la_condicion_que_lo_usa_no_se_cumple"),
            "estrato_sin_dato": (sd["forma"], sd["criterio_por_estrato"])
                                == ("servicio_faltante_o_edad_faltante_o_fuera_de_los_cortes", "no_entra"),
            "candidata": E8["candidata"] == "tres_condiciones_y_estratos_y_etiqueta_original_en_la_particion_42",
            "condicion_1": E8["condicion_1"] == "diferencia_sin_redondear",
            "publicacion_de_los_estratos": (list(pub["de_un_evaluable_se_publica"]), list(pub["de_un_no_evaluable_se_publica"]),
                                            int(pub["agrupar_si_tienen_menos_de"]), pub["del_grupo_se_publica"])
                                           == (PUBLICA_EVALUABLE, ["numero_de_epicrisis", "motivo"], eda_variables.MIN_CELDA,
                                               "su_total_de_epicrisis"),
            "orden_del_calculo": E8["orden_del_calculo"] == "la_actual_completa_y_escrita_antes_de_la_candidata",
            "control_de_reproduccion": list(ctl["metricas"]) == list(METRICAS) and float(ctl["tolerancia"]) == TOLERANCIA,
            "si_falla_un_control": E8["si_falla_un_control"] == "no_se_calcula_nada_y_la_causa_se_registra_antes_de_otro_intento",
            "marca_de_inicio": (mar["archivo"], mar["creacion"], list(mar["guarda"]))
                               == (INTENTO[0], "solo_si_no_existe", list(GUARDA_LA_MARCA)),
            "registro_de_cada_intento": E8["registro_de_cada_intento"]
                                        == "se_sube_a_github_con_lo_que_escribio_antes_de_otro_intento",
            "corte_despues_de_la_marca_de_inicio": (
                int(cor["repeticiones"]), cor["archivo_de_la_repeticion"], list(cor["que_es_un_corte"]),
                list(cor["no_se_repite_tras"]), list(cor["exige"]), cor["si_el_corte_fue_en_la_candidata"],
                dict(cor["si_la_repeticion_no_termina"]))
                == (1, INTENTO[4], ["registro_sin_linea_de_cierre", *[c.__name__ for c in CORTES]],
                    ["error_inesperado", "alto_despues_de_la_marca"], EXIGE_LA_REPETICION, "solo_se_calcula_la_candidata",
                    {"se_declara": "desviacion_en_el_informe_final",
                     "sin_resultados_de_la_actual": "he1_no_contrastada_hasta_una_nueva_enmienda",
                     "sin_la_candidata": "he1_decidida_con_la_actual_y_candidata_no_calculada"}),
            "error_del_codigo_despues_de_la_marca": E8["error_del_codigo_despues_de_la_marca"]
                                                    == "nueva_enmienda_con_el_script_corregido_y_lo_que_hace_con_la_marca",
            "sello": E8["sello"] == "commit_en_una_rama_de_github_comprobado_con_git_ls_remote",
            "sin_cambios_desde_el_registro": E8["sin_cambios_desde_el_registro"]
                                             == "codigo_referencias_y_secciones_de_config_del_calculo",
            "congelamiento": (list(E8["congelamiento"]["protocolo"]), E8["congelamiento"]["precisa"])
                             == (["fda6828", "enmiendas_1_a_8", "plan_del_17-11-2026"],
                                 "analisis_enmienda_7.congelamiento.protocolo")}
        agrupar, tolerancia = int(pub["agrupar_si_tienen_menos_de"]), float(ctl["tolerancia"])
        registrada, huella = str(E8["registrada"]), str(E8["sha256_del_script"])
    except (KeyError, TypeError, ValueError) as e:
        raise Alto(f"he1_enmienda_8 está incompleta ({e!r}).")
    malos = [k for k, ok in revisar.items() if not ok]
    if malos:
        raise Alto(f"he1_enmienda_8 no describe lo que este script calcula: {malos}.")

    def configuracion(c):
        c = dict(c)
        return {"C": float(c["C"]), "A_max_features": int(c["A_max_features"]), "A_min_df": int(c["A_min_df"]),
                "B_max_features": int(c["B_max_features"]), "B_min_df": int(c["B_min_df"]),
                "class_weight": str(c["class_weight"]), "sublinear_tf": bool(c["sublinear_tf"])}
    return {"S": S, "semillas_adicionales": [int(s) for s in E1["estabilidad"]["semillas"]],
            "remuestreos": int(ic["remuestreos_pacientes"]), "dif_min": float(E1["diferencia_minima_roc_auc"]),
            "li_mayor_que": float(E1["limite_inferior_ic95_mayor_que"]),
            "min_estrato": int(H0["min_epicrisis_estrato"]), "min_clase": int(H0["min_por_clase"]),
            "deciles": int(H0["estratos"]["decil_largo"]),
            "configuraciones": {"actual": configuracion(H0["configuracion"]),
                                "candidata": configuracion(H0["sensibilidad"]["configuracion_candidata"])},
            "test_size": float(cfg["preprocesado"]["test_size"]), "agrupar": agrupar, "tolerancia": tolerancia,
            "registrada": registrada, "sha256_del_script": huella, "huellas_de_los_datos": datos}


def enmienda_8_registrada(raiz, R):
    """Solo en el contraste real: la Enmienda 8 está registrada y la huella de este script es la que registra."""
    if R["registrada"].strip() in ("", "por_fijar", "None"):
        raise Alto("la Enmienda 8 no está registrada (config.yaml, he1_enmienda_8.registrada: por_fijar).")
    if not fecha_valida(R["registrada"]):
        raise Alto(f"la fecha de registro de la Enmienda 8 ({R['registrada']}) no es una fecha AAAA-MM-DD.")
    if "BORRADOR" in (Path(raiz) / DOC_E8).read_text(encoding="utf-8"):
        raise Alto(f"{DOC_E8} todavía dice BORRADOR: la Enmienda 8 no está registrada.")
    texto_cfg = (Path(raiz) / "config" / "config.yaml").read_text(encoding="utf-8")
    if "he1_enmienda_8:" not in texto_cfg or "BORRADOR" in texto_cfg[texto_cfg.index("he1_enmienda_8:"):]:
        raise Alto("la sección he1_enmienda_8 de config/config.yaml falta o todavía dice BORRADOR: la Enmienda 8 no está "
                   "registrada.")
    h = huella_normalizada(Path(raiz) / "src" / "he1_contraste.py")
    if h != R["sha256_del_script"]:
        raise Alto(f"la huella de este script ({h[:12]}…) no es la que registra la Enmienda 8 ({R['sha256_del_script'][:12]}…).")


def _git(raiz, *args, check=True):
    return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=check)


def traer_github(raiz):
    """Contraste real: el remoto origin es de GitHub y se trae su historial antes de buscar intentos anteriores, para que
    cuente también lo que se subió desde otra copia del repositorio."""
    url = _git(raiz, "remote", "get-url", "origin", check=False)
    if url.returncode != 0 or not es_github(url.stdout):
        raise Alto(f"el remoto origin no existe o no es de GitHub ({sin_credenciales((url.stdout or url.stderr).strip())[:200]}).")
    f = _git(raiz, "fetch", "--quiet", "origin", check=False)
    if f.returncode != 0:
        raise Alto(f"no se pudo traer el historial de GitHub con git fetch ({sin_credenciales(f.stderr.strip())[:200]}).")


def commit_de_registro(raiz, archivos, secciones):
    """Primer commit de la rama actual cuya sección he1_enmienda_8 trae una fecha de registro. Desde ese commit no deben haber
    cambiado los archivos del cálculo ni las secciones de config.yaml que usa (sin cambios silenciosos, Enmienda 7, 2.5)."""
    import yaml
    hist = _git(raiz, "log", "--format=%H", "--reverse", "HEAD", "--", "config/config.yaml", check=False)
    if hist.returncode != 0:
        raise Alto(f"no se pudo leer el historial de config/config.yaml ({hist.stderr.strip()[:200]}).")

    def seccion_en(commit):
        s = _git(raiz, "show", f"{commit}:config/config.yaml", check=False)
        try:
            return yaml.safe_load(s.stdout) or {} if s.returncode == 0 else {}
        except yaml.YAMLError:
            return {}
    registro_ = next((c for c in hist.stdout.split()
                      if fecha_valida((seccion_en(c).get("he1_enmienda_8") or {}).get("registrada", "por_fijar"))), None)
    if registro_ is None:
        raise Alto("no se encontró en git el commit que registra la Enmienda 8 (con su fecha en he1_enmienda_8.registrada).")
    d = _git(raiz, "diff", "--quiet", registro_, "HEAD", "--", *archivos, check=False)
    if d.returncode != 0:
        raise Alto(f"desde el commit de registro de la Enmienda 8 ({registro_[:7]}) cambiaron archivos del cálculo o no se pudo "
                   "comparar: un cambio exige una nueva enmienda (Enmienda 7, sección 2.5).")
    antes, ahora = seccion_en(registro_), seccion_en("HEAD")
    cambiadas = [s for s in secciones if antes.get(s) != ahora.get(s)]
    if cambiadas:
        raise Alto(f"desde el commit de registro de la Enmienda 8 ({registro_[:7]}) cambiaron las secciones {cambiadas} de "
                   "config.yaml: un cambio exige una nueva enmienda (Enmienda 7, sección 2.5).")
    return registro_


def commit_sin_cambios(raiz, extra=()):
    """Commit del repositorio, sus ramas de GitHub y el remoto, si los archivos del cálculo (y los de extra) están en git sin
    cambios pendientes, el remoto origin es el repositorio de GitHub del proyecto y el commit ya está en una de sus ramas."""
    try:
        sin_git = [f for f in (*RASTREADOS, *extra) if _git(raiz, "ls-files", "--error-unmatch", f, check=False).returncode != 0]
        if sin_git:
            raise Alto(f"{', '.join(sin_git)}: no está en git; se hace el commit antes de ejecutar el contraste.")
        pend = _git(raiz, "status", "--porcelain", "--", "src", "config", "docs", *REFERENCIAS, *extra).stdout.strip()
        if pend:
            raise Alto(f"hay cambios sin commit en los archivos del cálculo:\n{pend}")
        commit = _git(raiz, "rev-parse", "HEAD").stdout.strip()
        url = _git(raiz, "remote", "get-url", "origin", check=False)
        if url.returncode != 0 or not es_github(url.stdout):
            raise Alto(f"el remoto origin no existe o no es de GitHub ({sin_credenciales((url.stdout or url.stderr).strip())[:200]}): "
                       "no se comprueba que el commit esté publicado.")
        remoto = _git(raiz, "ls-remote", "--heads", "origin", check=False)
        if remoto.returncode != 0:
            raise Alto(f"no se pudo consultar GitHub con git ls-remote ({sin_credenciales(remoto.stderr.strip())[:200]}): no se "
                       "comprueba que el commit esté publicado.")
        puntas = [linea.split("\t") for linea in remoto.stdout.splitlines() if "\t" in linea]
        remotas = [ref for sha, ref in puntas if _git(raiz, "merge-base", "--is-ancestor", commit, sha, check=False).returncode == 0]
    except (OSError, subprocess.CalledProcessError) as e:
        raise Alto(f"no se pudo leer el estado de git ({e}).")
    if not remotas:
        raise Alto(f"el commit {commit[:7]} no está en ninguna rama de GitHub: se sube antes de ejecutar, para que el código "
                   "quede publicado antes de que exista ningún resultado.")
    return commit, remotas, url.stdout.strip()


def historial(raiz, archivos):
    """Commits de cualquier rama que contienen alguno de esos archivos (marca de inicio o resultados de la HE1)."""
    r = _git(raiz, "log", "--all", "--format=%h", "--", *archivos, check=False)
    if r.returncode != 0:
        raise Alto(f"no se pudo leer el historial de git ({r.stderr.strip()[:200]}).")
    return r.stdout.split()


def registros_en_git(raiz):
    """Registros de logs/ que, en cualquier rama, alguna vez trajeron la línea de la marca de inicio."""
    r = _git(raiz, "log", "--all", f"-S{MARCA}", "--name-only", "--format=", "--", "logs/", check=False)
    if r.returncode != 0:
        raise Alto(f"no se pudo buscar en el historial de git los registros con la marca ({r.stderr.strip()[:200]}).")
    return sorted({linea.strip() for linea in r.stdout.splitlines() if linea.strip()})


def registros_previos_en_git(raiz, carpeta, patron, propio):
    """Contraste real: el registro de todo intento anterior está en git sin cambios, de modo que la causa de un ALTO queda escrita
    en el repositorio antes de otro intento (Enmienda 8, sección 2.4, numerales 2 y 4). El registro de esta ejecución no cuenta."""
    raiz, propio = Path(raiz).resolve(), Path(propio).resolve()
    fuera = []
    for p in sorted(Path(carpeta).glob(patron)):
        if p.resolve() == propio:
            continue
        r = p.resolve().relative_to(raiz).as_posix()
        if _git(raiz, "ls-files", "--error-unmatch", r, check=False).returncode != 0 \
                or _git(raiz, "status", "--porcelain", "--", r, check=False).stdout.strip():
            fuera.append(r)
    if fuera:
        raise Alto(f"el registro de un intento anterior no está en git o cambió ({fuera}): se sube con su causa antes de otro "
                   "intento (Enmienda 8, sección 2.4).")


def registros_con_marca(carpeta, patron, propio):
    """Registros de intentos anteriores que escribieron la marca de inicio (el registro de esta ejecución no cuenta)."""
    propio = Path(propio).resolve()
    return sorted(p.resolve() for p in Path(carpeta).glob(patron)
                  if p.resolve() != propio and MARCA in p.read_text(encoding="utf-8", errors="replace"))


def cifras_del_registro(archivo, marca):
    """Cifras que el intento cortado alcanzó a escribir en su registro, y si alcanzó a escribir los resultados de la actual."""
    archivo = Path(archivo)
    if not archivo.is_file():
        raise Alto(f"no existe el registro del intento cortado ({archivo}).")
    if archivo.name != marca.get("registro"):
        raise Alto(f"el registro {archivo.name} no es el que guarda la marca de inicio ({marca.get('registro')}).")
    texto = archivo.read_text(encoding="utf-8")
    if f"Contraste de la HE1, versión {VERSION}" not in texto:
        raise Alto("el registro del intento cortado no es de esta versión del script.")
    if f"commit del cálculo: {marca.get('commit')}" not in texto:
        raise Alto("el registro del intento cortado no es del commit de la marca de inicio.")
    if MARCA not in texto:
        raise Alto("el registro no llegó a escribir la marca de inicio: no es un corte después de la marca.")
    if " · salidas: " in texto:
        raise Alto("el registro corresponde a un contraste que terminó: no se repite.")
    despues = texto.split(MARCA, 1)[1]
    if "| ERROR | ALTO:" in despues or "error inesperado" in despues:
        raise Alto("después de la marca, el intento terminó con un ALTO o con un error inesperado: no es un corte y no se repite. "
                   "Un error del código se corrige con una nueva enmienda (Enmienda 8, sección 2.4).")
    return {(m.group(1), m.group(2)): m.group(3) for m in FIRMA.finditer(texto)}, ACTUAL_ESCRITA in texto


# ------------------------------------------------------------------------------------------------ datos
def misma_particion(df, sp):
    """La columna split del corpus es la de data/processed/splits.csv, epicrisis por epicrisis."""
    a = df[["note_id", "split"]].astype({"note_id": str, "split": str})
    b = sp[["note_id", "split"]].astype({"note_id": str, "split": str})
    if len(a) != len(b) or a.note_id.duplicated().any() or b.note_id.duplicated().any():
        raise Alto("el corpus y data/processed/splits.csv no tienen las mismas epicrisis.")
    m = a.merge(b, on="note_id", how="inner", suffixes=("", "_csv"))
    if len(m) != len(a) or not (m.split == m.split_csv).all():
        raise Alto("la partición del corpus no es la de data/processed/splits.csv.")


def corpus_real(log, registradas):
    """El corpus del proyecto con las variables de dominio y las marcas de etiqueta dudosa (cachés de data/interim). Antes de
    leerlos, sus huellas deben ser las que registra la Enmienda 8."""
    import variables_dominio
    import variantes_datos as vd
    from utils import INTERIM, PROCESSED, sha256
    from utils import REPORTES
    archivos = {"dataset": PROCESSED / "dataset.parquet", "particion": PROCESSED / "splits.csv",
                "dominio": INTERIM / "dominio_hadm.parquet", "marcas": INTERIM / "variantes_datos.parquet"}
    faltan = [str(v) for v in archivos.values() if not v.exists()]
    if faltan:
        raise Alto(f"faltan archivos del corpus o de sus cachés {faltan}: no se recalculan desde MIMIC.")
    huellas = {k: sha256(v) for k, v in archivos.items()}
    datos_registrados(huellas, registradas)
    registrada = json.loads((REPORTES / "preprocesado_resumen.json").read_text(encoding="utf-8")).get("sha256_splits")
    if huellas["particion"] != registrada:
        raise Alto(f"la huella de splits.csv ({huellas['particion'][:12]}…) no es la de reportes/preprocesado_resumen.json.")
    df = pd.read_parquet(archivos["dataset"])
    misma_particion(df, pd.read_csv(archivos["particion"], dtype={"note_id": str}))
    hadm = set(df.hadm_id)
    for k in ("dominio", "marcas"):
        if not hadm <= set(pd.read_parquet(archivos[k], columns=["hadm_id"]).hadm_id):
            raise Alto(f"la caché {archivos[k].name} no cubre todas las hospitalizaciones del corpus: no se recalcula desde MIMIC.")
    df, _ = variables_dominio.calcular(df, log)
    df = vd.marcar(df, log)
    despues = {k: sha256(archivos[k]) for k in ("dominio", "marcas")}
    if any(despues[k] != huellas[k] for k in despues):
        raise Alto("una caché de data/interim cambió al leer el corpus: no se calcula con otros datos.")
    return df, huellas


def corpus_inventado(semilla, n_pacientes=5000):
    """Corpus INVENTADO con las columnas que usa el contraste. Ningún texto ni valor viene de MIMIC."""
    from variables_dominio import columnas, GRUPOS
    rng = np.random.default_rng(semilla)
    notas = rng.integers(1, 4, n_pacientes)
    subject = np.repeat(np.arange(100000, 100000 + n_pacientes), notas)
    n = len(subject)
    riesgo = np.repeat(rng.normal(0, 1, n_pacientes), notas)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(riesgo + rng.normal(0, 1, n))))).astype(int)
    comunes = np.array([f"pal{i:04d}" for i in range(800)])
    evento = np.array([f"eva{i:03d}" for i in range(60)])
    textos = []
    for yi in y:
        k = int(rng.integers(30, 450))
        w = rng.choice(comunes, k)
        m = rng.uniform(0, 1, k) < (0.08 if yi else 0.02)
        w[m] = rng.choice(evento, int(m.sum()))
        textos.append(" ".join(w))
    df = pd.DataFrame({"note_id": [f"N{i:06d}" for i in range(n)], "subject_id": subject, "hadm_id": np.arange(n) + 2000000,
                       "text": textos, "y": y})
    num, cat, b = columnas(list(GRUPOS))
    for c in num:
        v = rng.gamma(2.0, 3.0, n) + 1.5 * y * rng.uniform(0, 1, n)
        v[rng.uniform(0, 1, n) < 0.05] = np.nan
        df[c] = v
    df["edad"] = np.where(rng.uniform(0, 1, n) < 0.02, np.nan, rng.integers(18, 96, n))
    servicios = np.array(["MED", "SURG", "CMED", "NMED", "ORTHO", "OBS", "PSURG"])
    for c in cat:
        df[c] = rng.choice(servicios if c.startswith("servicio") else np.array(["uno", "dos", "tres"]), n,
                           p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02] if c.startswith("servicio") else None)
    df.loc[rng.uniform(0, 1, n) < 0.02, "servicio"] = np.nan
    for c in b:
        df[c] = (rng.uniform(0, 1, n) < 0.2 + 0.1 * y).astype(int)
    df["negativa_dudosa"] = (y == 0) & (rng.uniform(0, 1, n) < 0.03)
    df["positiva_dudosa"] = (y == 1) & (rng.uniform(0, 1, n) < 0.10)
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(df.text, df.y, groups=df.subject_id))
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    return df


# ------------------------------------------------------------------------------------------------ modelos y métricas
def modelo(nombre, prm, P, S):
    """A+B+C con los hiperparámetros de la configuración (experimentos_hp.modelo_oficial), o el modelo de contexto: el bloque C
    con la regresión logística del modelo adoptado y la C y los pesos de la configuración (detalle 3)."""
    import experimentos_hp as hp
    from features import bloque
    if nombre == "A+B+C":
        return hp.modelo_oficial(prm, P, S, "A+B+C")
    return Pipeline([("atributos", ColumnTransformer([bloque("C", P)], sparse_threshold=1.0)),
                     ("clf", LogisticRegression(C=prm["C"], class_weight=None if prm["class_weight"] == "ninguno" else prm["class_weight"],
                                                max_iter=2000, solver="liblinear", random_state=S))])


def particion(df, semilla, test_size):
    """Paso 6 de src/preprocesado.py con otra semilla: máscara de las epicrisis de prueba (detalle 1)."""
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semilla).split(df.text, df.y, groups=df.subject_id))
    m = np.zeros(len(df), dtype=bool)
    m[te] = True
    return m


def ic_diferencia(y, p1, p0, grupos, n, semilla):
    """IC 95 % por percentiles de ROC-AUC(p1) menos ROC-AUC(p0), remuestreando pacientes con reposición, los mismos para los dos.
    Un remuestreo con una sola clase se reemplaza por otro, con un tope de 2n sorteos (detalle 4)."""
    idx_g = list(pd.Series(np.arange(len(y))).groupby(grupos).indices.values())
    rng = np.random.default_rng(semilla)
    d, una_clase = [], 0
    while len(d) < n and len(d) + una_clase < 2 * n:
        idx = np.concatenate([idx_g[i] for i in rng.integers(0, len(idx_g), len(idx_g))])
        yy = y[idx]
        if yy.min() == yy.max():
            una_clase += 1
            continue
        d.append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx]))
    if len(d) < n:
        return [float("nan"), float("nan")], len(d), una_clase
    li, ls = np.percentile(d, [2.5, 97.5])
    return [float(li), float(ls)], len(d), una_clase


def analizar(y, p1, p0, grupos, B, S):
    r = {"epicrisis": int(len(y)), "con_evento": int(y.sum()), "sin_evento": int(len(y) - y.sum()),
         "pacientes": int(pd.unique(grupos).size), "roc_auc_abc": float(roc_auc_score(y, p1)),
         "roc_auc_c": float(roc_auc_score(y, p0))}
    r["diferencia"] = r["roc_auc_abc"] - r["roc_auc_c"]
    r["ic95"], r["remuestreos_validos"], r["remuestreos_reemplazados"] = ic_diferencia(y, p1, p0, grupos, B, S)
    return r


def estratos(te, y, p1, p0, grupos, R, B, S):
    """Sensibilidad por estrato en el conjunto de prueba del análisis (deciles y mínimos contados en ese conjunto). De los no
    evaluables se publican el número de epicrisis y el motivo; los de menos de R['agrupar'] se agrupan y del grupo se publica
    su total (detalle 8)."""
    deciles = pd.qcut(te.text.str.len().values, R["deciles"], labels=False) + 1
    edad = pd.Series(pd.cut(te.edad.values, CORTES_EDAD, labels=ETIQUETAS_EDAD)).astype(object)
    servicio = pd.Series(te.servicio.values, dtype=object)
    var = {"servicio_ingreso": servicio.where(servicio.notna(), SIN_DATO).values,
           "grupo_edad": edad.where(edad.notna(), SIN_DATO).values,
           "decil_largo": deciles}
    out = {}
    for nombre, v in var.items():
        out[nombre], agrupadas = {}, 0
        for valor in sorted(pd.unique(v), key=lambda x: (isinstance(x, str), x)):
            m = v == valor
            n, pos = int(m.sum()), int(y[m].sum())
            if n >= R["min_estrato"] and min(pos, n - pos) >= R["min_clase"]:
                out[nombre][str(valor)] = {"evaluable": True, **analizar(y[m], p1[m], p0[m], grupos[m], B, S)}
            elif n < R["agrupar"]:
                agrupadas += n
            else:
                motivo = f"menos de {R['min_estrato']} epicrisis" if n < R["min_estrato"] else f"menos de {R['min_clase']} en una clase"
                out[nombre][str(valor)] = {"evaluable": False, "epicrisis": n, "motivo": motivo}
        if agrupadas:
            out[nombre][OTROS] = {"evaluable": False, "epicrisis": agrupadas,
                                  "motivo": f"estratos de menos de {R['agrupar']} epicrisis cada uno, agrupados"}
    return out


def criterio_por_estrato(E, li_mayor_que):
    """Sensibilidad del prerregistro (he1_estratos.criterio): límite inferior mayor que cero en todos los estratos evaluables.
    El estrato «Sin dato» no entra (detalle 5)."""
    evaluables = {f"{var}={valor}": e["ic95"][0] for var, est in E.items() for valor, e in est.items()
                  if e["evaluable"] and valor != SIN_DATO}
    fallan = sorted(k for k, li in evaluables.items() if not (li is not None and li > li_mayor_que))
    return {"estratos_evaluables": len(evaluables), "estratos_con_limite_inferior_no_mayor_que_cero": fallan,
            "se_cumple": bool(evaluables) and not fallan}


def _4(x):
    return "nan" if x is None or x != x else f"{x:.4f}"


def firma(a):
    """Cifras de un análisis tal como se escriben en el registro (y como se comparan al repetir tras un corte)."""
    return (f"ROC-AUC A+B+C {_4(a['roc_auc_abc'])} · C {_4(a['roc_auc_c'])} · diferencia {_4(a['diferencia'])} "
            f"[{_4(a['ic95'][0])}, {_4(a['ic95'][1])}]")


def criterio(res_nombre, R):
    principal = res_nombre["particiones"][str(R["S"])]["sin_dudosa"]
    adicionales = {s: res_nombre["particiones"][str(s)]["sin_dudosa"]["ic95"][0] for s in R["semillas_adicionales"]}
    mayor = lambda v: v is not None and v == v and v > R["li_mayor_que"]          # noqa: E731
    c = {"1_diferencia_al_menos": bool(principal["diferencia"] >= R["dif_min"]),
         "2_limite_inferior_mayor_que": bool(mayor(principal["ic95"][0])),
         "3_estabilidad_en_las_cinco": bool(all(mayor(v) for v in adicionales.values()))}
    return {**c, "se_cumplen_las_tres": all(c.values()), "limite_inferior_por_particion": {str(s): v for s, v in adicionales.items()}}


# ------------------------------------------------------------------------------------------------ cálculo
def calcular(df, R, P, B, log, tamanos, reproducir, marcar_inicio, esperado=None, solo=("actual", "candidata"),
             al_terminar=None):
    from features import columnas_dominio
    import variantes_datos as vd
    S = R["S"]
    X_cols = ["text"] + [c for c in columnas_dominio(P["C"]["grupos"]) if c != "text"]
    faltan = [c for c in X_cols + ["subject_id", "y", "split", "negativa_dudosa", "positiva_dudosa", "edad", "servicio"]
              if c not in df.columns]
    if faltan:
        raise Alto(f"al corpus le faltan las columnas {faltan}.")
    df = df.reset_index(drop=True)
    configs = {k: v for k, v in R["configuraciones"].items() if k in solo}

    def partes(semilla):
        es_test = particion(df, semilla, R["test_size"])
        tr, te = df[~es_test].reset_index(drop=True), df[es_test].reset_index(drop=True)
        if set(tr.subject_id) & set(te.subject_id):
            raise Alto(f"la partición con la semilla {semilla} tiene pacientes en sus dos partes.")
        usar = vd.filas_de_entrenamiento(tr, "D2").values
        comun = ~(te.negativa_dudosa | te.positiva_dudosa).values
        return es_test, tr, te, usar, comun

    # control de la partición del proyecto y de sus tamaños
    es42, tr42, te42, usar42, comun42 = partes(S)
    if not np.array_equal(es42, (df.split == "test").values):
        raise Alto("la partición con la semilla 42 no reproduce la guardada en el corpus.")
    tam = {"entrenamiento": len(tr42), "filas_d2": int(usar42.sum()), "prueba": len(te42), "prueba_sin_dudosa": int(comun42.sum())}
    log.info(f"partición del proyecto reproducida · {tam}")
    if tamanos is not None and tam != tamanos:
        raise Alto(f"los tamaños de la partición del proyecto no son los registrados: {tam} frente a {tamanos}.")

    # control de reproducción: A+B+C de las configuraciones que se calculan, antes de cualquier modelo de contexto
    p1_42, control = {}, {}
    yt42 = te42.y.values.astype(int)
    for nombre, prm in configs.items():
        t0 = time.time()
        m = modelo("A+B+C", prm, P, S).fit(tr42.loc[usar42, X_cols], tr42.y.values[usar42])
        p = p1_42[nombre] = m.predict_proba(te42[X_cols])[:, 1]
        obtenido = {"sin_dudosa": {"roc_auc": float(roc_auc_score(yt42[comun42], p[comun42])),
                                   "pr_auc": float(average_precision_score(yt42[comun42], p[comun42]))},
                    "original": {"roc_auc": float(roc_auc_score(yt42, p)), "pr_auc": float(average_precision_score(yt42, p))}}
        control[nombre] = {a: {k: round(v, 6) for k, v in d.items()} for a, d in obtenido.items()}
        log.info(f"control de reproducción, {nombre}: A+B+C en la prueba {control[nombre]} ({time.time() - t0:.0f} s)")
        if reproducir is not None:
            fuera = fuera_de_tolerancia(obtenido, reproducir[nombre], R["tolerancia"])
            if fuera:
                raise Alto(f"A+B+C con la configuración {nombre} no reproduce los valores publicados dentro de la tolerancia de "
                           f"{R['tolerancia']}: diferencias {fuera}. No se entrena el modelo de contexto.")
    marcar_inicio()                       # desde aquí el contraste ya no se repite, salvo la regla de corte (Enmienda 8)

    # contraste: la actual completa (y escrita) y después la candidata, cada una en la partición del proyecto y en las cinco
    # adicionales
    res = {}
    for nombre, prm in configs.items():
        res[nombre] = {"parametros": prm, "control_de_reproduccion": control[nombre], "particiones": {}}
        for semilla in [S] + R["semillas_adicionales"]:
            t0 = time.time()
            if semilla == S:
                tr, te, usar, comun, p1 = tr42, te42, usar42, comun42, p1_42[nombre]
            else:
                _, tr, te, usar, comun = partes(semilla)
                p1 = modelo("A+B+C", prm, P, S).fit(tr.loc[usar, X_cols], tr.y.values[usar]).predict_proba(te[X_cols])[:, 1]
            p0 = modelo("C", prm, P, S).fit(tr.loc[usar, X_cols], tr.y.values[usar]).predict_proba(te[X_cols])[:, 1]
            yt, gt = te.y.values.astype(int), te.subject_id.values
            r = {"entrenamiento": len(tr), "filas_d2": int(usar.sum()), "prueba": len(te),
                 "sin_dudosa": analizar(yt[comun], p1[comun], p0[comun], gt[comun], B, S)}
            if semilla == S:
                r["original"] = analizar(yt, p1, p0, gt, B, S)
                r["estratos_sin_dudosa"] = estratos(te[comun], yt[comun], p1[comun], p0[comun], gt[comun], R, B, S)
                r["estratos_original"] = estratos(te, yt, p1, p0, gt, R, B, S)
            res[nombre]["particiones"][str(semilla)] = r
            f = firma(r["sin_dudosa"])
            log.info(f"{nombre}, semilla {semilla}: {f} · {time.time() - t0:.0f} s")
            previa = (esperado or {}).get((nombre, str(semilla)))
            if previa is not None and previa != f:
                raise Alto(f"la repetición no reproduce lo que escribió el intento cortado ({nombre}, semilla {semilla}): {f} "
                           f"frente a {previa}.")
        res[nombre]["criterio"] = criterio(res[nombre], R)
        p42 = res[nombre]["particiones"][str(S)]
        res[nombre]["criterio_por_estrato"] = {a: criterio_por_estrato(p42[f"estratos_{a}"], R["li_mayor_que"])
                                               for a in ("sin_dudosa", "original")}
        if al_terminar is not None:
            al_terminar(nombre, res[nombre], tam)
    return res, tam


# ------------------------------------------------------------------------------------------------ salida
def texto_resumen(R, res, tam, meta):
    L = [f"# metrics_he1.txt · {meta['generado']} · versión {VERSION}" + (" · DATOS INVENTADOS" if meta["inventado"] else ""),
         f"commit del cálculo: {meta['commit']}" + (" · repetición tras un corte (Enmienda 8)" if meta["repeticion_tras_corte"] else ""),
         f"partición del proyecto: {tam}",
         f"criterio (he1_enmienda_1): diferencia de al menos {R['dif_min']}, límite inferior del IC 95 % mayor que "
         f"{R['li_mayor_que']} y lo mismo en las semillas {R['semillas_adicionales']}; IC por percentiles, {meta['remuestreos']} "
         f"remuestreos de pacientes pareados", ""]
    for nombre, r in res.items():
        L.append(f"{nombre.upper()} {'(análisis principal)' if nombre == 'actual' else '(sensibilidad)'}")
        L.append(f"  control de reproducción de A+B+C: {r['control_de_reproduccion']}")
        for s, p in r["particiones"].items():
            for analisis in ("sin_dudosa", "original"):
                if analisis in p:
                    a = p[analisis]
                    L.append(f"  semilla {s} · {analisis:<10} n {a['epicrisis']:>6} · {firma(a)}")
        k = r["criterio"]
        L.append(f"  condición 1 {k['1_diferencia_al_menos']} · condición 2 {k['2_limite_inferior_mayor_que']} · condición 3 "
                 f"{k['3_estabilidad_en_las_cinco']} · se cumplen las tres: {k['se_cumplen_las_tres']}")
        for a_, ce in r["criterio_por_estrato"].items():
            L.append(f"  criterio por estrato ({a_}, sensibilidad): {ce['estratos_evaluables']} evaluables · límite inferior no "
                     f"mayor que cero en: {', '.join(ce['estratos_con_limite_inferior_no_mayor_que_cero']) or 'ninguno'} · "
                     f"se cumple: {ce['se_cumple']}")
        for analisis in ("estratos_sin_dudosa", "estratos_original"):
            L.append(f"  {analisis} (semilla {R['S']}):")
            for var, est in r["particiones"][str(R["S"])][analisis].items():
                for valor, e in est.items():
                    if e["evaluable"]:
                        L.append(f"    {var} = {valor}: n {e['epicrisis']} · diferencia {_4(e['diferencia'])} "
                                 f"[{_4(e['ic95'][0])}, {_4(e['ic95'][1])}]")
                    else:
                        L.append(f"    {var} = {valor}: n {e['epicrisis']} · no evaluable ({e['motivo']})")
        L.append("")
    L.append(f"HE1 (configuración actual): {'SE CUMPLE' if res['actual']['criterio']['se_cumplen_las_tres'] else 'NO SE CUMPLE'}")
    return "\n".join(L) + "\n"


PROTOCOLO = {"prerregistro": "he1_estratos (fda6828)", "enmienda_1": "he1_enmienda_1 (ad116a5; redacción vigente 977a1ce)",
             "detalles_de_ejecucion": f"he1_enmienda_8 ({DOC_E8})"}
NOTA = "la HE1 no es ciega respecto de su dirección (Enmienda 1, estado 3; Enmienda 8, estado); la candidata es sensibilidad"


def main():
    ap = argparse.ArgumentParser(description="Contraste de la HE1 (ver el docstring).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--datos-inventados", metavar="CARPETA", help="prueba del flujo con un corpus inventado")
    g.add_argument("--contraste", action="store_true", help="el contraste real, una sola vez")
    ap.add_argument("--remuestreos", type=int, help="solo con --datos-inventados: menos remuestreos para que la prueba sea corta")
    ap.add_argument("--repetir-tras-corte", metavar="REGISTRO",
                    help="una sola repetición tras un corte posterior a la marca de inicio (Enmienda 8, sección 2.4)")
    a = ap.parse_args()
    inventado = a.datos_inventados is not None
    repetir = a.repetir_tras_corte is not None
    if a.remuestreos is not None and not inventado:
        raise SystemExit("--remuestreos solo se usa con --datos-inventados: el contraste usa los remuestreos registrados.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if inventado:
        for v in ("MIMIC_DIR", "MIMIC_NOTE_DIR", "ETIQUETAS_DIR"):     # cargar_config las exige; la prueba no lee MIMIC
            os.environ.setdefault(v, "NO_SE_USA_EN_LA_PRUEBA")
    from utils import LOGS, RAIZ, REPORTES, cargar_config

    if inventado:
        carpeta = Path(a.datos_inventados).resolve()
        if RAIZ in carpeta.parents or carpeta == RAIZ:
            raise SystemExit("la prueba con datos inventados no escribe dentro del repositorio.")
        f = {k: carpeta / v for k, v in NOMBRES.items()}
        dir_logs, patron = carpeta, "*.log"
        archivo_log = carpeta / f"{ETAPA}_prueba_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    else:
        f = {k: (LOGS if k == "texto" else REPORTES) / v for k, v in NOMBRES.items()}
        dir_logs, patron = LOGS, f"{ETAPA}_*.log"
        archivo_log = LOGS / f"{ETAPA}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    log = registro(archivo_log)
    log.info(f"Contraste de la HE1, versión {VERSION}" + (" · PRUEBA CON DATOS INVENTADOS" if inventado else "")
             + (" · REPETICIÓN TRAS UN CORTE" if repetir else ""))

    def rel(x):
        return Path(x).resolve().relative_to(RAIZ.resolve()).as_posix()

    escritos = []

    def guardar_y_anotar(archivo, contenido, texto=False):
        escribir(archivo, contenido) if texto else guardar(archivo, contenido)
        escritos.append(Path(archivo).name)

    try:
        terminados = [str(x) for x in (f["resultados"], f["texto"]) if x.exists()]
        if terminados:
            raise Alto(f"el resultado ya existe ({terminados}): la prueba se mide una sola vez.")
        if f["repeticion"].exists():
            raise Alto(f"ya hubo una repetición tras un corte ({f['repeticion']}): no se repite otra vez.")
        if not inventado:
            traer_github(RAIZ)
            registros_previos_en_git(RAIZ, dir_logs, patron, archivo_log)
        previos = registros_con_marca(dir_logs, patron, archivo_log)
        marca, esperado, actual_escrita, reg = None, None, False, None
        if not repetir:
            hay = [str(x) for x in (f["inicio"], f["actual"], f["he1"]) if x.exists()]
            if hay:
                raise Alto(f"la marca de inicio o un resultado de la actual ya existen ({hay}): la prueba se mide una sola vez. "
                           "Si el intento anterior se cortó, se aplica la regla de corte de la Enmienda 8 (--repetir-tras-corte).")
            if previos:
                raise Alto(f"un intento anterior ya escribió la marca de inicio ({[p.name for p in previos]}): la prueba se mide "
                           "una sola vez.")
            if not inventado and historial(RAIZ, INTENTO):
                raise Alto("la marca de inicio o un resultado de la HE1 ya figuran en el historial de git: la prueba se mide una "
                           "sola vez.")
            if not inventado and registros_en_git(RAIZ):
                raise Alto(f"un registro del historial de git ya escribió la marca de inicio ({registros_en_git(RAIZ)}): la "
                           "prueba se mide una sola vez.")
        else:
            if not f["inicio"].exists():
                raise Alto("--repetir-tras-corte solo se usa si existe la marca de inicio de un intento cortado.")
            try:
                marca = json.loads(f["inicio"].read_text(encoding="utf-8"))
            except ValueError as e:              # JSON inválido o bytes que no son UTF-8
                raise Alto(f"la marca de inicio no se puede leer como JSON ({e}): no se repite. Se resuelve con una nueva "
                           "enmienda.")
            if not isinstance(marca, dict):
                raise Alto("la marca de inicio no tiene los datos de un intento: no se repite. Se resuelve con una nueva enmienda.")
            if marca.get("version") != VERSION or marca.get("inventado") is not inventado:
                raise Alto(f"la marca de inicio es de otra versión o de otro modo ({marca.get('version')}, "
                           f"inventado {marca.get('inventado')}).")
            reg = Path(a.repetir_tras_corte).resolve()
            if reg.parent != Path(dir_logs).resolve():
                raise Alto(f"el registro del intento cortado debe estar en {dir_logs}.")
            esperado, actual_escrita = cifras_del_registro(reg, marca)
            if previos != [reg]:
                raise Alto(f"el único registro que escribió la marca de inicio debe ser el del intento cortado: "
                           f"{[p.name for p in previos]}.")
            if actual_escrita != f["actual"].exists() or f["actual"].exists() != f["he1"].exists():
                raise Alto("los resultados escritos de la actual no coinciden con lo que dice el registro del intento cortado.")
            if not inventado and historial(RAIZ, FINALES):
                raise Alto("un resultado final o una repetición ya figuran en el historial de git: no se repite.")
            if not inventado and not set(registros_en_git(RAIZ)) <= {rel(reg)}:
                raise Alto(f"en el historial de git hay otro registro que escribió la marca de inicio ({registros_en_git(RAIZ)}): "
                           "no se repite.")
            log.info(f"repetición tras un corte: el intento cortado escribió {len(esperado)} cifras que se deben reproducir"
                     + ("; los resultados de la actual ya están escritos y solo se calcula la candidata" if actual_escrita else ""))
        solo = ("candidata",) if actual_escrita else ("actual", "candidata")

        if inventado:
            commit, remotas, remoto = "sin commit (prueba con datos inventados)", [], "sin remoto (prueba con datos inventados)"
        else:
            extra = [rel(x) for x in (reg, f["inicio"], f["actual"], f["he1"]) if x is not None and x.exists()] if repetir else []
            commit, remotas, remoto = commit_sin_cambios(RAIZ, extra)
        log.info(f"commit del cálculo: {commit}")
        cfg = cargar_config()
        R = protocolo(cfg)
        codigo = huellas_del_codigo(RAIZ, cfg)
        if inventado:
            registro_e8 = "sin registro (prueba con datos inventados)"
        else:
            enmienda_8_registrada(RAIZ, R)
            registro_e8 = commit_de_registro(RAIZ, sorted(codigo["archivos"]), SECCIONES)
        log.info(f"commit de registro de la Enmienda 8: {registro_e8}")
        P = cfg["feature_engineering"]
        import experimentos_hp as hp
        act_codigo = {k: hp.actual(P)[k] for k in HIPER}
        if act_codigo != R["configuraciones"]["actual"]:
            raise Alto(f"la configuración actual del código {act_codigo} no es la registrada {R['configuraciones']['actual']}.")
        B = a.remuestreos or R["remuestreos"]
        if inventado:
            df = corpus_inventado(20261124)
            huellas = {"corpus_inventado": huella_del_corpus(df)}
            tamanos, reproducir = TAMANOS_INVENTADO, REPRODUCIR_INVENTADO
        else:
            reproducir = valores_publicados(REPORTES)
            df, huellas = corpus_real(log, R["huellas_de_los_datos"])
            tamanos = TAMANOS
        versiones = versiones_de_librerias()

        if repetir:
            cambios = {q: d for q, d in (("commit_de_registro", [] if marca.get("commit_de_registro") == registro_e8
                                          else ["commit_de_registro"]),
                                         ("remoto", [] if marca.get("remoto") == remoto else ["remoto"]),
                                         ("huellas_de_los_datos", diferencias(marca.get("huellas_de_los_datos"), huellas)),
                                         ("huellas_del_codigo", diferencias(marca.get("huellas_del_codigo"), codigo)),
                                         ("versiones", diferencias(marca.get("versiones"), versiones))) if d}
            if cambios:
                raise Alto(f"desde la marca de inicio cambiaron {cambios}: no se repite.")
            if actual_escrita:
                previa = json.loads(f["actual"].read_text(encoding="utf-8"))
                firmas = {("actual", s): firma(p["sin_dudosa"]) for s, p in previa["resultados"]["actual"]["particiones"].items()}
                registradas = {k: v for k, v in esperado.items() if k[0] == "actual"}
                if firmas != registradas or previa.get("commit") != marca["commit"]:
                    raise Alto("los resultados escritos de la actual no coinciden con las cifras de su registro: no se repite.")

        def marcar_inicio():
            destino = f["repeticion"] if repetir else f["inicio"]
            datos = {"inicio": datetime.now().isoformat(timespec="seconds"), "commit": commit,
                     "commit_de_registro": registro_e8, "remoto": remoto, "version": VERSION, "inventado": inventado,
                     "registro": archivo_log.name, "huellas_de_los_datos": huellas, "huellas_del_codigo": codigo,
                     "versiones": versiones}
            if repetir:
                datos.update({"marca_de_inicio": marca.get("inicio"), "commit_de_la_marca_de_inicio": marca["commit"],
                              "registro_del_intento_cortado": reg.name, "cifras_que_debe_reproducir": len(esperado),
                              "configuraciones_que_calcula": list(solo)})
            crear_exclusivo(destino, datos)
            escritos.append(destino.name)
            log.info(f"controles superados; {MARCA} ({destino.name}): el contraste ya no se puede repetir"
                     + (" otra vez" if repetir else ""))

        def meta(t0):
            m_ = {"version": VERSION, "generado": datetime.now().isoformat(timespec="seconds"), "inventado": inventado,
                  "commit": commit, "commit_de_registro": registro_e8, "remoto": remoto, "ramas_remotas_con_el_commit": remotas,
                  "repeticion_tras_corte": repetir, "remuestreos": B, "versiones": versiones,
                  "segundos": round(time.time() - t0, 1), "huellas_de_los_datos": huellas, "huellas_del_codigo": codigo}
            if repetir:
                m_.update({"commit_de_la_marca_de_inicio": marca["commit"], "registro_del_intento_cortado": reg.name})
            return m_

        parametros = {**{k: v for k, v in R.items() if k != "configuraciones"}, "remuestreos": B,
                      "remuestreos_registrados": R["remuestreos"]}
        t0 = time.time()

        def al_terminar(nombre, r, tam):
            if nombre != "actual":
                return
            he1 = bool(r["criterio"]["se_cumplen_las_tres"])
            m = meta(t0)
            guardar_y_anotar(f["actual"], {**m, "protocolo": PROTOCOLO, "parametros": parametros, "particion_del_proyecto": tam,
                                           "resultados": {"actual": r}, "he1_se_cumple": he1, "candidata": "pendiente",
                                           "nota": NOTA})
            guardar_y_anotar(f["he1"], {"he1_se_cumple": he1, "commit": commit, "version": VERSION, "generado": m["generado"],
                                        "inventado": inventado, "fuente": f["actual"].name})
            log.info(f"{ACTUAL_ESCRITA} ({f['actual'].name}, {f['he1'].name}): la HE1 {'se cumple' if he1 else 'no se cumple'}; "
                     "sigue la candidata")

        res, tam = calcular(df, R, P, B, log, tamanos, reproducir, marcar_inicio, esperado, solo, al_terminar)
        if "actual" not in res:
            res = {"actual": json.loads(f["actual"].read_text(encoding="utf-8"))["resultados"]["actual"], **res}

        he1 = bool(res["actual"]["criterio"]["se_cumplen_las_tres"])
        m = meta(t0)
        guardar_y_anotar(f["resultados"], {**m, "protocolo": PROTOCOLO, "parametros": parametros, "particion_del_proyecto": tam,
                                           "resultados": res, "he1_se_cumple": he1, "candidata": "calculada", "nota": NOTA})
        guardar_y_anotar(f["texto"], texto_resumen(R, res, tam, m), texto=True)
    except Alto as e:
        log.error(f"ALTO: {e}")
        if escritos:
            log.error(f"En esta ejecución se escribieron {escritos} antes del ALTO; no se escribió ningún otro resultado. Un ALTO "
                      "después de la marca de inicio no se repite (Enmienda 8, sección 2.4).")
        else:
            log.error("No se escribió ningún resultado de la HE1 en esta ejecución.")
        sys.exit(1)
    except CORTES as e:
        log.error(f"corte ({type(e).__name__}): en esta ejecución se escribieron {escritos or 'ninguno'}; no se escribieron los "
                  "resultados que faltaban. Si la marca de inicio existe, se aplica la regla de corte de la Enmienda 8 (sección 2.4)")
        raise
    except Exception:
        log.exception(f"error inesperado: en esta ejecución se escribieron {escritos or 'ninguno'}; no se escribieron los "
                      "resultados que faltaban. Si la marca de inicio existe, el error del código se corrige con una nueva "
                      "enmienda (Enmienda 8, sección 2.4)")
        raise

    log.info(f"HE1 {'se cumple' if he1 else 'no se cumple'} · salidas: {f['actual'].name}, {f['he1'].name}, "
             f"{f['resultados'].name}, {f['texto'].name}")


if __name__ == "__main__":
    main()
