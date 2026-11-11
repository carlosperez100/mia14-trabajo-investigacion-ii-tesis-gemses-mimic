# Prototipo del sistema de detección y priorización de eventos adversos

El prototipo integra en una aplicación local los tres pasos del sistema, que son la detección de posibles eventos adversos en la epicrisis, la revisión de cada caso por una persona y la priorización de los eventos confirmados. La priorización se aplica a los ERSP (eventos relacionados con la seguridad del paciente) por medio de la Matriz de Priorización GEMSES, calculada según los num. 19.9 a 19.13 y el Anexo N.° 03 de la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01. La aplicación corre solo en la máquina del usuario, y la demostración usa epicrisis inventadas y un modelo entrenado con ellas, por lo que sus resultados no tienen valor clínico.

## 1. Flujo del sistema

Como se aprecia en la Tabla 1, cada paso tiene un responsable y una fuente para su criterio. El modelo solo marca la epicrisis, la decisión sobre la existencia del evento corresponde al revisor y la Matriz se calcula únicamente con los eventos que el revisor confirmó, salvo los ERSP inventados que carga la demostración.

Tabla 1. Pasos del prototipo

| Paso | Qué hace | Quién decide | Fuente del criterio |
|---|---|---|---|
| 1. Detección | Calcula la probabilidad de evento adverso, marca la epicrisis con el corte de 0.5 y, si la epicrisis marcada no tiene ningún código de evento, la envía a la cola “Posible evento sin código” | El modelo marca, pero no decide | Plan de tesis, que define la detección como “la epicrisis sin código con probabilidad de 0.5 o más” |
| 2. Revisión | Registra cada evento con su cita literal, gravedad, nexo con la atención, categoría ERSP, Id. de Evento del Anexo N.° 02, proceso y servicio, bloquea el registro que incumple las reglas del manual y envía a adjudicación los casos dudosos y los eventos centinela | El revisor | Manual de anotación v2 de la tesis (§1.2 a §5) y, para el Id. de Evento, el proceso y el servicio, Directiva N.° 7-OGCyH-ESSALUD-2020 V.01, num. 19.5 a 19.7 |
| 3. Priorización | Agrupa los ERSP confirmados por proceso e Id. de Evento, hace el corte de 15 días, calcula los índices y asigna el nivel de gestión, el responsable y el comité | Cálculo sobre los eventos confirmados | Directiva N.° 7-OGCyH-ESSALUD-2020 V.01, num. 19.9 a 19.13 y Anexo N.° 03 |

Fuente: elaboración propia a partir de la Directiva N.° 7-OGCyH-ESSALUD-2020 V.01 (EsSalud, 2020), del manual de anotación v2 y del plan de tesis.

## 2. Cómo ejecutar la demostración

Los comandos se ejecutan desde la raíz del repositorio con Python 3.13. La aplicación atiende solo en la dirección local de la máquina, según `prototipo/.streamlit/config.toml`, que Streamlit lee desde la carpeta de la aplicación aunque se arranque desde otra carpeta, y queda disponible en el navegador en http://127.0.0.1:8501.

1) Instalar las dependencias.
   ```bash
   pip install -r prototipo/requirements.txt
   ```
2) Entrenar el modelo de demostración con 600 epicrisis inventadas, de las cuales 322 tienen un evento, lo que toma unos segundos.
   ```bash
   python -m prototipo.modelo_demo
   ```
3) Iniciar la aplicación y abrir http://127.0.0.1:8501 en el navegador.
   ```bash
   streamlit run prototipo/app.py
   ```
4) En la barra lateral, cargar las 10 epicrisis inventadas y los ERSP inventados del semestre.
5) En la pestaña de revisión, elegir una epicrisis de la cola “Posible evento sin código”, copiar de su texto la cita ancla, completar el formulario, registrar el evento y cerrar el caso. Si la epicrisis no tiene ningún evento anotable, el caso se cierra sin eventos y con su motivo.
6) En la pestaña de priorización, revisar la Matriz con la fecha de corte del día y descargarla en CSV o en Excel. Si no se configuró la tabla local de impactos (sección 10, limitación 5), antes se valoran los cuatro impactos de cada evento en la tabla que muestra la misma pestaña.

La base de datos, la bitácora y los modelos se guardan en la carpeta local del prototipo, que por defecto es `~/.mia14_prototipo` y se puede cambiar con la variable PROTOTIPO_DATOS del `.env`. De esta manera, ninguno de esos archivos entra al repositorio.

## 3. Epicrisis y ERSP inventados de la demostración

Las 10 epicrisis de `prototipo/datos_demo/epicrisis_demo.csv` se redactaron para la demostración y no corresponden a ningún paciente. Como se aprecia en la Tabla 2, cubren casos con evento y sin él, con código de evento y sin código, y un caso con una explicación competidora que el revisor debe rechazar.

Tabla 2. Epicrisis inventadas y resultado esperado de la revisión

| Epicrisis | Caso | Códigos CIE-10 | Cola con el modelo de demostración | Resultado esperado de la revisión |
|---|---|---|---|---|
| DEMO-01 | Artroplastia de cadera con caída en el hospital y fractura de radio | M16.11; Z96.641; I10 | Posible evento sin código | 202 · Caída de paciente |
| DEMO-02 | Insuficiencia cardíaca sin evento | I50.23; I25.5 | No marcada | Sin evento |
| DEMO-03 | Lesión renal aguda atribuida a la vancomicina | L03.115; N17.9; E11.9 | Posible evento sin código | 7021 · Reacción adversa al medicamento |
| DEMO-04 | Colecistectomía laparoscópica con infección de la herida operatoria | K80.20; T81.41XA; B95.61 | Marcada, con código de evento | 604 · Infección del sitio quirúrgico superficial |
| DEMO-05 | Choque séptico con lesión renal aguda multifactorial | R65.21; N17.9; B96.20 | Posible evento sin código | Sin evento, por explicación competidora (R-11, nexo C2) |
| DEMO-06 | Neumotórax tras la colocación de un catéter venoso central | K26.4; D62 | Posible evento sin código | 8016 · Neumotórax |
| DEMO-07 | Artroplastia de rodilla sin complicaciones | M17.12; Z96.652 | No marcada | Sin evento |
| DEMO-08 | Úlcera por presión sacra en estadio 3 adquirida en el hospital | J44.1; Z99.81 | Posible evento sin código | 2018 · UPP III |
| DEMO-09 | Dosis duplicada de insulina glargina con hipoglucemia | E11.621; L08.9; E11.649 | Posible evento sin código | 705 · Dosis o frecuencia errónea |
| DEMO-10 | Fractura del cuello femoral con anafilaxia a la cefazolina | S72.001A; T88.6XXA; Z88.1 | Marcada, con código de evento | 7021 · Reacción adversa al medicamento |

Fuente: elaboración propia. La cola es la que asignó el modelo de demostración en la prueba del 11 de noviembre de 2026.

Asimismo, el botón de ERSP inventados carga 48 registros de 16 grupos de proceso e Id. de Evento, con fechas dentro de los 178 días que terminan en la fecha del día. De ellos, 8 grupos tienen un registro en los 14 días que terminan en esa fecha, de modo que la Matriz de la demostración parte con 8 filas, a las que se suman los eventos que se confirmen en la revisión. Cada vez que se pulsa, el botón reemplaza los ERSP inventados que se cargaron antes, con el fin de no duplicar la frecuencia de ningún grupo.

## 4. Detección por lote

La detección también se puede ejecutar sin la interfaz sobre un CSV con las columnas del contrato. La salida tiene una fila por epicrisis válida y no repite el texto, mientras que la consola solo muestra conteos, identificadores y motivos de rechazo.

```bash
python -m prototipo.lote entrada.csv salida.csv --guardar
```

Con `--guardar`, las epicrisis y sus detecciones se registran en la base local para revisarlas en la interfaz, y con `--reemplazar` se reemplazan las epicrisis que ya existían, lo cual reinicia su revisión y queda en la bitácora. Por otra parte, `--modelo` indica otro modelo con su ficha, el cual solo se abre si también se indica su huella SHA-256 esperada con `--sha256`.

La epicrisis que no cumple el contrato no detiene el lote, ya que se informa en la consola con su motivo y con su identificador o, si este no es válido, con su número de fila. En cambio, un identificador válido repetido detiene el lote antes de calcular, porque no se sabría a qué epicrisis corresponde cada detección.

## 5. Contrato de entrada y salida

La Tabla 3 resume el contrato del detector (versión 1.0), el cual se valida antes de calcular cualquier probabilidad. La epicrisis que no lo cumple no se procesa, y la aplicación informa el motivo sin detener el resto del lote.

Tabla 3. Contrato de entrada y salida del detector

| Campo | Contenido | Regla |
|---|---|---|
| id_epicrisis | Identificador que asigna el usuario | De 1 a 64 letras, dígitos, guiones, guiones bajos o puntos. El contrato pide no usar un identificador de MIMIC-IV, lo cual queda a cargo del usuario, porque la aplicación no lo comprueba |
| texto | Epicrisis en inglés | 100 caracteres o más, el largo mínimo con que se armó el corpus de entrenamiento |
| 18 variables de la hospitalización | Edad, horas en emergencia, número de servicios, estancia, traslados, unidades, días en UCI, medicamentos y cultivos; sexo, tipo de ingreso, procedencia, seguro, servicio al ingreso y al alta, y destino al alta; paso por un servicio quirúrgico y por UCI | Opcionales. Las numéricas deben ser números finitos y no negativos, y las de paso valen 0 o 1. El faltante se imputa igual que en el entrenamiento, y la categoría que el modelo no vio pasa a la categoría infrecuente, si el codificador la formó, o se codifica con ceros |
| codigos_cie y version_cie | Códigos CIE del alta, separados por punto y coma, y su versión | La versión vale 10 por defecto, o 9 |
| Salida | Probabilidad, marca, indicador de “sin código”, cola, códigos de evento, corte, huella SHA-256 y tipo del modelo, versión del contrato, fecha y hora | No repite el texto de la epicrisis |

Fuente: elaboración propia a partir del contrato del detector.

## 6. Modelo local entrenado con MIMIC-IV

El modelo local se entrena con MIMIC-IV, solo en la máquina donde están los datos, y con la configuración actual, que es la del lote de la revisión ciega y la de la HE1. Dicha configuración, registrada en la sección `he1_estratos` de `config/config.yaml`, comprende la variante D2 con los bloques A+B+C, la regresión logística con C de 1.0 y pesos balanceados, y los bloques A y B de 20 000 términos con frecuencia mínima de 3 epicrisis y *sublinear_tf*.

Las filas de entrenamiento se eligen con `--filas`. Con `todo` se usan las epicrisis que la variante D2 conserva en todo el corpus, que según el plan de tesis son 61 319 y con las cuales se reentrenó el modelo adoptado que marcó las detecciones de la HE2 y la HE3. Con `train` se usan solo las filas D2 de *train*, como en el modelo de la HE1.

```bash
python -m prototipo.entrenar_local --datos "<repositorio de trabajo>/data" --filas todo
```

El script guarda el modelo y su ficha, por defecto, en la carpeta local del prototipo, fuera del repositorio, cuyo `.gitignore` excluye además todo archivo `.joblib` y toda ficha de modelo. Al terminar, imprime las dos líneas que se copian al `.env`, es decir, PROTOTIPO_MODELO y PROTOTIPO_MODELO_SHA256.

A partir de ese momento, la aplicación usa el modelo local en lugar del de demostración y solo lo abre si su huella SHA-256 coincide con la de su ficha y con la fijada en el `.env`. Sin la huella fijada en el `.env`, el modelo local no se abre.

A criterio del autor, los modelos entrenados no se publican, dado que su vocabulario proviene de las notas de MIMIC-IV y publicarlos equivaldría a compartir parte de los datos. El uso de esos datos se rige por la licencia y el acuerdo de uso de datos (DUA) de PhysioNet, y por ello el repositorio solo contiene el código que entrena los modelos.

## 7. Criterios aplicados donde las fuentes no resuelven

En el cálculo de la Matriz, la Directiva deja abiertos algunos puntos o los resuelve de forma distinta en su texto y en sus anexos. Los ocho criterios que siguen los resuelven, cada uno con su rótulo de derivación o de criterio del autor, y se muestran también en la pestaña de priorización de la aplicación.

1) A criterio del autor, se aplica el texto del num. 19.11 a y f, que dice “entre”, y no la fórmula impresa en el Anexo N.° 03 (“B= (A+Σa)” e “I= H + Σh”). Es decir, B = A / ΣA e I = H / ΣH.
2) El num. 19.11 a pone las dos sumatorias en una oración que termina “en el último semestre, en el mismo proceso donde se identificó el ERSP”, de lo cual se deriva que esa ventana se aplica a las dos. A criterio del autor, el semestre son los 6 meses que terminan en la fecha de corte y el periodo de corte, los 15 días que terminan en ella.
3) Del num. 19.10 (“de los ERSP Notificados en dicho periodo”) se deriva que las filas de la Matriz son los grupos de proceso e Id. de Evento con al menos un ERSP en el periodo de corte. En cambio, A y ΣA cuentan todo el semestre.
4) El num. 19.12 no dice cómo se asignan los valores que caen entre percentiles. A criterio del autor, se usa la regla J > P75 → Nivel III, P50 < J ≤ P75 → Nivel II y J ≤ P50 → Nivel I, con percentiles por interpolación lineal.
5) A criterio del autor, ΣH y los percentiles se calculan sobre las filas que se muestran, sean de todos los procesos o de uno solo.
6) Se usan los nombres del num. 19.12 (“Servicio y/o Departamentos” y “Director de la IPRESS”), aunque el Anexo N.° 03 dice “Departamentos” y “Director del establecimiento” y llama “Nivel de Riesgo” al valor de riesgo del num. 19.11 e.
7) Del num. 19.11 d, que define el impacto como “la sumatoria de todas las ponderaciones obtenidas”, se deriva que G es la suma ponderada de C, D, E y F, fórmula que el Anexo N.° 03 imprime como “G=(cx0.4)+(dx0.20)+(ex0.15)+(fx0.25)”. El “Impacto” impreso en el Anexo N.° 02 no coincide con esa suma en 230 de los 231 eventos, por lo que solo se usa en el análisis de sensibilidad. A criterio del autor, el evento con valores fuera de la escala 9, 6, 3, 1 y 0 se calcula con la suma ponderada de esos valores, tal como están impresos.
8) A criterio del autor, al igual que en el plan de tesis, la Matriz se aplica a los ERSP que confirma la revisión, aunque los num. 19.9 y 19.10 hablan de los ERSP “Notificados”.

Adicionalmente, el prototipo simplifica la revisión respecto del manual de anotación v2, aunque conserva la regla por la cual el evento centinela pasa siempre a adjudicación (§5, pasos P5 y P6). A criterio del autor, las simplificaciones son las siguientes:

1) La cita ancla se pega en un campo y se verifica contra el texto, en lugar de capturarse por selección con sus posiciones de inicio y fin (§4.1), y debe tener tres palabras o más, para evitar anclas triviales.
2) Para EC-2 y EC-3 se marcan los cuatro elementos del art. 5.4 con una sola cita, la del desenlace, en lugar de una cita por casilla (§3.2.4). Asimismo, no se verifica que la cita contenga la frase de vínculo que pide EC-1 ni el desenlace que pide EC-2.
3) La vía A o B se deduce del nexo elegido, C4 o C3, y cada evento lleva un solo Id. de Evento del Anexo N.° 02 (num. 19.5), en lugar de varias naturalezas.
4) No se registran el origen del daño ni su marcador, ni los minutos y las secciones revisadas al cerrar el caso.
5) No hay bitácoras B-1, B-2 ni B-3. El caso sin eventos anotables se cierra con su motivo (sin daño de gravedad E o mayor, C1, C2 o D3), que queda en la bitácora general sin texto, y el caso C2 no pasa a adjudicación.
6) La adjudicación de los casos dudosos y de los eventos centinela la registra cualquier revisor, y su nombre queda en la bitácora.

De igual forma, la regla de “sin código” aproxima la población de la HE2 y la HE3 del plan de tesis. Según el plan, ninguna de esas epicrisis tiene “filas en las tablas de la etiqueta ni códigos de las familias de causa explícita o de efecto adverso de medicamento”. Por ello, en CIE-10 un código cuenta como código de evento si empieza con un código de la equivalencia con el Anexo N.° 02, validada el 29 de octubre de 2026.

También cuenta si pertenece a las familias de causa explícita de la etiqueta o si es un efecto adverso de medicamento, mientras que en CIE-9 cuentan las cinco familias de la etiqueta. Sin embargo, a criterio del autor, el código que la equivalencia validó como “no es evento adverso” no cuenta, salvo que caiga en una de esas familias, aunque en la tesis pertenece al estrato A2 de la tabla de la etiqueta. Por consiguiente, la cola “sin código” recibe epicrisis que la población del plan excluiría, como DEMO-03, cuyo código N17.9 tiene ese estado.

## 8. Gobernanza, privacidad y seguridad

Dado que un archivo `.joblib` puede ejecutar código al abrirse, el modelo solo se abre si su huella SHA-256 coincide con todas las que exige su origen. El modelo de demostración de la carpeta local se contrasta con su ficha, el modelo local con su ficha y con la huella fijada en el `.env`, y cualquier otro modelo con su ficha y con la huella que indica el usuario.

Además, la huella se calcula sobre los mismos *bytes* que luego se abren. Por otra parte, la aplicación no muestra términos, coeficientes ni resaltados del modelo, de modo que cada caso se decide con la lectura de la epicrisis.

La aplicación atiende solo en la dirección local de la máquina (127.0.0.1) y no envía estadísticas de uso. Fuera del contenedor, se detiene si la dirección de escucha es otra, y en el contenedor el puerto se publica solo en 127.0.0.1. Asimismo, ante un error no controlado, el navegador muestra solo el tipo de excepción y un mensaje genérico, mientras que el mensaje completo, que podría repetir texto de la epicrisis, se imprime solo en la consola local.

La base de datos y los modelos quedan fuera del repositorio, y la bitácora nunca guarda el texto de una epicrisis ni las citas. La bitácora registra cada acción con su fecha e identificador, las decisiones de revisión con el revisor, las detecciones con los primeros 12 caracteres de la huella del modelo y cada reemplazo de una epicrisis con el número de eventos y cierres borrados.

A criterio del autor, el juicio clínico no se delega al modelo. Por ello, ningún evento entra a la Matriz sin la decisión del revisor, salvo los ERSP inventados de la demostración, y los casos dudosos y los eventos centinela pasan a adjudicación.

Las reglas de bloqueo del manual impiden, entre otros casos, registrar un evento sin cita literal, con gravedad menor que E, con nexo C1 o C2, o intencional. Asimismo, los num. 19.5 a 19.7 de la Directiva describen cómo se asignan el Id. de Evento, el proceso y el servicio de cada ERSP, de lo cual se deriva que ningún evento se registra sin esos tres datos.

## 9. Contraste con el marco del autor para proyectos de salud

El marco del autor para proyectos de salud reúne cuatro criterios propios, con los cuales se contrasta toda incorporación de tecnología. Como se aprecia en la Tabla 4, el prototipo cumple parte de ellos, no cumple la interoperabilidad en esta etapa y deja pendiente el ensayo con revisores y con el modelo local.

Tabla 4. Contraste del prototipo con el marco del autor

| Criterio del marco del autor | Situación en el prototipo |
|---|---|
| 1. Simulación en los tres enfoques de seguridad del paciente | En el enfoque proactivo (S3), la demostración con epicrisis inventadas permite ensayar el recorrido completo antes de usarlo con epicrisis reales; falta repetir ese ensayo con revisores y con el modelo local. Los enfoques reactivo (S1) y cotidiano (S2) no aplican a esta etapa. |
| 2. Competencias no delegables a la IA | El juicio clínico adaptativo queda en el revisor, porque el modelo solo marca y ninguna epicrisis pasa a la Matriz sin su decisión. La alianza terapéutica, la seguridad psicológica y la paradoja tecnológica no aplican a esta etapa. |
| 3. Tres estrategias de impacto hacia 2027 | Se alinea con la seguridad integrada al proceso, por la trazabilidad de cada decisión y la analítica de los ERSP confirmados que hace la Matriz. El hospital diseñado alrededor del flujo y la experiencia con continuidad no aplican. |
| 4. Tecnología avanzada: invertir donde cambia el resultado | El problema que atiende es el evento adverso no registrado, que el plan de tesis define como el “evento confirmado en la revisión ciega en una hospitalización sin ningún código de evento adverso”. La interoperabilidad no se cumple en esta etapa, porque el prototipo no se conecta a la historia clínica electrónica. La gobernanza de la IA comprende la validación clínica prevista en la revisión ciega, la privacidad, la supervisión humana y la auditoría por medio de la bitácora. |

Fuente: elaboración propia a partir del marco del autor para proyectos de salud.

## 10. Limitaciones

1) El modelo de demostración se entrenó con 600 epicrisis inventadas por plantillas y no tiene valor clínico, ya que su único fin es mostrar el recorrido del sistema.
2) El detector solo admite epicrisis en inglés, como las de MIMIC-IV, por lo que su uso con epicrisis de EsSalud exigiría un modelo entrenado y validado con ellas.
3) La regla de “sin código” aproxima la población de la HE2 y la HE3, pero no la reproduce, porque las tablas de la etiqueta se arman por hospitalización con el *pipeline* de la tesis y el prototipo solo conoce los códigos que se ingresan. Por ejemplo, un código de la equivalencia validado con evento, como J18.9 (neumonía, organismo no especificado) o A41.9 (septicemia no especificada), saca la epicrisis de esa cola aunque la condición haya sido el motivo del ingreso. En ese caso, el revisor la encuentra en la cola de marcadas con código. En sentido contrario, el criterio de la sección 7 sobre los códigos validados como “no es evento adverso” deja en la cola epicrisis que la población del plan excluiría.
4) La sugerencia de Id. de Evento sale de la equivalencia y no indica si la condición estaba presente al ingreso, por lo que el revisor la confirma o la cambia.
5) La tabla de impactos del Anexo N.° 02 no está en el repositorio, y sin ella el usuario valora los cuatro impactos de cada evento con la escala del num. 19.11 b. Además, la tabla extraída del Anexo N.° 02 trae un evento, el 6024, con valores fuera de esa escala, cuyo impacto se calcula con la suma ponderada de esos valores (sección 7, criterio 7) y se señala en pantalla.
6) La fecha de cada ERSP es la que registra el revisor, que por defecto es la del día, y con ella se decide si el ERSP entra al periodo de corte y al semestre.
7) Con pocos grupos, los percentiles casi no separan los niveles. Por ello, a criterio del autor, la aplicación avisa cuando la Matriz tiene menos de cuatro grupos, aunque la Directiva no fija un mínimo.
8) El Dockerfile y el archivo de Compose no se probaron, porque la máquina de desarrollo no tiene Docker.

## 11. Estructura y pruebas

```
prototipo/
 ├── app.py               # interfaz local (Streamlit): detección, revisión, priorización y acerca del sistema
 ├── contrato.py          # contrato de entrada y salida del detector (versión 1.0)
 ├── detector.py          # carga del modelo con verificación de la huella y cálculo de la probabilidad
 ├── codigos.py           # regla de “sin código”: familias de la etiqueta y equivalencia validada
 ├── revision.py          # criterios y reglas de bloqueo del manual de anotación v2
 ├── gemses.py            # Matriz de Priorización (num. 19.9 a 19.13 y Anexo N.° 03)
 ├── catalogo.py          # 231 eventos del Anexo N.° 02, procesos del num. 19.6 y tabla local de impactos
 ├── almacen.py           # base SQLite local y bitácora sin texto clínico
 ├── lote.py              # detección por lote desde la línea de comandos
 ├── entrenar_local.py    # entrenamiento del modelo local con MIMIC-IV, solo en la máquina del investigador
 ├── modelo_demo.py       # epicrisis y ERSP inventados, y modelo de demostración
 ├── rutas.py             # carpeta local y variables del .env
 ├── datos_demo/          # las 10 epicrisis inventadas
 ├── .streamlit/          # config.toml: atiende solo en 127.0.0.1 y sin estadísticas de uso
 └── requirements.txt, Dockerfile y compose.yaml
tests/                    # 39 pruebas (pytest)
```

Las 39 pruebas se ejecutan con `python -m pytest tests` y cubren el cálculo de la Matriz con un ejemplo calculado a mano, la regla de “sin código”, el contrato y las reglas de bloqueo de la revisión. Asimismo, cubren el almacén con su bitácora, la verificación de la huella del modelo según su origen y la detección por lote de punta a punta, con un modelo pequeño que se entrena durante la prueba. Para usar un contenedor, se ejecuta `docker compose -f prototipo/compose.yaml up --build` desde la raíz, opción que no se probó.

## Referencias

- Seguro Social de Salud [EsSalud]. (2020). *Directiva de Gerencia General N.° 7-OGCyH-ESSALUD-2020 V.01: Registro, notificación y gestión de los eventos relacionados con la seguridad del paciente en ESSALUD*. Aprobada por la Resolución de Gerencia General N.° 402-GG-ESSALUD-2020, del 4 de marzo de 2020. Primera edición impresa: Lima, marzo de 2021.
