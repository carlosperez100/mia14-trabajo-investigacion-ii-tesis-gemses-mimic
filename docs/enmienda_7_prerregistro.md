# Enmienda 7 del prerregistro: tramos del puntaje, precisiones del análisis y regla de los cambios posteriores

- Fecha de registro: 24 de noviembre de 2026.
- Lo que se enmienda: del plan de tesis del 17 de noviembre de 2026, se precisa el pasaje de la sección 3.1.3 sobre los tramos del puntaje. De la Enmienda 5 (*commit* ed35d0f), se precisan las secciones 2.4 y 2.6, y se sustituye la última oración de la sección 2.8. De la Enmienda 6 (*commit* d5dcd14), se precisa la sección 3 y se sustituye su última oración. Las precisiones 2 y 4 de la sección 2.3 se añaden sin modificar ningún pasaje.
- Resumen estructurado: sección `analisis_enmienda_7` de `config/config.yaml`, en este mismo *commit*.
- Origen: revisión del protocolo al escribir el programa del análisis, el 22 de noviembre de 2026.
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda

1) La primera lectura de las 615 epicrisis está terminada. Un conteo del 24 de noviembre de 2026 confirma que los 615 casos del archivo de respuestas tienen veredicto. El conteo no muestra cuál es.
2) La re-revisión no ha empezado. El 24 de noviembre de 2026, su archivo de respuestas todavía no existe.
3) Los resultados de la revisión ciega no se han abierto. Nadie ha visto los veredictos separados por detecciones y controles.
4) El investigador conoce sus veredictos de las 615 epicrisis, pero no sabe a qué grupo pertenece cada una ni cuál es su puntaje. Podría reconocer parte de las detecciones, como declara la Enmienda 3 en la sección 2.5.
5) No se ha calculado ninguno de los análisis de las Enmiendas 5 y 6. El *script* de la HE1 no se ha ejecutado.
6) El programa que verifica el registro (Enmienda 6, sección 2.5) y el programa del análisis, que es el *script* del análisis de la Enmienda 5 (sección 2.8), están escritos. Se probaron solo con respuestas y claves inventadas, junto con el catálogo y la tabla de impactos del Anexo N.° 02. No se han ejecutado con los archivos de respuestas. Para escribirlos se leyeron el código de las dos versiones de la aplicación y el del *script* del sorteo. De los archivos de respuestas se leyeron solo sus nombres y su primera línea, que tiene los nombres de las columnas. La clave de la muestra no se abrió.
7) Ninguna persona distinta del investigador ha leído las epicrisis de la muestra.

## 2. Cambios

### 2.1 Motivo

La sección 3.1.3 del plan de tesis dice: “En cambio, el riesgo de un evento no registrado se describirá por tramos del puntaje en las 615 epicrisis revisadas, con los pesos del diseño (ver sección 4.7).” El plan no dice cuáles son esos tramos. En el texto del plan registrado, anexos incluidos, la palabra “tramo” aparece una sola vez, en singular o en plural, y es en esa oración. Tampoco aparece en la sección `he1_estratos` del prerregistro, en las Enmiendas 1 a 6 ni en sus secciones de `config/config.yaml`.

Sin esa definición, el análisis no se puede programar antes de abrir los resultados. La Enmienda 6 dice en su sección 3: “Cualquier otro cambio exigiría una nueva enmienda, con su razón, registrada antes de abrir los resultados.” Por ello, esta enmienda precisa los tramos antes de abrir.

### 2.2 Tramos del puntaje

Cada estrato del diseño se divide en tres tramos según el puntaje. Son seis tramos:

1) Detecciones, con puntaje de 0.5 o más. Son las 385 detecciones revisadas, tomadas de las 15 731 epicrisis marcadas del marco. Forman tres tramos.
2) Controles, con puntaje menor de 0.5. Son los 230 controles revisados, tomados de las 83 200 epicrisis sin marcar del marco. Forman otros tres tramos.

Los casos de cada estrato se ordenan por su puntaje, de menor a mayor, y se dividen en tres grupos consecutivos. Las 385 detecciones quedan, de menor a mayor puntaje, en tramos de 129, 128 y 128 casos, y los 230 controles, en tramos de 77, 77 y 76. Un empate de puntaje se ordena por el número del caso. El puntaje es el que guarda la clave de la muestra, el mismo con el que se formaron los estratos. Los tamaños dependen solo del número de casos de cada estrato, que fija la Enmienda 3. Los tramos se calculan al abrir los resultados.

Ningún texto del protocolo fija estos tramos, y ninguno los contradice. La decisión se toma a partir de cinco premisas. Las premisas 1, 3 y 5 contienen una derivación, que se señala en el texto.

1) El quinto párrafo de la sección 3.1.3 del plan empieza así: “El puntaje del modelo no es una probabilidad calibrada, por el enriquecimiento del corpus y los pesos balanceados.” Siguen dos oraciones sobre la calibración, una sobre su ajuste antes del uso operativo y otra sobre su evaluación frente a la etiqueta en la prueba. La oración de los tramos viene después y empieza con “En cambio”. Se deriva que, para el evento no registrado, el riesgo no se evalúa como una calibración, sino que se describe por tramos del puntaje. Para eso, los tramos tienen que permitir ver el riesgo observado a lo largo del puntaje. Si los tramos fueran los dos estratos, solo repetirían p1 y p0 de la HE2.
2) La misma oración dice dónde se forman los tramos: “en las 615 epicrisis revisadas”. Por ello, los tramos se forman en la muestra y no en el marco.
3) El plan fija el corte en la sección 3.1.2: “El puntaje se usa con el corte de 0.5, que se fijó antes del sorteo de la revisión ciega.” La sección 3.2 define la detección como “la epicrisis sin código con puntaje de 0.5 o más”. Se deriva que los tramos no deben cruzar ese corte. El corte separa los dos estratos del diseño, que tienen pesos distintos, y el cálculo de esta sección supone un solo peso por tramo. Además, como la muestra no es proporcional, un tercio de las 615 epicrisis no representaría un tercio del marco. Dentro de un estrato, en cambio, todos los casos tienen el mismo peso.
4) El diseño es un muestreo estratificado no proporcional (plan, sección 4.4), con un sorteo por estrato y una semilla para cada uno (Enmienda 3, sección 2.2). La sección 4.7 del plan fija los pesos: “cada caso se pondera por el inverso de su fracción de muestreo, que es 15 731/385 en las detecciones y 83 200/230 en los controles, es decir, 40.86 y 361.74”. Esos son los pesos del diseño que pide la oración de la 3.1.3.
5) Ningún texto del protocolo fija el número de tramos. El protocolo usa divisiones distintas para fines distintos. El prerregistro y la Enmienda 1 (sección 2.2) usan deciles del largo, y la Enmienda 5 resume el tiempo de lectura con la mediana y los cuartiles (sección 2.4). Para describir la HE2 y el VPP según una variable continua, la Enmienda 5 usa terciles en la misma sección: “Además, la HE2 y el VPP se describen por terciles del tiempo de lectura.” Esos terciles se calculan dentro de cada forma de lectura, como aquí dentro de cada estrato. Se deriva que los tramos del puntaje, que son una descripción de la misma clase, se forman también por terciles. Con terciles, los cortes salen del orden del puntaje y no de un valor elegido.

La Tabla 1 muestra cuánto se ensancha el intervalo de cada tramo al aumentar el número de tramos. Se calculó sin usar veredictos, con una proporción de 0.5, que da el intervalo más ancho posible. Con terciles, la semiamplitud del IC 95 % de Wilson en el tramo más pequeño es, como máximo, de 0.085 en las detecciones y de 0.110 en los controles. La tabla no fija el número de tramos, y solo muestra el costo en precisión de cada tramo que se añade.

Tabla 1. Semiamplitud máxima del IC 95 % de Wilson en el tramo más pequeño de cada estrato

| Tramos por estrato | Detecciones: casos del tramo más pequeño | Detecciones: semiamplitud | Controles: casos del tramo más pequeño | Controles: semiamplitud |
|---|---|---|---|---|
| 1 | 385 | 0.050 | 230 | 0.064 |
| 2 | 192 | 0.070 | 115 | 0.090 |
| 3 | 128 | 0.085 | 76 | 0.110 |
| 4 | 96 | 0.098 | 57 | 0.126 |
| 5 | 77 | 0.109 | 46 | 0.139 |

Fuente: elaboración propia, con la fórmula de Wilson y una proporción de 0.5, sin usar veredictos.

Para cada tramo se informan el puntaje menor y el mayor, el número de epicrisis revisadas, cuántas tienen al menos un evento confirmado y la proporción con su IC 95 % de Wilson. El veredicto “No se puede determinar” cuenta como sin evento, como en la HE2 (plan, sección 4.7). Se informan también las epicrisis del marco que representa el tramo, que son sus casos por el peso del estrato. Con los tamaños de esta sección, son 5 270.9, 5 230.0 y 5 230.0 en las detecciones, y 27 853.9, 27 853.9 y 27 492.2 en los controles. El número estimado de epicrisis con evento en el marco es la suma de los pesos de las epicrisis con evento del tramo. Su intervalo es el número de epicrisis del marco que representa el tramo multiplicado por cada límite del IC 95 % de Wilson de la proporción.

La suma de los números estimados de los tres tramos de un estrato es el número de epicrisis del estrato en el marco, 15 731 o 83 200, por p1 o por p0. La suma de los seis es el primer dato descriptivo de la sección 4.7 del plan, “15 731·p1 + 83 200·p0”. Esa suma conserva su intervalo por el método delta, y los intervalos de Wilson de los tramos no se suman.

Los seis tramos no se comparan con una prueba de hipótesis ni con una prueba de tendencia. Se sigue así la regla de la Enmienda 3 para las partes de la muestra, en la sección 2.3: “Las partes no se comparan con una prueba de hipótesis.” Como con los tramos no se contrasta ninguna hipótesis, sus intervalos no se ajustan por multiplicidad.

Después de terminar las dos lecturas y antes de abrir los resultados, se comprueba el puntaje en la clave. Cada detección debe tener un puntaje de 0.5 o más, y cada control, un puntaje menor de 0.5. La comprobación no une los veredictos con la clave. Este análisis es descriptivo y no decide la HE2.

### 2.3 Precisiones del análisis

Al escribir el programa del análisis se hallaron tres situaciones que el protocolo no resuelve, y se añadió un procedimiento. Se registran sin haber comprobado si esas situaciones se dan en los archivos de respuestas, salvo lo que dice el numeral 2.

1) Casos sin tiempo de lectura. La Enmienda 6 dice en la sección 2.2: “La versión 2 añade columnas al archivo de respuestas, y esos casos las conservan vacías.” El código de la aplicación, en sus dos versiones, deja también vacía la columna del tiempo en los casos guardados antes de que existiera. Sin embargo, la versión 1 ya creaba el archivo de respuestas con esa columna, y las dos versiones escriben el tiempo en cada caso que guardan. Por ello no se espera ningún caso sin tiempo, y la precisión es preventiva. La Enmienda 5 dice en la sección 2.4, al fijar los terciles del tiempo: “Ningún caso se excluye por su tiempo.” Un caso sin tiempo no puede asignarse a un tercil. Como única excepción a esa oración, ese caso queda fuera de los terciles y de la descripción del tiempo, y entra en todos los demás análisis. Se informa cuántos casos no tienen tiempo, por forma de lectura. Un tiempo escrito que no es un número finito, o que es negativo, detiene el análisis antes de abrir los resultados.
2) Casos fuera de alcance. Las dos versiones de la aplicación admiten, en el archivo de respuestas, una columna, `en_alcance`, que indica si el caso sigue en la muestra. Un caso fuera de alcance no se borra. La aplicación deja de pedirlo, pero lo conserva en el archivo. Si esa columna existe, todo valor distinto de `true`, `1`, `sí` o `si`, sin distinguir mayúsculas, cuenta como fuera de alcance, incluido el vacío, como en la aplicación. Una fila fuera de alcance que no corresponde a ninguno de los 615 casos de la muestra ni de los 60 de la re-revisión no cuenta en el análisis, porque la muestra es la de la Enmienda 3, y se informa cuántas son. Si uno de los 615 casos de la muestra o de los 60 de la re-revisión queda fuera de alcance, el análisis se detiene antes de abrir, porque la lectura no estaría completa. La Enmienda 3 dice en la sección 2.3: “Si la primera lectura de las 615 epicrisis no se completa, el análisis exigirá una nueva enmienda, registrada antes de abrir los resultados.” La primera línea del archivo de respuestas de la primera lectura, leída el 22 de noviembre de 2026, no tiene esa columna. La precisión es preventiva.
3) Eventos fuera del Anexo N.° 02. La Enmienda 5 establece en la sección 2.6 que la aplicación los registra como OTRO. La versión 2 guarda OTRO seguido del identificador del caso en el evento principal, y seguido además del orden en los eventos adicionales. El código de la versión 1 usa el mismo formato. Por ello, se cuenta como evento fuera del anexo todo identificador que es OTRO, solo o seguido de un carácter que no es una letra de la A a la Z ni un número. No se distinguen mayúsculas. Vale también para las 140 primeras epicrisis.
4) Ensayo previo. Después de terminar las dos lecturas y antes de registrar las huellas que fija la Enmienda 5 (sección 2.8), el programa del análisis puede correr un ensayo. El ensayo hace las comprobaciones previas a la apertura, salvo la de las huellas y la del resultado de la verificación, y termina sin unir los veredictos con la clave. No informa conteos por veredicto ni por grupo. Si una comprobación falla, la falla y lo que se hace con ella se registran por escrito en el repositorio antes de abrir los resultados. Si resolverla exige cambiar lo registrado, se hace con una nueva enmienda (sección 2.5). Los archivos de respuestas no se editan fuera de la aplicación (Enmienda 6, sección 2.5). La huella del programa del análisis se registra después de escribir en él el *commit* de esta enmienda.

### 2.4 Limitaciones que se declaran

1) Los tramos dividen la muestra y no el marco. Las epicrisis del marco que representa cada tramo se estiman con los pesos, y su rango de puntaje en el marco puede ser algo distinto del de la muestra.
2) El intervalo del número estimado en el marco toma como fijo el número de epicrisis que representa el tramo. No incluye la incertidumbre sobre cuántas epicrisis del marco tienen un puntaje dentro del rango del tramo. Tampoco usa la corrección por población finita. Se calculó que, con fracciones de muestreo de 2.4 % y 0.28 %, esa corrección estrecharía el error estándar en cerca de 1.2 % en las detecciones y de 0.1 % en los controles.
3) Cada tramo tiene de 76 a 129 casos. Sus intervalos son más anchos que los de p1 y p0 (Tabla 1).
4) El puntaje no es una probabilidad calibrada (plan, sección 3.1.3). Los tramos describen el riesgo observado según el orden del puntaje, y no lo convierten en una probabilidad.
5) Los tramos no se comparan con una prueba. Por ello, la descripción no permite afirmar que el riesgo cambia con el puntaje.
6) La decisión se registra con la primera lectura terminada. El investigador conoce sus veredictos y podría recordar cuántos casos juzgó con evento, aunque no el grupo ni el puntaje de cada epicrisis. La decisión no usa ningún conteo de veredictos. Los tamaños de los tramos dependen solo del número de casos de cada estrato, y la Tabla 1 se calculó con una proporción de 0.5.
7) En cada tramo pueden quedar casos de las dos formas de lectura. La Enmienda 6 declara en la sección 2.8, numeral 1, que el corte entre los 140 primeros casos y los siguientes separa la forma de leer, la aplicación y la ampliación de la muestra, y no distingue sus efectos.
8) Las precisiones de la sección 2.3 se registran sin haber comprobado si los archivos de respuestas tienen esos casos, salvo la columna del numeral 2.
9) El intervalo de Wilson trata cada epicrisis como independiente. Si un paciente aporta más de una epicrisis a un tramo, el intervalo puede quedar algo más estrecho de lo que corresponde. La sección 4.7 del plan prevé el remuestreo por paciente en el párrafo del análisis secundario.

### 2.5 Mejoras y cambios posteriores

Desde esta enmienda, ninguna mejora reemplaza un análisis registrado. El análisis registrado se ejecuta como está. Una mejora que surja después se informa aparte, rotulada como análisis no prerregistrado, y se compara con el análisis registrado.

Lo registrado solo se cambia cuando un error, una información que falta o un hecho imprevisto en los datos impide ejecutarlo. Son casos de este tipo las paradas de la sección 2.3 de esta enmienda y la primera lectura incompleta que prevé la Enmienda 3 en la sección 2.3. Antes de abrir los resultados, el cambio exige una nueva enmienda, con su razón. Después de abrirlos, se declara como desviación en el informe final.

Esta regla es una decisión del protocolo, y su razón es la siguiente. El investigador ya conoce sus veredictos de las 615 epicrisis (sección 1, numeral 4). Un cambio elegido desde ahora podría estar influido por ellos. En cambio, una mejora que se compara con lo registrado deja ver su efecto. Lakens (2024) distingue cinco situaciones en que se presenta una desviación de un prerregistro, entre ellas los hechos imprevistos, los errores del prerregistro y la información que falta, y propone evaluar sus consecuencias.

### 2.6 Pasajes que se precisan o se sustituyen

En el plan de tesis del 17 de noviembre de 2026:

1) En la sección 3.1.3, los tramos del puntaje son los de la sección 2.2 de esta enmienda.

En la Enmienda 5:

1) En la sección 2.4, los casos sin tiempo se tratan como fija el numeral 1 de la sección 2.3 de esta enmienda.
2) En la sección 2.6, el identificador OTRO se reconoce como fija el numeral 3 de la sección 2.3 de esta enmienda.
3) En la sección 2.8, la última oración se sustituye por la regla de la sección 2.5 de esta enmienda.

En la Enmienda 6:

1) En la sección 3, el protocolo queda formado por el prerregistro, las Enmiendas 1 a 7 y el plan de tesis del 17 de noviembre de 2026. Su última oración se sustituye por la regla de la sección 2.5 de esta enmienda.

En `config/config.yaml`, la sección `analisis_enmienda_7` precisa las claves `tiempo_de_lectura.exclusion_por_tiempo` e `impacto.eventos_fuera_del_anexo` de la sección `analisis_enmienda_5`. Sustituye además la clave `otro_cambio` de la sección `analisis_enmienda_5` y, en la sección `interfaz_enmienda_6`, la clave `protocolo` de `congelamiento` y la clave `otro_cambio`. Las dos secciones se conservan sin cambios como registro.

## 3. Lo que no cambia

Fuera de lo que declara la sección 2, no cambia nada de las Enmiendas 1 a 6 ni del plan de tesis. En particular, no cambian los criterios ni los umbrales de la HE1 y de la HE2, ni las semillas de la HE1. Tampoco cambian la muestra, sus pesos, la regla del resultado no concluyente, la hipótesis general ni la decisión confirmatoria en la HE2. Se mantienen la re-revisión, los análisis de la Enmienda 5 y los conteos de la Enmienda 6.

Con esta enmienda, el protocolo de la revisión ciega y de su análisis queda formado por tres partes. Son el prerregistro del 5 de noviembre de 2026 (*commit* fda6828), las Enmiendas 1 a 7 y el plan de tesis del 17 de noviembre de 2026. Todo cambio posterior sigue la regla de la sección 2.5.

## Referencia

Lakens, D. (2024). When and how to deviate from a preregistration. *Collabra: Psychology, 10*(1), Artículo 117094. https://doi.org/10.1525/collabra.117094
