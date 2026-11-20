# Enmienda 6 del prerregistro: interfaz de la aplicación de revisión y registro del fundamento

- Fecha de registro: 20 de noviembre de 2026.
- Lo que se enmienda: de la Enmienda 4 (*commit* 557bec8), el numeral 7 de la sección 1 y las secciones 2.3, 2.4 y 2.7. También, los numerales 8 y 10 de su sección 2.6 y el numeral 3 del plan en su sección 2.9. De la Enmienda 5 (*commit* ed35d0f), las secciones 2.1, 2.3, 2.4, 2.5 y 2.8. También se precisan los pasajes del plan de tesis que enumera la sección 2.9 de esta enmienda.
- Resumen estructurado: sección `interfaz_enmienda_6` de `config/config.yaml`, en este mismo *commit*.
- Origen: recomendación de la asesora de la tesis sobre la aplicación de revisión, recibida el 20 de noviembre de 2026 (comunicación personal).
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda

1) Se han leído 140 de las 615 epicrisis, todas antes de la Enmienda 4. El caso siguiente es el HE4-141. No se ha leído ningún caso con el procedimiento de la Enmienda 4.
2) Los resultados de la revisión ciega no se han abierto. Nadie ha visto los veredictos separados por detecciones y controles.
3) El investigador conoce sus veredictos de las 140 epicrisis leídas, pero no sabe a qué grupo pertenece cada una.
4) La re-revisión no ha empezado, y su archivo de respuestas no existe. Las 30 epicrisis que añadió la Enmienda 5 se sortearon y se cargaron el 19 de noviembre de 2026, después de su registro, con el *script* que esa enmienda identifica por su huella.
5) No se ha calculado ninguno de los análisis de la Enmienda 5.
6) El *script* de la HE1 no se ha ejecutado.
7) La aplicación instalada es la del 27 de octubre de 2026, que la Enmienda 5 identifica por su huella. La versión que registra esta enmienda no se ha ejecutado con las epicrisis de la muestra. Se probó solo con textos inventados.
8) Las 615 epicrisis de la muestra y las 60 de la re-revisión tienen traducción. Se comprobó con un programa el 20 de noviembre de 2026, sin leer las respuestas.
9) Ninguna persona distinta del investigador ha leído las epicrisis de la muestra.

## 2. Cambios

### 2.1 Motivo

La Enmienda 4 dejó la aplicación sin cambios y mandó escribir el fundamento en el campo de notas. La Enmienda 5 fijó el protocolo y no previó otro cambio. Después de ese registro, la asesora de la tesis recomendó modificar la interfaz de la aplicación, con estos elementos:

1) Mostrar la nota completa por defecto, porque el procedimiento exige leer cinco secciones completas.
2) Desactivar el resaltado de términos, que puede introducir un sesgo al identificar candidatos.
3) Mostrar el inglés y el castellano a la vez, para verificar las negaciones, la causa y el tiempo.
4) Un control de lectura, con una lista de las cinco secciones. Si una sección no existe, se registra su ausencia.
5) Un formulario para el evento, con las secciones revisadas, el veredicto, el tipo de nexo, la gravedad del daño, la pertenencia del daño al episodio y la evidencia del daño y del nexo.
6) Señaló como inconsistencias el encabezado, que nombraba la HE3, y el identificador con el prefijo HE4, porque el plan vigente solo tiene la HE1 y la HE2. Propuso además cambiar el texto de la pregunta.
7) Recalcular el tiempo de revisión, porque el procedimiento puede exigir más tiempo.

Esta enmienda se aparta de la recomendación en dos puntos. Los identificadores no cambian, porque están registrados en la Enmienda 3 y en las respuestas ya guardadas. Los rótulos de la gravedad son los del manual de anotación, que detallan el nivel F. Además, los numerales 8 a 12 y 14 a 16 de la sección 2.2 no proceden de la recomendación. El 14 ajusta la ayuda al procedimiento de la Enmienda 4. Los demás se añaden para asegurar el registro.

Esta enmienda registra esos cambios antes de leer el caso HE4-141. No se ha leído ningún caso con el procedimiento de la Enmienda 4 y la aplicación anterior. Por ello, la muestra tendrá dos formas de lectura y no tres. Las dos formas difieren ahora también en la aplicación.

### 2.2 Versión 2 de la aplicación

La aplicación de revisión pasa a su versión 2.0, del 20 de noviembre de 2026. La huella SHA-256 de su archivo es e9ec06159cd9fbf9c285096c2c279a2123fbf9fb240f904d3fbfd1c37f66dabb. La versión anterior es la que describe la Enmienda 5 (sección 2.5). Con ella se leyeron las 140 primeras epicrisis. Las 16 primeras se leyeron antes de la corrección de su ayuda del 27 de octubre de 2026.

La versión 2 cambia lo siguiente:

1) Muestra siempre la nota completa. Ya no tiene el extracto por secciones.
2) Ya no tiene el resaltado de términos.
3) Muestra por defecto el original en inglés y la traducción automática, lado a lado. Los dos cuadros se desplazan a la vez, en proporción a su largo, y una casilla permite desplazarlos por separado. Ese desplazamiento no alinea las oraciones. Se puede elegir ver solo uno de los dos. La aplicación guarda la vista que estaba elegida al guardar el caso, y no registra las que se usaron antes. Si un caso no tiene traducción, muestra solo el original.
4) Tiene una lista de comprobación con las cinco secciones del procedimiento. Cada una se marca como leída completa o como inexistente en la nota. Tiene además una marca para las otras partes de la nota, con dos valores, que son leídas o no hizo falta.
5) Pregunta “¿La epicrisis documenta al menos un evento adverso que cumple los criterios operacionales del estudio?”. Antes preguntaba “¿Hay un evento adverso real?”. Los tres veredictos no cambian.
6) Su encabezado dice “Revisión ciega de la HE2”. Los identificadores de los casos no cambian. Bajo el identificador, la aplicación indica que el prefijo HE4 no nombra una hipótesis del plan vigente y que la hipótesis que se contrasta es la HE2.
7) Registra el fundamento del veredicto en campos propios, como fija la sección 2.4 de esta enmienda.
8) No guarda un caso mientras falte un dato obligatorio o una evidencia no cumpla su regla. Lista lo que falta. Antes de guardar pide confirmar que el registro se revisó.
9) No permite modificar un caso guardado. Lo muestra tal como quedó. Ya no tiene el modo para verificar de nuevo los casos marcados.
10) Escribe cada archivo completo o no lo escribe. Si una escritura falla, repone los archivos como estaban y el caso sigue pendiente. Si no logra reponer un archivo auxiliar, lo avisa. Las filas de ese caso se reemplazan al guardarlo de nuevo. Antes de cada guardado copia los archivos de respuestas a una carpeta de respaldo, en el mismo equipo.
11) En cada ejecución comprueba el lote y los archivos de respuestas. Se detiene sin escribir nada si uno no se puede leer. Se detiene también si el archivo principal tiene casos repetidos o le faltan casos de la muestra. Si ese archivo falta y existen copias de respaldo suyas, se detiene. Si falta y no hay copias, lo crea vacío, como al abrir por primera vez la re-revisión.
12) Los botones para pasar a otro caso quedan bloqueados cuando hay datos sin guardar, salvo que el evaluador marque que quiere salir. El tiempo de un caso suma todas sus visitas dentro de la misma sesión.
13) Estima el tiempo que falta solo con los casos leídos con esta versión, cuando hay cinco o más. Muestra la mediana del tiempo por caso de esos casos, sin separar por veredicto.
14) Su ayuda en pantalla resume el procedimiento de la Enmienda 4, con las cuatro condiciones del “Sí”.
15) Vuelve a leer el lote si su archivo cambia. La versión anterior lo conservaba en memoria hasta cerrar la aplicación.
16) Pone una marca en cada caso que guarda. Con esa marca se separan las dos formas de lectura.

La versión 2 usa las mismas dos tablas del Anexo N.° 02 que la versión anterior, que son el catálogo de eventos y la tabla de impactos. Sus huellas SHA-256 son 219fc20b1cb70b3db9a95e0da07549463f323f29d28c0a10a6c4ed413835dfe4 y ba36303ac72a0d850599a4ff4bfbcfa81bf4dc3ff4741a0ba316fe859f1de19b. Se ejecuta con Python 3.13.9, pandas 2.3.3 y Streamlit 1.51.0.

No cambian la muestra, el orden de presentación ni el cegamiento. La aplicación sigue sin mostrar el veredicto del modelo, el código de diagnóstico, el grupo del caso ni el conteo de veredictos. No cambian el catálogo de eventos del Anexo N.° 02 ni la forma de tomar el impacto. El apartado de eventos adicionales conserva su tope de cinco y su archivo, con las columnas del fundamento. Tampoco cambia el umbral del aviso, que aparece cuando un caso lleva menos de 2 minutos. Su texto ya no estima minutos por caso. La medida del tiempo cambia en un punto, que fija la sección 2.6. La aplicación sigue ejecutándose solo en el equipo del investigador.

Las respuestas de las 140 epicrisis ya leídas no se modifican. La versión 2 añade columnas al archivo de respuestas, y esos casos las conservan vacías.

### 2.3 Procedimiento de lectura

El procedimiento de lectura de la Enmienda 4 (sección 2.3) se mantiene, con cuatro precisiones que siguen de la versión 2:

1) Se lee con el original y la traducción a la vista. Ya no hay que abrir la nota completa ni apagar el resaltado, porque la aplicación no tiene extracto ni resaltado.
2) En la comprobación del texto original, la oración se ubica en el cuadro del original, que está junto a la traducción. Ya no hay que apagar la traducción.
3) Al terminar la lectura se marca cada una de las cinco secciones como leída completa o como inexistente.
4) Todos los casos tienen traducción, como consta en la sección 1. Si alguna faltara, el caso no se lee hasta tenerla. Se informa el número de casos guardados con cada vista.

Los pasos, las cuatro preguntas, la regla del veredicto del caso y la comprobación del hecho, el tiempo, la causa y la duda no cambian. Tampoco cambian las reglas sobre dónde empieza y dónde termina cada sección.

### 2.4 Registro del fundamento

El fundamento deja de escribirse como líneas con códigos en el campo de notas. Se registra en campos de la aplicación. La información es la que fijó la Enmienda 4 (sección 2.4), con dos datos más, que son la gravedad y la evidencia del nexo.

Para el caso se registran:

1) El estado de cada una de las cinco secciones. Los códigos MOT, PROC, EVOL, DX, COND y NOTA de la Enmienda 4 se mantienen para indicar la sección de cada evidencia.
2) Si se leyeron otras partes de la nota. Equivale a la marca NC de la Enmienda 4, que ahora se registra por caso.
3) Si la traducción y el original no decían lo mismo en algún pasaje. Equivale a la marca TRAD, que ahora se registra por caso y no por candidato.

Para cada evento con “Sí”, el principal y los adicionales, se registran:

1) El evento del Anexo N.° 02 y su naturaleza, como antes.
2) El tipo de nexo, que es C3 o C4.
3) La gravedad del daño, de la E a la I, con las definiciones del manual de anotación.
4) La evidencia del daño y la evidencia del nexo, cada una con su sección.
5) El área y el día del evento, con los códigos de la Enmienda 4. El día se anota con DH o DPO y su número, o con NODICE. El número es obligatorio con DH o DPO y no tiene un valor por omisión.

Para el evento principal se registra además que el daño corresponde a la atención del episodio actual. Sin esa marca la aplicación no guarda un “Sí”. En los eventos adicionales, la condición del episodio no queda registrada. Los eventos adicionales siguen siendo hasta cinco por epicrisis, como en la versión anterior. Un evento que no está en el Anexo N.° 02 lleva su descripción, su naturaleza y sus cuatro indicadores, también cuando es adicional.

Los candidatos que no llegan a “Sí” se registran aparte, uno por candidato y hasta diez por epicrisis, en un archivo propio. Cada uno lleva su sección, su código y su evidencia. Los códigos son C2, C1, SIN-DANO, PREVISTO, NEGADO y PREVIO, con el significado de la Enmienda 4. Un candidato que no se puede decidir lleva su motivo, de ND-1 a ND-5 o ND-IDIOMA. Un candidato con el código ND-1 puede ir sin evidencia. La aplicación no guarda un “No” que tenga un candidato que no se puede decidir, porque en ese caso el veredicto es “No se puede determinar”. Si una epicrisis tiene más de diez candidatos, se registran los diez primeros en el orden de la nota y el total se anota en las notas.

El caso con veredicto “No” y sin candidatos se confirma con una marca propia. Equivale al código SC de la Enmienda 4.

Para el veredicto “No se puede determinar” se registra el motivo, que es uno de los cinco del manual o ND-IDIOMA. Se registra también el fragmento que no permite decidir, con su sección. El fragmento es obligatorio, salvo con el motivo ND-1, que es la sección ausente o truncada. Si hay más de un candidato que no se puede decidir, el que decide el veredicto va en el motivo del caso y los demás van con los otros candidatos. El conteo por motivo de la Enmienda 4 (sección 2.5) se toma del motivo del caso. En un caso con este veredicto, la aplicación no registra si hubo candidato.

Cada evidencia tiene de tres a ocho palabras seguidas, copiadas del original en inglés, como en la Enmienda 4. La aplicación comprueba que tenga ese número de palabras y que esté, con palabras enteras, en el original del mismo caso. La comparación no distingue mayúsculas ni saltos de renglón. La regla vale para las evidencias de los eventos, para las de los otros candidatos y para el fragmento del veredicto “No se puede determinar”. La aplicación no acepta una evidencia cuya sección conste como inexistente.

La aplicación exige además lo que el procedimiento ya mandaba. Si una de las cinco secciones no existe, las otras partes de la nota deben constar como leídas. Si hay al menos un candidato, también. Cuenta como candidato el que lleva a un veredicto “No se puede determinar” con los motivos ND-2 a ND-5.

Frente a la Enmienda 4, el registro pide más en tres puntos. Cada evento con “Sí” lleva dos evidencias y no una. La gravedad se registra. La pertenencia al episodio se marca de forma expresa. Además, el número de candidatos por epicrisis tiene un tope de diez.

El campo de notas queda para comentarios y para describir un evento que no está en el Anexo N.° 02. Las evidencias contienen texto de las epicrisis. Por ello quedan solo en el equipo del investigador, como fijó la Enmienda 4.

Después de abrir los resultados se informan los conteos que fijó la Enmienda 4, también por grupo. Se añaden los conteos por tipo de nexo y por gravedad de los eventos con “Sí”, y el número de epicrisis con cada sección inexistente. Estos datos son descriptivos y no entran en la HE1 ni en la HE2.

### 2.5 Control del registro

El control del registro lo hace la aplicación al guardar. No guarda un caso con un dato obligatorio vacío, con un código que no corresponda al veredicto o con una evidencia que no esté en el original.

El programa de comprobación de la Enmienda 4, cuya huella registró la Enmienda 5, verifica líneas del campo de notas. No se aplica a los casos leídos con la versión 2. Los tres conteos de control de la Enmienda 5 (sección 2.8) se mantienen para las 140 primeras epicrisis. Los informa el programa de esta sección. La aplicación no permite guardar un caso más de una vez.

Antes de abrir los resultados, un programa verifica de nuevo las reglas de la sección 2.4 sobre los archivos de respuestas, sin mostrar texto clínico. Ese programa todavía no está escrito. Solo informa cuántos casos cumplen y los identificadores de los que no. No informa conteos por código ni por veredicto. Si halla un caso que no cumple, el caso no se vuelve a abrir. Su veredicto cuenta y no entra en los conteos descriptivos, como fijó la Enmienda 4. La huella de ese programa se registra junto con las huellas que fija la Enmienda 5 (sección 2.8). Entre esas huellas se incluye la del archivo de los otros candidatos, que es un archivo de respuestas nuevo.

Los archivos de respuestas no se editan fuera de la aplicación. Si hay que reponer uno desde una copia de respaldo, se registran la fecha, la razón y las huellas de antes y de después. Las copias de respaldo contienen texto de las epicrisis y quedan solo en el equipo del investigador. Antes de leer el caso HE4-141 se comprueba que la huella del archivo instalado es la que registra esta enmienda.

### 2.6 Tiempo de lectura

El análisis del tiempo de lectura de la Enmienda 5 (sección 2.4) no cambia y se calcula al terminar las dos lecturas.

La aplicación muestra al evaluador la mediana del tiempo por caso de los casos leídos con la versión 2 y, con ella, el tiempo que falta. No separa por veredicto ni por grupo. Sirve para planificar la lectura y actualizar el cronograma. Es la misma medida que la Enmienda 5 describirá al final para esta forma de lectura. Se muestra durante la lectura porque no depende del veredicto ni del grupo.

La medida del tiempo cambia en un punto. En la versión anterior, el tiempo de un caso era el de la visita en que se guardó. En la versión 2 es la suma de sus visitas dentro de la misma sesión, e incluye el tiempo de llenar el formulario. La comparación del tiempo entre las dos formas de lectura de la Enmienda 5 (sección 2.4) se lee con esa diferencia.

La estimación de unas 24 horas de lectura de la Enmienda 3 (sección 2.1) vale para la forma anterior de leer. El tiempo con el procedimiento vigente se conocerá con los primeros casos.

### 2.7 Re-revisión

Las 60 epicrisis de la re-revisión se leen con la versión 2, que usa otro lote y otros tres archivos de respuestas. Los dos estratos de la Enmienda 5 (sección 2.1) no cambian. En el primero, la primera lectura se hizo con la versión anterior y la forma anterior de leer.

### 2.8 Limitaciones que se declaran

1) La aplicación cambió con 140 epicrisis leídas. Esas epicrisis se leyeron con el extracto, con el resaltado disponible y sin los campos del fundamento. El corte entre los 140 primeros casos y los siguientes separa ahora la forma de leer, la aplicación y la ampliación de la muestra, y no distingue sus efectos.
2) El cambio se decidió con 140 casos leídos, cuando el investigador ya conocía sus veredictos de esos casos, aunque no el grupo de cada uno.
3) La lista de las cinco secciones registra lo que el evaluador declara. No prueba que la sección se leyó ni que se entendió.
4) La comprobación de la evidencia es literal. Confirma que las palabras están en el original. No confirma que sostengan el daño o el nexo, ni que estén en la sección que se declara. Tampoco impide que las dos evidencias de un evento sean el mismo fragmento.
5) El formulario pide más datos para un “Sí” que para un “No”. Ese esfuerzo mayor puede inclinar un caso dudoso hacia el “No”. El registro con menos datos es el “No” sin candidatos. Ello puede llevar a registrar menos candidatos de los que se revisaron. El veredicto “No se puede determinar” también pide más datos que el “No”.
6) El formulario es el mismo para todos los casos, y el evaluador no conoce el grupo. Aun así, su efecto puede no ser igual en los dos grupos si los casos dudosos no se reparten igual entre ellos. Puede además bajar el VPP de los casos leídos con la versión 2 frente al de los 140 primeros.
7) El orden de presentación es aleatorio y la proporción de detecciones es casi la misma en los dos sorteos (151 de 241 y 234 de 374). Por ello no se espera que el cambio de la aplicación se concentre en un grupo. Sí puede cambiar la magnitud de la diferencia de proporciones de la HE2, porque un registro más exigente puede bajar la proporción de casos con evento en los dos grupos.
8) En las 14 epicrisis del primer estrato de la re-revisión, los cambios de veredicto mezclan la forma de leer con la aplicación y su formulario.
9) La versión 2 se probó con textos inventados y no con epicrisis de la muestra. Si falla durante la lectura, la lectura se detiene y el cambio se registra en una nueva enmienda.
10) El investigador conoce la recomendación que motivó este cambio y lee los casos restantes sabiéndolo.
11) El tipo de nexo, la gravedad, el área y el día dependen de un solo evaluador. La confiabilidad se mide sobre el veredicto y no sobre esos datos. La gravedad y el tipo de nexo se registran solo desde el caso HE4-141.
12) En los eventos adicionales, la condición del episodio no queda registrada. Esos eventos entran en la Matriz, como fijó la Enmienda 5 (sección 2.6).
13) Un error de digitación en un caso guardado ya no se puede corregir. El caso queda como se guardó.
14) Si la página se cierra o se recarga con un caso a medio llenar, se pierden lo llenado y el tiempo ya contado. El tiempo guardado sería entonces menor que el real. Si se cambia la lista de casos que se muestra, se pierde lo llenado, pero el tiempo se conserva.
15) La confianza del juicio tiene un valor por omisión, que es el intermedio. Un caso puede guardarse sin haberla tocado.

### 2.9 Pasajes que quedan sin efecto o que se precisan

En la Enmienda 4:

1) El numeral 7 de la sección 1 describe el estado a su registro. Desde esta enmienda la aplicación sí se modifica.
2) En la sección 2.3 quedan sin efecto el encendido de la traducción, el apagado del resaltado y la apertura de la nota completa, que la versión 2 hace innecesarios. Queda sin efecto también el apagado de la traducción en el paso 3. Del párrafo sobre el extracto valen solo como descripción de la versión anterior sus cinco primeras oraciones. Se mantienen las reglas sobre dónde empieza y dónde termina cada sección. Se mantiene también que un título que no se reconoce se ubica en el original. En la última oración de la sección, la marca TRAD se registra ahora por caso.
3) En la sección 2.4 queda sin efecto la escritura del fundamento en el campo de notas, con su formato de línea. Se sustituye por la sección 2.4 de esta enmienda. Para los casos leídos con la versión 2, los dos párrafos sobre el programa de comprobación y sobre el caso con error se sustituyen por la sección 2.5 de esta enmienda.
4) En la sección 2.6, el numeral 8 se sustituye, para los casos leídos con la versión 2, por la sección 2.5 de esta enmienda. El numeral 10 se precisa, porque la versión 2 muestra siempre la nota completa.
5) En la sección 2.7, la oración según la cual se lee y se registra como establece esa enmienda se precisa con las secciones 2.3 y 2.4 de esta.
6) En la sección 2.9, del numeral 3 del plan de tesis queda sin efecto, desde el caso HE4-141, la oración según la cual el motivo se escribe en las notas. La versión 2 registra el motivo en un campo propio.

En la Enmienda 5:

1) En la sección 2.1, la re-revisión usa otro lote y otros tres archivos de respuestas, y no uno.
2) En la sección 2.3, el corte entre los 140 primeros casos y los siguientes separa ahora también la aplicación.
3) En la sección 2.4, el tiempo por caso se precisa con la sección 2.6 de esta enmienda.
4) La sección 2.5 vale como descripción de la versión con la que se leyeron las 140 primeras epicrisis. La versión 2 es la de la sección 2.2 de esta enmienda.
5) En la sección 2.8, la versión y la huella de la aplicación son las de la sección 2.2 de esta enmienda desde el caso HE4-141. El programa de comprobación y sus conteos de control se precisan con la sección 2.5. Ese programa ya no se aplica a la re-revisión. Las huellas que se registran antes de abrir los resultados incluyen la del archivo de los otros candidatos y la del programa de la sección 2.5. La oración según la cual no se prevé otro cambio se cumple con esta enmienda, que da su razón y se registra antes de abrir los resultados.

En `config/config.yaml`, la sección `interfaz_enmienda_6` sustituye cuatro claves de `congelamiento`, en la sección `analisis_enmienda_5`. Son `aplicacion_de_revision`, `programa_de_comprobacion`, `protocolo` y `antes_de_abrir`. Precisa además la clave `tiempo_de_lectura` de esa sección. Sustituye también, en la sección `lectura_enmienda_4`, las claves `aplicacion_de_revision`, `registro_en_las_notas` y `programa_de_comprobacion`, cuatro claves de `lectura` y la clave `sin_candidato`. Las dos secciones se conservan sin cambios como registro.

En el plan de tesis del 17 de noviembre de 2026:

1) La descripción del sistema Revisión ciega de la sección 4.6 y del Anexo 8 vale para las 140 primeras epicrisis. Desde el caso HE4-141 rige la sección 2.2 de esta enmienda.
2) En la limitación 13 se precisa, desde el caso HE4-141, que la versión 2 registra las secciones que el evaluador declara y la vista elegida al guardar. No registra las vistas usadas antes. El extracto y el aviso de 5 a 8 minutos valen solo para las 140 primeras epicrisis.
3) En la sección 4.7, el modo para verificar de nuevo los casos es de la versión anterior. Se añade además esta enmienda a las registradas.

## 3. Lo que no cambia

Fuera de lo que declara la sección 2, no cambia nada de las Enmiendas 1 a 5 ni del plan de tesis. En particular, no cambian los criterios ni los umbrales de la HE1 y de la HE2, ni las semillas de la HE1. Tampoco cambian la muestra, sus pesos, la regla del resultado no concluyente, la hipótesis general ni la decisión confirmatoria en la HE2.

Se mantienen las definiciones de la Enmienda 4, incluida la condición del episodio y el tratamiento del veredicto “No se puede determinar”. Se mantienen la re-revisión y los análisis de la Enmienda 5, el impacto tomado del Anexo N.° 02 y la apertura de los resultados después de terminar las dos lecturas.

Con esta enmienda, el protocolo de la revisión ciega y de su análisis queda formado por tres partes. Son el prerregistro del 5 de noviembre de 2026 (*commit* fda6828), las Enmiendas 1 a 6 y el plan de tesis del 17 de noviembre de 2026. Cualquier otro cambio exigiría una nueva enmienda, con su razón, registrada antes de abrir los resultados.
