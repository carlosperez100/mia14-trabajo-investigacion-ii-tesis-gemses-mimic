# Enmienda 5 del prerregistro: confiabilidad intraevaluador, análisis de sensibilidad, sistema de revisión e impacto de los eventos

- Fecha de registro: 19 de noviembre de 2026.
- Lo que se enmienda: de la Enmienda 1 (*commit* ad116a5, con redacción vigente del *commit* 977a1ce), la sección 2.7. De la Enmienda 2 (*commit* b40b7ac), las secciones 2.2 y 2.3. De la Enmienda 3 (*commit* fb899b8), las secciones 2.1 y 2.4, el numeral 3 de la sección 2.5 y la sección 2.6. De la Enmienda 4 (*commit* 557bec8), las secciones 2.1, 2.4 y 2.8. También se precisan los pasajes del plan de tesis que enumera la sección 2.9 de esta enmienda.
- Resumen estructurado: sección `analisis_enmienda_5` de `config/config.yaml`, en este mismo *commit*.
- Origen: recomendaciones de la asesora de la tesis sobre el plan del 17 de noviembre de 2026, recibidas el 18 de noviembre de 2026 (comunicación personal). También, los asuntos que las Enmiendas 3 y 4 dejaron para un registro posterior.
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda

1) Se han leído 140 de las 615 epicrisis, todas antes de la Enmienda 4. El caso siguiente es el HE4-141.
2) Los resultados de la revisión ciega no se han abierto. Nadie ha visto los veredictos separados por detecciones y controles.
3) El investigador conoce sus veredictos de las epicrisis leídas, pero no sabe a qué grupo pertenece cada una.
4) La re-revisión no ha empezado, y su archivo de respuestas no existe. Las 30 epicrisis que añade esta enmienda no se han cargado. Su sorteo solo se ha ensayado, sin escribir ningún archivo ni mostrar ningún caso.
5) No se ha calculado ninguno de los análisis de esta enmienda.
6) El *script* de la HE1 no se ha ejecutado.
7) La Enmienda 4 se registró el 19 de noviembre de 2026. Desde ese registro no se ha leído ningún caso.
8) Ninguna persona distinta del investigador ha leído las epicrisis de la muestra.

## 2. Cambios

### 2.1 Re-revisión y confiabilidad intraevaluador

La asesora recomendó ampliar la segunda lectura a unas 60 epicrisis, si es viable, sorteadas al azar y procurando incluir los dos bloques de revisión. La re-revisión pasa de 30 a 60 epicrisis.

Las 60 epicrisis se forman así:

1) Se conservan las 30 epicrisis ya sorteadas. El *script* del sorteo las tomó de las 241 primeras el 26 de octubre de 2026, junto con el lote y antes de cualquier lectura. Es el *script* que la Enmienda 3 identifica por su huella. Usó la función `DataFrame.sample`, sin reemplazo y con la semilla 2027. Tienen los identificadores HE4R-01 a HE4R-30. El sorteo tomó las 30 del lote ya mezclado, sin distinguir el grupo. Así se corrige el numeral 1 de la sección 2.2 de la Enmienda 2, que describía otro sorteo.
2) Se añaden 30 epicrisis de los 374 casos añadidos por la Enmienda 3. Se sortean con la regla que registró la Enmienda 4 (sección 2.8). Se ordenan de menor a mayor los identificadores HE4-242 a HE4-615 y se sortean con la función `DataFrame.sample`, sin reemplazo y con la semilla 2029. Se toman las 30 primeras del sorteo. El sorteo no usa la clave de la muestra ni los veredictos. Se ejecuta después del registro de esta enmienda. Reciben los identificadores HE4R-31 a HE4R-60. El *script* que las sortea y las carga se identifica por su huella SHA-256, que es 07e7ea4751a3f3c12ae9b4ade937dc8c20a77388630bb0cae5602e2437695d8a. Su ensayo, que no escribe ningún archivo, se hizo el 19 de noviembre de 2026 y todas sus comprobaciones pasaron.

El investigador no ve la lista de las epicrisis sorteadas. La aplicación las presenta sin su identificador de la primera lectura. La correspondencia entre los dos identificadores está en la carpeta de la clave.

Cada epicrisis se lee de nuevo al menos dos semanas después de terminar la primera lectura de su bloque. Para las 30 primeras, el plazo corre desde el final de la primera lectura de las 241. Para las 30 añadidas, corre desde el final de la primera lectura de las 615. Se informan la mediana y el rango de los días entre las dos lecturas, en cada estrato.

La segunda lectura se hace a ciegas y sin ver el veredicto ni las notas de la primera. Para ello la aplicación usa otro lote y otro archivo de respuestas. Presenta primero las 30 ya sorteadas y después las 30 añadidas, cada grupo en el orden de su sorteo. Se usa el procedimiento de lectura de la Enmienda 4, también en las epicrisis cuya primera lectura fue anterior.

Se informan tres medidas, como recomendó la asesora. Son el kappa de Cohen con su IC 95 %, el porcentaje de acuerdo y la matriz entre la primera y la segunda lectura. La matriz usa las tres categorías del veredicto. El kappa se calcula con dos categorías, con evento y sin evento, como en el análisis principal de la HE2. El veredicto “No se puede determinar” cuenta como sin evento. Se informa también el kappa de las tres categorías. El IC se calcula con el error estándar de Cohen (1960), como en el plan de tesis (sección 4.4).

Las tres medidas se informan en dos estratos y en conjunto. El primer estrato reúne las epicrisis cuya primera lectura fue anterior a la Enmienda 4. El segundo reúne las demás. En el primero, la segunda lectura cambia también la forma de leer. Por ello, la confiabilidad intraevaluador es la del segundo estrato. El primero describe la concordancia entre las dos formas de leer, y el conjunto mezcla las dos cosas.

Con el kappa supuesto de 0.70 y la prevalencia de 0.28 a 0.29 del plan (sección 4.4), la semiamplitud del IC 95 % del kappa sería de 0.20 con 60 epicrisis. Con 30 era de 0.29. De las 30 primeras, 14 tuvieron su primera lectura antes de la Enmienda 4. Las otras 16 están entre los casos 141 a 241. Con 14 epicrisis en el primer estrato, la semiamplitud sería de 0.42. Con 46 en el segundo, sería de 0.23.

Se declaran tres límites. Los dos bloques aportan 30 epicrisis cada uno, aunque el primero tiene 241 casos y el segundo 374, y las medidas no se ponderan. El tiempo entre las dos lecturas es mayor en el primer estrato. El primer estrato es pequeño, y su kappa solo describe.

La referencia de la HE2, del VPP y de los eventos confirmados sigue siendo la primera lectura. La segunda se usa para la confiabilidad y para el análisis descriptivo de la sección 2.3. Los resultados de la revisión ciega se abren después de terminar la primera lectura de las 615 epicrisis y la re-revisión de las 60.

### 2.2 Interpretación de la HE2 y precisión

No existe un patrón de referencia independiente. Por ello, los resultados de la HE2 se interpretan como eventos identificados mediante la evaluación del investigador. No se interpretan como eventos verificados mediante consenso clínico independiente.

El margen de error de 0.05 de la Enmienda 3 (sección 2.1) vale para el VPP de las 385 detecciones, bajo un muestreo aleatorio simple de ese estrato. No es la precisión de la diferencia de proporciones de la HE2.

Con 385 detecciones y 230 controles, se calculó la semiamplitud esperada del IC 95 % de Newcombe de esa diferencia. Es de 0.066 si el VPP es de 0.35 y la proporción en los controles es de 0.15. Es de 0.070 si esa proporción es de 0.20. En el caso más desfavorable, con las dos proporciones en 0.50, es de 0.081.

### 2.3 Análisis de sensibilidad del sesgo por señales

Las 140 primeras epicrisis se leyeron buscando las señales de un evento. Para describir el efecto de esa forma de leer se prevén dos análisis:

1) La HE2 y el VPP se calculan por separado en los 140 casos leídos antes de la Enmienda 4 y en los leídos después. Es el segundo corte de la sección 2.3 de la Enmienda 3. La parte leída después muestra si el resultado se sostiene sin los casos leídos por señales.
2) En la re-revisión, y solo en el estrato cuya primera lectura fue por señales, se cuentan los casos que pasan de “No” a “Sí” y de “Sí” a “No”. Después de abrir los resultados, ese conteo se informa por grupo.

Los dos análisis son descriptivos y no deciden la HE2. El primero no separa la forma de leer de la ampliación de la muestra, como declara la Enmienda 4 (sección 2.6). El segundo cuenta con 14 epicrisis, y con ese número solo puede describir.

### 2.4 Tiempo de lectura

El tiempo de lectura se describe con la mediana y los cuartiles del tiempo por caso. Se informan por forma de lectura y por veredicto. Después de abrir los resultados, se informan también por grupo. Las diferencias se evalúan con la diferencia de medianas y su IC 95 % de percentiles, con 2 000 remuestreos de casos dentro de cada grupo comparado. Se comparan las dos formas de lectura, y los casos con evento con los casos sin evento. Después de abrir los resultados, se comparan también los dos grupos. No se hace una prueba de hipótesis sobre el tiempo.

Además, la HE2 y el VPP se describen por terciles del tiempo de lectura. Los terciles se calculan dentro de cada forma de lectura. Ningún caso se excluye por su tiempo.

Se declaran dos límites. La aplicación mide el tiempo que el caso permanece abierto, sin descontar la inactividad. Además, el procedimiento de la Enmienda 4 añade pasos cuando hay un candidato. Por ello, un tiempo mayor en los casos con evento es lo esperado y no mide la calidad de la lectura.

### 2.5 Descripción de la aplicación de revisión

La aplicación de revisión es el sistema que el plan de tesis denomina Revisión ciega. Se ejecuta solo en el equipo del investigador, sin acceso desde la red. Su ayuda en pantalla se corrigió el 27 de octubre de 2026, con 16 epicrisis leídas, y se añadió un modo para verificar de nuevo esos casos (plan, sección 4.7). Desde esa fecha su archivo no se ha modificado. Hace lo siguiente:

1) Presenta una epicrisis por vez, en el orden que fija la Enmienda 3. No muestra el veredicto del modelo, el código de diagnóstico, el grupo del caso ni el conteo de veredictos. La clave está en una carpeta distinta de la que lee la aplicación.
2) Ofrece tres veredictos, que son “Sí, es evento adverso”, “No es evento adverso” y “No se puede determinar”.
3) Muestra por defecto un extracto con cinco secciones. Son el motivo de consulta, el procedimiento mayor, la evolución, los diagnósticos al alta y la condición al alta. La regla es fija e igual para todos los casos. La nota completa se abre con un interruptor, y se muestra entera si no se reconocen esas secciones. El extracto termina cada sección en la primera línea con forma de título. Por ello puede mostrar cortada una sección con subtítulos, o no mostrarla. La Enmienda 4 fija que las cinco secciones se leen en la nota completa.
4) Tiene dos ayudas de lectura, apagadas por defecto. Una resalta una lista fija de términos. La otra muestra una traducción automática al castellano, hecha en el propio equipo y sin leer la clave. No registra si se usaron las ayudas ni si se abrió la nota completa.
5) Registra el tiempo que cada caso permanece abierto y muestra el avance y los tiempos de lectura. Avisa cuando un caso lleva menos de 2 minutos, pero no impide guardarlo.
6) Guarda, por cada caso, el veredicto, el evento del Anexo N.° 02, su naturaleza, cuatro indicadores de impacto, la confianza, las notas, la fecha y el tiempo. No tiene campos propios para el nexo, la sección ni la cita, que se escriben en las notas como fija la Enmienda 4.
7) Tiene un apartado para los eventos adicionales de una misma epicrisis. Admite hasta cinco, cada uno con su evento del Anexo N.° 02 y su nota, y los guarda en un archivo aparte.
8) Para cada evento con “Sí”, toma los cuatro indicadores del impacto de la tabla del Anexo N.° 02. Solo se valoran a mano cuando el evento no está en ese anexo. La aplicación no calcula la prioridad.
9) Permite volver a un caso guardado y corregirlo, y guarda la fecha del cambio. La Enmienda 4 fija que el veredicto no se cambia.
10) Presenta las epicrisis de la re-revisión sin la marca del modelo, sin el primer veredicto y sin sus notas, porque lee otro lote y otro archivo de respuestas.

La huella SHA-256 del archivo de la aplicación es 556768326de6813ac2ea80557c7dfbe5bf94000cfe50781d412ba399739b90c9.

### 2.6 Impacto de cada evento

La Directiva valora cada indicador del impacto como nulo (0), bajo (1), medio (3), alto (6) o muy alto (9), y lo pondera (num. 19.11). No fija cuántos días de estancia corresponden a cada valor. Su Anexo N.° 02 relaciona cada identificador de evento con una tabla de impactos (num. 15.1).

El impacto de cada evento confirmado se toma del Anexo N.° 02. Se usan los cuatro indicadores que el anexo da para ese evento. Son la prolongación de la estancia, las complicaciones, los sobrecostos y la insatisfacción. Se ponderan con los pesos del num. 19.11, que son 0.40, 0.20, 0.15 y 0.25. La columna del impacto impreso en el anexo se usa solo como sensibilidad, como ya fija el plan de tesis. Entran en la Matriz todos los eventos con “Sí” que tienen identificador, el principal y los adicionales.

Es lo que registra la aplicación de revisión. Al elegir el evento, la aplicación toma sus cuatro indicadores de una tabla del anexo, y el evaluador no los valora.

Con ello no hacen falta cortes del impacto. La diferencia de estancia y el traslado a la UCI ya no se convierten a la escala del num. 19.11, y no se informan en este estudio.

Los eventos que no están en el Anexo N.° 02 se informan aparte y no entran en la Matriz. La aplicación los registra como OTRO, con una descripción, y pide valorar su impacto a mano. La Matriz usa el identificador de evento del anexo (num. 19.11), y esos eventos no lo tienen. Se informan su número y su naturaleza. La valoración a mano no se usa en la Matriz. El índice de frecuencia conserva el divisor de la Enmienda 1 (sección 2.6), que son todos los eventos confirmados en el proceso, incluidos los que quedan fuera de la Matriz. Ese divisor es común a todos los eventos, de modo que no cambia el índice de gestión ni el orden de los eventos. Se informa además qué parte de los eventos confirmados, con los pesos del diseño, queda fuera de la Matriz.

El evento 6024 se usa con los valores que trae el anexo, que son 4, 7, 7 y 8. Están fuera de la escala del num. 19.11, y es el único de los 231 eventos del anexo en ese caso. Con los pesos del num. 19.11, su impacto es de 6.05.

Se declara un límite. El impacto es el del tipo de evento según la norma. No mide lo que le ocurrió a cada paciente. La validación de esos impactos con datos de los pacientes queda para una fase posterior. Por ello, en la Matriz los datos de MIMIC-IV aportan solo la frecuencia de cada evento. El impacto depende del evento que elige el evaluador, y la confiabilidad se mide sobre el veredicto y no sobre esa elección.

### 2.7 Epicrisis con un código de efecto adverso fuera de la regla

En la población de 98 931 epicrisis sin código hay 35 con un código de efecto adverso de medicamento que la regla del proyecto no alcanza (plan, Anexo 3).

Esas epicrisis se quedan en la población. Cumplen su definición registrada, que excluye los códigos que identifica la regla del proyecto (plan, sección 4.4). La población, la muestra y sus pesos siguen como los registró la Enmienda 3.

Después de terminar la primera lectura y la re-revisión, un programa cuenta cuántas de las 35 están en la muestra de 615. Si hay alguna, la HE2 y el VPP se calculan también sin ellas, como sensibilidad, con los mismos pesos. El cruce no se hace antes, porque mostraría un código de diagnóstico de casos concretos.

### 2.8 Congelamiento del protocolo y apertura de los resultados

Con esta enmienda queda fijado el protocolo de la revisión ciega y de su análisis. Lo forman el prerregistro del 5 de noviembre de 2026 (*commit* fda6828), las Enmiendas 1 a 5 y el plan de tesis del 17 de noviembre de 2026. Cada enmienda tiene su *commit* y su fecha. La huella SHA-256 del PDF de ese plan es f8d2216386c34542d02cbe6192e49cb4c903c6797ecd9ed6987b38559aa83f96.

Se registran además la versión y la huella SHA-256 de la aplicación de revisión y del programa de comprobación de la Enmienda 4 (sección 2.4). La aplicación es la del 27 de octubre de 2026, con la huella 556768326de6813ac2ea80557c7dfbe5bf94000cfe50781d412ba399739b90c9. El programa de comprobación es la versión 1.1, del 19 de noviembre de 2026, con la huella 8f2ac4cf33d5b4acb229a3e84584310faaa417693208892ca0e88406e482c46b.

El programa de comprobación informa además tres conteos de control. Son los casos con veredicto, los leídos antes del registro de la Enmienda 4 y los guardados más de una vez. Deja para el investigador un detalle de cada error, sin texto clínico. Se aplica también a la re-revisión.

Ninguno de los análisis de esta enmienda se calcula antes de terminar la primera lectura de las 615 epicrisis y la re-revisión. Tampoco se calculan antes las medidas de la confiabilidad ni los conteos por veredicto.

Abrir los resultados es unir los veredictos con la clave. Antes de hacerlo se registran en el repositorio las huellas SHA-256 de los archivos de respuestas y de eventos adicionales ya terminados, de la primera lectura y de la re-revisión. Se registra también la del *script* del análisis.

No se prevé otro cambio. Cualquier cambio exigiría una nueva enmienda, con su razón, registrada antes de abrir los resultados.

### 2.9 Pasajes que quedan sin efecto o que se precisan

En la Enmienda 2, la sección 2.2 queda sin efecto en cuanto al número de epicrisis, a la descripción del sorteo, a la precisión del kappa y al uso de la segunda lectura.

En la Enmienda 3:

1) En la sección 2.1, el margen de error de 0.05 se precisa con la sección 2.2 de esta enmienda.
2) La sección 2.4 queda sin efecto en cuanto al número de epicrisis de la re-revisión y a que ninguna se toma de los casos añadidos.
3) El numeral 3 de la sección 2.5 queda sin efecto. El kappa intraevaluador ya no mide solo la lectura del primer bloque.
4) La sección 2.6 queda cumplida entre la Enmienda 4 y esta. En lugar de los cortes del impacto se registra la sección 2.6 de esta enmienda.

En la Enmienda 4:

1) La última oración de la sección 2.1 queda cumplida con la sección 2.1 de esta enmienda.
2) En la sección 2.4, la huella del programa de comprobación es la de la sección 2.8 de esta enmienda.
3) La sección 2.8 queda cumplida con esta enmienda. En lugar de los cortes del impacto se registra la sección 2.6.

En la Enmienda 1, la sección 2.7 queda sin efecto. El impacto ya no se obtiene con cortes sobre la diferencia de estancia y el traslado a la UCI. En la Enmienda 2, la sección 2.3 queda sin efecto, porque ya no hacen falta cortes del impacto. La sección 2.6 de la Enmienda 1 no cambia.

En `config/config.yaml`, la sección `analisis_enmienda_5` sustituye tres claves de la sección `lectura_enmienda_4`. Son `rerevision_intraevaluador`, `enmienda_5` y `cortes_impacto`. Sustituye también la clave `rerevision_intraevaluador` de las secciones `revision_ciega_enmienda_2` y `muestra_enmienda_3`. Las tres secciones se conservan sin cambios como registro.

En el plan de tesis del 17 de noviembre de 2026:

1) Se precisan los pasajes sobre la re-revisión de 30 epicrisis y la precisión de su kappa. Están en la limitación 2, en la Tabla 1 (sección 3.2) y en las secciones 4.4, 4.6 y 4.7. Están también en la etapa 6 de la Tabla 4, en la Figura 1, en la Figura 2 y en el Anexo 6.
2) En la limitación 12 queda sin efecto que el kappa intraevaluador mide solo el primer bloque.
3) La limitación 2 y la fila de la HE2 de la Tabla 2 (sección 3.3.1) se precisan con la interpretación de la sección 2.2 de esta enmienda.
4) En la sección 4.4, el margen de error de 0.05 se precisa con la sección 2.2 de esta enmienda.
5) Las remisiones de la sección 4.7 y de la limitación 13 a una enmienda posterior quedan cumplidas con las secciones 2.1, 2.4, 2.5 y 2.6 de esta enmienda.
6) La remisión del Anexo 3 sobre las 35 epicrisis queda cumplida con la sección 2.7 de esta enmienda.
7) En la sección 3.1.4 queda sin efecto que solo dos indicadores del impacto se aproximan en MIMIC-IV, con la estancia y el traslado a la UCI. Queda sin efecto también que los cortes del impacto se prerregistrarán. En la limitación 9 queda sin efecto que los indicadores del impacto se derivan de la estancia y del paso por la UCI. El impacto se toma del Anexo N.° 02, con sus cuatro indicadores. En la sección 3.1.4, en la limitación 9, en la Tabla 3 y en la Figura 1 queda sin efecto que el impacto es parcial.

## 3. Lo que no cambia

Fuera de lo que declara la sección 2, no cambia nada de las Enmiendas 1 a 4 ni del plan de tesis. En particular, no cambian los criterios ni los umbrales de la HE1 y de la HE2, ni las semillas de la HE1. Tampoco cambian la muestra, sus pesos, la regla del resultado no concluyente, la hipótesis general ni la decisión confirmatoria en la HE2.

Se mantienen el procedimiento de lectura de la Enmienda 4, el tratamiento del veredicto “No se puede determinar” y la lectura sin conocer la marca del modelo.

## Referencia

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46. https://doi.org/10.1177/001316446002000104
