# Enmienda 3 del prerregistro: ampliación de la muestra de la revisión ciega

- Fecha de registro: 17 de noviembre de 2026. La decisión de ampliar la muestra se tomó el 16 de noviembre de 2026 y el tamaño se fijó el 17 de noviembre.
- Lo que se enmienda: de la Enmienda 1 (*commit* ad116a5, con redacción vigente del *commit* 977a1ce), la sección 2.4, el tercer párrafo de la sección 2.5 y la sección 3. De la Enmienda 2 (*commit* b40b7ac), el numeral 2 de la sección 2.1, el numeral 4 de la sección 2.2, las secciones 2.3 y 2.4 y la sección 3. También se enmiendan los pasajes del plan de tesis y del informe parcial que enumera la sección 2.7 de esta enmienda.
- Resumen estructurado: sección `muestra_enmienda_3` de `config/config.yaml`, en este mismo *commit*. El *script* de la ampliación no lee esa sección, porque lleva los tamaños y las semillas en su código.
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda

1) La revisión ciega sigue en curso con el manual de anotación (*codebook* v2), que no ha cambiado. Se han leído 140 de las 241 epicrisis. La última lectura se guardó el 11 de noviembre de 2026.
2) Los resultados de la revisión ciega no se han abierto. Nadie ha visto los veredictos separados por detecciones y controles.
3) El investigador conoce sus veredictos de las 140 epicrisis leídas, pero no sabe a qué grupo pertenece cada una. La aplicación le muestra el avance y los tiempos de lectura. No muestra el grupo ni el conteo de veredictos, y la clave se guarda en una carpeta distinta de la que lee la aplicación.
4) El *script* de la HE1 no se ha ejecutado.
5) Ninguna persona distinta del investigador ha leído las epicrisis de la muestra.
6) El 17 de noviembre de 2026 se ejecutó un ensayo de la ampliación. El ensayo leyó la clave y el lote, cargó el archivo de anotaciones para comprobar sus identificadores y cargó el texto de los casos nuevos. No escribió ningún archivo. Solo mostró conteos y el resultado de sus comprobaciones, sin identificadores, grupos, veredictos ni texto.

## 2. Cambios

### 2.1 Tamaño de la muestra

La muestra pasa de 151 detecciones y 90 controles a 385 detecciones y 230 controles. Son 615 epicrisis. Se añaden 234 detecciones y 140 controles, es decir, 374 casos. No se prevé otra ampliación. Cualquier cambio del tamaño exigiría una nueva enmienda, registrada antes de abrir los resultados.

La razón es la precisión del VPP y la potencia de la HE2. La decisión no se tomó a partir de los veredictos, y los resultados por grupo no se han abierto. El investigador conocía sus 140 veredictos, sin el grupo de cada caso, lo cual se declara como limitación (ver sección 2.5).

La decisión sí usó el tiempo de lectura que registra la aplicación. Eran 8.9 horas en 140 casos, con una media de 3.8 minutos y una mediana de 1.4 minutos por caso. Con esa media, los 374 casos añadidos suponen unas 24 horas de lectura, lo cual se consideró viable.

El tamaño de 151 se calculó con un margen de error de 0.08. Naing et al. (2006, p. 10) señalan que, según su experiencia, en los estudios de prevalencia es apropiada una precisión de 0.05 cuando la proporción esperada está entre 0.10 y 0.90. Con la fórmula n = z²·P·(1 − P) / e², z = 1.96, P = 0.5 y e = 0.05, resultan 384.16 casos, que se redondean hacia arriba a 385 detecciones.

Los controles se amplían en la misma proporción. El cálculo es 90 × 385 / 151 = 229.47, que se redondea hacia arriba a 230. El marco tiene 15 731 epicrisis marcadas y 83 200 sin marcar. Cada detección pasa de representar 104.18 epicrisis a representar 40.86. Cada control pasa de 924.44 a 361.74. Así, la razón entre los pesos de los dos grupos casi no cambia: era de 8.87 y pasa a 8.85.

Con 385 y 230, la potencia de la HE2 es de 0.982 para un VPP de 0.35 frente a una proporción de 0.20 en los controles. Con 151 y 90 era de 0.714. Con un VPP de 0.35 y una potencia de 0.80, la diferencia mínima detectable pasa de 0.164 a 0.107. Se usó el mismo cálculo binomial exacto sobre el IC 95 % de Newcombe, sin corrección por continuidad.

### 2.2 Regla de la ampliación

El sorteo original, del 26 de octubre de 2026, tomó 151 epicrisis marcadas con la semilla 42 y 90 controles con la semilla 77. Se hizo con la función `DataFrame.sample` de pandas, sin reemplazo y con una semilla entera. Con esa forma de uso, la función toma los primeros casos de una permutación que depende de la semilla y del número de filas. Por ello, pedir 385 y 230 con las mismas semillas devuelve los 241 casos ya sorteados y 374 casos nuevos.

La ampliación usa esa regla. No se elige ninguna semilla nueva para decidir qué epicrisis entran. El marco, el modelo y el corte de 0.5 son los mismos. El puntaje del modelo adoptado no se vuelve a calcular: se usan las probabilidades guardadas al generar el lote original. Para los casos nuevos solo se calcula el puntaje del modelo de contexto, que va a la clave y se usa en el análisis secundario.

Los 241 casos ya sorteados conservan su identificador, su orden y sus veredictos. Los 374 casos nuevos se mezclan entre sí en orden aleatorio, con la semilla 2028, que sigue la serie de las semillas 2026 y 2027 del sorteo original. Reciben los identificadores HE4-242 a HE4-615. El identificador indica el bloque del caso, pero no su grupo.

El orden de lectura es aleatorio dentro de cada bloque, sin agrupar detecciones y controles. La aplicación presenta el bloque añadido después de los 241 primeros casos.

Antes de cargar los casos nuevos, el *script* comprueba, entre otras cosas, lo siguiente:

1) El marco tiene 15 731 epicrisis marcadas y 83 200 sin marcar.
2) Los primeros 151 y los primeros 90 de las listas ampliadas son, en el mismo orden, los del sorteo original, y coinciden con la clave vigente.
3) Ningún caso nuevo está en la clave vigente.
4) Las probabilidades guardadas coinciden con las de la clave.

Si alguna comprobación falla, la ampliación no se hace. En el ensayo del 17 de noviembre de 2026 se cumplieron todas.

El *script* respalda antes los archivos del lote y de la clave. Después añade las filas de los casos nuevos al final del lote, de la clave y del archivo de anotaciones, sin reescribir las filas de los 241 casos. También añade un bloque al archivo del diseño de la muestra y dos líneas al archivo de huellas. Si la verificación final falla, retira las filas añadidas. No toca los archivos de la re-revisión.

Los dos *scripts* no están en el repositorio. Se identifican por su huella SHA-256. La del *script* del sorteo es 211f47ec5606457b4dfef0a3cf3c54a5216c3dd83ae540607057543ff79ba3d3. La del *script* de la ampliación es 0473848a38c35dfa938114f29e25750bbcd82bdad5e80fd6cf013bb5e240006f.

### 2.3 Análisis

El criterio de la HE2 no cambia. La HE2 se cumple si, en las 385 detecciones y los 230 controles, el límite inferior del IC 95 % de Newcombe de la diferencia de proporciones es mayor que cero. La regla del resultado no concluyente tampoco cambia.

La referencia de la HE2, del VPP y de los eventos confirmados de la aplicación de la Matriz GEMSES es la lectura del investigador en las 615 epicrisis. Las estimaciones se calculan con los pesos nuevos.

Como análisis de sensibilidad, la HE2 y el VPP se calculan también por separado en dos cortes. El primero separa los 241 casos del primer sorteo de los 374 añadidos. El segundo separa los 140 casos leídos antes de la decisión de los leídos después. En cada parte se informan p1, p0, su diferencia con el IC 95 % de Newcombe y el VPP con su IC 95 % de Wilson.

Las partes no se comparan con una prueba de hipótesis. Este análisis es descriptivo y no decide la HE2. Si las partes discrepan, se informa como limitación.

Si la primera lectura de las 615 epicrisis no se completa, el análisis exigirá una nueva enmienda, registrada antes de abrir los resultados.

### 2.4 Re-revisión y apertura de los resultados

Esta enmienda no cambia la re-revisión de 30 epicrisis. Las 30 siguen siendo de las 241 primeras, y ninguna se toma de los casos añadidos. La espera de dos semanas se sigue contando desde el final de la primera lectura de esas 241.

Los resultados de la revisión ciega se abren después de terminar la primera lectura de las 615 epicrisis y la re-revisión. Así se precisan el numeral 4 de la sección 2.2 de la Enmienda 2 y la sección 4.7 del plan de tesis.

### 2.5 Limitaciones que se declaran

1) El investigador decidió la ampliación cuando ya conocía sus veredictos de 140 casos, aunque no el grupo de cada uno. Además, podría reconocer parte de las detecciones, como declara la Enmienda 2.
2) Los dos bloques se leen en momentos distintos. Los casos 141 a 241 se leen sabiendo de la ampliación.
3) El kappa intraevaluador mide solo la lectura del primer bloque.
4) El tiempo de lectura registrado es menor que el que estima la propia aplicación. Su aviso de pantalla indica de 5 a 8 minutos por caso, y 87 de los 140 casos tomaron menos de 2 minutos. La aplicación muestra por defecto la evolución y los diagnósticos al alta, y no la nota completa.

### 2.6 Asuntos que se registran en la Enmienda 4

La sección 2.3 de la Enmienda 2 anunciaba que los cortes del impacto se registrarían en la Enmienda 3. Esta enmienda registra la ampliación de la muestra y sus consecuencias en el orden de lectura y en la apertura de los resultados. Los asuntos siguientes se registran en la Enmienda 4, antes de abrir los resultados de la revisión ciega:

1) La descripción del sorteo de la re-revisión. El *script* del sorteo ya había generado las 30 epicrisis junto con el lote, con una regla distinta de la que describe la Enmienda 2 (sección 2.2, numeral 1).
2) El tratamiento en el análisis de los casos con el veredicto “No se puede determinar”.
3) La descripción completa de la aplicación de revisión.
4) Los cortes del impacto.

### 2.7 Pasajes que quedan sin efecto o que se precisan

En la Enmienda 1:

1) En la sección 2.4 quedan sin efecto los tamaños de 151 detecciones y 90 controles.
2) En el tercer párrafo de la sección 2.5 se precisa el orden de lectura, como indica la sección 2.2 de esta enmienda. Además, queda sin efecto la frase según la cual la aplicación “no registra el tiempo de lectura ni admite la opción de no evaluable”. La aplicación registra el tiempo dedicado a cada caso y ofrece el veredicto “No se puede determinar”.
3) En la sección 3 queda sin efecto la mención de la muestra de 151 detecciones y 90 controles.

En la Enmienda 2:

1) En el numeral 2 de la sección 2.1, el número de 241 epicrisis pasa a 615.
2) El numeral 4 de la sección 2.2 se precisa con la sección 2.4 de esta enmienda.
3) La sección 2.3 queda sin efecto en cuanto al número de la enmienda que registra los cortes del impacto.
4) En la sección 2.4 queda sin efecto la oración que mantiene el tercer párrafo de la sección 2.5 de la Enmienda 1, en lo que esta enmienda precisa de ese párrafo.
5) En la sección 3 queda sin efecto la mención de la muestra de 151 detecciones y 90 controles con sus pesos. La mención del orden aleatorio se precisa como en la Enmienda 1.

En `config/config.yaml`, la sección `muestra_enmienda_3` fija los tamaños nuevos, que antes no figuraban en ese archivo. Además, sustituye tres claves de la sección `revision_ciega_enmienda_2`: `cortes_impacto`, `apertura_resultados` y `referencia_he2_vpp_gemses`. Las secciones `he1_enmienda_1` y `revision_ciega_enmienda_2` se conservan sin cambios como registro.

En el plan de tesis del 16 de noviembre de 2026 quedan sin efecto las cifras que dependen del tamaño de la muestra. Son los tamaños de 151, 90 y 241, los pesos de 104.2 y 924.4, el margen de error de 0.08 y las cifras de potencia. Están en los pasajes siguientes:

1) las limitaciones 10 y 11;
2) el numeral del OE2 de la sección 2.2;
3) la sección 3.1.3, en cuanto a las 241 epicrisis revisadas;
4) la fila de la HE2 de la Tabla 2 (sección 3.3.1);
5) la sección 4.4, salvo el párrafo de la re-revisión;
6) la sección 4.7;
7) la etapa 6 de la Tabla 4 (sección 4.8);
8) la matriz de consistencia del Anexo 1;
9) la Figura 1 (sección 4.3) y la Figura 2 (sección 5.1).

En ese plan quedan sin efecto, además, tres afirmaciones. La primera está en la sección 4.4 y dice que el tamaño se fijó antes de iniciar la revisión ciega. El tamaño de 385 y 230 se fijó el 17 de noviembre de 2026, con 140 epicrisis leídas. La segunda está en la sección 4.6 y se refiere al orden aleatorio, que se precisa como en la Enmienda 1. La tercera está en la sección 4.7 y se refiere a la apertura de los resultados, que se precisa con la sección 2.4 de esta enmienda.

En el informe parcial del 13 de noviembre de 2026, los pasajes equivalentes quedan superados por esta enmienda.

## 3. Lo que no cambia

Fuera de lo que declara la sección 2, no cambia nada de las Enmiendas 1 y 2 ni del plan de tesis. En particular, no cambian los criterios ni los umbrales de la HE1 y de la HE2. Tampoco cambian las semillas de la HE1, la regla del resultado no concluyente de la HE2, la hipótesis general ni la decisión confirmatoria en la HE2.

Se mantienen el modelo, el corte de 0.5 y el manual de anotación (*codebook* v2). Se mantienen también la revisión con un solo evaluador, la lectura sin conocer la marca del modelo y el proceso de la aplicación de la Matriz GEMSES.

## Referencia

Naing, L., Winn, T., y Rusli, B. N. (2006). Practical issues in calculating the sample size for prevalence studies. *Archives of Orofacial Sciences, 1*, 9–14.
