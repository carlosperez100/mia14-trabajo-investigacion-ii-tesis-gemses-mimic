# Enmienda 1 del prerregistro de la HE1

- Fecha de registro: 12 de noviembre de 2026.
- Redacción vigente: 13 de noviembre de 2026, con la numeración y los términos del plan de tesis, la segunda lectura con condición y aclaraciones tomadas del plan. Los criterios, los umbrales, las semillas y las fechas no cambian, y el texto registrado el 12 de noviembre está en el *commit* ad116a5.
- Prerregistro que se enmienda: sección `he1_estratos` de `config/config.yaml`, *commit* fda6828, del 5 de noviembre de 2026.
- Versión legible por el *script*: sección `he1_enmienda_1` de `config/config.yaml`, en este mismo *commit*.
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda

1) La revisión ciega de las 241 epicrisis, 151 detecciones y 90 controles, empezó antes del 5 de noviembre de 2026 con el manual de anotación (*codebook* v2), el cual no ha cambiado desde entonces. Sus resultados no se han abierto, y nadie ha visto los veredictos separados por detecciones y controles.
2) El *script* de la HE1, que el prerregistro dejó por escribir, no se ha ejecutado, de modo que no se ha calculado la diferencia de ROC-AUC con su IC por paciente ni la ROC-AUC de ningún estrato.
3) La HE1 no es ciega respecto de su dirección, puesto que en la prueba ya se conocían tres valores de ROC-AUC. El de la línea de base del 30 de septiembre de 2026, con solo las palabras, fue de 0.864, y el del análisis exploratorio del 16 de octubre, sin el contenido textual, de 0.760. El del modelo adoptado fue de 0.8706, que la nota de la matriz de consistencia v4 registra como 0.871.

## 2. Cambios

### 2.1 Criterio de la HE1

El criterio por estrato del prerregistro pasa a ser un análisis de sensibilidad. La HE1 se considera cumplida si, en las 12 321 epicrisis de prueba sin etiqueta dudosa, se cumplen a la vez tres condiciones:

1) La diferencia global de ROC-AUC entre el modelo adoptado (A+B+C, variante D2) y el modelo de contexto (bloque C, entrenado con las mismas filas) es de al menos 0.05.
2) El límite inferior de su IC 95 % por paciente es mayor que cero.
3) Ese límite inferior también es mayor que cero en cada una de cinco particiones por paciente adicionales, con las semillas 43 a 47, en las que ambos modelos se reentrenan, lo cual constituye la condición de estabilidad.

### 2.2 Definiciones que el prerregistro dejó pendientes

1) El IC 95 % de la diferencia se obtiene por percentiles, con 2 000 remuestreos de pacientes pareados, es decir, con los mismos pacientes para ambos modelos en cada remuestreo.
2) Los cortes de los deciles de largo se calculan en el conjunto de prueba de cada análisis, es decir, en las 12 321 epicrisis sin etiqueta dudosa en el análisis principal y en las 14 085 en el de la etiqueta original.
3) Los mínimos de 200 epicrisis y de 30 por clase se cuentan de nuevo en cada análisis de sensibilidad.
4) En la sensibilidad con la configuración candidata, el modelo de contexto usa la misma regresión logística que la candidata, con C de 1.8734 y sin pesos.

### 2.3 Hipótesis general

La hipótesis general se considera cumplida si se cumplen a la vez la HE1 y la HE2, y no las tres hipótesis del prerregistro. La HE2 de la matriz v4, que exigía que el límite inferior del IC del VPP alcanzara 0.25, pasa a ser una estimación descriptiva del OE2, sin umbral, y la HE3 de la matriz v4 pasa a ser la HE2.

Dado que la HE1 no es ciega respecto de su dirección, la decisión confirmatoria de la hipótesis general recae en la HE2.

### 2.4 HE2

El criterio de la HE2 no cambia. Como fijó la matriz v4 para la entonces HE3, la HE2 se cumple si, en las 151 detecciones y los 90 controles, el límite inferior del IC 95 % de Newcombe de la diferencia de proporciones es mayor que cero.

Si no se cumple, se descarta una diferencia de 0.10 o mayor solo si el límite superior de ese IC es menor que esa cifra, y en otro caso el resultado se informa como no concluyente. El umbral de 0.05 de la HE1 y el de 0.10 de la HE2 no proceden de una fuente, por lo cual se informan los intervalos completos.

### 2.5 Lectura de referencia, adjudicación y segundo evaluador

Si hay segunda lectura, en las 100 epicrisis previstas, 63 detecciones y 37 controles, las discrepancias entre el investigador y el segundo evaluador se discutirán sin conocer la marca del modelo, y si no se alcanza un acuerdo prevalecerá la lectura del segundo evaluador. En ese caso, la referencia de la HE2 y del VPP será esa lectura adjudicada en las 100 epicrisis y la del investigador en las demás.

El segundo evaluador leerá solo después de obtener su propia credencial de PhysioNet y de firmar a su nombre el acuerdo de uso. Si al 30 de noviembre de 2026 no la tiene, se invitará a otro evaluador clínico con credencial propia. Si al 31 de diciembre de 2026 ninguno la tiene, el kappa interevaluador no se informará y la referencia será la lectura del investigador en las 241 epicrisis, lo cual se declarará como limitación.

La aplicación de la revisión presenta las epicrisis en orden aleatorio, sin agrupar detecciones y controles, y no registra el tiempo de lectura ni admite la opción de no evaluable.

### 2.6 Proceso de la aplicación de la Matriz GEMSES

Todas las epicrisis se asignan al proceso asistencial de Hospitalización, uno de los que enumera el num. 19.6 de la Directiva N.° 7-OGCyH-ESSALUD-2020, dado que cada epicrisis resume una hospitalización. Esta asignación es una simplificación, pues un evento quirúrgico o de emergencia podría pertenecer a otro proceso.

Por consiguiente, el índice de frecuencia de un evento es la suma de los pesos de sus casos confirmados, dividida entre la de todos los eventos confirmados en el proceso, y los percentiles de la regla del nivel abarcan todos los grupos del proceso.

Según el num. 19.7, una vez seleccionado el proceso se selecciona el departamento o servicio al cual pertenece el evento. Como aproximación a ese servicio, el servicio al ingreso se informa como atributo y como análisis de sensibilidad.

### 2.7 Cortes del impacto

Los cortes con los cuales la diferencia de estancia y el traslado a la UCI se convierten en los valores del num. 19.11 (0, 1, 3, 6 o 9) se registrarán en la Enmienda 2, antes de abrir los resultados de la revisión ciega.

## 3. Lo que no cambia

No cambian la configuración actual (C de 1.0, 20 000 palabras y 20 000 bigramas con min_df de 3, pesos balanceados y sublinear_tf), la partición de prueba, la etiqueta sin dudosa ni los estratos de servicio al ingreso, grupo de edad y decil de largo. Tampoco cambian los mínimos de 200 epicrisis y 30 por clase, la candidata y la etiqueta original como sensibilidad, el corte de 0.5, la muestra de 151 detecciones y 90 controles ni el *codebook* v2.
