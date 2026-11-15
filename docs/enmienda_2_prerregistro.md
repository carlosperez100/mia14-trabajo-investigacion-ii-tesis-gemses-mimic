# Enmienda 2 del prerregistro: revisión ciega con un solo evaluador

- Fecha de registro: 15 de noviembre de 2026. La decisión se tomó el 14 de noviembre de 2026.
- Lo que se enmienda: las secciones 2.5 y 2.7 de la Enmienda 1, registrada el 12 de noviembre de 2026 (*commit* ad116a5) y con redacción vigente del 13 de noviembre (*commit* 977a1ce). Esa enmienda modifica el prerregistro de la sección `he1_estratos` de `config/config.yaml` (*commit* fda6828, del 5 de noviembre de 2026). También se enmiendan los pasajes del plan de tesis del 13 de noviembre de 2026 que enumera la sección 2.5.
- Versión legible por el *script*: sección `revision_ciega_enmienda_2` de `config/config.yaml`, en este mismo *commit*.
- Autor: Carlos Pérez Pérez.

## 1. Estado al registrar la enmienda, el 15 de noviembre de 2026, sin cambios desde la decisión del 14 de noviembre

1) La revisión ciega de las 241 epicrisis, 151 detecciones y 90 controles, sigue en curso con el manual de anotación (*codebook* v2), el cual no ha cambiado. Sus resultados no se han abierto, y nadie ha visto los veredictos separados por detecciones y controles.
2) El *script* de la HE1 no se ha ejecutado.
3) Ninguna persona distinta del investigador ha leído las epicrisis de la muestra.
4) El segundo evaluador previsto no tiene credencial propia de PhysioNet ni ha firmado a su nombre el acuerdo de uso, y no podrá obtenerla. Asimismo, no se dispone de otro evaluador clínico con credencial propia ni de otro que pueda tramitarla.

## 2. Cambios

### 2.1 Revisión ciega con un solo evaluador

El acuerdo de uso no permite compartir el acceso a los datos (*PhysioNet Credentialed Health Data Use Agreement 1.5.0*, cláusula 3), por lo cual solo puede leer las epicrisis quien tenga su propia credencial y haya firmado a su nombre ese acuerdo.

Dado que el segundo evaluador previsto no podrá obtener la credencial, no se espera el plazo del 30 de noviembre de 2026. Asimismo, no se invita a otro evaluador clínico ni se espera al 31 de diciembre de 2026, porque no se dispone de otro con credencial propia de PhysioNet ni de otro que pueda tramitarla.

Por consiguiente, sin esperar los plazos de la sección 2.5 de la Enmienda 1, se adopta la consecuencia que esa sección preveía si al 31 de diciembre de 2026 ningún evaluador tenía credencial propia. La revisión ciega se completa con un solo evaluador, el investigador. La decisión se tomó el 14 de noviembre de 2026, antes de abrir los resultados de la revisión ciega y de ejecutar el *script* de la HE1, y no depende de ellos. Sus consecuencias son tres:

1) No hay segunda lectura de las 100 epicrisis previstas, 63 detecciones y 37 controles, ni adjudicación de discrepancias, y no se informan el kappa interevaluador ni los acuerdos positivo y negativo.
2) La referencia de la HE2, del VPP y de los eventos confirmados de la aplicación de la Matriz GEMSES es la lectura del investigador en las 241 epicrisis, y ninguno se recalcula con otra lectura.
3) Las conclusiones, incluida la decisión confirmatoria de la hipótesis general, que recae en la HE2, descansan en la lectura del investigador, sin confirmación independiente. El investigador lee sin conocer la marca del modelo, pero lo desarrolló, y los términos de mayor coeficiente se publicaron en el README de la semana 7 (*commit* cf4178b). Por ello podría reconocer parte de las detecciones, lo cual se declara como limitación.

### 2.2 Confiabilidad de la lectura y re-revisión de 30 epicrisis

Se mantiene el kappa de Cohen intraevaluador en 30 epicrisis revisadas de nuevo a ciegas al menos dos semanas después de su primera lectura, como establecen el plan de tesis (sección 4.6) y el informe parcial (Tabla 20). Dado que ese kappa mide la estabilidad de la lectura de un mismo evaluador, no sustituye la confirmación independiente.

Con el kappa supuesto de 0.70 y la prevalencia de 0.28 a 0.29 del plan (sección 4.4), se calculó que la semiamplitud del IC 95 % del kappa intraevaluador sería de 0.29 (Cohen, 1960). Asimismo, la re-revisión se define como sigue:

1) Las 30 epicrisis se sortean por muestreo aleatorio simple entre las 241, sin usar la clave de la muestra, con `numpy.random.default_rng(42).choice` sin reemplazo, sobre sus identificadores de MIMIC-IV-Note (`note_id`) en orden ascendente.
2) La re-revisión empieza al menos dos semanas después de terminar la primera lectura de las 241. Así se precisa la Tabla 18 del informe parcial (p. 51), que indica “Terminar la revisión de las 241 y repetir 30 casos a ciegas al menos dos semanas después”. La aplicación presenta las 30 en orden aleatorio, sin la marca del modelo ni el primer veredicto.
3) En esas 30 epicrisis, la referencia de la HE2, del VPP y de los eventos confirmados es la primera lectura. La segunda solo se usa para el kappa intraevaluador, que se calcula con ambas lecturas sin adjudicar sus discrepancias.
4) Los resultados de la revisión ciega se abren después de la re-revisión, en el orden que fija la misma Tabla 18, sin la segunda lectura ni la adjudicación.

### 2.3 Cortes del impacto

La sección 2.7 de la Enmienda 1 anunciaba que los cortes del impacto se registrarían en la Enmienda 2. Dado que esta enmienda registra solo los cambios en la revisión ciega, los cortes se registrarán en la Enmienda 3, también antes de abrir los resultados de la revisión ciega.

### 2.4 Pasajes de la Enmienda 1 que quedan sin efecto

En la Enmienda 1 quedan sin efecto los dos primeros párrafos de la sección 2.5, sobre la adjudicación y el segundo evaluador, y la sección 2.7 en cuanto al número de la enmienda. Se mantiene el tercer párrafo de la sección 2.5, sobre el orden aleatorio y las opciones de la aplicación.

En `config/config.yaml`, la sección `revision_ciega_enmienda_2` sustituye las claves `adjudicacion`, `segundo_evaluador` y `cortes_impacto` de la sección `he1_enmienda_1`, la cual se conserva sin cambios como registro.

### 2.5 Pasajes del plan de tesis que quedan sin efecto

En el plan de tesis del 13 de noviembre de 2026 quedan sin efecto, en cuanto a la segunda lectura, los siguientes pasajes:

1) la limitación 2, en lo que se refiere a las 100 epicrisis con segunda lectura y a los plazos del 30 de noviembre y del 31 de diciembre de 2026;
2) el último párrafo de la sección 4.4, sobre la segunda lectura de 100 epicrisis y la precisión de su kappa;
3) la oración de la sección 4.6 sobre la credencial del segundo evaluador;
4) la segunda lectura de 100 en la etapa 6 de la Tabla 4;
5) la última oración de la sección 5.1 y la fila del segundo evaluador clínico en la Tabla 5 (sección 5.2);
6) la definición del kappa interevaluador en la Tabla 1 (sección 3.2);
7) la segunda lectura prevista y el kappa interevaluador en la matriz de consistencia del Anexo 1;
8) el kappa interevaluador en 100 epicrisis de la Figura 1 (sección 4.3) y la segunda lectura de 100 de la Figura 2 (sección 5.1).

Los pasajes condicionados a que haya segunda lectura, en las secciones 2.2, 4.6 y 4.7, no se aplican, incluida la oración de la sección 4.6 sobre las discrepancias. En el informe parcial del 13 de noviembre de 2026, los pasajes equivalentes quedan superados por esta enmienda.

## 3. Lo que no cambia

Fuera de lo que declara la sección 2, no cambia nada de la Enmienda 1 ni del plan de tesis. En particular, no cambian los criterios ni los umbrales de la HE1 y de la HE2, las semillas de la HE1, la regla del resultado no concluyente de la HE2, la hipótesis general ni la decisión confirmatoria en la HE2.

Tampoco cambian la configuración actual, el corte de 0.5, la muestra de 151 detecciones y 90 controles con sus pesos, el *codebook* v2, el orden aleatorio, la lectura sin conocer la marca del modelo ni el proceso de la aplicación de la Matriz GEMSES.
