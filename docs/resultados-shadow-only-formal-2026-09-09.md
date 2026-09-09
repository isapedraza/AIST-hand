# Resultados Shadow-only — validación funcional pick_bucket, 2026-09-09

Serie formal completa de diez intentos, siguiendo el protocolo acordado en
[ThesisGCN/protocolo-validacion-funcional-registro-2026-09-07.md](../../ThesisGCN/protocolo-validacion-funcional-registro-2026-09-07.md).
Evalúa el checkpoint `stage1_shadow_solo_ckpt17_run18b_13k7.pt` sobre `pick_bucket`
con la mano Shadow, control del brazo por waypoints preparados por el operador
con correcciones manuales, dedos por retargeting, sin compensación de
deslizamiento ni de gravedad del objeto. Cada intento arranca con el balde en
su posición de reposo; el criterio de éxito exige que el sensor del objeto
quede dentro de la caja delimitadora del balde y que este se levante al menos
0.15 m sobre su altura de reposo, ambos leídos del criterio de tarea del
simulador (`shadow_ext/tasks.py`, adaptación local documentada de DexJoCo).

## Medida principal: éxitos / intentos evaluables

| Mano | Tarea | Control del brazo | Comp. deslizamiento | Éxitos/intentos | Tiempo real a éxito | Interrupciones/errores |
| --- | --- | --- | --- | --- | --- | --- |
| Shadow | pick_bucket | Waypoints + correcciones manuales | No | 8/10 | mediana 178.8 s, rango 152.7–235.2 s (n=8) | 3 interrupciones manuales, 0 errores técnicos |

**8/10 = 80.0%.** Intervalo de confianza de Wilson al 95% para la proporción:
[49.0%, 94.3%] — con diez intentos el resultado es exploratorio, no una
estimación precisa de tasa de éxito poblacional.

| Intento | Resultado | Terminación |
|---|---|---|
| shadow-only-formal-001 | Éxito | success |
| shadow-only-formal-002 | Fallo | manual_interruption |
| shadow-only-formal-003 | Éxito | success |
| shadow-only-formal-004 | Éxito | success |
| shadow-only-formal-005 | Éxito | success |
| shadow-only-formal-006 | Éxito | success |
| shadow-only-formal-007 | Fallo | manual_interruption |
| shadow-only-formal-008 | Éxito | manual_interruption (ver nota) |
| shadow-only-formal-009 | Éxito | success |
| shadow-only-formal-010 | Éxito | success |

Las interrupciones manuales de un intento formal cuentan como fallo por regla
declarada, salvo evidencia física directa de éxito dentro del intento, como en
008 (ver nota siguiente). Los intentos sin éxito no tienen tiempo a éxito
ausente ni se les asigna cero.

## Medida secundaria: tiempo real a éxito

El tiempo total mide el sistema completo -- posicionamiento del brazo,
decisiones del operador, control de dedos y retrasos de cómputo -- no aisla la
calidad del retargeter. Para que sea comparable entre los diez intentos, la
frontera de inicio no es la marca manual del operador (arranca en momentos muy
distintos según cuánto tiempo real usó cada uno en preparar cámara y postura
antes de empezar), sino el primer instante, leído de la traza física completa
guardada por intento, en que el punto de agarre de la mano se desplaza más de
1 cm de su pose de reposo -- el operador empezando a actuar de verdad,
verificado estable frente a 0.5-3 cm de umbral. El intervalo hasta el primer
éxito (mismo criterio de la tarea, mismo dato crudo) es la duración real del
intento en los diez casos, con o sin marca manual bien ubicada:

| Intento | Tiempo real (movimiento → éxito) |
|---|---:|
| shadow-only-formal-001 | 178.4 s |
| shadow-only-formal-003 | 218.8 s |
| shadow-only-formal-004 | 179.3 s |
| shadow-only-formal-005 | 170.8 s |
| shadow-only-formal-006 | 187.8 s |
| shadow-only-formal-008 | 235.2 s |
| shadow-only-formal-009 | 152.7 s |
| shadow-only-formal-010 | 168.5 s |

Mediana 178.8 s, rango 152.7–235.2 s, n=8 (los ocho éxitos, ninguno excluido).

## Nota: referencia física y marca manual de inicio (intento 008)

En 008 el operador marcó el inicio de evaluación después de que el agarre y
levantamiento ya habían ocurrido. La lógica de tarea recapturaba la referencia
de altura de reposo del balde en cada marca de inicio; al llegar tarde, la
recapturó desde el balde ya en el aire, y todo el criterio de levantamiento
posterior a ese punto quedó medido contra una referencia incorrecta -- el
registro dice `lifted: false` para un intento que sí tuvo éxito. Corregimos la
lógica de tarea (`dexjoco-shadow` commit `7d961b2`, `shadow_ext/tasks.py`) para
que la referencia se capture una sola vez, al inicio real del intento, y ya no
dependa de cuándo se marca el inicio de evaluación. Confirmamos el éxito de 008
reproduciendo su traza física completa contra la referencia real, y auditamos
los diez intentos con el mismo método: solo 008 tenía la referencia
corrompida. El intento 010 marcó el inicio igual de tarde y confirma la
corrección en vivo, sin necesitar reconstrucción posterior.

Fuente: `recordings/shadow-only-formal-001` a `010`, metadatos de simulación y
cámara. Auditoría de referencia física reproducible con
`shadow_ext/verify_task_reference.py`; tiempo real a éxito reproducible con
`shadow_ext/attempt_timing.py`. Consolidado completo en
`resultados-shadow-only-formal-2026-09-09.json`. Grabaciones y checkpoints son
archivos locales y no se incluyen en Git.
