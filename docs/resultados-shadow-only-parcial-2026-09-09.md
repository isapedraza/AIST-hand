# Resultados parciales Shadow only — 2026-09-09

Corte: primeros ocho intentos terminados de una serie de diez. No es el resultado final.

Checkpoint: `stage1_shadow_solo_ckpt17_run18b_13k7.pt`. Tarea: `pick_bucket`; operador: `yahel`.

**6/8 = 75.0% de éxito; 2 fallos.**

| Intento | Resultado | Tiempo real hasta éxito |
|---|---|---:|
| shadow-only-formal-001 | Éxito | 184.8 s |
| shadow-only-formal-002 | Fallo: interrupción manual | — |
| shadow-only-formal-003 | Éxito | 235.0 s |
| shadow-only-formal-004 | Éxito | 190.6 s |
| shadow-only-formal-005 | Éxito | 172.6 s |
| shadow-only-formal-006 | Éxito | 198.8 s |
| shadow-only-formal-007 | Fallo: interrupción manual | — |
| shadow-only-formal-008 | Éxito (corregido, ver nota) | no comparable |

Mediana entre los 5 éxitos con tiempo comparable: 190.6 s. Rango: 172.6–235.0 s. El 008
se cuenta en el éxito pero se excluye de esta mediana/rango (ver nota).

## Nota sobre 008: referencia corrompida por F5 tardío, corregida por replay físico

En 008 el operador presionó F5 (arranque de evaluación) *después* de que el agarre y
levantamiento ya habían ocurrido. Bug de instrumentación (`shadow_ext/tasks.py`,
`PickBucket.reset`, arreglado en `dexjoco-shadow` commit `7d961b2`): F5 volvía a capturar
la altura de referencia de la mesa desde la posición *actual* del balde, no la de reposo.
Si el balde ya estaba en el aire al presionar F5, toda la métrica de "levantado" quedaba
medida contra una referencia equivocada — el JSON grabado decía `lifted: false` para un
intento que sí tuvo éxito.

Se repitió la física cruda del trial (`shadow_ext/verify_task_reference.py`, trace de
estado completo guardado por trial) usando la referencia real (altura del balde en el
primer frame grabado, antes de cualquier input del operador): confirma `inside=True,
lifted=True` justo en el instante de F5. Los otros 7 trials de la serie se auditaron con
el mismo script: ninguno tiene la referencia corrompida (script y resultado crudo en
`recordings/shadow-only-formal-008/sim/task_reference_correction.json`, local, no
versionado).

Como el éxito real ocurrió antes de F5, el tiempo hasta éxito medido desde F5 no refleja
el intento real (~0 s desde F5) y no es comparable con el resto — se excluye de la
mediana/rango, pero el trial cuenta como éxito en la tasa.

El bug ya está corregido para trials futuros (F5 ya no puede corromper la referencia).

Las interrupciones manuales de pruebas formales cuentan como fallos por la regla declarada
por el operador (002, 007), salvo cuando hay evidencia física directa de éxito dentro de la
ventana de evaluación, como en 008. La 002 carece de marca F5: cuenta como fallo, conserva
esa incidencia y no se le infiere un tiempo de evaluación. Los problemas de video se
registran aparte y no eliminan intentos del denominador.

El JSON conserva agrupaciones por configuración y procedencia del código. El total anterior
suma estos ocho intentos de la serie Shadow only; no constituye una comparación causal
entre configuraciones.

Fuente: `recordings/shadow-only-formal-001` a `008`, metadatos de simulación y cámara. El
consolidado se conserva en `resultados-shadow-only-parcial-2026-09-09.json`. Las
grabaciones y los checkpoints son archivos locales y no se incluyen en Git.
