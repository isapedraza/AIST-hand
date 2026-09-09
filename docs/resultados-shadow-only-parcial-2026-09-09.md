# Resultados parciales Shadow only — 2026-09-09

Corte: primeros siete intentos terminados de una serie de diez. No es el resultado final.

Checkpoint: `stage1_shadow_solo_ckpt17_run18b_13k7.pt`. Tarea: `pick_bucket`; operador: `yahel`.

**5/7 = 71.4% de éxito; 2 fallos.**

| Intento | Resultado | Tiempo real hasta éxito |
|---|---|---:|
| shadow-only-formal-001 | Éxito | 184.8 s |
| shadow-only-formal-002 | Fallo: interrupción manual | — |
| shadow-only-formal-003 | Éxito | 235.0 s |
| shadow-only-formal-004 | Éxito | 190.6 s |
| shadow-only-formal-005 | Éxito | 172.6 s |
| shadow-only-formal-006 | Éxito | 198.8 s |
| shadow-only-formal-007 | Fallo: interrupción manual | — |

Mediana entre los éxitos: 190.6 s. Rango: 172.6–235.0 s.

Las interrupciones manuales de pruebas formales cuentan como fallos por la regla declarada por el operador. La 002 carece de marca F5: cuenta como fallo, conserva esa incidencia y no se le infiere un tiempo de evaluación. Los problemas de video se registran aparte y no eliminan intentos del denominador.

El JSON conserva agrupaciones por configuración y procedencia del código. El total anterior suma estos siete intentos de la serie Shadow only; no constituye una comparación causal entre configuraciones.

Fuente: `recordings/shadow-only-formal-001` a `007`, metadatos de simulación y cámara. El consolidado se conserva en `resultados-shadow-only-parcial-2026-09-09.json`. Las grabaciones y los checkpoints son archivos locales y no se incluyen en Git.
