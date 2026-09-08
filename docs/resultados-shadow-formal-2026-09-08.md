# Resultados formales Shadow — 5 pruebas, 2026-09-08

Protocolo: 5 intentos consecutivos, sin práctica intercalada, condition-id
fijo (`shadow-bucket-formal-v1`), sin re-intentos descartados. Ver
`chapters/Metodos.tex` (`subsec:validacion-funcional-banco`) en ThesisGCN
para el criterio completo.

## Número a citar

**3/5 = 60% de éxito.** Interrupción manual cuenta como fallo, igual que
cualquier otro desenlace que no sea éxito — así quedó decidido, sin
excepción por cobertura de cámara incompleta.

| Sesión | Resultado | Tiempo a éxito |
|---|---|---|
| shadow-bucket-eval-001 | Éxito | 229.3 s |
| shadow-bucket-eval-002 | Éxito | 177.4 s |
| shadow-bucket-eval-003 | Fallo (interrupción manual, objeto caído) | — |
| shadow-bucket-eval-004 | Éxito | 151.6 s |
| shadow-bucket-eval-005 | Fallo (interrupción manual) | — |

Mediana tiempo a éxito (de los 3 éxitos): 177.4 s. Rango: 151.6–229.3 s.

## Nota de cobertura de cámara (no afecta el número anterior)

`resultados-shadow-formal-2026-09-08.json` (`shadow_ext.trial`, herramienta
automática) reporta `success_rate: 0.75 (3/4)` porque excluye
`shadow-bucket-eval-005` de su cálculo estricto: la webcam se cerró ~55s
antes que el simulador para esa sesión específica, así que no hay video que
cubra el tramo final. El resultado del intento (fallo) viene de la traza de
física del simulador, no depende del video — la exclusión de la herramienta
es solo sobre evidencia en video complementaria, no sobre si el intento
cuenta. Para el número reportado en la tesis, cuenta como fallo (ver arriba).

Fuente completa (todas las condiciones, hashes de checkpoint/código,
exclusiones por sesión): `resultados-shadow-formal-2026-09-08.json` en este
mismo directorio.
