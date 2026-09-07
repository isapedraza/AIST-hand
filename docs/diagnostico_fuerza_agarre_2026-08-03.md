# Diagnóstico — por qué falla el agarre en sim (2026-08-03)

Continúa de `estado_grasp_slip_manual_control_2026-07-06.md`. Fix de fricción
(`condim=4`, cono elíptico) aplicado y confirmado insuficiente: control manual
(pose canónica Feix, sin pasar por el modelo/retargeter) también falla en
agarrar correctamente.

## Descarte de hipótesis por evidencia, no por intuición

Tres hipótesis en juego: (1) falta módulo de control extra, (2) la forma que
produce el modelo es incoherente, (3) fuerza de agarre insuficiente.

**(2) descartada por el propio test manual.** El control manual usa pose
canónica de clase Feix ("Parallel Extension"), sin retargeter ni checkpoint de
la tesis. Si el problema fuera la postura que produce el modelo, una pose hecha
a mano debería haber sostenido el objeto. No lo hizo. La postura no es la causa.

**(3) confirmada por el mecanismo de actuación.** Dedos = actuadores de
posición MuJoCo puros: `<position kp="2"/>`
(`dexjoco/dexjoco/sim/envs/xmls/panda_allegro_copy.xml:56`), sin `forcerange`.
Par = `kp * (target − actual)`. Si el dedo ya está en el ángulo comandado, el
error ≈0 y el par de sujeción ≈0: es posible alcanzar el ángulo correcto con
fuerza de sujeción insuficiente para resistir perturbación externa. Con
`kp` bajo se requiere más desplazamiento para generar par de reacción
significativo — coincide con el patrón observado: agarre estático aguanta (sin
perturbación), se desarma al mover el brazo (inercia desplaza el dedo, la
respuesta del PD es lenta/débil).

**(1) matizada: no falta OSC (existe, `dexjoco/dexjoco/sim/controllers/opspace.py`,
usado para el brazo desde 2026-07-02) ni transmisión de muñeca en vivo (UDP
5012, `live_retarget.py --emit-wrist`, desde 2026-06-26). Lo que falta
específicamente es una capa de fuerza/impedancia en los DEDOS**, no en el
brazo. El brazo sí tiene control operacional; los dedos son PD de posición
desnudo sin modulación de fuerza reactiva al contacto.

## Literatura: el hueco es conocido, no idiosincrático

- **DexPilot** (Handa et al. 2020): manda ángulos retargeted del Allegro a un
  controlador de articulación a **nivel de torque** (impedance controller), no
  PD de posición simple; brazo vía RMPs → impedance controller a 200Hz. Aun
  así, en su propia sección de limitaciones: *"Grasp and manipulation control
  algorithms could be implemented on the hand that automate force modulating
  control to reduce burden on the user and minimizing unintentional part drops
  from the application of incorrect grip forces."* — el baseline más maduro
  del campo reconoce la modulación de fuerza como trabajo futuro abierto, no
  como algo resuelto de forma estándar.
- **AnyTeleop** (Qin et al. 2024): mismo patrón, ejecución final vía
  "impedance controllers" sobre el robot (real o simulado), no PD de posición
  desnudo.

Conclusión: el gap identificado (fuerza/impedancia de dedos) es el mismo que
el campo deja abierto en sus sistemas de referencia. No es un fallo específico
del checkpoint ni de la representación latente de la tesis.

## Medición pendiente para confirmar cuantitativamente

Loggear `qpos` objetivo vs. `qpos` real de los dedos durante la fase de lift.
Si el real se aleja del target bajo carga (dedos empujados a abrirse por el
torque del objeto), confirma que el PD no genera suficiente par de reacción.
Si el real sigue fielmente al target y aun así hay resbalón, el problema
residual es puramente de contacto/solver, no de ganancia de actuador.

## Próximo paso (si se decide seguir esta vía)

Opciones, de menor a mayor esfuerzo:
1. Subir `kp` de los actuadores de dedo (riesgo: oscilación/inestabilidad,
   ya descartado antes como "no estándar" sin medir primero — medir antes de
   tocar).
2. Reemplazar `<position kp=...>` por controlador de impedancia explícito por
   articulación (par = kp·error_pos + kd·error_vel con ganancias por fase:
   suaves en approach, rígidas en hold), siguiendo el patrón DexPilot/AnyTeleop.
3. Explicit force/torque feedback closed-loop (fuera de alcance razonable para
   esta tesis; anotar como trabajo futuro).

Decisión de alcance para la tesis: la tabla 2.1 y el capítulo de resultados
deben describir esto como limitación de la capa de ejecución (actuación de
dedos sin modulación de fuerza), no como falla del modelo de retargeting ni de
la representación latente — ambos quedan fuera de causa según la evidencia de
arriba.

## ¿Vale la pena implementar impedancia estilo DexPilot/AnyTeleop?

Depende de qué "uso real" se busque. Para la tesis, el retargeting cross-
embodiment (la contribución central) ya es evaluable sin esto — el eval
cinemático (RS/NDS/NVS/MPJPE, `docs/sim_eval_findings.md`) mide fidelidad de
seguimiento, no éxito de agarre físico, y no depende de esta capa. Implementar
impedancia de dedos es trabajo de integración de sistemas (controlador +
tuning de ganancias por fase), no de la representación latente que es el
aporte de la tesis — desviaría tiempo del capítulo de resultados sin fortalecer
la contribución principal.

Si el objetivo es demo de agarre físico funcional (fuera del alcance mínimo de
la tesis pero deseable como validación cualitativa), sí conviene: es la
diferencia entre "el retargeting es fiel" (ya demostrado) y "el sistema agarra
objetos" (no demostrado, y el gap identificado arriba es la causa concreta,
no un misterio). La opción de menor esfuerzo (subir kp con la medición
qpos target/real primero) es razonable como prueba de concepto rápida antes de
comprometerse a construir un controlador de impedancia completo.

## Actualización (2026-09-06): primer sostenimiento exitoso, Allegro

Se subió `kp` (2→20) y se agregó `kv=2` en `panda_allegro_copy.xml:56`
(`dexjoco-shadow`), opción 1 de la lista de arriba, primera pasada sin
medición previa de qpos target/real. Yahel reporta que logró sostener la
comida por primera vez ("nunca jamás lo había logrado"), atribuido al cambio
de ganancias más una mejor posición del pulgar contra el objeto. Sin capturas
de este intento. Pendiente: repetir con capturas para tener evidencia visual,
y confirmar que no fue un caso aislado (un solo run, no validado en repetición
todavía).
