# Fuerzas de contacto y prueba de concepto bucket — 2026-09-08

Alcance acordado: sólo bucket con Shadow y Allegro. Las sesiones actuales son prácticas/pruebas de concepto, no la serie final.

**Recomendación:** registrar fuerzas de contacto puede aportar diagnóstico a la justificación física del artículo/tesis, pero no es un requisito general de publicación ni una condición previa para probar Allegro. La recomendación anterior de añadirlas antes de seguir era demasiado fuerte para esta fase.

Evidencia consultada:

- [AnyTeleop, RSS 2023, sección VI y tabla IV](https://www.roboticsproceedings.org/rss19/p015.pdf): evalúa teleoperación real mediante éxito en tareas con XArm6 y Allegro; reporta diez intentos por tarea. Es un precedente de evaluación funcional centrada en resultados. Sus repeticiones no son una cuota que deba imponerse a nuestra práctica.
- [DexJoCo, preprint 2026, secciones 3–5 y tablas 2–3](https://arxiv.org/html/2605.16257v1): presenta estados/poses/comandos, condiciones estructuradas de éxito y evaluación mediante tasas de éxito. También estudia variaciones de dinámica. En discusión identifica información de contacto ausente en políticas visuales y deja mayor realismo sim-to-real como trabajo futuro. La importancia de contacto para una política no equivale a exigir un log de fuerzas en toda prueba de concepto.
- [ReForce, preprint agosto 2026](https://arxiv.org/abs/2608.15560): su contribución explícita es retargeting consciente de fuerzas; por ello informa error de seguimiento de fuerza e interacción de múltiples dedos. Es un ejemplo donde esos datos responden directamente a la afirmación científica. No es una norma editorial general.

La conclusión es una recomendación metodológica para nuestro alcance, no una exigencia declarada por estos artículos. Ninguno de estos ejemplos garantiza aceptación de otro artículo.

Para la prueba de concepto actual sirven: resultado del bucket, tiempo desde F5, medidas de elevación/contención, estados y comandos, intervenciones y configuración reproducible. La toma Shadow ya comprobada aporta esa evidencia; sigue Allegro.

Para afirmaciones específicas sobre carga, fricción, saturación o realismo, conviene añadir posteriormente fuerzas/torques de contacto y actuación como diagnóstico. Las fuerzas simuladas no son mediciones físicas reales. Su registro directo tampoco valida por sí mismo masas, materiales, actuadores o montaje.

Estado del software: no se añadió logging directo de fuerzas en esta revisión. Los estados actuales permiten reconstrucción, que debe identificarse como tal. La prueba Allegro continúa con el mismo esquema de registro de práctica.

[Comandos de prueba Allegro](prueba-concepto-allegro-bucket-2026-09-08.md).
