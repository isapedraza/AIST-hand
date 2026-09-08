# Auditoría de física — Shadow/Panda, 2026-09-08

**Resultado: hay evidencia de agarre mediante dinámica y contacto, pero todavía no se puede defender esta escena como una réplica físicamente validada del montaje real.** La prioridad descubierta es la carga del Panda; después, la masa de los objetos y la calibración del contacto. Cambiar el damping de 4 a 1 no resuelve esas cuestiones: la sesión exitosa ya usaba 1.

Se auditó exclusivamente la física y la actuación del simulador. No se evaluó ni modificó el modelo de retargeting. Tampoco se cambió la configuración de trabajo. Todas las perturbaciones se hicieron en memoria, cargando estados de la grabación.

Referencia: `recordings/shadow-live-check-20260907-224629/sim`, MuJoCo 3.6.0, commit del simulador `3b6db0431ff574708c9574c7ae24c1804faf6641`, 31 649 estados. SHA-256 del MJB verificado por el lector: `ba398303441e04ee8f639443db3504a0293b9094e9b067b0af7b5f0bde70c601`.

Datos completos: [auditoria-fisica-shadow-2026-09-08.json](auditoria-fisica-shadow-2026-09-08.json). Incluyen opciones compiladas, arrays de masas/inercia, geometrías, articulaciones, actuadores, tendones, exclusiones, contactos reconstruidos y resultados de las pruebas. [Script reproducible](../experiments/audit_simulator_physics.py).

**Hallazgo prioritario: montaje y carga.** El constructor adjunta todo el subárbol `rh_forearm` al sitio de montaje del Panda. Ese antebrazo pesa 3 kg; el conjunto Shadow pesa 3.794 kg. El cubo con asa pesa 1.659808 kg y la comida 0.588057 kg. Por tanto:

| Carga soportada en el montaje | Masa |
|---|---:|
| Shadow solo | 3.794 kg |
| Shadow + comida | 4.382057 kg |
| Shadow + cubo cargado | 6.041865 kg |

El manual del fabricante especifica 3 kg de payload para Panda y pares pico repetibles de 87 Nm en articulaciones 1–4 y 12 Nm en 5–7. La masa modelada ya supera la carga nominal con la mano sola. Esto no demuestra que cada postura sea dinámicamente imposible: sí impide justificar el montaje completo como operación dentro de la capacidad nominal del hardware. La simulación no reproduce todas las protecciones, límites térmicos o de funcionamiento continuo. Fuente primaria: [Franka Emika, Panda User Handbook, octubre de 2018, pp. 42–43; copia alojada por Panda Safe HRC](https://panda-safe-hrc.readthedocs.io/en/latest/_downloads/deab6dbdb4210b88c2db1053111209f6/FrankaPandaManual.pdf#page=42).

No se debe borrar o reducir la masa del antebrazo solamente para pasar este chequeo. Primero hay que identificar qué hardware Shadow se pretende montar, qué partes quedan realmente sobre la brida y si existe soporte externo o un brazo diferente. Cualquier corrección debe representar ese montaje y conservar una variante separada para replicar la sesión histórica.

**Inventario y razón de cada familia de parámetros.** “Heredado” describe procedencia; no significa identificado experimentalmente.

| Ajuste efectivo de la sesión | Procedencia / función | Evidencia y decisión |
|---|---|---|
| Gravedad `[0,0,-9.81]` m/s²; gravedad habilitada | Arena terrestre | Caída libre comprobada. Conservar para condiciones terrestres. |
| Objetos con articulaciones libres; `body_gravcomp=0` | Objetos dinámicos | Cero en toda la grabación. Conservar; no hay flotación activa. |
| `qfrc_applied=xfrc_applied=0`; cero equalities | MJB y estados | Ningún empuje externo registrado ni soldadura mano–objeto. Motor del asa también a cero. |
| Masa comida 0.588057 kg; inercia principal `[0.001101589,0.000945313,0.000430859]` kg·m² | Dos cajas de 0.1 kg más masa inferida de malla visual | Contabilidad inesperada confirmada por recompilación. Definir objeto físico de referencia antes de fijar masa/inercia. |
| Cubo 1.62832 kg + asa 0.031488 kg | Geometrías de colisión y densidades/masas del asset DexJoCo | Son masas compiladas, no pesadas en laboratorio. Comprobar objeto y centro de masa. |
| Shadow 3.794 kg; inerciales por eslabón explícitos | Asset Shadow importado de Menagerie | Procedencia trazable; revisar montaje y payload antes de declarar equivalencia física. |
| Geometría comida: dos cajas; envolvente aprox. 56 × 149.4 × 90 mm | Aproximación de colisión del asset | Comparar con medidas del objeto real. La malla visible no determina exactamente el contacto. |
| Asa y cubo: varias cajas de colisión | Aproximación del asset | Revisar grosor del asa y superficies que forman el agarre; afectan soporte geométrico. Arrays en JSON. |
| `cone=elliptic` | Ya presente en el Shadow importado; propagado por el constructor | Mecanismo defendible. No constituye calibración de coeficientes. |
| Mano `condim=4`, fricción `[1,0.02,0.005]` | Modificación local de colisiones plásticas | Deslizamiento + torsión. Medir materiales y parche de contacto; valor uniforme en superficies plásticas. |
| Fricción efectiva mano–objeto: `[1,1,0.02,0.005,0.005]` | Mezcla de propiedades de ambos geoms | Observada en contactos reconstruidos. Con dim 4 los dos componentes de rodadura no actúan. |
| `impratio=10` | Shadow/Menagerie | Reduce deriva tangencial numérica; no es multiplicar por diez el coeficiente de fricción. Provisional, con sensibilidad local registrada. |
| `noslip_iterations=5`, `noslip_tolerance=0` | Opción de la arena, distinta del controlador opcional | Postproceso numérico activo en la sesión. Declararlo; su desactivación conserva ambos agarres durante la prueba local de 1 s. No extrapolar a toda la tarea. |
| Mano `solref=[0.005,1]`, `solimp=[0.5,0.99,0.0001,0.5,2]` | Contactos endurecidos del asset Shadow | Parámetros numéricos de cumplimiento; no equivalen por sí solos a rigidez medida de piel/plástico. |
| Contacto mano–comida `solref=[0.0125,1]`; mano–asa `[0.003,1.5]`; ambos `solimp=[0.7,0.97,0.00055,0.5,2]` | Mezcla efectiva comprobada en contactos | Usar estos valores al describir la interacción, además de los defaults por geom. |
| `refsafe` habilitado | Default MuJoCo; flags de desactivación cero | Para mano–asa el tiempo de referencia de 3 ms se limita a 4 ms con timestep de 2 ms. Reducir timestep cambia también esta rigidez efectiva. |
| `timestep=0.002` s, integrador Euler | Arena de la sesión Shadow | Caída converge al reducir paso; continuaciones a 1 ms muy próximas. Evidencia local, no convergencia global de todas las tareas. |
| Solver Newton, 100 iteraciones máximas, tolerancia `1e-8`; búsqueda 50/`0.01` | Opciones compiladas | Presupuesto numérico razonable como punto de partida; no se midió convergencia de todas las restricciones durante toda la sesión. |
| CCD 35 iteraciones, tolerancia `1e-6`; Jacobiano auto | Opciones compiladas | Mantener explícito para réplica. No valida precisión geométrica de assets. |
| `disableflags=0`, `enableflags=0`; override de contacto desactivado | Defaults compilados | Gravedad, contacto, límites y clamp de controles activos. Opciones `o_*` almacenadas no sustituyen contactos sin override. |
| Damping pasivo Shadow: dedos 0.05, muñeca 0.5; armature 0.0002; frictionloss 0.01 | Heredado del asset, valores por DOF en JSON | Representan resistencia y masa rotacional efectiva. No hay identificación de hardware para justificar exactamente estos números. |
| 20 servos Shadow; ganancias según articulación, `kv=0`; todos con límites de fuerza | Asset Shadow, incluidos tendones J2+J1 | No son actuadores de fuerza infinita. Límites nominales del modelo, no una validación de capacidad continua real. |
| Ejemplos Shadow: WRJ2 kp 10/±10; WRJ1 kp 8/±5; THJ5 kp 0.4/±3; FFJ3 kp 1/±1; FFJ0 kp 0.5/±1 | Parámetros compilados | El último actúa sobre transmisión tendinosa; no interpretar todos los límites como una fuerza normal de pinza. |
| Panda: motores gear 1, gain 1, ctrl limitado a ±87 Nm (1–4), ±12 Nm (5–7) | Modelo Panda | Coinciden con pares pico del manual citado. No bastan para validar payload, calentamiento, velocidad ni respuesta continua. |
| Brazo OSC: posición 400, orientación 200, nullspace 0.5, damping ratio 1 | Driver congelado; controlador actual sin diferencias frente al commit grabado | Razón de amortiguamiento del lazo, distinta del damping pasivo. Valor 1 busca respuesta críticamente amortiguada idealizada; el sistema acoplado con saturación no tiene garantía de esa respuesta. |
| Compensación del brazo mediante `qfrc_bias`; sin caps explícitos de aceleración en OSC | Driver/controlador | Aplica torques a través de actuadores limitados. Supone conocimiento perfecto del modelo; faltan identificación y límites de implementación real. |
| Sin dinámica de activación en servos; comandos ideales a cada paso | Arrays compilados | No modela fielmente retardo, ancho de banda, histéresis, backlash ni dinámica completa de transmisión Shadow. Declarar simplificación. |
| Aire: densidad 0, viscosidad 0, viento 0 | Defaults | Se omiten fuerzas fluidas. Aproximación para manipulación lenta; no usarla para validar objetos ligeros con gran área. Campo magnético no sostiene los objetos. |
| Límites articulares, tendones y exclusiones | Assets | Inventario en JSON; conservación de topología no demuestra fidelidad mecánica o de autocolisiones. |

La documentación oficial explica la torsión de contacto como una aproximación a un parche finito; el coeficiente torsional tiene unidades de longitud. Aquí `0.02` corresponde a una escala de 20 mm y a una cota torsional aislada de 0.02 Nm por newton normal. No es un coeficiente adimensional ni una medición del parche real. Además, varias restricciones de contacto pueden aportar torsión: hay que evaluar el efecto agregado, no solamente un punto. Fuente: [MuJoCo 3.6, contacto](https://mujoco.readthedocs.io/en/3.6.0/computation/index.html#contact).

El cono elíptico, `impratio` y el postproceso no-slip tienen funciones numéricas diferentes. No-slip modifica la resolución de fricción después del solver principal; el controlador de cierre opcional estaba apagado. La legitimidad de la opción no identifica su valor con la realidad. Fuente: [MuJoCo 3.6, solver y prevención de deslizamiento](https://mujoco.readthedocs.io/en/3.6.0/modeling.html#preventing-slip).

Menagerie documenta que añadió actuadores de posición, aumentó `impratio` y endureció contactos durante la conversión del modelo Shadow. Es una procedencia útil, pero esas decisiones de simulación no son identificación de servos reales. En el historial local, desde la importación `11b3a38` hasta HEAD, el XML de Shadow sólo cambió en la clase de colisión plástica para añadir `condim=4` y la fricción citada. Fuente: [README oficial Shadow de Menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/shadow_hand/README.md). El historial local fija la evidencia de nuestra versión; el enlace upstream puede evolucionar.

La mezcla de contactos, la inclusión de geometrías en la inferencia de inercia, el clamp de controles y `refsafe` explican por qué inspeccionar solamente un valor del XML puede ser insuficiente. El JSON incluye el modelo compilado y parámetros de contactos ya mezclados. Fuente: [Referencia MJCF de MuJoCo 3.6](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#option-flag-refsafe).

**Comprobaciones realizadas.**

Los 31 649 estados muestran cero fuerzas externas generalizadas/cartesianas, cero compensación gravitatoria de objetos, cero sesgo de cierre y cero comando del motor del asa. No hay restricciones de igualdad en el MJB. Las traslaciones libres son consistentes con el avance Euler: residuo máximo comida 4.58e-15 m, cubo 1.57e-15 m. Esto descarta saltos de posición translacional no explicados por el estado de velocidad entre muestras, dentro de esa precisión; no certifica por sí solo toda la implementación física.

Se reconstruyeron 318 estados, cada 100 pasos más el último, con `mj_forward`. En 339 contactos mano–objeto: penetración mediana 0.316 mm, percentil 95 de 0.452 mm, máximo 0.541 mm. Fuerza normal mediana 3.35 N, percentil 95 de 7.87 N y máximo 27.13 N por contacto. **Son fuerzas recalculadas a partir del estado, no telemetría original de fuerzas ni máximos garantizados de la sesión completa.** Penetración pequeña es evidencia contra un agarre por interpenetración grande en estas muestras; no valida por sí misma fricción, rigidez o materiales.

En 11/318 estados muestreados el brazo pidió algún torque fuera de ctrlrange, que MuJoCo limita. Ningún actuador Shadow alcanzó el 99.9% de su límite de fuerza en esa muestra. El error máximo brazo–objetivo muestreado fue 72.25 mm. No se debe describir el seguimiento del brazo como perfecto.

La recompilación actual no presenta diferencias frente al MJB grabado en ninguna de las opciones o arrays físicos seleccionados por el script. Esta comparación no es una prueba de identidad byte a byte de todos los assets.

En una copia en memoria, eliminar únicamente la contribución inercial de la geometría visual de comida reduce su masa de 0.588057 a 0.2 kg y su inercia a `[0.000495670,0.000422570,0.000190967]` kg·m². No se aplicó esa modificación a la escena. La masa de 0.2 kg tampoco es automáticamente la correcta: requiere referencia física.

Caída libre de comida durante 0.1 s, colocada sin contactos:

| Paso | Caída simulada | Solución analítica | Velocidad final |
|---|---:|---:|---:|
| 2 ms | 50.031 mm | 49.050 mm | -0.981 m/s |
| 1 ms | 49.5405 mm | 49.050 mm | -0.981 m/s |
| 0.5 ms | 49.29525 mm | 49.050 mm | -0.981 m/s |

Aceleración inicial -9.81 m/s² en los tres casos; cero contactos con comida. El error de posición se reduce con el paso, como cabe esperar del integrador. La caída libre no identifica masa: la aceleración gravitatoria es independiente de ella.

**Sensibilidad local, 18 continuaciones de un segundo.**

Se eligió para cada objeto el estado muestreado de mayor elevación con al menos dos contactos con la mano: comida índice 18100 (36.2 s), cubo índice 31648 (63.296 s). Se mantuvieron los comandos de dedos y el objetivo mocap grabados, recalculando OSC en cada paso. Las perturbaciones son una por una, dejando activado el resto del modelo. En particular no-slip sigue activo al reducir torsión. No son repeticiones completas con operador ni evidencia de robustez ante cambios combinados.

| Variante | Desplazamiento comida relativo a palma | Cambio de altura del cubo | Contactos finales comida / asa |
|---|---:|---:|---:|
| Base | 0.600 mm | +92.34 mm | 2 / 2 |
| No-slip numérico 0 | 1.614 mm | +92.16 mm | 2 / 2 |
| impratio 1 | 0.801 mm | +93.72 mm | 2 / 2 |
| Torsión de mano 0.005 m | 2.269 mm | +88.62 mm | 2 / 2 |
| Fricción de deslizamiento limitada a 0.5 en todos los geoms | 323.389 mm; cae | +90.75 mm | 0 / 2 |
| Paso de 1 ms | 0.605 mm | +92.29 mm | 2 / 2 |
| implicitfast | 0.600 mm | +92.34 mm | 2 / 2 |
| Damping ratio del brazo 4 | 0.597 mm | +78.11 mm | 2 / 4 |
| Orden de abrir mano | 313.955 mm; cae | -157.48 mm; cae | 0 / 0 |

En la prueba de comida con fricción 0.5, ésta desciende 305.40 mm y pierde contacto con la mano. Por tanto la capacidad de sostener esa postura depende materialmente de la fricción de deslizamiento. No implica que 0.5 sea el valor verdadero ni que 1 sea incorrecto; identifica qué parámetro exige mejor evidencia.

El cubo aún estaba en movimiento al detenerse la grabación y su asa está articulada. En la continuación base sigue subiendo mientras el brazo llega al objetivo; el desplazamiento del origen del asa respecto a la palma es 86.75 mm, mezclando movimiento, giro y asentamiento. No se interpreta como 86.75 mm de deslizamiento puro ni como un ensayo estático. Abrir la mano produce descenso y pérdida de contacto en ambos objetos.

Las 18 continuaciones completaron sus pasos sin warnings de MuJoCo. La dependencia dm_robotics emitió el aviso preexistente de NumPy sobre `where` sin `out`. La proximidad entre pasos de 2 y 1 ms es favorable en estas dos condiciones, pero `refsafe` cambia el contacto del asa a 1 ms y se requiere un ensayo controlado adicional para separar integración y rigidez. No se comprobó convergencia global mediante barrido de tolerancia/iteraciones.

**Qué resolver antes de congelar una configuración con pretensión física.**

1. Definir montaje real y carga: Shadow completo o variante, antebrazo y adaptador, brazo portador y objetos. Corregir la representación únicamente contra esas especificaciones. Es el problema prioritario para una afirmación de réplica Panda–Shadow.
2. Fijar masas, dimensiones, centros de masa e inercias desde medidas o especificaciones identificadas. Evitar masa visual incidental cuando se defina explícitamente el objeto.
3. Justificar fricción de deslizamiento por pareja de materiales: medición de inicio de movimiento con carga normal conocida, o rango documentado bajo condiciones comparables. Priorizarlo porque ya apareció sensibilidad funcional.
4. Justificar torsión mediante tamaño/material del parche y resistencia al giro. No asumir que todas las superficies plásticas merecen 20 mm.
5. Comprobar cumplimiento de contacto y dinámica de actuadores mediante ensayos de carga/deflexión y respuesta temporal; validar además los límites del brazo. Si no hay hardware, etiquetar los valores como supuestos y reportar sensibilidad, sin llamarlos calibrados.
6. Tras fijar parámetros físicos, repetir evaluación con condiciones predefinidas, documentar solver/timestep y ampliar sensibilidad numérica a las fases relevantes. Congelar versiones y hashes antes de los ensayos formales.

La redacción defendible hoy es: “Se obtuvo éxito funcional en una simulación de cuerpos rígidos con gravedad, contacto friccional y actuación limitada. Se verificó ausencia de fuerzas externas y compensación gravitatoria de objetos, liberación al abrir la mano y sensibilidad local a parámetros de contacto e integración. La correspondencia con hardware permanece limitada por la carga del montaje y por parámetros de materiales y actuación no identificados.”

No presentar todavía el resultado como predicción cuantitativa de éxito real. Esta limitación deriva de comprobaciones concretas; no de que el cono elíptico o la torsión sean mecanismos ilegítimos.

Para reproducir desde AIST-hand:

```sh
.venv/bin/python experiments/audit_simulator_physics.py \
  --session recordings/shadow-live-check-20260907-224629 \
  --simulator-root /home/yareeez/dexjoco-shadow \
  --output /tmp/auditoria-fisica-shadow.json
```

El script necesita el checkout sibling y la grabación local, verifica versión MuJoCo y checksum del MJB, y guarda un JSON sin modificar archivos de la sesión. El checksum de su propia fuente y del controlador OSC quedan en los resultados. Alcance actual: Shadow/pick_bucket; estos resultados no validan las otras tareas ni Allegro.
