# Evaluación cinemática: RS/NDS/NVS — Reporte experimental (2026-07-01)

## Qué se hizo

Se corrió evaluación offline sobre el conjunto de prueba de HOGraspNet (sujetos 74–99, sin overlap con entrenamiento) usando el script `models/latent-retargeting/scripts/eval_retarget.py`. Para cada checkpoint, el modelo toma poses humanas del test split, las retargetea al robot via `E_h → D_X → D_r`, corre FK sobre las poses robot resultantes, y compara las dos posturas en el espacio cinemático compartido. Se usaron 10 batches de 3000 muestras cada uno (30k muestras totales).

CSV: `hograspnet_abl14_r6.csv`, representación r6 (6D continua).

---

## Métricas — definición y origen

Las tres métricas siguen a Yan et al. (2026), adaptadas al dominio de manos.

### RS — Rotation Similarity

```
RS = Σ_j  (1 - cos θ_j) / 2
```

Por joint j, `(1 - cos θ) / 2` vale 0 cuando las orientaciones coinciden y 1 cuando difieren 180°. Se suma sobre todos los joints compartidos entre human y robot.

**Espacio de comparación**: los joints no se comparan directamente (Shadow joint_1 ≠ Allegro joint_0.0 ≠ humano landmark). En su lugar, ambas posturas se proyectan al espacio Dong — ángulos por segmento (mcp_flex, pip_flex, dip_flex, mcp_abd por dedo) en el marco local del carpo. Esto permite comparar cross-embodiment. Es la adaptación de Yan al dominio de manos: Yan compara joints de humanoides que tienen correspondencia directa (codo humano = codo robot); aquí no existe esa correspondencia directa, y Dong la construye.

**Máximo teórico**: N joints compartidos (todos a 180°). Shadow comparte ~20 slots Dong, Allegro ~16.

### NDS — Normalized Distance Similarity

```
NDS = Σ_{dedos} Σ_{segmentos} || p_humano - p_robot ||
```

Posiciones cartesianas de la cadena de dedo (MCP → PIP → DIP → TIP) en el marco del carpo, normalizadas por `hand_length`. No es espacio Dong — es FK cartesiano directo. Mide si los links del dedo quedan en el mismo lugar relativo, independientemente de la morfología (la normalización elimina diferencias de escala pero no de forma).

### NVS — Normalized Velocity Similarity

```
NVS = || v_tip_humano - v_tip_robot ||
```

Velocidad de las yemas entre frames consecutivos. Mide coherencia temporal del retargeting: que el robot acelere y desacelere cuando el humano lo hace, no solo que acierte la postura instantánea.

---

## Resultados

| modelo | robot | RS | NDS | NVS | rec |
|---|---|---|---|---|---|
| Shadow solo (ckpt17 = run18b, 13.7k steps) | shadow | 0.2054 | 1.9300 | 0.0212 | 0.9327 |
| Allegro solo (bodex_objbal, 15k steps) | allegro | 0.7420 | 3.7528 | 0.0241 | 1.2529 |
| Allegro solo (ckpt24, 14.5k steps) | allegro | 0.6839 | 3.8437 | 0.0235 | 1.2474 |
| Multi-robot (shadow+allegro, 15k steps) | shadow | 0.3394 | 2.3781 | 0.0229 | 0.9248 |
| Multi-robot (shadow+allegro, 15k steps) | allegro | 0.7110 | 3.7262 | 0.0261 | 1.0935 |

`rec` = error de reconstrucción del robot-AE (E_r → E_X → D_X → D_r sobre poses robot reales). Mide calidad del espacio latente robot, independiente de la percepción humana.

---

## Lectura de los números

**Shadow solo vs multi-robot (shadow)**:
- RS: 0.2054 → 0.3394 (+65%)
- NDS: 1.9300 → 2.3781 (+23%)

El latente compartido con Allegro degrada Shadow. El costo es real pero moderado — cualitativamente Shadow en el modelo multi-robot sigue siendo funcional.

**Allegro solo vs multi-robot (allegro)**:
- RS: 0.7420 → 0.7110 (leve mejora)
- NDS: 3.7528 → 3.7262 (leve mejora)

Allegro en el modelo conjunto es marginalmente mejor o igual. Posible: el latente compartido con Shadow exporta información útil al decoder de Allegro.

**Shadow vs Allegro (mismo modelo multi)**:
Shadow RS=0.34 vs Allegro RS=0.71. Shadow NDS=2.38 vs Allegro NDS=3.73.

La diferencia refleja distancia morfológica al humano, no fallo del sistema. Shadow Hand es antropomórfica (5 dedos, proporciones similares); Allegro es industrial (4 dedos, palma y cadenas distintas). El espacio Dong compartido mapea mejor cuando la morfología es más cercana.

**ckpt24 vs bodex_objbal (allegro solo)**:
ckpt24 mejora RS (0.684 vs 0.742) pero empeora NDS (3.844 vs 3.753). Ninguno domina al otro en ambas métricas.

---

## Limitaciones del protocolo

**No hay baseline de comparación.** RS/NDS/NVS son métricas relativas útiles para comparar configuraciones entre sí (solo vs multi-robot, o distintos hyperparams). Sin un método de referencia (IK clásico, retarget naive) que corra sobre el mismo test set, los números absolutos no dicen si el retargeting es "bueno" o "malo" en sentido absoluto.

**No hay ground truth de retargeting.** La comparación es humano vs robot retargeteado en espacio Dong. No existe una pose robot "correcta" para cada pose humana — ese es el problema cross-embodiment. El criterio de calidad es necesariamente proxy.

**RS en manos ≠ RS en Yan.** Yan compara joints de humanoides con correspondencia directa. Aquí la correspondencia se construye via Dong. Los valores no son comparables numéricamente con la tabla de Yan.

**Evaluación cualitativa y funcional son primarias.** Los números de RS/NDS/NVS confirman tendencias (Shadow > Allegro, solo ≈ multi) pero no predicen calidad de teleop. El viewer y DexJoCo son la evidencia de calidad real.

---

## Pendientes

- Ablación formal: variar lambdas (`lam_udhm`, `lam_tip`, `lam_pinch`) y reportar RS/NDS por configuración sobre el mismo test set.
- Evaluar ckpts resultantes de runs con UDHM uniflex y MCP boost (en entrenamiento ahora).
- Definir baseline naive (IK directo fingertip) para anclar los números absolutos.
