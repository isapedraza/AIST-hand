# Metodología — frases guía por sección

Frase guía = oración ancla que fija el contenido de cada sección. Escritas en
**N-robot/general** para que Allegro (u otro robot) entre después sin reescribir.
Encuadre: Yan & Lee (2025, 2026) — espacio latente compartido cross-embodiment —
adaptado de cuerpos humanoides a **manos diestras** (5 dedos como segmentos).

Diagrama de referencia: `docs/assets/training_pipeline.mmd`.

---

## Marco metodológico
> Adoptamos el paradigma de espacio latente compartido aprendido por contraste de
> Yan & Lee (2025, 2026), originalmente formulado para cuerpos humanoides, y lo
> adaptamos al dominio de manos diestras tratando los cinco dedos como segmentos
> desacoplados.

## Diseño general del sistema
> El sistema retargetea poses de mano humana a configuraciones articulares de
> robot (f: pose_h → q_r) mediante un espacio latente común de N embodiments,
> entrenado sin datos pareados.

## Datos humanos
> Las poses humanas provienen de HOGraspNet (ablación 14, representación R6)
> aumentadas con 10% de HaGRID, preprocesadas a 20 articulaciones estilo Dong.

## Validación preliminar de la representación mediante grafos
> _(Pendiente — clasificador clásico previo; no cubierto por el pipeline de retargeting.)_

## Representación de poses
> La pose humana se representa en rotaciones continuas R6 por articulación; la del
> robot, como vector de ángulos articulares (qpos); ambas se proyectan a un vector
> canónico UDHM de 22 ángulos con nombre para la métrica cross-embodiment.

## Generación de poses de robot
> Las poses válidas de robot se obtienen muestreando el espacio de eigengrasps,
> filtrando colisiones en MuJoCo y precomputando offline (equivalente a
> EIGENGRASP_ONLINE) para evitar el coste por step.

## Arquitectura del espacio latente cross-embodiment
> La arquitectura comprende un encoder humano basado en GNN con atención contextual
> (CAM), encoders/embeddings ligeros por robot (E_r/D_r) y un encoder-decoder
> compartido (E_X/D_X) que define un latente desacoplado en cinco subespacios por dedo.

## Métrica de similitud y función objetivo
> El espacio se estructura por aprendizaje contrastivo guiado por una métrica de
> similitud cross-embodiment (Xin: tip/pinch + D_R adaptativo / UDHM), combinada con
> pérdidas de reconstrucción, consistencia cíclica latente y consistencia temporal.

## Procedimiento de entrenamiento
> El latente compartido se entrena conjuntamente sobre los embodiments disponibles;
> robots adicionales se incorporan congelando las redes compartidas y entrenando
> únicamente su embedding E_r/D_r.

## Pipeline de inferencia y deploy
> En inferencia solo se requieren el encoder humano y el decoder del robot objetivo:
> una pose humana estimada se codifica al latente y se decodifica directamente a
> qpos ejecutable.

---

## Cobertura vs diagrama

| Sección | Pieza del diagrama |
|---|---|
| Diseño general | pipeline completo |
| Datos humanos | nodo Human dataset (HOGraspNet + HaGRID) |
| Representación de poses | aristas pose_h R6 / q_r qpos + UDHM-22 |
| Generación poses robot | subgrafo OFFLINE (eigengrasp → MuJoCo → FK/Dong → npz) |
| Arquitectura latente | E_h (CAM-GNN), E_r, E_X/D_X, D_r, latente 5-subespacios |
| Métrica + objetivo | L_cont(xin S_k), L_rec, L_ltc, L_temp → Adam |
| Procedimiento entrenamiento | Adam + freeze-add (resume 18B, freeze shared) |
| Inferencia/deploy | E_h + D_X + D_r (subconjunto; flujo no dibujado aún) |

**Marco metodológico** y **Validación preliminar** no salen del diagrama
(encuadre teórico y clasificador previo, respectivamente).
