# Deploy Experiment Protocol

## Objetivo

Evaluar el comportamiento del sistema en deploy bajo distintos estimadores de pose
y distintos conjuntos de features. Cada experimento aísla una variable para entender
qué parte del pipeline cierra el domain gap.

## Bateria de experimentos

| Exp | Estimador  | Features GCN              | Estado     |
|-----|------------|---------------------------|------------|
| 1   | MediaPipe  | xyz                       | Completado |
| 2   | HaMeR      | xyz                       | Pendiente  |
| 3   | HaMeR      | xyz + bone vectors        | Pendiente  |
| 4   | HaMeR      | xyz + bone + velocity     | Pendiente  |
| 5   | HaMeR      | xyz + bone + vel + pose   | Pendiente  |

## Condiciones de captura

- Viewpoints: exocentrico (webcam frontal) y near-egocentrico (smartphone tripod ~ojo)
- Condiciones: air (sin objeto) y object (con objeto)
- Duracion por fase: 5s (MediaPipe, ~150 frames) | 20s (HaMeR, ~66 responses)
- Countdown: 3 segundos antes de cada captura
- Takes: 1 por fase (con opcion de retry)

## Metrica

Frame accuracy por clase x condicion x viewpoint:

    accuracy = predicciones_correctas / total_observaciones

Donde cada observacion es una prediccion independiente del estimador
(frame para MediaPipe, response de HaMeR para HaMeR).

Justificacion estadistica: Cochran (1952) -- celdas esperadas >= 5 para
cualquier accuracy entre 8% y 92% con n=66. Cohen (1988) -- 80% poder
estadistico para detectar diferencias >= 25pp con n=64.

## Objetos (condicion object)

Cada objeto se usa para los agarres Feix indicados. Mantener los mismos
objetos en todos los experimentos 2-5.

| Objeto           | Agarres Feix                                        |
|------------------|-----------------------------------------------------|
| Bola de navidad  | Power Sphere, Sphere 3-Finger, Extension Type, Precision Sphere |
| Tijeras          | Distal                                              |
| Lapiz/pluma      | Writing Tripod, Light Tool, Adducted Thumb, Stick   |
| Tapa de cafe     | Power Disk, Precision Disk, Palmar                  |
| Palillo          | Tip Pinch, Inferior Pincer                          |
| Libro            | Parallel Extension                                  |
| Tarjeta          | Lateral, Adduction Grip, Palmar Pinch               |
| Goma/borrador    | Lateral Tripod                                      |
| Termo            | Large Diameter, Ring                                |
| Rollo plastico   | Medium Wrap                                         |
| Pritt            | Small Diameter                                      |

## Scripts

- Exp 1 (MediaPipe): `grasp-app/domain_gap_experiment.py`
- Exp 2-5 (HaMeR):   `grasp-app/domain_gap_experiment_hamer.py --url <cloudflare_url>`

## Sesiones completadas

### Exp 1 -- MediaPipe, xyz
- Sesion exocentrica:     `experiment/20260311_145054/`
- Sesion near-egocentrica: `experiment/20260311_155647/`
