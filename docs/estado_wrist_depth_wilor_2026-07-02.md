# Estado — profundidad de muñeca (WiLoR) verificada con datos reales

**Fecha:** 2026-07-02

**Contexto:** control de brazo vía webcam monocular (`--emit-wrist`, UDP 5012) resultó
"demasiado problema" en la práctica — el brazo apenas reaccionaba al reach (adelante/
atrás), mientras la orientación de la mano sí funcionaba bien. Esta sesión diagnosticó
la causa raíz con datos reales (no especulación) y aplicó un primer fix.

---

## Diagnóstico

Dos ejes con sensibilidad a escala completamente distinta en `wrist_target()`
(`dexjoco/shadow_ext/teleop_driver.py`):

- **Orientación (`dR`):** rotación pura, unit-free — no depende de la escala de
  profundidad de WiLoR. Por eso funciona bien.
- **Traslación (`dt = pose_scale * delta[:3,3]`):** depende de `cam_t.z` de WiLoR
  (weak-perspective, derivado del tamaño aparente de la mano en píxeles). El gain
  `_WRIST_POSE_SCALE = 1.0` estaba copiado de la calibración VIVE (tracker físico en
  metros reales) y reusado sin verificar para WiLoR, cuyas unidades de `cam_t` **no
  son metros** salvo que se le pase el focal length real de la cámara.

## Verificación con datos reales (no simulados)

Se clonó `kaist-uvr-lab/HOGraspNet` (`/home/pc_pro/HOGraspNet`), se bajó 1 sujeto
(S1, Source_data + Labeling_data) y se extrajo 1 trial completo (reach → agarre →
soltar), 11 frames con ground-truth 3D real + intrínsecos de cámara reales (Kinect
Azure, fx=942.4px):

- **GT real:** el reach mueve la muñeca ~8.3cm en profundidad (71.6cm → 63.2cm).
- **WiLoR `cam_t.z` (focal default=5000, wilor_mini):** correlación con GT **r=0.981**
  — la señal de profundidad SÍ es buena, pero en unidades arbitrarias, no metros.
- **WiLoR `cam_t.z` (focal length real pasado al constructor):** mismas unidades que
  metros reales (~0.68-0.79m vs GT 0.63-0.72m), r=0.981 igual. Ajuste lineal fino
  (`gt = 0.8116*pred + 0.087`) da RMSE 0.66cm.
- **Depth Anything V2 (metric, indoor/Hypersim, Small):** **descartado**. r=-0.218
  (básicamente ruido), y error de escala absoluta ~2x (da 1.16-1.37m cuando la
  realidad es 0.63-0.72m). Entrenado en escenas de habitación completa (Hypersim),
  no generaliza a close-range de una sola mano.

**Conclusión:** MANO (Romero et al. 2017, calibrado con ~1000 escaneos reales,
0.2mm RMS) da a WiLoR una "regla" en mm reales vía el parámetro de forma `β`,
regresado junto con pose y cámara por frame — por eso el tracking de profundidad es
bueno, no es casualidad ni "magia de WiLoR". No hace falta modelo de profundidad
externo ni multi-cámara (DexPilot/RGB-D) para este caso de uso: la señal ya es buena,
solo estaba mal alimentada y sin calibrar.

## Fix aplicado (ya cableado, no depende de tener la webcam)

`servers/wilor_colab_server.ipynb`, Cell 2: se pasa `focal_length=179.4` al
constructor de `WiLorHandPose3dEstimationPipeline`, calculado desde specs de la
Dell Inspiron 15 3511 (webcam integrada, 1280x720, FOV diagonal 78.6°):

```
diag_px = sqrt(1280**2 + 720**2) = 1468.6
fx_px   = (diag_px/2) / tan(78.6deg/2) ≈ 897 px
focal_length_arg = fx_px * 256/max(1280,720) ≈ 179.4
```

**Esto es un valor ESTIMADO por specs, no medido.** Corrige la mayor parte del error
(unidades pasan de arbitrarias a metros reales), pero el ajuste lineal muestra que
todavía queda un **error de escala real de ~19-28%** (`scale=0.8116` en el fit, no
1.0) — esto NO se cancela con el control delta/ancla (`tracker_start` en
`teleop_driver.py`), porque es multiplicativo, no un offset constante.

## Pendiente — validación en casa (necesita la webcam real)

1. Con la webcam real conectada: 1 foto con objeto de ancho conocido a distancia
   medida (la idea del "brazo estirado + algo sostenido al pecho" sirve). Da 2 cosas
   a la vez:
   - `focal_length` medido real, para reemplazar la estimación por specs (179.4) si
     difiere.
   - El punto de calibración para `_WRIST_POSE_SCALE` en
     `dexjoco/shadow_ext/teleop_driver.py:60` (hoy en 1.0, necesita bajar ~20-30%
     según el fit de hoy, pero debe remedirse con la cámara real, no asumir el
     mismo número que salió con la Kinect de HOGraspNet).
2. Repetir la cadena de 3 procesos de `docs/estado_pendientes_casa_2026-07-01.md`
   (WiLoR Colab + `live_retarget.py --emit-wrist` + `teleop_driver.py --recv-wrist`)
   con el gain corregido, confirmar que el reach ahora sí desplaza el brazo en sim.

## Caveat honesto

Verificado con 1 sujeto, 1 trial, cámara de estudio (Kinect, iluminación controlada)
— no con la webcam Dell real ni la iluminación de casa. El mecanismo y la dirección
del fix están confirmados con datos reales, pero el número final de
`_WRIST_POSE_SCALE` hay que remedirlo en vivo, no asumir que el 0.81 de hoy
traslada 1:1 a la cámara real.

## Herramientas/datos nuevos en este equipo (no en AIST-hand, repos separados)

- `/home/pc_pro/HOGraspNet` — clon de `kaist-uvr-lab/HOGraspNet`, con 1 sujeto (S1)
  descargado en `data/zipped/` (~5.4GB) y 1 trial extraído en
  `data/extracted_sample/`.
- `/home/pc_pro/Depth-Anything-V2` — clon oficial + checkpoint metric-indoor-small
  (99MB) en `metric_depth/checkpoints/`. Ya evaluado y descartado para este caso de
  uso (ver arriba); se deja clonado por si sirve para otra cosa más adelante.
- `wilor-mini` instalado en `.venv` de AIST-hand (antes solo corría en Colab) —
  permite correr WiLoR local en CPU para verificación offline sin depender de vivo.

Script de verificación: `experiments/verify_wrist_depth.py`.
