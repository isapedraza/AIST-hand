# grasp-app — GraphGrasp

Real-time perception app for **GraphGrasp: A Framework for Grasp Intent-Driven Teleoperation**.

**Status:** pending implementation

---

## Structure

```
grasp-app/
├── perception/
│   ├── mediapipe_backend.py  ← reference implementation (RGB camera + MediaPipe)
│   └── apertura.py           ← AperturaCalculator
├── main.py                   ← inference loop entry point
└── models/                   ← checkpoints de deploy (no confundir con grasp-model/models/ que tiene checkpoints de entrenamiento/experimentos viejos)
```

---

## Changing the perception backend

This app is sensor-agnostic. The only contract is 21 XYZ landmarks in ToGraph format.

To use a different sensor:
1. Create `perception/my_backend.py`
2. Implement `PerceptionBackend` from `grasp_gcn`
3. Swap `MediaPipeBackend` for your class in `main.py` — one line change

```python
# main.py
from perception.my_backend import MyBackend  # ← change this
backend = MyBackend()                         # ← and this
```

Everything else stays the same.

## Visualization

Visualization is decoupled from the inference loop. `main.py` calls `backend.render(token)` every frame without knowing what will be shown — or whether anything will be shown at all.

Each backend owns its visualization:
- `MediaPipeBackend` → OpenCV window with landmarks, class, confidence, aperture
- A Gradio backend → updates a web UI
- A haptic glove backend → skips visualization entirely (`render()` is a no-op by default)

This means `main.py` stays clean regardless of the sensor or UI framework used.

---

## Dependencies
- `grasp-model` (installed from monorepo root)
- `mediapipe`
- `opencv-python`

---

## Running

From the monorepo root:

```bash
source .venv/bin/activate
python grasp-app/main.py
```

## Camera Selection

Both `main.py` and `mujoco_canonical_demo.py` support selecting the camera source.

Use a camera index:

```bash
python grasp-app/main.py --camera 0
python grasp-app/mujoco_canonical_demo.py --camera 1
```

Use a stream URL, for example with DroidCam:

```bash
python grasp-app/mujoco_canonical_demo.py --camera http://PHONE_IP:4747/video
```

You can also set the default source via environment variable:

```bash
GRAPHGRASP_CAMERA_INDEX=1 python grasp-app/main.py
```

Priority order:
- `--camera` overrides everything
- `GRAPHGRASP_CAMERA_INDEX` is used if `--camera` is not provided
- otherwise the default is `0`

If OpenCV opens a device but no frames arrive, the UI now keeps running and shows a diagnostic message instead of exiting immediately. This is useful when switching between a laptop webcam and DroidCam virtual devices.
