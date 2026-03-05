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
└── models/                   ← place best_model.pth here (download from Releases)
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
