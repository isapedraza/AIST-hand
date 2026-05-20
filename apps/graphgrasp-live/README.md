# graphgrasp-live

Live GraphGrasp entrypoints. This app composes the pipeline domains:

```text
human.perception -> grasp_gcn model -> control policy -> render/robot adapter
```

Run from the repository root:

```bash
python apps/graphgrasp-live/main.py --camera 0
```

Camera source can be a numeric OpenCV device index or a stream URL:

```bash
python apps/graphgrasp-live/main.py --camera 1
python apps/graphgrasp-live/mujoco_canonical_demo.py --camera http://PHONE_IP:4747/video
```

Perception backends live in `human/perception/`. Dong and hand preprocessing
utilities live in `human/kinematics/`. Deploy checkpoints for this app remain
under `apps/graphgrasp-live/models/`.
