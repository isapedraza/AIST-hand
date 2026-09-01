"""Fig 4.1 (ThesisGCN, Resultados -- Fidelidad del retargeting): un par cualitativo
operador-camara / robot-MuJoCo por tipo de agarre.

Pipeline real de punta a punta (no simulado): foto -> WiLoR (server remoto por
defecto, o WiLoR-mini local con --local) -> Dong kinematics -> Retargeter
(checkpoint multi-robot real) -> render MuJoCo (Shadow + Allegro) -> tira de
3 paneles (foto | Shadow | Allegro), sin recortar ninguno.

Camara calibrada 2026-08-26 (aprobada): palma al frente (azimuth=180,
elevation=0). Shadow trae pedestal de muneca, Allegro no -- por eso su
camara esta mas cerca y mas arriba (si no, el pedestal ausente deja la
mano chica y baja en el cuadro).

Uso (desde AIST-hand/):
    .venv/bin/python models/grasp-intent-classification/scripts/render_fig41_pair.py \\
        <foto.jpg> <nombre-pose> [--url https://xxxx.trycloudflare.com] [--local]

Ejemplo:
    .venv/bin/python models/grasp-intent-classification/scripts/render_fig41_pair.py \\
        ~/Downloads/tripod.jpg tripod --url https://xxxx.trycloudflare.com

Salida: ~/Downloads/fig41-preview-<nombre-pose>.png (tira de preview, revisar
antes de copiar a ThesisGCN/figures/).
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import mujoco
import numpy as np
import requests
import torch
from PIL import Image

ROOT = pathlib.Path("/home/yareeez/AIST-hand")
CKPT = ROOT / "models/latent-retargeting/checkpoints/active/stage1_shadow_allegro_bodex_objbal_15k.pt"
OUT_DIR = pathlib.Path.home() / "Downloads"

SHADOW_JOINT_ORDER = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]
ALLEGRO_JOINT_ORDER = [
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
]

# Camara calibrada y aprobada (palma al frente, mano completa en cuadro).
SHADOW_CAM = dict(azimuth=180.0, elevation=0.0, distance=0.40, lookat=np.array([0.0, -0.01, 0.30]))
ALLEGRO_CAM = dict(azimuth=180.0, elevation=0.0, distance=0.36, lookat=np.array([0.02, 0.02, 0.05]))

SCENE_XML = """<mujoco model="fig41_scene">
  <include file="{inc}"/>
  <statistic extent="0.3" center="0 0 0.15"/>
  <visual>
    <rgba haze="1 1 1 1"/>
    <quality shadowsize="8192"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/>
    <texture type="2d" name="groundplane" builtin="flat" rgb1="1 1 1" width="32" height="32"/>
    <material name="groundplane" texture="groundplane" texuniform="true" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light pos="0 0 1" diffuse="0.6 0.6 0.6"/>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" pos="0 0 -0.02" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def get_keypoints_remote(img_path: str, url: str) -> tuple[np.ndarray, int]:
    with open(img_path, "rb") as f:
        resp = requests.post(f"{url.rstrip('/')}/infer", files={"frame": ("frame.jpg", f, "image/jpeg")}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if "keypoints" not in body:
        raise RuntimeError(f"No hand detected by remote server: {body}")
    return np.array(body["keypoints"], dtype=np.float64), int(body["is_right"])


def get_keypoints_local(img_path: str) -> tuple[np.ndarray, int]:
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cpu"), dtype=torch.float32, verbose=False)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    outs = pipe.predict(img)
    if not outs:
        raise RuntimeError("WiLoR (local) found no hand in the image.")
    best = max(outs, key=lambda o: (o["hand_bbox"][2] - o["hand_bbox"][0]) * (o["hand_bbox"][3] - o["hand_bbox"][1]))
    preds = best["wilor_preds"]
    return preds["pred_keypoints_3d"][0].astype(np.float64), int(round(best["is_right"]))


def photo_to_qpos(img_path: str, url: str | None) -> dict[str, np.ndarray]:
    from human.kinematics.dong_kinematics import DongKinematics, canonicalize_to_right_hand
    from cross_emb.inference.retarget import Retargeter
    from cross_emb.rotations import quat_wxyz_to_rot6d

    kp3d, is_right = get_keypoints_remote(img_path, url) if url else get_keypoints_local(img_path)
    hand_label = "Right" if is_right else "Left"
    points_w, _, _ = canonicalize_to_right_hand(kp3d, hand_label)

    dk = DongKinematics(calibration_frames=1)
    res = dk.process(points_w)
    quats = np.array([res["quaternions"][j] for j in res["joint_order"]], dtype=np.float32)
    r6 = quat_wxyz_to_rot6d(torch.from_numpy(quats).unsqueeze(0))

    qpos = {}
    for robot in ("shadow", "allegro"):
        rt = Retargeter(str(CKPT), robot_name=robot)
        qpos[robot] = rt(r6)[0]
    return qpos


def _render(scene_dir: pathlib.Path, upright_rel: str, joint_order: list[str], qvals: np.ndarray, cam_cfg: dict) -> Image.Image:
    scene_path = scene_dir / ".tmp_fig41_scene.xml"
    if not scene_path.exists():
        scene_path.write_text(SCENE_XML.format(inc=upright_rel))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 900
    model.vis.global_.ipd = 0.0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=900, width=900)
    q = np.zeros(model.nq)
    for name, val in zip(joint_order, qvals):
        q[model.joint(name).qposadr[0]] = val
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance, cam.lookat = (
        cam_cfg["azimuth"], cam_cfg["elevation"], cam_cfg["distance"], cam_cfg["lookat"],
    )
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = Image.fromarray(renderer.render().copy())
    renderer.close()
    return img


def render_shadow(qvals: np.ndarray) -> Image.Image:
    hand_dir = ROOT / "third_party/mujoco_menagerie/shadow_hand"
    upright = hand_dir / ".tmp_fig41_upright.xml"
    if not upright.exists():
        orig_tag = '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">'
        upright_tag = '<body name="rh_forearm" childclass="right_hand" pos="0 0 0.05" quat="1 0 0 0">'
        upright.write_text((hand_dir / "right_hand.xml").read_text().replace(orig_tag, upright_tag, 1))
    return _render(hand_dir, upright.name, SHADOW_JOINT_ORDER, qvals, SHADOW_CAM)


def render_allegro(qvals: np.ndarray) -> Image.Image:
    hand_dir = ROOT / "third_party/mujoco_menagerie/wonik_allegro"
    return _render(hand_dir, ".right_hand_upright.xml", ALLEGRO_JOINT_ORDER, qvals, ALLEGRO_CAM)


def fit_panel(img: Image.Image, size: int = 900, bg: str = "white") -> Image.Image:
    """Resize preserving aspect ratio to fit within size x size, pad with bg. Never crops."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", help="path to the operator's photo")
    ap.add_argument("pose_name", help="short label, e.g. tripod / pinza-lateral / pinza-yema / abduccion")
    ap.add_argument("--url", default=None, help="WiLoR remote server URL (trycloudflare); omit with --local")
    ap.add_argument("--local", action="store_true", help="run WiLoR-mini locally on CPU instead of --url")
    args = ap.parse_args()
    if not args.url and not args.local:
        raise SystemExit("pass --url <server> or --local")

    qpos = photo_to_qpos(args.photo, args.url)

    photo = fit_panel(Image.open(args.photo).convert("RGB"))
    shadow = fit_panel(render_shadow(qpos["shadow"]))
    allegro = fit_panel(render_allegro(qpos["allegro"]))

    strip = Image.new("RGB", (2700, 900), "white")
    strip.paste(photo, (0, 0))
    strip.paste(shadow, (900, 0))
    strip.paste(allegro, (1800, 0))

    out_path = OUT_DIR / f"fig41-preview-{args.pose_name}.png"
    strip.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
