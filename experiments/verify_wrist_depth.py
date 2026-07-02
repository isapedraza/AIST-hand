#!/usr/bin/env python3
"""Compare WiLoR cam_t (weak-perspective) and Depth Anything V2 metric depth
against HOGraspNet ground-truth wrist depth, on the same 11 real frames of a
real reach-grasp-release trial (subject 1, obj_16_grasp_14, trial_0, view=mas).

Question: does either monocular method track the real ~8cm depth excursion of
the wrist during a real reach, or is the signal too weak/noisy to be useful for
arm teleop translation control?
"""

from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import torch

SAMPLE_DIR = (
    "/home/pc_pro/HOGraspNet/data/extracted_sample/"
    "230905_S01_obj_16_grasp_14/trial_0"
)
RGB_DIR = os.path.join(SAMPLE_DIR, "rgb", "mas")
ANN_DIR = os.path.join(SAMPLE_DIR, "annotation", "mas")

DA2_CKPT = (
    "/home/pc_pro/Depth-Anything-V2/metric_depth/checkpoints/"
    "depth_anything_v2_metric_hypersim_vits.pth"
)


def load_frames():
    files = sorted(
        glob.glob(os.path.join(ANN_DIR, "*.json")),
        key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
    )
    frames = []
    for f in files:
        frame_id = int(os.path.basename(f).split("_")[1].split(".")[0])
        d = json.load(open(f, encoding="utf-8-sig"))
        img_path = os.path.join(RGB_DIR, f"mas_{frame_id}.jpg")
        gt_z_cm = d["hand"]["mano_xyz_root"][2]
        fx = d["calibration"]["intrinsic"][0][0]
        fy = d["calibration"]["intrinsic"][1][1]
        cx = d["calibration"]["intrinsic"][0][2]
        cy = d["calibration"]["intrinsic"][1][2]
        # 21 3D keypoints (cam space, cm) -> use for a 2D wrist pixel via projection.
        kp3d = np.asarray(d["hand"]["3D_pose_per_cam"], dtype=np.float64)  # some var name; root separate
        frames.append(
            dict(
                frame_id=frame_id,
                img_path=img_path,
                gt_z_cm=gt_z_cm,
                fx=fx, fy=fy, cx=cx, cy=cy,
                root_xyz_cm=np.asarray(d["hand"]["mano_xyz_root"], dtype=np.float64),
            )
        )
    return frames


def project(xyz_cm, fx, fy, cx, cy):
    x, y, z = xyz_cm
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v


def run_wilor(frames):
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    device = torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)

    results = []
    for fr in frames:
        img = cv2.imread(fr["img_path"])
        outs = pipe.predict(img)
        if not outs:
            results.append(None)
            print(f"frame {fr['frame_id']}: WiLoR found no hand")
            continue
        # largest hand
        best = max(outs, key=lambda o: abs(
            (o["hand_bbox"][2] - o["hand_bbox"][0]) * (o["hand_bbox"][3] - o["hand_bbox"][1])
        ))
        cam_t = np.asarray(best["wilor_preds"]["pred_cam_t_full"], dtype=np.float64).reshape(-1)[:3]
        results.append(cam_t)
        print(f"frame {fr['frame_id']}: cam_t = {cam_t}")
    return results


def run_depth_anything(frames):
    import sys
    sys.path.insert(0, "/home/pc_pro/Depth-Anything-V2/metric_depth")
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**{"encoder": "vits", "features": 64,
                                "out_channels": [48, 96, 192, 384], "max_depth": 20})
    model.load_state_dict(torch.load(DA2_CKPT, map_location="cpu"))
    model.eval()

    results = []
    for fr in frames:
        img = cv2.imread(fr["img_path"])
        depth_m = model.infer_image(img)  # HxW, meters
        u, v = project(fr["root_xyz_cm"], fr["fx"], fr["fy"], fr["cx"], fr["cy"])
        u, v = int(round(u)), int(round(v))
        u = np.clip(u, 0, depth_m.shape[1] - 1)
        v = np.clip(v, 0, depth_m.shape[0] - 1)
        z_pred_m = float(depth_m[v, u])
        results.append(z_pred_m)
        print(f"frame {fr['frame_id']}: DepthAnythingV2 z at wrist px=({u},{v}) -> {z_pred_m:.3f} m "
              f"(GT z={fr['gt_z_cm']/100:.3f} m)")
    return results


def main():
    frames = load_frames()
    print(f"{len(frames)} frames loaded.\n")

    print("=== Ground truth wrist z (cm) ===")
    for fr in frames:
        print(f"frame {fr['frame_id']:>3}: z_gt = {fr['gt_z_cm']:.2f} cm")

    print("\n=== WiLoR cam_t (weak-perspective) ===")
    wilor_res = run_wilor(frames)

    print("\n=== Depth Anything V2 (metric, indoor) ===")
    da2_res = run_depth_anything(frames)

    print("\n=== Summary ===")
    gt = np.array([fr["gt_z_cm"] for fr in frames]) / 100.0  # meters
    print(f"GT z range: {gt.min():.3f} - {gt.max():.3f} m (delta={gt.max()-gt.min():.3f} m)")

    valid_wilor = [r[2] for r in wilor_res if r is not None]
    if valid_wilor:
        wz = np.array(valid_wilor)
        print(f"WiLoR cam_t.z range: {wz.min():.3f} - {wz.max():.3f} (delta={wz.max()-wz.min():.3f}, arbitrary units)")

    da2 = np.array(da2_res)
    print(f"DepthAnythingV2 z range: {da2.min():.3f} - {da2.max():.3f} m (delta={da2.max()-da2.min():.3f} m)")

    # Correlation between each method's delta and GT delta (order-consistent trend check).
    if valid_wilor and len(valid_wilor) == len(gt):
        corr_wilor = np.corrcoef(gt, wz)[0, 1]
        print(f"corr(GT, WiLoR cam_t.z) = {corr_wilor:.3f}")
    corr_da2 = np.corrcoef(gt, da2)[0, 1]
    print(f"corr(GT, DepthAnythingV2 z) = {corr_da2:.3f}")


if __name__ == "__main__":
    main()
