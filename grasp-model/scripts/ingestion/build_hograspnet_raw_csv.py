"""
build_hograspnet_raw_csv.py
---------------------------
Genera un único CSV RAW desde los zips de anotaciones de HOGraspNet.

Formato de salida (alineado a hograspnet_hamer_s1 pero sin MANO_pose):
  subject_id  : int (1-99)
  date_id     : int (ej. 231023)
  object_id   : int (obj_XX)
  grasp_type  : int (ID FEIX original del dataset, grasp_XX)
  trial_id    : int (trial_X)
  cam         : str (mas|sub1|sub2|sub3)
  frame_id    : int
  WRIST_x ... PINKY_TIP_z : float (63 columnas XYZ RAW, sin normalización)

Notas:
  - XYZ se toma tal cual del JSON: anno["hand"]["3D_pose_per_cam"].
  - NO se aplica root-relative, escala, mirror ni post-procesamiento.
  - El split S1 y filtros de contacto NO se aplican aquí.

Uso:
  python scripts/ingestion/build_hograspnet_raw_csv.py \
      --zips_dir ~/HOGraspNet/data/zipped/annotations \
      --out      data/raw/hograspnet_raw.csv
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

# ── Joints (orden MediaPipe / HOGraspNet) ─────────────────────────────────────
JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HEADER = (
    ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam", "frame_id"]
    + [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]
)


CAM_ORDER = {"mas": 0, "sub1": 1, "sub2": 2, "sub3": 3}
SEQ_PATTERN = re.compile(r"^(\d+)_S(\d+)_obj_(\d+)_grasp_(\d+)$")
TRIAL_PATTERN = re.compile(r"^trial_(\d+)$")
FRAME_PATTERN = re.compile(r"_(\d+)\.json$")


def _parse_int_suffix(text: str, pattern: str, default: int = 10**9) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def _parse_path_metadata(jname: str):
    """
    Parse HOGraspNet annotation path:
      <date>_S<subject>_obj_<object>_grasp_<grasp>/trial_<id>/annotation/<cam>/<cam>_<frame>.json
    """
    parts = jname.split("/")
    if len(parts) < 5:
        return None

    seq, trial, cam, fname = parts[0], parts[1], parts[3], parts[-1]
    mseq = SEQ_PATTERN.match(seq)
    mtrial = TRIAL_PATTERN.match(trial)
    mframe = FRAME_PATTERN.search(fname)
    if not (mseq and mtrial and mframe):
        return None

    date_id, subject_id, object_id, grasp_id = map(int, mseq.groups())
    trial_id = int(mtrial.group(1))
    frame_id = int(mframe.group(1))
    return {
        "subject_id": subject_id,
        "date_id": date_id,
        "object_id": object_id,
        "grasp_type": grasp_id,
        "trial_id": trial_id,
        "cam": cam,
        "frame_id": frame_id,
    }


def _json_sort_key(jname: str):
    """
    Deterministic frame order within each subject:
      subject_id -> date_id -> object_id -> grasp_type -> trial_id -> cam -> frame_id
    """
    meta = _parse_path_metadata(jname)
    if meta is None:
        return (10**9, 10**9, 10**9, 10**9, 10**9, 99, 10**9, jname)

    return (
        meta["subject_id"],
        meta["date_id"],
        meta["object_id"],
        meta["grasp_type"],
        meta["trial_id"],
        CAM_ORDER.get(meta["cam"], 99),
        meta["frame_id"],
        jname,
    )


def iter_rows_from_zip(zip_path: Path, subject_id: int):
    """Yield rows parsed from a single HOGraspNet subject zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = sorted(
            (n for n in zf.namelist() if n.endswith(".json")),
            key=_json_sort_key,
        )

        for jname in json_names:
            meta = _parse_path_metadata(jname)
            if meta is None:
                continue

            try:
                raw = zf.read(jname)
                anno = json.loads(raw.decode("utf-8-sig"))
            except Exception:
                continue

            try:
                pts = anno["hand"]["3D_pose_per_cam"]
            except KeyError:
                continue
            if not isinstance(pts, list) or len(pts) != 21:
                continue
            if any((not isinstance(p, (list, tuple)) or len(p) != 3) for p in pts):
                continue

            # Keep original FEIX id in CSV; remapping to 0..27 is handled in the training dataloader.
            row = [
                meta["subject_id"],
                meta["date_id"],
                meta["object_id"],
                meta["grasp_type"],
                meta["trial_id"],
                meta["cam"],
                meta["frame_id"],
            ]
            row.extend([coord for joint in pts for coord in joint])
            yield row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera hograspnet_raw.csv (XYZ raw sin normalización) desde zips HOGraspNet."
    )
    parser.add_argument(
        "--zips_dir",
        required=True,
        help="Directorio con los zips de anotaciones de HOGraspNet",
    )
    parser.add_argument(
        "--out",
        default="data/raw/hograspnet_raw.csv",
        help="Ruta del CSV de salida (default: data/raw/hograspnet_raw.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zips_dir = Path(args.zips_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_pattern = re.compile(r"GraspNet_S(\d+)_\d+_Labeling_data\.zip")
    zip_entries = []
    for zip_path in zips_dir.glob("GraspNet_S*_Labeling_data.zip"):
        match = zip_pattern.match(zip_path.name)
        if not match:
            continue
        zip_entries.append((int(match.group(1)), zip_path))

    zip_entries.sort(key=lambda x: (x[0], x[1].name))
    if not zip_entries:
        raise FileNotFoundError(f"No se encontraron zips en {zips_dir}")

    total = 0
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(HEADER)

        for subject_id, zip_path in zip_entries:
            print(f"  S{subject_id:02d} — {zip_path.name} ...", end=" ", flush=True)

            n_rows = 0
            for row in iter_rows_from_zip(zip_path, subject_id):
                writer.writerow(row)
                n_rows += 1

            total += n_rows
            print(f"{n_rows} frames")

    print(f"\nTotal: {total} frames guardados en {out_path}")


if __name__ == "__main__":
    main()
