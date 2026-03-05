# test_tograph_from_csv.py
import pandas as pd
import numpy as np
from grasp_gcn.transforms.tograph import ToGraph
from pathlib import Path

# calcula la raíz del proyecto (2 niveles arriba del archivo actual)
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "examples" / "data" / "grasps_sample_test.csv"

FEATURES = "xyz"


JOINTS = [
    'WRIST',
    'THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP',
    'INDEX_FINGER_MCP','INDEX_FINGER_PIP','INDEX_FINGER_DIP','INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP','MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_FINGER_TIP',
    'PINKY_MCP','PINKY_PIP','PINKY_DIP','PINKY_TIP'
]

AXES = ["X", "Y"] if FEATURES == "xy" else ["X", "", "Z"]

def main():
    df = pd.read_csv(CSV)

    # ---- detectar columna de etiqueta (soporta 'grasp_type', 'label' o 'grasp_type_ ')
    y_col = None
    for c in df.columns:
        c_clean = c.strip().lower()
        if c_clean in ("grasp_type", "label"):
            y_col = c
            break
    if y_col is None:
        candidates = [c for c in df.columns if c.strip().lower().startswith("grasp_type")]
        if candidates:
            y_col = candidates[0]
    if y_col is None:
        print("Columnas disponibles:", list(df.columns))
        raise KeyError("No encontré columna de etiqueta ('grasp_type'/'label').")

    # toma una fila de ejemplo
    row = df.iloc[0]

    # ---- construir el sample en el formato que espera ToGraph
    sample = {}
    for j in JOINTS:
        vals = []
        for ax in AXES:
            col = f"{j}_{ax}"
            if col not in df.columns:
                raise KeyError(f"Falta la columna '{col}' en el CSV.")
            vals.append(float(row[col]))
        sample[j] = np.array(vals, dtype=float)

    sample["grasp_type"] = int(row[y_col])

    # 👇 sin normalize porque tu clase ya no lo usa
    tg = ToGraph(features=FEATURES, make_undirected=True)
    data = tg(sample)

    print("Etiqueta:", data.y.item())
    print("x.shape:", tuple(data.x.shape))
    print("edge_index.shape:", tuple(data.edge_index.shape))
    print("mask mean:", float(data.mask.float().mean()))
    print("x mean:", data.x.mean(0))
    valid = data.mask.squeeze(1) > 0.5
    print("x std(validos):", data.x[valid].std(0))

if __name__ == "__main__":
    main()
