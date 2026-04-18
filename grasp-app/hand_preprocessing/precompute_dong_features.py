"""
Pre-computation script for Dong kinematics.

Reads the raw HOGraspNet dataset and computes Dong kinematic features
(quaternions and angles) for each frame. The result is saved as a new
CSV file that can be merged with the raw data during training.

Calibration logic:
The dataset is grouped by `subject_id`. A new DongKinematics instance is
used per subject so that each subject has their own frozen palm/finger
proportions derived from their first valid frame.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dong_kinematics import DongKinematics


def main():
    raw_csv_path = "/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_raw.csv"
    out_csv_path = "/home/yareeez/AIST-hand/grasp-model/data/processed/hograspnet_dong.csv"

    print(f"Leyendo dataset original: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    
    # Identificamos columnas
    xyz_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]
    meta_cols = ["subject_id", "date_id", "object_id", "grasp_type", "trial_id", "cam", "frame_id"]

    if len(xyz_cols) != 63:
        print(f"Error: Se esperaban 63 columnas de landmarks, se encontraron {len(xyz_cols)}")
        sys.exit(1)

    print(f"Total de filas: {len(df)}")
    print(f"Total de sujetos: {df['subject_id'].nunique()}")

    # Aseguramos que el directorio exista y borramos cualquier archivo previo que haya fallado a medias
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_csv_path).exists():
        Path(out_csv_path).unlink()

    # Preparar almacenamiento de resultados
    results = []
    degenerate_count = 0
    processed_count = 0
    first_write = True

    # Iteramos agrupando por sujeto
    grouped = df.groupby("subject_id")
    
    with tqdm(total=len(df), desc="Procesando Dong Kinematics") as pbar:
        for subject_id, group in grouped:
            # Nuevo instanciador sin limite para calibracion offline
            dk = DongKinematics(calibration_frames=None)
            
            # Fase 1: Calibracion robusta (Mediana de los primeros 50 frames utiles del sujeto)
            calib_limit = 50
            calib_count = 0
            for idx, row in group.iterrows():
                if calib_count >= calib_limit:
                    break
                points_w = row[xyz_cols].values.astype(np.float64).reshape(21, 3)
                try:
                    dk.calibrate(points_w)
                    calib_count += 1
                except ValueError:
                    pass
            dk.force_freeze()
            
            # Fase 2: Procesamiento de todos los frames con el esqueleto ya bloqueado
            for idx, row in group.iterrows():
                points_w = row[xyz_cols].values.astype(np.float64).reshape(21, 3)
                
                try:
                    res = dk.process(points_w)
                except ValueError:
                    degenerate_count += 1
                    pbar.update(1)
                    continue

                # Extraer metadatos para hacer JOIN después
                out_row = {c: row[c] for c in meta_cols}
                
                # Invertimos los dicts y arreglos a formato tabular
                
                # 1. Wrist Euler Angles
                for k in ("alpha", "beta", "gamma"):
                    out_row[f"wrist_{k}_deg"] = res["wrist_euler_deg"][k]
                    
                # 2. Quaternions (20 joints)
                for j in res["joint_order"]:
                    q = res["quaternions"][j]
                    out_row[f"q{j}_w"] = q[0]
                    out_row[f"q{j}_x"] = q[1]
                    out_row[f"q{j}_y"] = q[2]
                    out_row[f"q{j}_z"] = q[3]
                    
                # 3. Angles (beta / gamma)
                for j, angles in res["angles_deg"].items():
                    out_row[f"beta{j}_deg"] = angles["beta"]
                    if "gamma" in angles:
                        out_row[f"gamma{j}_deg"] = angles["gamma"]
                        
                results.append(out_row)
                pbar.update(1)

            # Guardar el bloque del sujeto actual en disco para liberar RAM
            if results:
                out_df = pd.DataFrame(results)
                out_df.to_csv(out_csv_path, mode='a', header=first_write, index=False)
                first_write = False
                processed_count += len(results)
                results = []

    print(f"\nFinalizado. Filas procesadas exitosamente: {processed_count}")
    if degenerate_count > 0:
        print(f"Atención: {degenerate_count} frames fueron omitidos por ser degenerados (ruido extremo).")

    print(f"Resultados guardados por lotes de exitosamente a disco en: {out_csv_path}")
    print("¡Listo!")


if __name__ == "__main__":
    main()
