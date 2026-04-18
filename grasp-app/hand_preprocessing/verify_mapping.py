import pandas as pd
import numpy as np
import sys

def main():
    print("Leyendo 1 sola fila del dataset...")
    df = pd.read_csv('/home/yareeez/AIST-hand/grasp-model/data/raw/hograspnet_raw.csv', nrows=1)
    
    xyz_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]
    points_w = df[xyz_cols].iloc[0].values.astype(np.float64).reshape(21, 3)

    log_file = "mapping_log.txt"
    with open(log_file, 'w') as f:
        f.write("=== MAPEO EXPLICITO DEL DATASET A LA MEMORIA ===\n")
        f.write(f"Verificando que las columnas del CSV no se asignen al dedo incorrecto en el Array(21,3)\n\n")
        
        col_idx = 0
        for joint_idx in range(21):
            x_col = xyz_cols[col_idx]
            y_col = xyz_cols[col_idx+1]
            z_col = xyz_cols[col_idx+2]
            
            val_x = df[x_col].iloc[0]
            val_y = df[y_col].iloc[0]
            val_z = df[z_col].iloc[0]
            
            arr_x = points_w[joint_idx][0]
            arr_y = points_w[joint_idx][1]
            arr_z = points_w[joint_idx][2]
            
            f.write(f"DONG JOINT INDEX [{joint_idx}]:\n")
            f.write(f"  -> X: Columna leida: '{x_col.ljust(25)}' (Valor: {val_x:+.4f}) ===> Se inyecto en: points_w[{joint_idx}][0] (Valor: {arr_x:+.4f})\n")
            f.write(f"  -> Y: Columna leida: '{y_col.ljust(25)}' (Valor: {val_y:+.4f}) ===> Se inyecto en: points_w[{joint_idx}][1] (Valor: {arr_y:+.4f})\n")
            f.write(f"  -> Z: Columna leida: '{z_col.ljust(25)}' (Valor: {val_z:+.4f}) ===> Se inyecto en: points_w[{joint_idx}][2] (Valor: {arr_z:+.4f})\n")
            f.write("-" * 110 + "\n")
            col_idx += 3
            
    print(f"Log generado con exito en: {log_file}")

if __name__ == "__main__":
    main()
