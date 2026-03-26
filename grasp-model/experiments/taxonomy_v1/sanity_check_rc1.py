"""
sanity_check_rc1.py -- Verificar RC1: Large_Diameter tiene menor flexion distal que Medium_Wrap?

RC1 en el analisis de sinergias cargo positivo en PIP/DIP de dedos 2-5.
Large_Diameter (class 0) tiene RC1 = -2.57 (negativo extremo).
Medium_Wrap   (class 6) tiene RC1 = +0.41 (cerca del centro).

Si RC1 captura flexion distal, Large_Diameter deberia tener PIP/DIP menores.
Pero intuitivamente large diameter es un agarre cerrado con dedos envolventes...
Este script verifica directamente en los datos crudos.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CSV_PATH, CONTACT_SUM_THRESHOLD, ANGLE_TRIPLETS, ANGLE_NAMES, ALL_ANGLE_NAMES
from features import compute_flexion_angles, compute_abduction_angles

# Indices de angulos de interes (en ANGLE_NAMES)
# INDEX_PIP=4, INDEX_DIP=5, MIDDLE_PIP=7, MIDDLE_DIP=8
# RING_PIP=10, RING_DIP=11, PINKY_PIP=13, PINKY_DIP=14
PIP_INDICES = [4, 7, 10, 13]   # INDEX/MIDDLE/RING/PINKY PIP
DIP_INDICES = [5, 8, 11, 14]   # INDEX/MIDDLE/RING/PINKY DIP
PIP_NAMES   = ["INDEX_PIP", "MIDDLE_PIP", "RING_PIP", "PINKY_PIP"]
DIP_NAMES   = ["INDEX_DIP", "MIDDLE_DIP", "RING_DIP", "PINKY_DIP"]

TARGET_CLASSES = {
    0:  "Large_Diameter",
    6:  "Medium_Wrap",
    11: "Power_Disk",     # vecino mas cercano a Medium_Wrap en espacio sinergico (d=2.122)
}

print("=" * 60)
print("RC1 Sanity Check: DIP/PIP flexion por clase")
print("=" * 60)

# --- Cargar CSV ---
print(f"\nLoading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"Total frames: {len(df):,}")

# Filtro contact_sum (igual que en el pipeline de sinergias)
df = df[df["contact_sum"] > CONTACT_SUM_THRESHOLD].reset_index(drop=True)
print(f"After contact_sum > 0 filter: {len(df):,}")

# Solo clases objetivo
df = df[df["grasp_type"].isin(TARGET_CLASSES.keys())].reset_index(drop=True)
print(f"Filtered to target classes: {len(df):,}")
for cls, name in TARGET_CLASSES.items():
    n = (df["grasp_type"] == cls).sum()
    print(f"  Class {cls:2d} ({name}): {n:,} frames")

# --- Extraer XYZ ---
xyz_cols = [c for c in df.columns if c not in ("subject_id","sequence_id","cam","grasp_type","contact_sum")]
xyz = df[xyz_cols].values.astype(np.float32).reshape(-1, 21, 3)

# --- Calcular angulos de flexion ---
print("\nComputing flexion angles...")
flex = compute_flexion_angles(xyz)  # (N, 15)

# --- Comparar por clase ---
print("\n--- Angulos PIP (radianes, mediana por clase) ---")
print(f"{'Clase':<20}", end="")
for n in PIP_NAMES:
    print(f"  {n:>12}", end="")
print(f"  {'Mean_PIP':>10}")

for cls, name in TARGET_CLASSES.items():
    mask = (df["grasp_type"].values == cls)
    vals = flex[mask][:, PIP_INDICES]
    meds = np.median(vals, axis=0)
    print(f"{name:<20}", end="")
    for m in meds:
        print(f"  {m:>12.4f}", end="")
    print(f"  {meds.mean():>10.4f}")

print("\n--- Angulos DIP (radianes, mediana por clase) ---")
print(f"{'Clase':<20}", end="")
for n in DIP_NAMES:
    print(f"  {n:>12}", end="")
print(f"  {'Mean_DIP':>10}")

for cls, name in TARGET_CLASSES.items():
    mask = (df["grasp_type"].values == cls)
    vals = flex[mask][:, DIP_INDICES]
    meds = np.median(vals, axis=0)
    print(f"{name:<20}", end="")
    for m in meds:
        print(f"  {m:>12.4f}", end="")
    print(f"  {meds.mean():>10.4f}")

print("\n--- Diferencia Large_Diameter - Medium_Wrap ---")
mask_ld = (df["grasp_type"].values == 0)
mask_mw = (df["grasp_type"].values == 6)

pip_ld = np.median(flex[mask_ld][:, PIP_INDICES], axis=0)
pip_mw = np.median(flex[mask_mw][:, PIP_INDICES], axis=0)
dip_ld = np.median(flex[mask_ld][:, DIP_INDICES], axis=0)
dip_mw = np.median(flex[mask_mw][:, DIP_INDICES], axis=0)

pip_diff = pip_ld - pip_mw
dip_diff = dip_ld - dip_mw

print(f"Mean PIP diff (LD - MW): {pip_diff.mean():.4f} rad = {np.degrees(pip_diff.mean()):.1f} deg")
print(f"Mean DIP diff (LD - MW): {dip_diff.mean():.4f} rad = {np.degrees(dip_diff.mean()):.1f} deg")

if pip_diff.mean() < 0 and dip_diff.mean() < 0:
    print("\n[CONFIRMADO] Large_Diameter tiene MENOR flexion distal que Medium_Wrap.")
    print("RC1 negativo para LD = correcto. La interpretacion del espacio sinergico es valida.")
elif pip_diff.mean() > 0 and dip_diff.mean() > 0:
    print("\n[CONTRADICION] Large_Diameter tiene MAYOR flexion distal que Medium_Wrap.")
    print("RC1 negativo para LD seria inconsistente. Revisar el analisis de sinergias.")
else:
    print("\n[MIXTO] Patron no uniforme entre PIP y DIP. Revisar por dedo.")

print("\n(Recordatorio: RC1 en el PCA cargo positivo en PIP/DIP dedos 2-5)")
print("(RC1 = -2.57 para LD => LD proyecta en direccion NEGATIVA del eje de flexion)")
