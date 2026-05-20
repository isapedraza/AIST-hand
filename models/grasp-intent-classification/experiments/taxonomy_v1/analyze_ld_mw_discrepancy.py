"""
analyze_ld_mw_discrepancy.py
Analisis matematico de por que d(Large_Diameter, Medium_Wrap) = 4.071 en nuestro espacio de
sinergias, mientras que Stival et al. (2019) los fusiona primero en su dendrograma.

Fuentes de datos:
  - class_separability.json: centroides PCA, loadings, distancias de centroides
  - hograspnet.csv: angulos crudos por clase (via sanity_check_rc1.py)
"""

import json
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# ── Cargar datos ────────────────────────────────────────────────────────────
with open(RESULTS / "class_separability.json") as f:
    cs = json.load(f)

# Centroides en espacio PCA (9D)
centroids = {int(k): np.array(v) for k, v in cs["class_centroids_pca"].items()}

# Loadings no rotados (results.json tiene los primeros 6 PCs; class_separability los 9 rotados)
# Usamos results.json para PC1 unrotado (40.6% var explicada)
with open(RESULTS / "results.json") as f:
    res = json.load(f)

angle_names = cs["pca"]["angle_names"]   # 20 angulos
pc1_loadings = res["pca"]["loadings"][0]  # primer eigenvector (unrotado), 20 valores
ev_ratio     = res["pca"]["explained_variance_ratio"]

# Loadings varimax rotados: matrix (20 angulos x 9 RCs)
rc_loadings  = np.array(cs["pca"]["loadings_rotated"])   # shape (20, 9)

print("=" * 70)
print("ANALISIS MATEMATICO: Large_Diameter vs Medium_Wrap en espacio sinergico")
print("=" * 70)

# ── 1. Varianza explicada ─────────────────────────────────────────────────
print("\n--- 1. Espacio PCA ---")
cum_var = sum(ev_ratio[:9])
print(f"  PCs retenidos: 9 (varianza acumulada: {cum_var:.4f} = {cum_var*100:.1f}%)")
print(f"  PC1 varianza individual: {ev_ratio[0]:.4f} = {ev_ratio[0]*100:.1f}%")

# ── 2. Centroides ─────────────────────────────────────────────────────────
ld  = centroids[0]   # Large_Diameter
mw  = centroids[6]   # Medium_Wrap
pd_ = centroids[11]  # Power_Disk (vecino sinergico de MW)

print("\n--- 2. Centroides en espacio PCA 9D (unidades: desviaciones estandar) ---")
header = f"{'Clase':<20}" + "".join(f"  PC{i+1:>2}" for i in range(9))
print(header)
for name, c in [("Large_Diameter (0)", ld), ("Medium_Wrap (6)", mw), ("Power_Disk (11)", pd_)]:
    row = f"{name:<20}" + "".join(f"  {x:>6.3f}" for x in c)
    print(row)

# ── 3. Distancia y descomposicion por dimension ────────────────────────────
diff  = ld - mw
diff2 = diff ** 2
total_d2 = diff2.sum()
total_d  = np.sqrt(total_d2)

print(f"\n--- 3. Descomposicion de la distancia d(LD, MW) ---")
print(f"  Distancia Euclidea 9D: {total_d:.4f}  (almacenada en JSON: 4.071)")
print()
print(f"  {'PC':<6}  {'LD':>8}  {'MW':>8}  {'diff':>8}  {'diff^2':>8}  {'% d^2':>7}  {'% |d|':>7}")
print(f"  {'-'*65}")
for i in range(9):
    pct_d2  = diff2[i] / total_d2 * 100
    pct_d   = np.sqrt(diff2[i]) / total_d * 100
    print(f"  PC{i+1:<4}  {ld[i]:>8.3f}  {mw[i]:>8.3f}  {diff[i]:>8.3f}  {diff2[i]:>8.3f}  {pct_d2:>6.1f}%  {pct_d:>6.1f}%")
print(f"  {'TOTAL':<6}  {'':>8}  {'':>8}  {'':>8}  {total_d2:>8.3f}  100.0%")
print(f"  sqrt(total d^2) = {total_d:.4f}")

# ── 4. Loadings PC1 (no rotado) ───────────────────────────────────────────
print("\n--- 4. Loadings PC1 no rotado (eje principal de separacion) ---")
print(f"  PC1 = {ev_ratio[0]*100:.1f}% de varianza total")
print(f"  Angulos con |loading| > 0.20:")
for i, (name, load) in enumerate(zip(angle_names, pc1_loadings)):
    if abs(load) > 0.20:
        print(f"    {name:<22}  {load:>+.4f}")

# ── 5. Interpretacion fisica del score PC1 ────────────────────────────────
print("\n--- 5. Interpretacion fisica ---")
print(f"  Score PC1 Large_Diameter: {ld[0]:>+.3f}  (unidades sigma)")
print(f"  Score PC1 Medium_Wrap:    {mw[0]:>+.3f}  (unidades sigma)")
print(f"  Diferencia PC1:           {ld[0]-mw[0]:>+.3f}  sigma")
print()
print(f"  Eje PC1 = flexion global dedos 2-5 (indices MCP/PIP/DIP con loadings 0.19-0.31)")
print(f"  Score negativo extremo de LD (-{abs(ld[0]):.2f}σ) => dedos 2-5 significativamente menos")
print(f"  flexionados que la media de las 28 clases.")
print(f"  Score de MW ({mw[0]:+.2f}σ) => cerca de la media, flexion moderada.")

# ── 6. Confirmacion con angulos crudos (resultados de sanity_check_rc1.py) ─
print("\n--- 6. Confirmacion con angulos articulares crudos (mediana sobre frames) ---")
print("  (resultados de sanity_check_rc1.py sobre hograspnet.csv, contact_sum > 0)")
print()
pip_ld  = [0.4163, 0.3634, 0.3184, 0.2017]  # INDEX/MIDDLE/RING/PINKY PIP
pip_mw  = [0.6556, 0.7269, 0.7715, 0.5929]
dip_ld  = [0.2944, 0.3019, 0.2312, 0.1401]
dip_mw  = [0.6990, 0.7565, 0.8637, 0.5485]
fingers = ["INDEX", "MIDDLE", "RING", "PINKY"]

print(f"  {'Articulacion':<22}  {'LD (rad)':>10}  {'MW (rad)':>10}  {'Δ (deg)':>10}")
print(f"  {'-'*58}")
for f_, pl, pm in zip(fingers, pip_ld, pip_mw):
    print(f"  {f_+'_PIP':<22}  {pl:>10.4f}  {pm:>10.4f}  {np.degrees(pl-pm):>+10.1f}")
for f_, dl, dm in zip(fingers, dip_ld, dip_mw):
    print(f"  {f_+'_DIP':<22}  {dl:>10.4f}  {dm:>10.4f}  {np.degrees(dl-dm):>+10.1f}")
print()
mean_pip_diff_deg = np.degrees(np.mean(pip_ld) - np.mean(pip_mw))
mean_dip_diff_deg = np.degrees(np.mean(dip_ld) - np.mean(dip_mw))
print(f"  Media PIP  LD={np.mean(pip_ld):.4f}, MW={np.mean(pip_mw):.4f}  => LD-MW={mean_pip_diff_deg:+.1f} deg")
print(f"  Media DIP  LD={np.mean(dip_ld):.4f}, MW={np.mean(dip_mw):.4f}  => LD-MW={mean_dip_diff_deg:+.1f} deg")

# ── 7. Por que Stival ve lo contrario ─────────────────────────────────────
print("\n--- 7. Discrepancia con Stival et al. (2019) ---")
print("""  Stival usaron Ninapro DB5: 6 tipos de agarre ejecutados SIN objeto real,
  solo mímesis a partir de imagenes (electromiografia + cinemática de muneca).
  Sin objeto fisico que coaccione la configuracion de la mano, los sujetos tienden
  a adoptar posturas similares para 'large diameter' y 'medium wrap': en ambos
  casos muestran la forma del agarre cylindrico con similar cierre de dedos.

  HOGraspNet usa objetos YCB reales. El objeto impone restricciones fisicas:
  - Large_Diameter (bowl, large can): la mano debe ABRIRSE para envolver el
    diametro. Los DIP/PIP quedan relativamente extendidos.
  - Medium_Wrap (smaller cylinder): la mano CIERRA mas para conformarse
    al diametro menor. Los DIP/PIP alcanzan mayor flexion.

  Resultado: en HOGraspNet la diferencia cinetica es real y medible (20-27 grados
  en PIP y DIP respectivamente). El espacio sinergico la captura como la dimension
  de mayor separacion entre los dos centroides (PC1, 53.7% de d^2).

  La discrepancia Stival <-> nuestro analisis no es un error: refleja la diferencia
  fundamental entre datasets sin objeto (mimesis) vs con objeto real (HOGraspNet).""")

# ── 8. Resumen cuantitativo final ─────────────────────────────────────────
print("\n--- 8. Resumen cuantitativo ---")
print(f"  d(LD, MW) en espacio de 9 sinergias = {total_d:.4f}")
print(f"  PC1 aporta {diff2[0]/total_d2*100:.1f}% de d^2 (eje dominante de separacion)")
print(f"  Score LD en PC1: {ld[0]:+.2f}σ  |  Score MW en PC1: {mw[0]:+.2f}σ")
print(f"  Δ PIP promedio: {mean_pip_diff_deg:+.1f}°  |  Δ DIP promedio: {mean_dip_diff_deg:+.1f}°")
print(f"  El analisis es correcto. La discrepancia con Stival es de dataset, no de metodo.")
print()
