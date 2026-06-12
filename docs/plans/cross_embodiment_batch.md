# Cross-Embodiment Batch Training Plan

## Context

Run 38 results showed MCP closure problem. Diagnosis: perception issue (camera angle) + likely
structural issue in training loop. We discovered our multi-robot loop iterates robots SEQUENTIALLY
(separate batches per robot), while Yan 2026 mixes ALL embodiments in ONE batch per step.
Triplets must be mined across embodiments, not within each robot separately.

Also: EIGENGRASP_ONLINE sampler was 50 min/50 steps → replaced with precomputed 1M-pose npz
(trimmed from 10M). Files renamed: old 10M files kept as *_10M.npz backups.

Similarity metric analysis (Xin 2025 paper):
- D_R: cross-robot safe (semantic joint labels align: thumb_mcp/pip/dip same for Shadow+Allegro)
- D_joints: NOT cross-robot safe (intermediate chain positions, different bone proportions)
- D_ahg: cross-robot safe (angle-based, scale-invariant)
- Xin lam_tip / lam_tip_rot / lam_pinch: cross-robot safe (tip-based, direction-based, relative)
- Xin lam_finger: NOT cross-robot safe (dense MCP/PIP/DIP chain, same as D_joints problem)

Shadow has 5 fingers (common_labels: 15 joints), Allegro has 4 (12 joints, no pinky).
Per subspace, joint labels are IDENTICAL for shared fingers → D_R and D_ahg work.

---

## Task List

### 1. Fix loop.py — cross-embodiment batch structure

**File:** `models/latent-retargeting/src/cross_emb/training/loop.py`

**What:** Split `for cfg in robot_cfgs:` loop into two passes:

**Pass 1 (per-robot, unchanged):**
- Sample batch per robot
- Forward pass: z_h = E_h(pose_h), z_r = E_X(E_r(q_r))
- Compute L_rec, L_ltc, L_temp per robot
- Store latents + chain/tips for cross-robot contrastive:
  ```python
  all_rd.append({
      "z_t_subs":      z_t.chunk(5, dim=-1),
      "z_r_subs":      z_r.chunk(5, dim=-1),
      "chain_h_sub":   chain_h_sub,    # [B, Fc, 4, 3] normalized
      "chain_r_sub":   chain_r_sub,    # [B, Fc, 4, 3] normalized
      "tips_h_sub":    tips_h_sub,     # [B, Fc, 3] normalized
      "common_fingers": common_fingers,
      "common_labels":  common_labels,
      "pose_h_sub":    pose_h_sub,     # [B, K, F] for D_R
      "pose_r_sub":    pose_r_sub,     # [B, K, F] for D_R
      "cfg_name":      cfg["name"],
  })
  ```
- L_robot = lambda_rec*L_rec + lambda_ltc*L_ltc + lambda_tmp*L_temp (NO L_cont here)
- L_total += L_robot

**Pass 2 (cross-robot contrastive, NEW):**
- For each subspace k in (thumb, index, middle, ring, pinky):
  - For each robot in all_rd: collect z_h_k, z_r_k, chain_h_k, chain_r_k, pose_h_k, pose_r_k
    - jidx = joints in subspace k from common_labels
    - tidx = fingers in subspace k from common_fingers
    - Skip robot if jidx empty (e.g. Allegro skips pinky)
  - Pool ALL collected: z_all_k, q_all_k, chain_all_k (cat dim=0)
  - Mine triplets from pooled batch
  - Similarity metric (see Task 2 below)
  - L_cont += triplet_loss(z_all_k, ...)
- L_total += lambda_c * L_cont
- accum_cont = L_cont.item()

**Key:** single backward() after both passes — unchanged.

---

### 2. Fix similarity metric for cross-robot (xin + ahg paths)

**File:** `models/latent-retargeting/src/cross_emb/training/loop.py`

**Problem:** `xin_sk_per_finger` expects `tips[N, 5, 3]` but Shadow=5 fingers, Allegro=4.
Can't cat directly. Also `lam_finger` uses intermediate chain → must be 0 cross-robot.

**Fix for xin path:**
Build per-robot mini-tips `[B, 2, 3]` = stack([thumb_tip, current_finger_tip], dim=1).
Pool across robots → `[N, 2, 3]`. Use `finger_idx = 0 if sub=="thumb" else 1`.
Call xin_sk_per_finger with `lam_finger=0` (override, never pass lam_finger for cross-robot).

```python
# per robot, for subspace k:
thumb_tidx = [i for i, f in enumerate(rd["common_fingers"]) if f == "thumb"]
curr_tidx  = tidx  # already computed from common_fingers for subspace k
if not curr_tidx or not thumb_tidx:
    continue
tip_thumb   = rd["chain_h_sub"][:, thumb_tidx[0], 3, :]  # [B, 3]
tip_current = rd["chain_h_sub"][:, curr_tidx[0],  3, :]  # [B, 3]
mini_tips_h = torch.stack([tip_thumb, tip_current], dim=1)  # [B, 2, 3]
# same for robot side
```

For chain: `chain_all_k` is `[N, 1, 4, 3]` (already per-finger). Use `finger_idx=0`.

**Fix for ahg path:**
`S_k = w_r * D_R + w_ahg * D_ahg` — drop `w_joints * D_joints`.
D_R uses q_all_k (same joint semantics per subspace ✓).
D_ahg uses chain_all_k (angle-based, scale-invariant ✓).

---

### 3. Run 1: cross-embodiment batch, AHG metric, same config as Run 20

**Purpose:** Isolate structural fix (cross-robot batching) from metric changes.
Baseline = Run 20 config. Only change = cross-robot batch pooling.

Config:
- `--robots shadow allegro`
- `--sk_metric ahg`
- `--w_r 1.0 --w_joints 0.0 --w_ahg 1.0`  ← w_joints=0 for cross-robot safety
- Same lambda_c, lambda_rec, lambda_ltc, lambda_tmp as Run 20
- `--n_steps 15000 --b 50000`
- `--skip_final_eval`

Expected: verify both robots train together, loss curves stable, checkpoint saves shadow(24)+allegro(16).

---

### 4. Run 2: cross-embodiment batch, Xin metric

**Purpose:** Test Xin similarity metric for cross-robot triplet mining.

Config: same as Run 3 but:
- `--sk_metric xin`
- `--lam_finger 0.0`  ← explicitly zero for cross-robot safety
- `--lam_tip`, `--lam_pinch`, `--lam_tip_rot` at Run 20 values
- `--enable_switching True`

Compare against Run 3 (ahg cross-robot) on RS/NDS/NVS metrics.

---

## Files Changed So Far (before Tasks 1-4)

- `robot/hands/allegro_hand/datasets/processed/valid_robot_poses_allegro_dong.npz` — trimmed to 1M
- `robot/hands/allegro_hand/datasets/processed/valid_robot_poses_allegro_dong_10M.npz` — backup
- `robot/shadow-hand/datasets/processed/valid_robot_poses_eigengrasp_dong.npz` — trimmed to 1M
- `robot/shadow-hand/datasets/processed/valid_robot_poses_eigengrasp_dong_10M.npz` — backup

## Order of Attack

1. **[PENDING] Verify Leap pipeline end-to-end** — viewer (all PCs look sane), FK/Dong smoke,
   EIGENGRASP_ONLINE sampler smoke with `--robots leap`. Only then proceed.
2. **[PENDING] Tasks 1-4 below** — cross-embodiment batch fix (loop.py + metric).
   Upgrade `--robots shadow allegro` → `--robots shadow allegro leap` once Leap verified.
3. **[PENDING] docs/siguientes_pasos.md ideas** — after cross-embodiment batch is running.

---

## Status

- [x] NPZ files trimmed to 1M
- [x] Leap eigengrasp basis built (BODex, 1.36M poses, k=9 → 92.5% var)
- [x] Leap valid_robot_poses_leap.npz (1M collision-free poses, 91% acceptance)
- [x] Leap valid_robot_poses_leap_dong.npz (Dong features precomputed, 784 MB)
- [x] leap.yaml wired (eigengrasp + valid_poses + mjcf)
- [x] --robots flag refactor (per-hand yaml, combo yamls deleted)
- [ ] Leap pipeline verification (viewer + FK smoke + sampler smoke)
- [ ] Task 1: loop.py cross-embodiment batch
- [ ] Task 2: similarity metric cross-robot fix
- [ ] Task 3: Run 1 (ahg, cross-robot batch, shadow+allegro+leap)
- [ ] Task 4: Run 2 (xin, cross-robot batch)
