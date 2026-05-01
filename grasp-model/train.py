# ==============================================================
# train.py — Training script for GCN_8_8_16_16_32
# with train/val/test splits, validation metrics, and checkpoints
# ==============================================================

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ✅ IMPORTS CORREGIDOS (paquete)
from grasp_gcn.dataset.grasps import GraspsClass
from grasp_gcn.transforms.tograph import ToGraph
from grasp_gcn.network.utils import get_network

print('--------------------------------')
print('🚀 Import libraries: OK')

# ====================== Run config (set via env vars from notebook) ==========
# GG_RUN_NAME:       e.g. "run008_c28_xyz_bone"
# GG_COLLAPSE:       "none" | "feix" | "taxonomy_v1"
# GG_CHECKPOINT:     filename for saved model, e.g. "best_model_run008_c28_xyz_bone.pth"
# GG_BONE_VECTORS:   "true" | "false"
# GG_VELOCITY:       "true" | "false"
# GG_MANO_POSE:      "true" | "false"
# GG_GLOBAL_SWING:   "true" | "false"
# GG_MANO_PKL:       path to MANO_RIGHT.pkl (required when GG_GLOBAL_SWING=true)
# GG_AHG_ANGLES:     "true" | "false"  -- 10 wrist-relative angles per node (Aiman & Ahmad 2024)
# GG_AHG_DISTANCES:  "true" | "false"  -- 10 distances to critical joints per node
# GG_DONG_EULER:     "true" | "false"  -- 2 Dong euler angles (beta/gamma) per node, zero-padded

RUN_NAME      = os.getenv("GG_RUN_NAME",      "run_unnamed")
_collapse     = os.getenv("GG_COLLAPSE",      "none").strip().lower()
COLLAPSE      = False if _collapse == "none" else _collapse
CHECKPOINT    = os.getenv("GG_CHECKPOINT",   f"best_model_{RUN_NAME}.pth")
JOINT_ANGLES  = os.getenv("GG_JOINT_ANGLES", "true" ).strip().lower() == "true"
BONE_VECTORS  = os.getenv("GG_BONE_VECTORS", "false").strip().lower() == "true"
VELOCITY      = os.getenv("GG_VELOCITY",     "false").strip().lower() == "true"
MANO_POSE     = os.getenv("GG_MANO_POSE",    "false").strip().lower() == "true"
GLOBAL_SWING  = os.getenv("GG_GLOBAL_SWING", "false").strip().lower() == "true"
AHG_ANGLES    = os.getenv("GG_AHG_ANGLES",   "false").strip().lower() == "true"
AHG_DISTANCES = os.getenv("GG_AHG_DISTANCES","false").strip().lower() == "true"
DONG_QUATS    = os.getenv("GG_DONG_QUATS",   "false").strip().lower() == "true"
DONG_EULER    = os.getenv("GG_DONG_EULER",   "false").strip().lower() == "true"
DONG_CSV_PATH = os.getenv("GG_DONG_CSV_PATH","").strip() or None
CSV_FILENAME  = os.getenv("GG_CSV_FILENAME", "").strip() or None
ADD_XYZ       = os.getenv("GG_ADD_XYZ",      "").strip().lower() or None
if ADD_XYZ is not None:
    ADD_XYZ = ADD_XYZ == "true"
NORMALIZE_XYZ = os.getenv("GG_NORMALIZE_XYZ","false").strip().lower() == "true"

print(f"Run:           {RUN_NAME}")
print(f"Collapse:      {COLLAPSE}")
print(f"Checkpoint:    {CHECKPOINT}")
print(f"Joint angles:  {JOINT_ANGLES}")
print(f"Bone vectors:  {BONE_VECTORS}")
print(f"Velocity:      {VELOCITY}")
print(f"MANO pose:     {MANO_POSE}")
print(f"Global swing:  {GLOBAL_SWING}")
print(f"AHG angles:    {AHG_ANGLES}")
print(f"AHG distances: {AHG_DISTANCES}")
print(f"Dong quats:    {DONG_QUATS}" + (f" ({DONG_CSV_PATH})" if DONG_QUATS else ""))
print(f"Dong euler:    {DONG_EULER}" + (f" ({DONG_CSV_PATH})" if DONG_EULER else ""))
print(f"Add XYZ:       {ADD_XYZ}  |  Normalize XYZ: {NORMALIZE_XYZ}")

# ====================== Hyperparameters =========================

lr = 1e-3
network_type = os.getenv("GG_NETWORK_TYPE", "GCN_CAM_8_8_16_16_32")
weight_decay = 5e-4
seed = 42
BATCH_SIZE = 256
NUM_WORKERS = 0
num_epochs = int(os.getenv("GG_NUM_EPOCHS", "40"))
EARLY_STOPPING_PATIENCE = int(os.getenv("GG_PATIENCE", "10"))
LR_SCHEDULER = os.getenv("GG_LR_SCHEDULER", "none").strip().lower()  # "none" | "plateau"
LR_PLATEAU_FACTOR   = float(os.getenv("GG_LR_PLATEAU_FACTOR",   "0.5"))
LR_PLATEAU_PATIENCE = int(os.getenv("GG_LR_PLATEAU_PATIENCE",   "5"))

# Reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}, LR: {lr}, Patience: {EARLY_STOPPING_PATIENCE}")
print(f"LR scheduler: {LR_SCHEDULER}" + (f" (factor={LR_PLATEAU_FACTOR}, patience={LR_PLATEAU_PATIENCE})" if LR_SCHEDULER == "plateau" else ""))
print('--------------------------------')

# ==================== TensorBoard ================================
writer = SummaryWriter(log_dir=f'experiments/runs/{RUN_NAME}')

# ==================== Datasets ===================================
print("📦 Loading datasets...")

datasetTrain = GraspsClass(root='data/', split='train', collapse=COLLAPSE, add_joint_angles=JOINT_ANGLES, add_bone_vectors=BONE_VECTORS, add_velocity=VELOCITY, add_mano_pose=MANO_POSE, add_global_swing=GLOBAL_SWING, add_ahg_angles=AHG_ANGLES, add_ahg_distances=AHG_DISTANCES, add_dong_quats=DONG_QUATS, add_dong_euler=DONG_EULER, dong_csv_path=DONG_CSV_PATH, csv_filename=CSV_FILENAME, add_xyz=ADD_XYZ, normalize_xyz=NORMALIZE_XYZ)
datasetVal   = GraspsClass(root='data/', split='val',   collapse=COLLAPSE, add_joint_angles=JOINT_ANGLES, add_bone_vectors=BONE_VECTORS, add_velocity=VELOCITY, add_mano_pose=MANO_POSE, add_global_swing=GLOBAL_SWING, add_ahg_angles=AHG_ANGLES, add_ahg_distances=AHG_DISTANCES, add_dong_quats=DONG_QUATS, add_dong_euler=DONG_EULER, dong_csv_path=DONG_CSV_PATH, csv_filename=CSV_FILENAME, add_xyz=ADD_XYZ, normalize_xyz=NORMALIZE_XYZ)

print(f"Train: {len(datasetTrain)}, Val: {len(datasetVal)}")
print(f"Num classes:         {datasetTrain.num_classes}")

# Verify feature count matches ToGraph.F (catches stale cache)
from grasp_gcn.transforms.tograph import ToGraph as _ToGraph
_tg = _ToGraph(
    add_xyz=datasetTrain._add_xyz,
    add_joint_angles=JOINT_ANGLES,
    add_bone_vectors=BONE_VECTORS,
    add_velocity=VELOCITY,
    add_mano_pose=MANO_POSE,
    add_global_swing=GLOBAL_SWING,
    add_ahg_angles=AHG_ANGLES,
    add_ahg_distances=AHG_DISTANCES,
    add_dong_quats=DONG_QUATS,
    add_dong_euler=DONG_EULER,
    normalize_xyz=NORMALIZE_XYZ,
)
expected_F = _tg.F
actual_F   = datasetTrain.num_features
if actual_F != expected_F:
    raise RuntimeError(
        f"Feature count mismatch: dataset has {actual_F} but ToGraph expects {expected_F}. "
        "Delete stale .pt cache and re-run."
    )
print(f"Features per node:   {actual_F}  (verified)")

# DataLoaders
train_loader = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(datasetVal,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==================== Device =====================================
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device_}")
print('--------------------------------')

# ==================== Model ======================================
model_ = get_network(network_type, datasetTrain.num_features, datasetTrain.num_classes).to(device_)

def reset_weights(m):
    """Reset model weights to avoid leakage between runs."""
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

model_.apply(reset_weights)

optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_ = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_, mode='max', factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE
    )
    if LR_SCHEDULER == "plateau" else None
)


# ==================== Joint Dropout Augmentation =================
JOINT_DROPOUT_P = 0.10   # probability of masking each joint during training

def random_joint_dropout(batch, p=JOINT_DROPOUT_P):
    """
    Randomly zero out individual joints to simulate occlusion.
    Updates both x and mask so the model learns to handle missing landmarks.
    Applied only during training; eval uses full landmarks.
    """
    if p <= 0:
        return batch
    drop = torch.rand(batch.x.size(0), device=batch.x.device) < p
    if drop.any():
        batch.x    = batch.x.clone()
        batch.x[drop] = 0.0
        if hasattr(batch, 'mask') and batch.mask is not None:
            batch.mask = batch.mask.clone()
            batch.mask[drop] = 0.0
    return batch

# ==================== Evaluation Function ========================
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_)
            out = model(batch)
            label = batch.y.view(-1)
            loss = F.nll_loss(out, label)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += batch.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# ==================== Training Loop ==============================
def train(model_, train_loader_, val_loader_, writer):
    best_val_acc = 0.0
    epochs_no_improve = 0
    time_start_ = timer()
    print("🎯 Starting training...")

    for epoch in range(num_epochs):
        model_.train()
        epoch_loss = 0.0

        for batch in train_loader_:
            batch = batch.to(device_)
            batch = random_joint_dropout(batch)
            optimizer_.zero_grad()
            pred = model_(batch)
            label = batch.y.view(-1)
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer_.step()
            epoch_loss += loss.item() * batch.num_graphs

        epoch_loss /= len(train_loader_.dataset)
        writer.add_scalar("Train/Loss", epoch_loss, epoch)

        # Validation phase
        val_loss, val_acc = evaluate(model_, val_loader_)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        current_lr = optimizer_.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"LR: {current_lr:.2e}")

        if scheduler_ is not None:
            scheduler_.step(val_acc)
        writer.add_scalar("Train/LR", current_lr, epoch)

        # Save best model + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs("experiments", exist_ok=True)
            torch.save(model_.state_dict(), f"experiments/{CHECKPOINT}")
            print(f"💾 Saved best model (Val Acc = {val_acc:.3f}) -> {CHECKPOINT}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    total_time = timer() - time_start_
    print(f"\n⏱️  Training completed in {total_time/60:.2f} min.")

    # Load best checkpoint before returning
    model_.load_state_dict(torch.load(f"experiments/{CHECKPOINT}", map_location=device_))
    print(f"✅ Loaded best model (Val Acc = {best_val_acc:.3f})")
    return model_

# ==================== Run Training ===============================
trained_model = train(model_, train_loader, val_loader, writer)

# Free train/val memory before loading test set
del datasetTrain, datasetVal, train_loader, val_loader
import gc; gc.collect()

# ==================== Load Test Set (lazy) =======================
print("📦 Loading test dataset...")
datasetTest = GraspsClass(root='data/', split='test', collapse=COLLAPSE, add_joint_angles=JOINT_ANGLES, add_bone_vectors=BONE_VECTORS, add_velocity=VELOCITY, add_mano_pose=MANO_POSE, add_global_swing=GLOBAL_SWING, add_ahg_angles=AHG_ANGLES, add_ahg_distances=AHG_DISTANCES, add_dong_quats=DONG_QUATS, add_dong_euler=DONG_EULER, dong_csv_path=DONG_CSV_PATH, csv_filename=CSV_FILENAME, add_xyz=ADD_XYZ, normalize_xyz=NORMALIZE_XYZ)
test_loader = DataLoader(datasetTest, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Test: {len(datasetTest)}")

# ==================== Final Evaluation ===========================
test_loss, test_acc = evaluate(trained_model, test_loader)
print(f"\n✅ Test set → Loss: {test_loss:.4f}, Accuracy: {test_acc:.3f}")

# ==================== Classification Report ======================
y_true, y_pred = [], []
trained_model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device_)
        preds = trained_model(batch).argmax(1).cpu().numpy()
        labels = batch.y.view(-1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=3)

print("\n" + "="*60)
print("📊 CONFUSION MATRIX")
print("="*60)
print(cm)
print("\n" + "="*60)
print("📈 CLASSIFICATION REPORT")
print("="*60)
print(report)
print("="*60)


print("\n🎉 Training process complete. Ready for analysis!")
