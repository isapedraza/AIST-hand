# src/grasp_gcn/check_pipeline.py
import os
import torch
from torch_geometric.loader import DataLoader

# Imports del paquete local
from grasp_gcn.dataset.grasps import GraspsClass
from grasp_gcn.network.utils import get_network

def main():
    print("\n🧠 === Verificación del pipeline grasp_gcn ===")

    # --- 1️⃣ Cargar dataset ---
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    csv_path = os.path.join(data_root, "raw", "grasps_sample_train.csv")

    datasetTrain = GraspsClass(root=data_root, csvs=csv_path, normalize=True)

    print(f"✅ Dataset cargado: {len(datasetTrain)} muestras")
    print(f"   Features: {datasetTrain.num_features}, Classes: {datasetTrain.num_classes}")

    # --- 2️⃣ Inicializar modelo ---
    model_ = get_network("GCN_8_8_16_16_32", datasetTrain.num_features, datasetTrain.num_classes)
    print(f"✅ Modelo creado: {model_.__class__.__name__}")

    # --- 3️⃣ Probar un batch ---
    loader = DataLoader(datasetTrain, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    pred = model_(batch)

    print(f"✅ Pred shape: {pred.shape}, Label shape: {batch.y.view(-1).shape}")

    # --- 4️⃣ Calcular pérdida de prueba ---
    loss = torch.nn.functional.nll_loss(pred, batch.y.view(-1))
    print(f"✅ Loss de prueba: {loss.item():.4f}")

    print("\n🎯 Todo el pipeline está correctamente conectado.")

if __name__ == "__main__":
    main()
