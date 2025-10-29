# test_gcn_network.py  (o test_model_pooling.py)
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from network.gcn_network import GCN_8_8_16_16_32

# Reproducibilidad básica
torch.manual_seed(0)

# === Verbosidad controlada ===
VERBOSE = True
def _dbg(*args):
    if VERBOSE:
        print(*args)

def make_chain_edges(n: int) -> torch.Tensor:
    """Aristas bidireccionales en cadena: 0-1-2-...-(n-1)."""
    rows, cols = [], []
    for i in range(n - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
    return torch.tensor([rows, cols], dtype=torch.long)

def test_A_smoke_and_grads_single_graph():
    """Smoke test: forward + gradientes con un solo grafo."""
    n, f, k = 21, 3, 3               # 21 nodos, 3 features, 3 clases
    x = torch.randn(n, f)
    ei = make_chain_edges(n)
    y = torch.tensor([1])

    data = Data(x=x, edge_index=ei, y=y)
    # Si no usamos DataLoader, creamos batch manual (todo ceros)
    data.batch = torch.zeros(n, dtype=torch.long)

    model = GCN_8_8_16_16_32(numFeatures=f, numClasses=k)
    model.train()

    out = model(data)  # [1, k]
    _dbg("\n[A] out.shape:", out.shape)
    _dbg("[A] logsumexp(out) (≈0):", torch.logsumexp(out, dim=1))
    _dbg("[A] softmax sums (≈1):", torch.exp(out).sum(1))
    assert out.shape == (1, k)

    loss = F.nll_loss(out, y)
    _dbg("[A] loss:", float(loss.item()))
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    _dbg("[A] grad norms:", [round(g, 4) for g in grad_norms])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_B_batch_two_graphs():
    """Batch real: 2 grafos en el mismo forward."""
    n, f, k = 21, 3, 3

    # Grafo 1
    x1 = torch.randn(n, f)
    ei1 = make_chain_edges(n)
    b1 = torch.zeros(n, dtype=torch.long)

    # Grafo 2 (desplazar índices de aristas en +n)
    x2 = torch.randn(n, f)
    ei2 = make_chain_edges(n) + n
    b2 = torch.ones(n, dtype=torch.long)

    # Concatenar para formar el batch
    x = torch.cat([x1, x2], dim=0)             # [42, f]
    ei = torch.cat([ei1, ei2], dim=1)          # [2, E1+E2]
    batch = torch.cat([b1, b2], dim=0)         # [42]
    y = torch.tensor([2, 0])                   # dos etiquetas, una por grafo

    data = Data(x=x, edge_index=ei, y=y, batch=batch)

    model = GCN_8_8_16_16_32(numFeatures=f, numClasses=k)
    model.eval()

    out = model(data)  # [2, k]
    _dbg("\n[B] out.shape:", out.shape)
    _dbg("[B] logsumexp(out) (≈0):", torch.logsumexp(out, dim=1))
    _dbg("[B] softmax sums (≈1):", torch.exp(out).sum(1))
    _dbg("[B] preds:", out.argmax(1).tolist())
    assert out.shape == (2, k)

def test_C_missing_nodes_with_mask():
    """Robustez a nodos faltantes: usar mask para enmascarar."""
    n, f, k = 21, 3, 3
    x = torch.randn(n, f)

    # Simular nodos perdidos (valores 0 y mask=0 en esos nodos)
    missing = [3, 7, 14]
    mask = torch.ones(n, 1)
    x[missing] = 0.0
    mask[missing] = 0.0

    ei = make_chain_edges(n)
    y = torch.tensor([0])

    data = Data(x=x, edge_index=ei, y=y)
    data.batch = torch.zeros(n, dtype=torch.long)
    data.mask = mask

    model = GCN_8_8_16_16_32(numFeatures=f, numClasses=k)
    model.train()

    _dbg("\n[C] missing idx:", missing, "| mask_valid_count:", int(mask.sum().item()))
    out = model(data)  # [1, k]
    _dbg("[C] out.shape:", out.shape)
    _dbg("[C] logsumexp(out) (≈0):", torch.logsumexp(out, dim=1))
    _dbg("[C] softmax sums (≈1):", torch.exp(out).sum(1))
    assert out.shape == (1, k)

    loss = F.nll_loss(out, y)
    _dbg("[C] loss:", float(loss.item()))
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    _dbg("[C] grad norms:", [round(g, 4) for g in grad_norms])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def main():
    print("A) single graph: forward + grads")
    test_A_smoke_and_grads_single_graph(); print("OK")
    print("B) batch=2 graphs")
    test_B_batch_two_graphs(); print("OK")
    print("C) missing nodes + mask")
    test_C_missing_nodes_with_mask(); print("OK")
    print("=== TODO OK ===")

if __name__ == "__main__":
    main()
