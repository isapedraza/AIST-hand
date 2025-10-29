import io
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import mediapipe as mp

from grasp_gcn.transforms.tograph import ToGraph  # tu clase corregida

# --- MediaPipe Hands (una sola mano, imagen estática) ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Mapea índices MediaPipe -> nombres de tus JOINTS
JOINTS = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
]

to_graph = ToGraph()

def build_sample_from_landmarks(landmarks: np.ndarray) -> dict:
    """
    landmarks: (21, 3) con (x,y,z) NORMALIZADOS de MediaPipe.
    Devuelve dict {JOINT: (x,y,z)} + grasp_type=0 (dummy).
    """
    sample = {}
    for i, name in enumerate(JOINTS):
        x, y, z = landmarks[i]
        sample[name] = np.array([x, y, z], dtype=np.float32)
    sample["grasp_type"] = 0
    return sample

def coords_dataframe(landmarks: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, jn in enumerate(JOINTS):
        x, y, z = landmarks[i]
        rows.append([jn, float(x), float(y), float(z)])
    return pd.DataFrame(rows, columns=["joint", "x", "y", "z"])

def render_graph_image(g_data: torch.Tensor, xy: np.ndarray) -> np.ndarray:
    """
    g_data: Data(x, edge_index, y)
    xy: (21, 2) posiciones 2D (x,y) normalizadas de MediaPipe
    Devuelve PNG BGR (np.ndarray) del grafo.
    """
    G = nx.Graph()
    for i in range(21):
        G.add_node(i)
    edges = g_data.edge_index.cpu().numpy().T
    for (u, v) in edges:
        G.add_edge(int(u), int(v))

    pos = {i: (float(xy[i,0]), float(xy[i,1])) for i in range(21)}

    fig = plt.figure(figsize=(4,4), dpi=150)
    ax = fig.add_subplot(111)
    nx.draw(G, pos=pos, with_labels=True, node_size=300, font_size=6, ax=ax)
    ax.set_title("Graph (nodes=21)")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    buf = io.BytesIO()
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    file_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
    return png

def process(image: np.ndarray):
    """
    image entra como RGB (Gradio).
    Salidas: imagen con landmarks (RGB), tabla coords, imagen grafo (RGB).
    """
    img_rgb = image.copy()
    result = hands.process(img_rgb)  # MediaPipe espera RGB

    if not result.multi_hand_landmarks:
        raise gr.Error("No se detectó ninguna mano en la imagen.")

    handLms = result.multi_hand_landmarks[0]  # primera mano
    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in handLms.landmark], dtype=np.float32)  # (21,3)

    # Construir grafo
    sample = build_sample_from_landmarks(landmarks)
    g = to_graph(sample)

    # Imagen con landmarks (dibujado en RGB)
    drawn = img_rgb.copy()
    mp_draw.draw_landmarks(drawn, handLms, mp_hands.HAND_CONNECTIONS)

    # Tabla (x,y,z)
    df = coords_dataframe(landmarks)

    # Render del grafo (usa (x,y) normalizados para posicionar)
    xy = landmarks[:, :2]
    graph_img_bgr = render_graph_image(g, xy)
    graph_img_rgb = cv2.cvtColor(graph_img_bgr, cv2.COLOR_BGR2RGB)

    return drawn, df, graph_img_rgb

# --------- UI Gradio ----------
with gr.Blocks(title="Hand → ToGraph Demo") as demo:
    gr.Markdown("## Hand Landmarks → PyG Graph (MediaPipe, demo simple)")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Sube una imagen", type="numpy")
            btn = gr.Button("Procesar")
        with gr.Column():
            out_img = gr.Image(label="Imagen con landmarks", interactive=False)
            out_tbl = gr.Dataframe(label="Coordenadas (x,y,z)")
            out_graph = gr.Image(label="Grafo (21 nodos)", interactive=False)

    btn.click(process, inputs=inp, outputs=[out_img, out_tbl, out_graph])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
