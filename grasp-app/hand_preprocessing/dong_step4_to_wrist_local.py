"""
Paso 4 (Dong-lite): transformar world -> wrist local.

Basado en Dong & Payandeh (2025):
- Eq. (5)-(7): construccion de ejes X0, Y0, Z0
- Eq. (9): T0w = [R0w d0w; 0 1]
- Eq. (16): d_i^0 = (T0w)^(-1) d_i^w

Controles:
- SPACE: captura y muestra T0w, T0w^-1 y 21 puntos en marco local
- Q / ESC: salir
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Reduce ruido de logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)


def _open_capture(camera_index: int | str):
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if os.name == "posix" else [cv2.CAP_ANY]
    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
        except TypeError:
            cap = cv2.VideoCapture(camera_index)
        if cap and cap.isOpened():
            return cap
        last_cap = cap
        if cap is not None:
            cap.release()
    return last_cap


def _camera_source(raw: str) -> int | str:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Vector degenerado (norm={n:.3e})")
    return v / n


def _world_points(hand_world_landmarks) -> np.ndarray:
    return np.asarray([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark], dtype=float)


def _compute_wrist_frame(points_w: np.ndarray) -> dict:
    # Dong Eq. (5)-(7):
    # Y0 = normalize(d13 - d5)
    # Z0 = normalize(normalize(d9 - d0) x Y0)
    # X0 = Y0 x Z0
    d0 = points_w[0]
    d5 = points_w[5]
    d9 = points_w[9]
    d13 = points_w[13]

    v_13_5 = d13 - d5
    v_9_0 = d9 - d0

    y0 = _normalize(v_13_5)
    v_9_0_hat = _normalize(v_9_0)
    z0 = _normalize(np.cross(v_9_0_hat, y0))
    x0 = _normalize(np.cross(y0, z0))

    r0w = np.column_stack((x0, y0, z0))

    # Dong Eq. (8):
    # beta  = asin(-X0_z)
    # gamma = atan2(X0_y, X0_x)
    # alpha = atan2(Y0_z, Z0_z)
    beta = float(np.arcsin(np.clip(-x0[2], -1.0, 1.0)))
    gamma = float(np.arctan2(x0[1], x0[0]))
    alpha = float(np.arctan2(y0[2], z0[2]))

    return {
        "d0": d0,
        "d5": d5,
        "d9": d9,
        "d13": d13,
        "v_13_5": v_13_5,
        "v_9_0": v_9_0,
        "v_9_0_hat": v_9_0_hat,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "r0w": r0w,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "gamma_rad": gamma,
        "alpha_deg": float(np.degrees(alpha)),
        "beta_deg": float(np.degrees(beta)),
        "gamma_deg": float(np.degrees(gamma)),
        "det_r0w": float(np.linalg.det(r0w)),
        "orth_err": float(np.linalg.norm(r0w.T @ r0w - np.eye(3))),
    }


def _build_t0w(r0w: np.ndarray, d0w: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = r0w
    t[:3, 3] = d0w
    return t


def _transform_world_to_local(points_w: np.ndarray, t0w_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((points_w.shape[0], 1), dtype=float)
    pw_h = np.hstack((points_w, ones))  # [21,4]
    p0_h = (t0w_inv @ pw_h.T).T
    return p0_h[:, :3]


def _transform_local_to_world(points_0: np.ndarray, t0w: np.ndarray) -> np.ndarray:
    ones = np.ones((points_0.shape[0], 1), dtype=float)
    p0_h = np.hstack((points_0, ones))  # [21,4]
    pw_h = (t0w @ p0_h.T).T
    return pw_h[:, :3]


def _set_equal_axes_3d(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])) / 2.0
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _set_shared_axes_3d(axes, points_list: list[np.ndarray]) -> None:
    all_pts = np.vstack(points_list)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])) / 2.0
    radius = max(radius, 1e-3)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)


def _plot_points_3d(
    ax,
    points: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    set_equal_axes: bool = True,
) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=28)
    for i, p in enumerate(points):
        ax.text(p[0], p[1], p[2], str(i), fontsize=7)
    for i, j in edges:
        seg = points[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if set_equal_axes:
        _set_equal_axes_3d(ax, points)


def _plot_world_vs_local(
    points_w: np.ndarray,
    points_0: np.ndarray,
    capture_id: int,
    edges: list[tuple[int, int]],
    block: bool = False,
    save_dir: str | None = None,
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] matplotlib no disponible: {exc}", flush=True)
        return
    backend = str(matplotlib.get_backend())
    print(f"[PLOT] backend={backend} | block={block}", flush=True)

    fig = plt.figure(figsize=(12, 5))
    ncols = 2
    ax_w = fig.add_subplot(1, ncols, 1, projection="3d")
    ax_l = fig.add_subplot(1, ncols, 2, projection="3d")

    _plot_points_3d(ax_w, points_w, edges, "World (d_i^w)", set_equal_axes=False)
    _plot_points_3d(ax_l, points_0, edges, "Wrist Local (d_i^0)", set_equal_axes=False)
    axes = [ax_w, ax_l]
    points_for_limits = [points_w, points_0]
    _set_shared_axes_3d(axes, points_for_limits)

    fig.suptitle(f"Dong Step4 - Captura {capture_id}")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(save_dir, f"step4_capture_{capture_id:03d}_{stamp}.png")
        fig.savefig(out_path, dpi=170)
        print(f"[PLOT] guardado en: {out_path}", flush=True)

    backend_lower = backend.lower()
    non_interactive_backends = {
        "agg",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "cairo",
    }
    if backend_lower in non_interactive_backends:
        print("[PLOT] backend no interactivo; se guarda PNG pero no se abre ventana.", flush=True)
        plt.close(fig)
        return

    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.2)
        print("[PLOT] ventana solicitada (si no aparece, revisa DISPLAY/entorno grafico).", flush=True)


def _print_matrix(name: str, m: np.ndarray) -> None:
    print(f"{name} =", flush=True)
    for i in range(m.shape[0]):
        print("  [" + ", ".join(f"{m[i,j]:+.6f}" for j in range(m.shape[1])) + "]", flush=True)


def _print_vector(name: str, v: np.ndarray) -> None:
    print(f"{name} = [{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]", flush=True)


def _print_points_world(points_w: np.ndarray) -> None:
    print("Entrada d_i^w (21 puntos world):", flush=True)
    for i, p in enumerate(points_w):
        print(f"[{i:02d}] x={p[0]:+.6f}  y={p[1]:+.6f}  z={p[2]:+.6f}", flush=True)


def _print_points_local(points_0: np.ndarray) -> None:
    print("d_i^0 (21 puntos en marco de muneca):", flush=True)
    for i, p in enumerate(points_0):
        print(f"[{i:02d}] x={p[0]:+.6f}  y={p[1]:+.6f}  z={p[2]:+.6f}", flush=True)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paso 4 Dong-lite: world -> wrist local")
    parser.add_argument("--camera", type=str, default="0", help="Indice de camara (0,1,2,...) o URL.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta de imagen para procesar un solo frame (sin camara).",
    )
    parser.add_argument("--mirror", action="store_true", help="Espejar vista de camara.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Mostrar plot 3D world vs local en cada captura (SPACE).",
    )
    parser.add_argument(
        "--plot-block",
        action="store_true",
        help="Bloquear ejecucion hasta cerrar cada plot.",
    )
    parser.add_argument(
        "--plot-save-dir",
        type=str,
        default="/home/yareeez/AIST-hand/grasp-app/plots_step4",
        help="Directorio donde se guarda PNG de cada captura cuando --plot esta activo.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    edges = sorted(list(mp_hands.HAND_CONNECTIONS))

    def run_pipeline(points_w: np.ndarray, has_2d: bool, capture_id: int) -> None:
        wf = _compute_wrist_frame(points_w)
        r0w = wf["r0w"]
        d0w = wf["d0"]
        t0w = _build_t0w(r0w, d0w)
        t0w_inv = np.linalg.inv(t0w)
        points_0 = _transform_world_to_local(points_w, t0w_inv)
        points_w_rec = _transform_local_to_world(points_0, t0w)

        print(
            f"\n=== Step4 World->Local | Captura {capture_id} ===",
            flush=True,
        )
        print(
            f"[SOURCE] using=WORLD has_2d={has_2d} has_world=True n_points={len(points_w)}",
            flush=True,
        )
        print("\n[1] ENTRADA", flush=True)
        _print_points_world(points_w)

        print("\n[2] WRIST FRAME (Eq. 5-8)", flush=True)
        _print_vector("d0 (kp0)", wf["d0"])
        _print_vector("d5 (kp5)", wf["d5"])
        _print_vector("d9 (kp9)", wf["d9"])
        _print_vector("d13 (kp13)", wf["d13"])
        _print_vector("v_13_5 = d13-d5", wf["v_13_5"])
        _print_vector("v_9_0 = d9-d0", wf["v_9_0"])
        _print_vector("v_9_0_hat", wf["v_9_0_hat"])
        _print_vector("Y0", wf["y0"])
        _print_vector("Z0", wf["z0"])
        _print_vector("X0", wf["x0"])
        _print_matrix("R0w = [X0 Y0 Z0]", r0w)
        print(
            f"Euler (Dong Eq.8): alpha={wf['alpha_rad']:+.6f} rad ({wf['alpha_deg']:+.2f} deg) | "
            f"beta={wf['beta_rad']:+.6f} rad ({wf['beta_deg']:+.2f} deg) | "
            f"gamma={wf['gamma_rad']:+.6f} rad ({wf['gamma_deg']:+.2f} deg)",
            flush=True,
        )
        print(
            f"check R0w: det={wf['det_r0w']:+.6f} | ||R^T R - I||={wf['orth_err']:.3e}",
            flush=True,
        )

        print("\n[3] TRANSFORMACIONES (Eq. 9 y Eq. 16)", flush=True)
        _print_matrix("T0w", t0w)
        _print_matrix("T0w_inv", t0w_inv)

        print("\n[4] SALIDA LOCAL d_i^0", flush=True)
        _print_points_local(points_0)

        print("\n[5] CHECKS DE CONSISTENCIA", flush=True)
        p0_norm = float(np.linalg.norm(points_0[0]))
        inv_err = float(np.linalg.norm(t0w @ t0w_inv - np.eye(4)))
        rec_err = float(np.max(np.linalg.norm(points_w_rec - points_w, axis=1)))
        print(
            f"check: ||d0^0||={p0_norm:.3e} (deberia estar cerca de 0)",
            flush=True,
        )
        print(
            f"check: ||T0w*T0w_inv - I||={inv_err:.3e} (deberia ser ~0)",
            flush=True,
        )
        print(
            f"check: max_i ||d_i^w(rec) - d_i^w||={rec_err:.3e} (deberia ser ~0)",
            flush=True,
        )
        if args.plot:
            _plot_world_vs_local(
                points_w=points_w,
                points_0=points_0,
                capture_id=capture_id,
                edges=edges,
                block=args.plot_block,
                save_dir=args.plot_save_dir,
            )

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"No se pudo leer la imagen: {args.image}", flush=True)
            return 1
        if args.mirror:
            image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            result = hands.process(rgb)
            if not result.multi_hand_world_landmarks:
                print("No hay world_landmarks en la imagen.", flush=True)
                return 1
            has_2d = bool(result.multi_hand_landmarks)
            points_w = _world_points(result.multi_hand_world_landmarks[0])
            try:
                run_pipeline(points_w=points_w, has_2d=has_2d, capture_id=1)
            except (ValueError, KeyError) as exc:
                print(f"Frame degenerado: {exc}", flush=True)
                return 1
        return 0

    source = _camera_source(args.camera)

    cap = _open_capture(source)
    if not cap or not cap.isOpened():
        print(f"No se pudo abrir la camara: {source!r}", flush=True)
        return 1

    capture_id = 0
    win = "Dong Step 4 - World to Wrist Local"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("Iniciado. SPACE: transformar a marco de muneca | Q/ESC: salir.", flush=True)
    if args.plot:
        print(
            f"[PLOT] activo: se intentara abrir ventana y guardar PNG en {args.plot_save_dir}",
            flush=True,
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_2d = None
            hand_world = None
            if result.multi_hand_landmarks:
                hand_2d = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_2d, mp_hands.HAND_CONNECTIONS)
                h, w = frame.shape[:2]
                for idx, lm in enumerate(hand_2d.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.putText(
                        frame,
                        str(idx),
                        (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if result.multi_hand_world_landmarks:
                hand_world = result.multi_hand_world_landmarks[0]

            msg = f"SPACE: world->local | WORLD={'YES' if hand_world is not None else 'NO'} | Q/ESC"
            if hand_2d is None:
                msg = "No hand detected | " + msg
            cv2.putText(
                frame,
                msg,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key == 32:  # SPACE
                if hand_world is None:
                    print("\nNo hay world_landmarks en este frame.", flush=True)
                    continue

                try:
                    capture_id += 1
                    points_w = _world_points(hand_world)
                    run_pipeline(points_w=points_w, has_2d=(hand_2d is not None), capture_id=capture_id)
                except (ValueError, KeyError) as exc:
                    print(f"\nFrame degenerado: {exc}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
