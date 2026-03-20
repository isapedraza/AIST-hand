from __future__ import annotations

import os
from pathlib import Path

from grasp_gcn.dataset.grasps import GRASP_CLASS_NAMES, TAXONOMY_V1_CLASS_NAMES


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANT = "c28"

# Registry: variant_name -> spec dict
# Each spec contains everything needed to load and run a model:
#   model_path:       path to .pth checkpoint (relative to grasp-app/models/)
#   num_classes:      number of output classes
#   num_node_features: number of input features per node
#   class_names:      dict {class_id: class_name}
#   tograph_kwargs:   kwargs forwarded to ToGraph (must match training config exactly)
_REGISTRY: dict[str, dict] = {
    "c28": {
        "model_path": "best_model_run006_c28_xyz.pth",
        "num_classes": 28,
        "num_node_features": 4,
        "class_names": GRASP_CLASS_NAMES,
        "tograph_kwargs": {
            "features": "xyz",
            "add_joint_angles": True,
            "add_cmc_angle": True,
            "add_bone_vectors": False,
        },
    },
    "taxonomy_v1": {
        "model_path": "best_model_run007_c17_xyz.pth",
        "num_classes": 17,
        "num_node_features": 4,
        "class_names": TAXONOMY_V1_CLASS_NAMES,
        "tograph_kwargs": {
            "features": "xyz",
            "add_joint_angles": True,
            "add_cmc_angle": True,
            "add_bone_vectors": False,
        },
    },
    "c28_bone": {
        "model_path": "best_model_run008_c28_xyz_bone.pth",
        "num_classes": 28,
        "num_node_features": 7,
        "class_names": GRASP_CLASS_NAMES,
        "tograph_kwargs": {
            "features": "xyz",
            "add_joint_angles": True,
            "add_cmc_angle": True,
            "add_bone_vectors": True,
            "add_velocity": False,
        },
    },
    "c28_bone_vel": {
        "model_path": "best_model_run009_c28_xyz_bone_vel.pth",
        "num_classes": 28,
        "num_node_features": 10,  # xyz(3) + flex(1) + bone(3) + vel(3)
        "class_names": GRASP_CLASS_NAMES,
        "tograph_kwargs": {
            "features": "xyz",
            "add_joint_angles": True,
            "add_cmc_angle": True,
            "add_bone_vectors": True,
            "add_velocity": True,
        },
    },
    "c28_bone_vel_pose": {
        "model_path": "best_model_run010_c28_xyz_bone_vel_pose.pth",
        "num_classes": 28,
        "num_node_features": 13,  # xyz(3) + flex(1) + bone(3) + vel(3) + pose(3)
        "class_names": GRASP_CLASS_NAMES,
        "tograph_kwargs": {
            "features": "xyz",
            "add_joint_angles": True,
            "add_cmc_angle": True,
            "add_bone_vectors": True,
            "add_velocity": True,
            "add_mano_pose": True,
        },
    },
}

SUPPORTED_VARIANTS = set(_REGISTRY.keys())


def resolve_model_spec(variant: str | None = None) -> dict:
    if variant is None:
        variant = os.getenv("GRAPHGRASP_MODEL_VARIANT", DEFAULT_VARIANT).strip().lower()
    if variant not in _REGISTRY:
        raise ValueError(
            f"Unsupported GRAPHGRASP_MODEL_VARIANT={variant!r}. "
            f"Expected one of {sorted(SUPPORTED_VARIANTS)}."
        )

    spec = dict(_REGISTRY[variant])
    model_path_override = os.getenv("GRAPHGRASP_MODEL_PATH", "").strip()
    if model_path_override:
        spec["model_path"] = Path(model_path_override).expanduser()
    else:
        spec["model_path"] = ROOT / "grasp-app" / "models" / spec["model_path"]

    spec["variant"] = variant
    return spec
