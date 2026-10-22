import numpy as np
from smplx import MANO


def build_mano(model_path, is_rhand, flat_hand_mean=False):
    """Build the differentiable MANO layer used by Contact Inner Guidance."""
    # The official MANO v1.2 pickle imports chumpy, which still references the
    # NumPy aliases removed in NumPy 1.24.
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    return MANO(
        model_path=model_path,
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        create_transl=False,
    )
