from typing import Dict, Any

def encode_theta(theta: Dict[str, Any]) -> Dict[str, float]:
    enc = {}
    for k, v in theta.items():
        if isinstance(v, bool):
            enc[f"th__{k}__bool"] = float(v)
        elif isinstance(v, (int, float)):
            enc[f"th__{k}__num"] = float(v)
        else:
            enc[f"th__{k}__cat={v}"] = 1.0
    return enc

def encode_fx(features: dict) -> dict:
    return {f"fx__{k}": float(v) for k, v in features.items()}