from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from pfhedge.nn import Hedger


# -------------------------- small utilities --------------------------------- #

def _get_device(device: Optional[str] = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def _ensure_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if path.is_dir():
        return
    if path.suffix:  # file path -> ensure its parent exists
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------- build & train --------------------------------- #

def build_hedger(
    model: nn.Module,
    features: Sequence[str],
    *,
    device: Optional[str] = "auto",
) -> Hedger:
    """
    Create a pfhedge.nn.Hedger with your model and feature list.
    No magic: mirrors what you do in the notebook.
    """
    dev = _get_device(device)
    model = model.to(dev)
    hedger = Hedger(model, list(features))
    return hedger


def train_model(
    hedger: Hedger,
    derivative: Any,
    *,
    n_paths: int = 50_000,
    n_epochs: int = 50,
    init_state: Optional[Tuple] = None,   # e.g. (S0,) or whatever you already pass
    seed: Optional[int] = 42,
    save_path: Optional[Union[str, Path]] = None,  # if provided, saves at end
    save_meta: Optional[Dict[str, Any]] = None,    # anything you want to stash with the ckpt
    device: Optional[str] = "auto",
    **fit_kwargs: Any,                     # passthrough to Hedger.fit (lr, batch_size, etc.)
) -> Dict[str, Any]:
    """
    Thin wrapper around Hedger.fit exactly like you do in the notebook.
    - Sets seed if provided
    - Calls .fit(...) once
    - Saves checkpoint if save_path is given
    - Returns the log dict from training (if Hedger.fit returns it), plus 'save_path'
    """
    _set_seed(seed)
    _ = _get_device(device)  # ensure CUDA init if needed; hedger.model already moved in build_hedger()

    # Fit (1:1 with your current call pattern)
    # Example in notebook:
    #   hedger.fit(derivative, n_paths=..., n_epochs=..., init_state=(S0,), ...)
    log = hedger.fit(
        derivative,
        n_paths=n_paths,
        n_epochs=n_epochs,
        init_state=init_state,
        **fit_kwargs,
    )

    # Save (optional)
    if save_path is not None:
        save_checkpoint(save_path, hedger.model, meta=save_meta)

    # Standardize return
    out = {"log": log, "save_path": str(save_path) if save_path else None}
    return out


# ------------------------------ save / load --------------------------------- #

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves a checkpoint with:
      - 'state_dict': model.state_dict()
      - 'meta': any dictionary you want to stash (features, tc_range, cfg, etc.)
    """
    _ensure_dir(path)
    obj = {
        "state_dict": model.state_dict(),
        "meta": meta or {},
        "torch_version": torch.__version__,
    }
    torch.save(obj, str(path))


def load_checkpoint_if_exists(
    path: Union[str, Path],
    model: nn.Module,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    If a checkpoint file exists at 'path', load it into 'model' and return its meta dict.
    If it doesn't exist, return None (and do nothing).
    """
    path = Path(path)
    if not path.exists():
        return None

    ckpt = torch.load(str(path), map_location=map_location)
    state = ckpt.get("state_dict", ckpt)  # tolerate raw state_dict files
    model.load_state_dict(state, strict=strict)
    return ckpt.get("meta", {})


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and return its meta dict. Raises FileNotFoundError if missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location=map_location)
    state = ckpt.get("state_dict", ckpt)  # tolerate raw state_dict files
    model.load_state_dict(state, strict=strict)
    return ckpt.get("meta", {})


# ------------------------------ convenience --------------------------------- #

def load_or_build(
    maybe_path: Union[str, Path],
    model: nn.Module,
    features: Sequence[str],
    *,
    device: Optional[str] = "auto",
    strict: bool = True,
) -> Hedger:
    """
    Convenience:
      - if 'maybe_path' exists, load weights into 'model'
      - then build and return a Hedger(model, features)

    This lets you write in the notebook:
        hedger = load_or_build("runs/base.pt", model, FEATURES)
    """
    meta = load_checkpoint_if_exists(maybe_path, model, strict=strict)
    # You can inspect 'meta' in the notebook if you want (e.g., to check features match)
    return build_hedger(model, features, device=device)
