"""Soft bridge to ComfyUI-MiniMax-H3-PDD-Acc.

PDD Acc is optional: when ``pdd_acc_lora`` is ``None`` the Extender never
imports the pack. Selecting a PDD file requires
``ComfyUI-MiniMax-H3-PDD-Acc`` to already be installed and loaded by ComfyUI;
we call its Apply node rather than vendoring the head-bank / sigma math.
"""

from __future__ import annotations

import logging
import os
import sys

import folder_paths
import torch

PDD_NONE = "None"
_logger = logging.getLogger("MiniMaxH3Extender.pdd")

_APPLY_CLS = None
_APPLY_IMPORT_ERROR = None


def ensure_pdd_folder():
    """Create/register ``models/pdd_acc`` so the dropdown works without the pack."""
    pdd_dir = os.path.join(folder_paths.models_dir, "pdd_acc")
    os.makedirs(pdd_dir, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("pdd_acc", pdd_dir, is_default=True)
    except TypeError:
        folder_paths.add_model_folder_path("pdd_acc", pdd_dir)
    except Exception:
        pass
    return pdd_dir


def pdd_acc_choices():
    ensure_pdd_folder()
    names = []
    try:
        names = list(folder_paths.get_filename_list("pdd_acc") or [])
    except Exception:
        names = []
    return [PDD_NONE] + list(names)


def is_pdd_enabled(pdd_acc_lora) -> bool:
    name = str(pdd_acc_lora or "").strip()
    return bool(name) and name != PDD_NONE


def _trunk_from_generation_mode(generation_mode: str) -> str:
    return "fl2va" if str(generation_mode).lower().strip() == "fl2va" else "ref2va"


def validate_pdd_file_for_mode(pdd_file: str, generation_mode: str) -> None:
    """Hard-fail when the filename clearly targets the other H3 trunk."""
    expected = _trunk_from_generation_mode(generation_mode)
    other = "ref2va" if expected == "fl2va" else "fl2va"
    low = str(pdd_file).lower().replace("-", "_")
    has_expected = expected in low
    has_other = other in low
    if has_other and not has_expected:
        raise ValueError(
            f"MiniMax H3 Extender: PDD Acc file '{pdd_file}' looks like a {other.upper()} "
            f"distill, but generation_mode is {expected}. Use the matching "
            f"{expected.upper()} Acc LoRA (models/pdd_acc/)."
        )
    if not has_expected and not has_other:
        _logger.warning(
            "PDD Acc filename '%s' does not contain fl2va/ref2va; relying on "
            "ComfyUI-MiniMax-H3-PDD-Acc trunk fingerprint check.",
            pdd_file,
        )


def _resolve_pdd_apply_cls():
    global _APPLY_CLS, _APPLY_IMPORT_ERROR
    if _APPLY_CLS is not None:
        return _APPLY_CLS
    if _APPLY_IMPORT_ERROR is not None:
        raise _APPLY_IMPORT_ERROR

    try:
        try:
            import nodes as comfy_nodes

            cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(
                "MiniMaxH3PDDAccApply"
            )
            if cls is not None:
                _APPLY_CLS = cls
                return cls
        except Exception:
            pass

        for mod in list(sys.modules.values()):
            if mod is None:
                continue
            cls = getattr(mod, "MiniMaxH3PDDAccApply", None)
            if cls is not None and hasattr(cls, "apply"):
                _APPLY_CLS = cls
                return cls

        raise ImportError(
            "MiniMaxH3PDDAccApply was not found in loaded custom nodes"
        )
    except Exception as exc:
        err = ImportError(
            "MiniMax H3 Extender: PDD Acc requires the ComfyUI-MiniMax-H3-PDD-Acc "
            "custom node pack. Install "
            "https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc "
            "into ComfyUI/custom_nodes/, put Acc LoRAs in models/pdd_acc/, "
            f"and restart ComfyUI. ({exc})"
        )
        _APPLY_IMPORT_ERROR = err
        raise err from exc


def trim_pdd_sigmas(sigmas, denoise: float):
    """Keep the last round(nfe * denoise) PDD blocks (same as PDD Acc Scheduler)."""
    denoise = float(denoise)
    if sigmas is None:
        return None
    if denoise >= 1.0:
        return sigmas
    if denoise <= 0.0:
        return torch.FloatTensor([])
    nfe = max(1, int(sigmas.shape[-1]) - 1)
    keep = max(1, int(round(nfe * denoise)))
    return sigmas[-(keep + 1) :]


def apply_pdd_acc(
    model,
    pdd_file: str,
    *,
    generation_mode: str,
    nfe: str | int = "8",
    lora_strength: float = 1.0,
    head_strength: float = 1.0,
    denoise: float = 1.0,
):
    """Apply PDD Acc via the external pack. Returns ``(model, sigmas, info)``."""
    ensure_pdd_folder()
    validate_pdd_file_for_mode(pdd_file, generation_mode)
    apply_cls = _resolve_pdd_apply_cls()

    try:
        folder_paths.get_full_path_or_raise("pdd_acc", pdd_file)
    except Exception as exc:
        raise ValueError(
            f"MiniMax H3 Extender: PDD Acc file '{pdd_file}' was not found in "
            f"models/pdd_acc/. Place the Alibaba Acc LoRA there "
            f"(FL2VA or Ref2VA matching generation_mode)."
        ) from exc

    nfe_s = str(int(nfe))
    patched, sigmas, info = apply_cls().apply(
        model,
        pdd_file,
        nfe_s,
        float(lora_strength),
        float(head_strength),
        "error",
        partition="",
        enabled=True,
        bypass_sigmas=None,
        partition_check="error",
    )
    sigmas = trim_pdd_sigmas(sigmas, denoise)
    _logger.info(
        "PDD Acc enabled (%s, nfe=%s): using trained sigma grid; "
        "steps/scheduler widgets are ignored for sampling.",
        pdd_file,
        nfe_s,
    )
    return patched, sigmas, info
