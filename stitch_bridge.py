"""Soft bridge to Comfyui-MinimaxUtils SeamlessVideoStitcher.

Stitching is optional: Final Decode never imports MinimaxUtils unless
``original_images`` is connected. Without that input the Extender path is
unchanged and Frame-Interpolation is not required.
"""

from __future__ import annotations

import logging
import sys

_LOG = logging.getLogger("MiniMaxH3Extender.stitch")

_DEFAULT_RIFE_CKPTS = [
    "rife47.pth",
    "rife49.pth",
    "rife417.pth",
    "rife426.pth",
    "sudo_rife4_269.662_testV1_scale1.pth",
]

_STITCHER_CLS = None


def rife_ckpt_choices():
    """Checkpoint names for the Final Decode widget (fallback list if pack absent)."""
    # Soft probe only — never cache failure (Extender may load before MinimaxUtils).
    try:
        cls = _probe_stitcher_cls()
        if cls is not None and hasattr(cls, "INPUT_TYPES"):
            spec = cls.INPUT_TYPES().get("required", {}).get("rife_ckpt")
            if isinstance(spec, tuple) and spec and isinstance(spec[0], (list, tuple)):
                names = list(spec[0])
                if names:
                    return names
    except Exception:
        pass
    return list(_DEFAULT_RIFE_CKPTS)


def default_rife_ckpt():
    names = rife_ckpt_choices()
    return "rife49.pth" if "rife49.pth" in names else names[0]


def coerce_rife_ckpt(rife_ckpt) -> str:
    """Return a real RIFE checkpoint name.

    Legacy/shifted widgets_values can feed a boolean (``True``) or empty value
    into ``rife_ckpt``. ``str(True) == "True"`` would otherwise try to download
    a nonexistent GitHub asset named True.
    """
    if isinstance(rife_ckpt, bool) or rife_ckpt is None:
        return default_rife_ckpt()
    name = str(rife_ckpt).strip()
    if not name or name.lower() in ("true", "false", "none", "null"):
        return default_rife_ckpt()
    allowed = set(rife_ckpt_choices())
    if name in allowed:
        return name
    # Accept unknown *.pth names (custom drops) but never non-weight strings.
    if name.lower().endswith(".pth"):
        return name
    _LOG.warning(
        "Invalid rife_ckpt %r; falling back to %s",
        rife_ckpt,
        default_rife_ckpt(),
    )
    return default_rife_ckpt()


def _probe_stitcher_cls():
    """Return SeamlessVideoStitcher class if already loaded, else None."""
    global _STITCHER_CLS
    if _STITCHER_CLS is not None:
        return _STITCHER_CLS

    try:
        import nodes as comfy_nodes

        cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(
            "SeamlessVideoStitcher"
        )
        if cls is not None:
            _STITCHER_CLS = cls
            return cls
    except Exception:
        pass

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        cls = getattr(mod, "SeamlessVideoStitcher", None)
        if cls is not None and hasattr(cls, "stitch"):
            _STITCHER_CLS = cls
            return cls
        mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mappings, dict) and "SeamlessVideoStitcher" in mappings:
            _STITCHER_CLS = mappings["SeamlessVideoStitcher"]
            return _STITCHER_CLS
    return None


def _resolve_stitcher_cls():
    cls = _probe_stitcher_cls()
    if cls is not None:
        return cls
    raise ImportError(
        "MiniMax H3 Extender: optional original_images stitching requires "
        "Comfyui-MinimaxUtils (Seamless Video Stitcher) and "
        "ComfyUI-Frame-Interpolation. Install both into custom_nodes/ and "
        "restart ComfyUI."
    )


def stitch_original_with_ai(
    original_images,
    ai_images,
    *,
    ref_frames_offset: int = 20,
    rife_multiplier: int = 2,
    rife_ckpt: str | None = None,
    fast_mode: bool = False,
    ensemble: bool = True,
    ai_skip_first: int = 1,
):
    """Call external SeamlessVideoStitcher. Returns ``(stitched, bridge)``."""
    cls = _resolve_stitcher_cls()
    ckpt = coerce_rife_ckpt(rife_ckpt)
    stitched, bridge = cls().stitch(
        original_images=original_images,
        ai_images=ai_images,
        ref_frames_offset=int(ref_frames_offset),
        rife_multiplier=int(rife_multiplier),
        rife_ckpt=ckpt,
        fast_mode=bool(fast_mode),
        ensemble=bool(ensemble),
        ai_skip_first=int(ai_skip_first),
    )
    _LOG.info(
        "Seamless stitch: original=%d ai=%d → stitched=%d bridge=%d",
        int(original_images.shape[0]),
        int(ai_images.shape[0]),
        int(stitched.shape[0]),
        int(bridge.shape[0]),
    )
    return stitched, bridge
