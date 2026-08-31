"""
MiniMax H3 Extender
===================

One horizontal, JS-driven sequence node that replaces the repeated
Ref2VA -> Motion Context -> Sampler -> Disk Join graph while keeping the
validated disk cache and the separate Final Decode / Preview node.

The node intentionally accepts an already-patched H3 MODEL. Sigma-shift,
upstream LoRA, Spectrum or other model patches therefore compose normally
before the Extender; optional card-local LoRAs can be stacked on top per clip.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path
import secrets
import shutil
import time
import uuid
import zipfile
import numpy as np
import torch
import torchaudio
import comfy.model_management
import comfy.sd
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import folder_paths
import node_helpers
from aiohttp import web
from PIL import Image, ImageEnhance, ImageOps
from server import PromptServer

from .motion_context_ram import MiniMaxH3MotionContextRAM
from .prompt_bridge import MAX_PROMPTS, PROMPT_PACK_TYPE, _prompt_pack_signature
from .reference_bridge import MAX_REFERENCE_SLOTS, REF_PACK_TYPE
from .motion_context_disk import (
    CACHE_VERSION,
    CACHE_TYPE,
    _DATA_START,
    _chain_paths,
    _decoded_audio_cache_path,
    _decoded_audio_cache_end,
    _decoded_preview_cache_path,
    _decoded_preview_video_cache_path,
    _latest_preview_temp_path,
    PREVIEW_AUDIO_MODE,
    _ensure_cache_root,
    _safe_name,
    _write_json_atomic,
    MiniMaxH3MotionContextDiskJoin,
    _cache_size_mb,
    _load_manifest_from_paths,
    _manifest_for_first,
    _truncate_chain,
    _final_frame_count,
)
from .fl2va_engine import (
    normalize_mode as _normalize_generation_mode,
    cache_owner_id as _fl2va_cache_owner_id,
    make_fl2va_conditioning,
    sync_fl2va_manifest,
    cached_fl2va_ids,
    store_fl2va_segment,
    set_fl2va_validation,
    fl2va_cache_state,
    resolve_fl2va_previous_frame,
    drop_fl2va_cached_ids,
    fl2va_last_frame_path,
    fl2va_continuity_meta,
    continuity_signatures_for_segments,
    compact_fl2va_cache,
    fl2va_project_continuity_files,
    install_fl2va_project_continuity,
)

BUILD = "minimax-h3-extender-v2.3.2"
FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
REF_VIDEO_BASE_SHORT_EDGE = 768
REF_VIDEO_MAX_PIXELS = 768 * 1344
MAX_VIDEO_REFS = 3
MAX_STANDALONE_AUDIO_REFS = 3
MAX_REF_VIDEO_FRAMES = 362
MAX_MIXED_REF_ITEMS = 12
MIN_REF_AUDIO_SECONDS = 2.0


class _LazyUnconnected:
    """Sentinel used to distinguish an empty optional socket from a lazy input.

    ComfyUI passes ``None`` for a connected lazy input that has not been
    evaluated yet. An omitted optional socket never reaches the method, so its
    default value can safely use this distinct sentinel.
    """

    def __repr__(self):
        return "<unconnected>"


_LAZY_UNCONNECTED = _LazyUnconnected()
MAX_REF_AUDIO_SECONDS = 15.0
MAX_CLIPS = 512
MAX_FL2VA_GUIDES = 3
DEFAULT_DURATION = 10.0
DEFAULT_MEGAPIXELS = 0.40
MAX_RESOLUTION = 4096
DEFAULT_SEED_MAX = (1 << 53) - 1  # exact integer range in browser JS

PROJECT_FORMAT = "MiniMax H3 Extender Project"
PROJECT_FORMAT_VERSION = 2
PROJECT_SUPPORTED_VERSIONS = {1, 2}
PROJECT_JSON_MAX_BYTES = 16 * 1024 * 1024
PROJECT_DOWNLOAD_TTL_SECONDS = 2 * 60 * 60
PROJECT_COPY_CHUNK = 8 * 1024 * 1024
MAX_IMAGE_REFS = MAX_REFERENCE_SLOTS
REFS_JSON_VERSION = 2
MAX_REF_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_REF_PIXELS = 120_000_000
_PROJECT_DOWNLOADS = {}


def _align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _duration_to_frames(seconds: float) -> int:
    raw = max(5, int(round(float(seconds) * FPS)))
    return _align_frame_count(raw)


def _empty_av_latent(width: int, height: int, frame_count: int):
    frame_count = _align_frame_count(frame_count)
    latent_t = _video_latent_t(frame_count)
    duration = frame_count / float(FPS)
    audio_t = round(duration * AUDIO_LATENT_FPS)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [1, 24, latent_t, int(height) // 16, int(width) // 16],
        device=device,
    )
    audio = torch.zeros(
        [1, 32, 2, int(audio_t)],
        device=device,
    )
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _manual_effective_resolution(width: int, height: int):
    """Snap Manual/fallback resolution to MiniMax H3's 32-pixel canvas grid."""
    step = CANVAS_MULTIPLE
    w = max(step, min(MAX_RESOLUTION, (int(width) // step) * step))
    h = max(step, min(MAX_RESOLUTION, (int(height) // step) * step))
    return w, h


def _auto_resolution_from_dimensions(src_w: int, src_h: int, megapixels: float):
    """Auto resolution on H3's 32-pixel grid without exceeding the MP budget.

    H3's standard workflows use resolution_steps=32. For the Extender we snap
    DOWN to that grid rather than to the nearest value: this keeps every Auto
    canvas divisible by 32 while avoiding an upward size jump on borderline
    Dynamic-VRAM/AIMDO setups.

    Manual/fallback mode uses the same 32-pixel canvas grid, so every newly
    requested H3 resolution follows the same alignment rule.
    """
    src_w = int(src_w)
    src_h = int(src_h)
    if src_w <= 0 or src_h <= 0:
        raise ValueError("MiniMax H3 Extender: reference image has invalid dimensions.")

    mp = max(0.01, min(16.0, float(megapixels)))
    total = mp * 1024.0 * 1024.0
    scale_by = math.sqrt(total / float(src_w * src_h))
    scaled_w = float(src_w) * scale_by
    scaled_h = float(src_h) * scale_by

    if scaled_w > MAX_RESOLUTION or scaled_h > MAX_RESOLUTION:
        shrink = min(MAX_RESOLUTION / scaled_w, MAX_RESOLUTION / scaled_h)
        scaled_w *= shrink
        scaled_h *= shrink

    step = int(CANVAS_MULTIPLE)
    w = max(step, min(MAX_RESOLUTION, int(math.floor(scaled_w / step)) * step))
    h = max(step, min(MAX_RESOLUTION, int(math.floor(scaled_h / step)) * step))
    return w, h




def _refs_root():
    root = _ensure_cache_root() / "_refs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ref_id_is_safe(value):
    value = str(value or "").lower()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _ref_path(ref_id):
    ref_id = str(ref_id or "").lower()
    if not _ref_id_is_safe(ref_id):
        raise ValueError("MiniMax H3 Extender: invalid internal reference id.")
    return _refs_root() / f"{ref_id}.png"


def _empty_refs():
    return [None for _ in range(MAX_IMAGE_REFS)]


def _normalize_ref_descriptor(value):
    if not isinstance(value, dict):
        return None
    ref_id = str(value.get("id") or value.get("ref_id") or "").lower().strip()
    if not _ref_id_is_safe(ref_id):
        return None
    try:
        width = int(value.get("width", 0) or 0)
        height = int(value.get("height", 0) or 0)
    except Exception:
        width = height = 0
    try:
        size_bytes = int(value.get("size_bytes", 0) or 0)
    except Exception:
        size_bytes = 0
    source_id = str(value.get("source_id") or value.get("original_id") or ref_id).lower().strip()
    if not _ref_id_is_safe(source_id):
        source_id = ref_id

    def _adjustment(name):
        try:
            number = float(value.get(name, 100) or 100)
        except Exception:
            number = 100.0
        if not math.isfinite(number):
            number = 100.0
        return max(0.0, min(200.0, number))

    descriptor = {
        "id": ref_id,
        "source_id": source_id,
        "original_name": str(value.get("original_name") or value.get("name") or "reference.png"),
        "width": max(0, width),
        "height": max(0, height),
        "size_bytes": max(0, size_bytes),
        "saturation": _adjustment("saturation"),
        "contrast": _adjustment("contrast"),
        "brightness": _adjustment("brightness"),
    }
    external_signature = str(value.get("external_signature") or "").lower().strip()
    if _ref_id_is_safe(external_signature):
        descriptor["external_signature"] = external_signature
    return descriptor


def _normalize_ref_descriptors(refs):
    """Normalize nine stable logical ref slots without compacting holes."""
    source = list(refs or [])[:MAX_IMAGE_REFS]
    normalized = []
    for value in source:
        normalized.append(_normalize_ref_descriptor(value))
    normalized += [None] * (MAX_IMAGE_REFS - len(normalized))
    return normalized


def _parse_refs_json(raw):
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw or "{}"))
        except Exception:
            payload = {}
    values = payload.get("refs") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        values = []
    refs = [_normalize_ref_descriptor(value) for value in values[:MAX_IMAGE_REFS]]
    refs += [None] * (MAX_IMAGE_REFS - len(refs))
    return _normalize_ref_descriptors(refs)


def _refs_json(refs):
    return json.dumps(
        {"version": REFS_JSON_VERSION, "refs": _normalize_ref_descriptors(refs)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _refs_signature(refs):
    ids = [ref.get("id") if isinstance(ref, dict) else None for ref in _normalize_ref_descriptors(refs)]
    raw = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reference_count(refs):
    return sum(1 for ref in refs or [] if ref is not None)


def _validate_reference_file(path):
    path = Path(path)
    with Image.open(path) as image:
        width, height = map(int, image.size)
        if width <= 0 or height <= 0:
            raise ValueError("MiniMax H3 Extender: reference image has invalid dimensions.")
        if width * height > MAX_REF_PIXELS:
            raise ValueError(
                f"MiniMax H3 Extender: reference image is too large ({width}x{height})."
            )
        image.verify()
    return width, height


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(PROJECT_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _store_uploaded_reference(source_path, original_name):
    """Normalize an uploaded reference to RGB PNG and store it content-addressed."""
    source_path = Path(source_path)
    temp_png = _refs_root() / f".upload_{uuid.uuid4().hex}.png"
    try:
        with Image.open(source_path) as source:
            source = ImageOps.exif_transpose(source)
            width, height = map(int, source.size)
            if width <= 0 or height <= 0:
                raise ValueError("MiniMax H3 Extender: reference image has invalid dimensions.")
            if width * height > MAX_REF_PIXELS:
                raise ValueError(
                    f"MiniMax H3 Extender: reference image is too large ({width}x{height})."
                )
            rgb = source.convert("RGB")
            rgb.save(temp_png, format="PNG", optimize=False, compress_level=4)

        ref_id = _hash_file(temp_png)
        target = _ref_path(ref_id)
        if target.exists():
            temp_png.unlink(missing_ok=True)
        else:
            os.replace(temp_png, target)
        return {
            "id": ref_id,
            "source_id": ref_id,
            "original_name": str(original_name or source_path.name or "reference.png"),
            "width": int(width),
            "height": int(height),
            "size_bytes": int(target.stat().st_size),
            "saturation": 100.0,
            "contrast": 100.0,
            "brightness": 100.0,
        }
    finally:
        try:
            temp_png.unlink(missing_ok=True)
        except Exception:
            pass



def _canonical_external_reference(image, slot_index):
    """Return the first IMAGE batch item as canonical uint8 RGB pixels + hash."""
    if not torch.is_tensor(image):
        raise ValueError(
            f"MiniMax H3 Extender: external Ref {int(slot_index)} is not a valid IMAGE tensor."
        )

    tensor = image.detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[0]) < 1:
        raise ValueError(
            f"MiniMax H3 Extender: external Ref {int(slot_index)} has an invalid IMAGE shape."
        )

    frame = tensor[0].to(device="cpu", dtype=torch.float32)
    if int(frame.shape[-1]) == 1:
        frame = frame.repeat(1, 1, 3)
    elif int(frame.shape[-1]) >= 3:
        frame = frame[..., :3]
    else:
        raise ValueError(
            f"MiniMax H3 Extender: external Ref {int(slot_index)} must contain RGB image data."
        )

    height = int(frame.shape[0])
    width = int(frame.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"MiniMax H3 Extender: external Ref {int(slot_index)} has invalid dimensions."
        )
    if width * height > MAX_REF_PIXELS:
        raise ValueError(
            f"MiniMax H3 Extender: external Ref {int(slot_index)} is too large ({width}x{height})."
        )

    frame = torch.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pixels = torch.round(frame * 255.0).to(torch.uint8).numpy()
    pixels = np.ascontiguousarray(pixels)

    digest = hashlib.sha256()
    digest.update(f"h3-external-ref-v1:{width}x{height}:rgb8:".encode("ascii"))
    digest.update(memoryview(pixels))
    return pixels, digest.hexdigest(), width, height


def _store_external_reference(image, slot_index, existing_ref=None):
    """Import one external IMAGE into the normal content-addressed ref store.

    ``external_signature`` hashes canonical RGB pixels before PNG encoding.  If
    the same connected image is seen on a later Queue, the current internal
    descriptor is kept untouched.  This also preserves any brightness/contrast/
    saturation edit made in the Extender while the upstream image is unchanged.
    """
    pixels, external_signature, width, height = _canonical_external_reference(
        image, slot_index
    )
    existing = _normalize_ref_descriptor(existing_ref)
    if existing is not None and existing.get("external_signature") == external_signature:
        current_path = _ref_path(existing.get("id"))
        source_path = _ref_path(existing.get("source_id") or existing.get("id"))
        if current_path.exists() and source_path.exists():
            return existing, False

    temp_png = _refs_root() / f".external_{uuid.uuid4().hex}.png"
    try:
        Image.fromarray(pixels, mode="RGB").save(
            temp_png,
            format="PNG",
            optimize=False,
            compress_level=4,
        )
        ref_id = _hash_file(temp_png)
        target = _ref_path(ref_id)
        if target.exists():
            temp_png.unlink(missing_ok=True)
        else:
            os.replace(temp_png, target)

        descriptor = {
            "id": ref_id,
            "source_id": ref_id,
            "original_name": f"External Ref {int(slot_index)}.png",
            "width": int(width),
            "height": int(height),
            "size_bytes": int(target.stat().st_size),
            "saturation": 100.0,
            "contrast": 100.0,
            "brightness": 100.0,
            "external_signature": external_signature,
        }
        return descriptor, True
    finally:
        try:
            temp_png.unlink(missing_ok=True)
        except Exception:
            pass


def _normalize_external_ref_pack(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("MiniMax H3 Extender: external reference pack is invalid.")

    raw_slots = value.get("slots")
    if not isinstance(raw_slots, (list, tuple)):
        raise ValueError("MiniMax H3 Extender: external reference pack has no slot list.")

    slots = list(raw_slots)[:MAX_IMAGE_REFS]
    slots += [None] * (MAX_IMAGE_REFS - len(slots))
    return {
        "type": REF_PACK_TYPE,
        "version": int(value.get("version", 1) or 1),
        "source": str(value.get("source") or "External reference pack"),
        "count": sum(1 for image in slots if image is not None),
        "slots": slots,
    }


def _sync_refs_from_ref_pack(refs, pack):
    """Inject connected external slots into the existing internal Ref N slots.

    Empty external slots are deliberately no-ops: they never clear or compact an
    internal reference.  Connected slots keep their exact logical number.
    """
    refs = _normalize_ref_descriptors(refs)
    if pack is None:
        return refs, []

    imported_slots = []
    for index, image in enumerate(pack.get("slots") or [], start=1):
        if index > MAX_IMAGE_REFS:
            break
        if image is None:
            continue
        try:
            descriptor, changed = _store_external_reference(
                image,
                index,
                refs[index - 1],
            )
        except Exception as exc:
            raise ValueError(
                f"MiniMax H3 Extender: failed to import external Ref {index}: {exc}"
            ) from exc
        refs[index - 1] = descriptor
        if changed:
            imported_slots.append(index)

    return refs, imported_slots


def _edit_internal_reference(source_id, original_name, brightness, contrast, saturation, external_signature=""):
    """Render absolute photographic adjustments from the immutable source ref.

    Every edited descriptor keeps ``source_id`` pointing at the pixels that were
    originally loaded. Re-opening the editor therefore has an actual baseline:
    Reset = 100/100/100 against those original pixels, rather than 100% against
    the already edited derivative.
    """
    source_id = str(source_id or "").lower().strip()
    if not _ref_id_is_safe(source_id):
        raise ValueError("MiniMax H3 Extender: invalid source reference id.")

    source_path = _ref_path(source_id)
    if not source_path.exists():
        raise ValueError("MiniMax H3 Extender: original reference image not found.")

    def _factor(value, label):
        try:
            number = float(value)
        except Exception as exc:
            raise ValueError(f"MiniMax H3 Extender: invalid {label} value.") from exc
        if not math.isfinite(number) or number < 0.0 or number > 200.0:
            raise ValueError(f"MiniMax H3 Extender: {label} must be between 0 and 200 percent.")
        return number, number / 100.0

    brightness_value, brightness_factor = _factor(brightness, "brightness")
    contrast_value, contrast_factor = _factor(contrast, "contrast")
    saturation_value, saturation_factor = _factor(saturation, "saturation")

    temp_png = _refs_root() / f".edit_{uuid.uuid4().hex}.png"
    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            width, height = map(int, image.size)
            if width <= 0 or height <= 0:
                raise ValueError("MiniMax H3 Extender: reference image has invalid dimensions.")
            if width * height > MAX_REF_PIXELS:
                raise ValueError(
                    f"MiniMax H3 Extender: reference image is too large ({width}x{height})."
                )

            # Keep the order identical to the browser preview filter chain.
            if abs(brightness_factor - 1.0) > 1e-9:
                image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            if abs(contrast_factor - 1.0) > 1e-9:
                image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            if abs(saturation_factor - 1.0) > 1e-9:
                image = ImageEnhance.Color(image).enhance(saturation_factor)

            image.save(temp_png, format="PNG", optimize=False, compress_level=4)

        new_id = _hash_file(temp_png)
        target = _ref_path(new_id)
        if target.exists():
            temp_png.unlink(missing_ok=True)
        else:
            os.replace(temp_png, target)

        descriptor = {
            "id": new_id,
            "source_id": source_id,
            "original_name": str(original_name or "reference.png"),
            "width": int(width),
            "height": int(height),
            "size_bytes": int(target.stat().st_size),
            "saturation": float(saturation_value),
            "contrast": float(contrast_value),
            "brightness": float(brightness_value),
        }
        external_signature = str(external_signature or "").lower().strip()
        if _ref_id_is_safe(external_signature):
            descriptor["external_signature"] = external_signature
        return descriptor
    finally:
        try:
            temp_png.unlink(missing_ok=True)
        except Exception:
            pass

def _store_project_reference(source_path, original_name):
    """Validate an archived PNG and preserve its exact bytes/hash on import."""
    source_path = Path(source_path)
    width, height = _validate_reference_file(source_path)
    ref_id = _hash_file(source_path)
    target = _ref_path(ref_id)
    if not target.exists():
        temp = _refs_root() / f".import_{uuid.uuid4().hex}.png"
        shutil.copyfile(source_path, temp)
        try:
            os.replace(temp, target)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass
    return {
        "id": ref_id,
        "source_id": ref_id,
        "original_name": str(original_name or "reference.png"),
        "width": int(width),
        "height": int(height),
        "size_bytes": int(target.stat().st_size),
        "saturation": 100.0,
        "contrast": 100.0,
        "brightness": 100.0,
    }


def _load_reference_tensor(ref):
    if not isinstance(ref, dict):
        raise ValueError("MiniMax H3 Extender: invalid internal reference metadata.")
    path = _ref_path(ref.get("id"))
    if not path.exists():
        raise FileNotFoundError(
            f"MiniMax H3 Extender: internal reference '{ref.get('original_name') or ref.get('id')}' is missing. "
            "Reload the reference image or load the portable .ext project that contains it."
        )
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _refs_from_project_payload(project_payload):
    extender = project_payload.get("extender", {}) if isinstance(project_payload, dict) else {}
    raw = extender.get("refs_json") if isinstance(extender, dict) else None
    if not raw:
        settings = extender.get("settings", {}) if isinstance(extender, dict) else {}
        raw = settings.get("refs_json") if isinstance(settings, dict) else None
    if not raw and isinstance(extender, dict) and isinstance(extender.get("references"), list):
        raw = {"version": REFS_JSON_VERSION, "refs": extender.get("references")}
    return _parse_refs_json(raw)


def _write_refs_to_project_payload(project_payload, refs):
    refs = _normalize_ref_descriptors(refs)
    extender = project_payload.setdefault("extender", {})
    raw = _refs_json(refs)
    extender["refs_json"] = raw
    extender["references"] = copy.deepcopy(refs)
    settings = extender.setdefault("settings", {})
    if isinstance(settings, dict):
        settings["refs_json"] = raw
    return refs


def _reference_dimensions(ref):
    if ref is None:
        return None
    if isinstance(ref, dict):
        try:
            width = int(ref.get("width", 0) or 0)
            height = int(ref.get("height", 0) or 0)
        except Exception:
            return None
        if width > 0 and height > 0:
            return width, height
        path = _ref_path(ref.get("id"))
        if path.exists():
            width, height = _validate_reference_file(path)
            ref["width"] = int(width)
            ref["height"] = int(height)
            return width, height
        return None
    if getattr(ref, "ndim", 0) >= 4:
        return int(ref.shape[2]), int(ref.shape[1])
    return None


def _select_resolution_guide(refs):
    """Ref 1 wins; otherwise the first available internal reference wins."""
    if refs and refs[0] is not None:
        return 1, refs[0]
    for index, image in enumerate(refs or [], start=1):
        if image is not None:
            return index, image
    return None, None


def _resolve_generation_resolution(resolution_mode, megapixels, width, height, refs):
    manual_w, manual_h = _manual_effective_resolution(width, height)
    mode = str(resolution_mode or "auto_from_ref")
    if mode == "manual":
        return {
            "width": manual_w,
            "height": manual_h,
            "mode": "manual",
            "guide_ref": None,
            "guide_src_width": 0,
            "guide_src_height": 0,
            "fallback": False,
            "megapixels": float(megapixels),
        }

    guide_index, guide = _select_resolution_guide(refs)
    dims = _reference_dimensions(guide) if guide is not None else None
    if guide is None or dims is None:
        return {
            "width": manual_w,
            "height": manual_h,
            "mode": "manual_fallback",
            "guide_ref": None,
            "guide_src_width": 0,
            "guide_src_height": 0,
            "fallback": True,
            "megapixels": float(megapixels),
        }

    src_w, src_h = dims
    resolved_w, resolved_h = _auto_resolution_from_dimensions(src_w, src_h, megapixels)
    return {
        "width": resolved_w,
        "height": resolved_h,
        "mode": "auto_from_ref",
        "guide_ref": int(guide_index),
        "guide_src_width": int(src_w),
        "guide_src_height": int(src_h),
        "fallback": False,
        "megapixels": float(megapixels),
    }

def _resolution_from_manifest(manifest):
    if not isinstance(manifest, dict):
        return None
    geom = manifest.get("geometry")
    if not isinstance(geom, dict):
        return None
    try:
        w = int(geom.get("video_w", 0)) * 16
        h = int(geom.get("video_h", 0)) * 16
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"width": w, "height": h}


def _resize(image, width: int, height: int):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", "disabled"
    )
    return samples.movedim(1, -1)


def _adapt_ref_video_canvas(width: int, height: int):
    """Native H3 Ref2VA reference-video canvas (ComfyUI compatible).

    Reference videos use a 768-pixel short edge with a 768x1344 pixel-area
    ceiling, then snap each axis to the H3 32-pixel canvas grid.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    ratio = width / float(height)
    if ratio >= 1.0:
        nominal_w, nominal_h = REF_VIDEO_BASE_SHORT_EDGE * ratio, REF_VIDEO_BASE_SHORT_EDGE
    else:
        nominal_w, nominal_h = REF_VIDEO_BASE_SHORT_EDGE, REF_VIDEO_BASE_SHORT_EDGE / ratio
    if nominal_w * nominal_h > REF_VIDEO_MAX_PIXELS:
        scale = math.sqrt(REF_VIDEO_MAX_PIXELS / float(nominal_w * nominal_h))
        nominal_w *= scale
        nominal_h *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _normalize_ref_video_fps(value, label: str):
    try:
        fps = float(value)
    except Exception:
        fps = float(FPS)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(
            f"MiniMax H3 Extender: {label} must be a positive source FPS value; got {value!r}."
        )
    return fps


def _ref_video_h3_frame_count(video_frames, source_fps: float, label: str):
    """Return the H3 24-fps frame count without materializing a resampled batch."""
    fps = _normalize_ref_video_fps(source_fps, label)
    if not torch.is_tensor(video_frames) or video_frames.ndim != 4:
        raise ValueError(f"MiniMax H3 Extender: {label} must be an IMAGE batch [frames,H,W,C].")
    source_count = int(video_frames.shape[0])
    if abs(fps - float(FPS)) < 1e-6 or source_count <= 1:
        return fps, source_count
    duration = source_count / fps
    return fps, max(1, int(round(duration * float(FPS))))


def _take_ref_video_h3_frames(video_frames, source_fps: float, start: int, end: int):
    """Materialize only the requested H3-frame window from a source video."""
    start = max(0, int(start))
    end = max(start, int(end))
    if end <= start:
        return video_frames[:0]
    if abs(float(source_fps) - float(FPS)) < 1e-6 or int(video_frames.shape[0]) <= 1:
        return video_frames[start:end]

    positions = torch.arange(
        start, end, device=video_frames.device, dtype=torch.float32
    )
    idx = torch.round(positions * (float(source_fps) / float(FPS))).to(torch.long)
    idx = torch.clamp(idx, 0, int(video_frames.shape[0]) - 1)
    return video_frames.index_select(0, idx)


def _resize_ref_video_h3_frames(
    video_frames, source_fps: float, count: int, width: int, height: int, chunk_frames: int = 32
):
    """Resize a bounded H3 prefix without a full 24-fps intermediate tensor."""
    count = max(0, int(count))
    if count <= 0:
        return video_frames[:0, :int(height), :int(width), :3]
    chunk_frames = max(1, int(chunk_frames))
    out = torch.empty(
        (count, int(height), int(width), 3),
        device=video_frames.device,
        dtype=video_frames.dtype,
    )
    for start in range(0, count, chunk_frames):
        end = min(count, start + chunk_frames)
        source_chunk = _take_ref_video_h3_frames(video_frames, source_fps, start, end)
        resized_chunk = _resize(source_chunk, width, height)
        out[start:end].copy_(resized_chunk)
        del source_chunk, resized_chunk
    return out


def _resize_ref_video_qwen_frames(
    video_frames, source_fps: float, count: int, width: int, height: int
):
    """Resize only the half-second Qwen samples for a cached video latent."""
    sample_step = max(1, FPS // 2)
    target_positions = list(range(0, int(count), sample_step))
    if not target_positions:
        return video_frames[:0, :int(height), :int(width), :3], target_positions

    if abs(float(source_fps) - float(FPS)) < 1e-6 or int(video_frames.shape[0]) <= 1:
        selected = video_frames[target_positions]
    else:
        positions = torch.tensor(
            target_positions, device=video_frames.device, dtype=torch.float32
        )
        idx = torch.round(positions * (float(source_fps) / float(FPS))).to(torch.long)
        idx = torch.clamp(idx, 0, int(video_frames.shape[0]) - 1)
        selected = video_frames.index_select(0, idx)
    resized = _resize(selected, width, height)
    return resized, target_positions

def _audio_duration_seconds(audio):
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("MiniMax H3 Extender: invalid AUDIO reference payload.")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform) or waveform.ndim < 2:
        raise ValueError("MiniMax H3 Extender: invalid AUDIO waveform tensor.")
    sr = int(audio.get("sample_rate", 0) or 0)
    if sr <= 0:
        raise ValueError("MiniMax H3 Extender: invalid AUDIO sample rate.")
    return float(waveform.shape[-1]) / float(sr)


def _slice_ref_audio(audio, start_seconds: float, duration_seconds: float, label: str, require_full: bool):
    """Return an AUDIO payload cropped before Audio-VAE encoding.

    Long standalone references use require_full=True so every clip gets exactly
    its own sequential timeline window. Video soundtracks use require_full=False
    because container audio can be a few samples shorter than the video stream.
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError(f"MiniMax H3 Extender: {label} is not a valid AUDIO payload.")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform) or waveform.ndim < 2:
        raise ValueError(f"MiniMax H3 Extender: {label} has an invalid waveform tensor.")
    sr = int(audio.get("sample_rate", 0) or 0)
    if sr <= 0:
        raise ValueError(f"MiniMax H3 Extender: {label} has an invalid sample rate.")

    start_seconds = max(0.0, float(start_seconds))
    duration_seconds = max(0.0, float(duration_seconds))
    total_samples = int(waveform.shape[-1])
    start = int(round(start_seconds * sr))
    wanted = max(1, int(round(duration_seconds * sr)))

    if start >= total_samples:
        total_seconds = total_samples / float(sr)
        raise ValueError(
            f"MiniMax H3 Extender: {label} is exhausted at {total_seconds:.3f}s; "
            f"the current clip needs audio starting at {start_seconds:.3f}s."
        )

    available = total_samples - start
    if require_full and available < wanted:
        total_seconds = total_samples / float(sr)
        required_end = start_seconds + duration_seconds
        raise ValueError(
            f"MiniMax H3 Extender: {label} is too short for the current clip. "
            f"Source duration is {total_seconds:.3f}s but this clip needs the window "
            f"{start_seconds:.3f}s -> {required_end:.3f}s."
        )

    end = min(total_samples, start + wanted)
    return {
        "waveform": waveform[..., start:end],
        "sample_rate": sr,
    }


def _encode_ref_audio(audio_vae, audio):
    """Encode one already-cropped H3 audio reference."""
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return latent, int(latent.shape[-1])


def _prepare_standalone_audio_refs(
    audio_vae,
    ref_audios,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    cache=None,
):
    """Build per-clip standalone audio refs without reusing illegal long audio.

    Native-sized refs (<=15s) remain reusable references: they start at 0 for
    every clip and are cropped to the current clip duration when useful. A long
    source (>15s) is treated as a timeline and automatically advanced by the
    cumulative H3-aligned duration of preceding cards that selected the same
    logical Audio slot.
    """
    active = [
        (slot, audio)
        for slot, audio in enumerate(list(ref_audios or [])[:MAX_STANDALONE_AUDIO_REFS], start=1)
        if audio is not None
    ]
    if not active:
        return [], []
    if audio_vae is None:
        raise ValueError(
            "MiniMax H3 Extender: standalone audio reference inputs are connected but audio_vae is not. "
            "Connect the MiniMax H3 Audio VAE to audio_vae."
        )

    clip_duration_seconds = float(clip_duration_seconds)
    # clip_start_seconds may be a scalar (legacy/global timeline) or a mapping
    # {logical_audio_slot: start_seconds}. The mapping form lets each long
    # standalone Audio reference keep its own independent timeline cursor.
    clip_start_offsets = clip_start_seconds if isinstance(clip_start_seconds, dict) else None
    default_clip_start = 0.0 if clip_start_offsets is not None else float(clip_start_seconds)
    prepared = []
    total_effective_audio = 0.0

    for slot, audio in active:
        label = f"ref_audio_{slot}"
        source_duration = _audio_duration_seconds(audio)
        timeline_mode = source_duration > MAX_REF_AUDIO_SECONDS + 1e-6

        if timeline_mode:
            if clip_duration_seconds > MAX_REF_AUDIO_SECONDS + 1e-6:
                raise ValueError(
                    f"MiniMax H3 Extender: {label} cannot cover this clip as one H3 audio reference: "
                    f"effective clip duration is {clip_duration_seconds:.3f}s, above the {MAX_REF_AUDIO_SECONDS:.0f}s reference-audio limit."
                )
            start = float(clip_start_offsets.get(slot, 0.0)) if clip_start_offsets is not None else default_clip_start
            duration = clip_duration_seconds
            sliced = _slice_ref_audio(audio, start, duration, label, require_full=True)
        else:
            # Preserve classic short-reference behavior across every clip, but
            # never feed more audio than the current generated clip requires.
            start = 0.0
            duration = min(source_duration, clip_duration_seconds, MAX_REF_AUDIO_SECONDS)
            sliced = _slice_ref_audio(audio, start, duration, label, require_full=False)

        effective_duration = _audio_duration_seconds(sliced)
        if effective_duration + 1e-6 < MIN_REF_AUDIO_SECONDS:
            raise ValueError(
                f"MiniMax H3 Extender: {label} provides only {effective_duration:.3f}s for this clip; "
                f"MiniMax H3 reference audio requires at least {MIN_REF_AUDIO_SECONDS:.0f}s."
            )
        total_effective_audio += effective_duration
        prepared.append((slot, sliced, start, duration, timeline_mode, id(audio["waveform"])))

    if total_effective_audio > MAX_REF_AUDIO_SECONDS + 1e-6:
        raise ValueError(
            "MiniMax H3 Extender: the standalone audio references for this clip total "
            f"{total_effective_audio:.3f}s, above MiniMax H3's {MAX_REF_AUDIO_SECONDS:.0f}s cumulative audio-reference limit."
        )

    ref_items = []
    ref_blocks = []
    cache = cache if isinstance(cache, dict) else {}
    for slot, sliced, start, duration, timeline_mode, source_waveform_id in prepared:
        waveform = sliced["waveform"]
        sr = int(sliced["sample_rate"])
        # Short reusable refs with equal clip durations can reuse their encoded
        # latent; long timeline refs naturally use a different start each card.
        key = (
            int(slot),
            int(source_waveform_id),
            int(sr),
            int(waveform.shape[-1]),
            round(float(start), 6),
            round(float(duration), 6),
            bool(timeline_mode),
        )
        cached = cache.get(key)
        if cached is None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, sliced)
            cache[key] = (audio_latent, int(ref_audio_t))
        else:
            audio_latent, ref_audio_t = cached
        ref_items.append({"type": "audio"})
        ref_blocks.append(
            {
                "kind": "audio",
                "ref_audio_t": int(ref_audio_t),
                "audio_latent": audio_latent,
            }
        )
    return ref_items, ref_blocks


_AUDIO_TAG_RE = re.compile(r"<Audio\s+(\d+)>", re.IGNORECASE)


def _select_standalone_audio_refs_for_prompt(prompt: str, ref_audios):
    """Select standalone Audio refs for one clip without accidental stacking.

    Rules:
    - With exactly one connected standalone audio ref, that ref is always used.
      This preserves the long-audio timeline behavior across every clip even if
      a prompt contains a stale/different <Audio N> tag.
    - With several connected refs, matching <Audio N> tags explicitly select
      the requested refs for that clip.
    - With several connected refs and no matching <Audio N> tag, only the first
      connected logical slot is used as the deterministic default. This avoids
      silently stacking every connected audio reference on the same clip.
    """
    audios = list(ref_audios or [])[:MAX_STANDALONE_AUDIO_REFS]
    if len(audios) < MAX_STANDALONE_AUDIO_REFS:
        audios.extend([None] * (MAX_STANDALONE_AUDIO_REFS - len(audios)))

    referenced = {
        int(match.group(1))
        for match in _AUDIO_TAG_RE.finditer(str(prompt or ""))
        if 1 <= int(match.group(1)) <= MAX_STANDALONE_AUDIO_REFS
    }
    connected = sorted(
        slot
        for slot, audio in enumerate(audios, start=1)
        if audio is not None
    )

    if not connected:
        selected_slots = []
    elif len(connected) == 1:
        # A single connected ref is the sequence/global ref. In particular, a
        # long source keeps advancing through the clip timeline independently of
        # Audio tags left in individual prompts.
        selected_slots = list(connected)
    else:
        selected_slots = sorted(referenced & set(connected))
        if not selected_slots:
            # Several refs with no usable explicit tag: choose the first logical
            # connected slot instead of combining all refs and causing an
            # unexpected layered/superimposed audio conditioning.
            selected_slots = [connected[0]]

    selected_set = set(selected_slots)
    selected = [
        audio if slot in selected_set else None
        for slot, audio in enumerate(audios, start=1)
    ]
    return selected, selected_slots


def _build_standalone_audio_clip_plan(clips, ref_audios):
    """Precompute per-clip Audio selection and independent source offsets.

    Each logical standalone Audio slot owns its own timeline cursor. A slot only
    advances on clips that actually select/use that slot. This keeps rerenders
    deterministic because validated/cached cards are included in the plan even
    when they are skipped by the sampler.
    """
    cursors = {slot: 0.0 for slot in range(1, MAX_STANDALONE_AUDIO_REFS + 1)}
    plan = []
    for clip_cfg in clips:
        selected_ref_audios, selected_audio_slots = _select_standalone_audio_refs_for_prompt(
            clip_cfg.get("prompt", ""), ref_audios
        )
        offsets = {slot: float(cursors[slot]) for slot in selected_audio_slots}
        duration = _duration_to_frames(clip_cfg["duration"]) / float(FPS)
        for slot in selected_audio_slots:
            cursors[slot] += duration
        plan.append((selected_ref_audios, selected_audio_slots, offsets))
    return plan


def _prepare_shared_refs(
    vae,
    audio_vae,
    width: int,
    height: int,
    ref_image_size: str,
    refs,
    ref_videos=None,
    ref_video_fps=None,
    ref_video_audios=None,
    standalone_audio_count=0,
    frame_count=None,
    cached_image_blocks=None,
    cached_video_blocks=None,
):
    """Encode shared H3 Ref2VA image/video refs and paired soundtracks.

    Standalone audio is deliberately prepared per clip by
    _prepare_standalone_audio_refs(), because long sources advance through the
    sequence timeline. Image/video payloads remain cached across equal-duration
    cards so this feature does not force expensive visual re-encoding.
    """
    active_images = [
        (slot, image)
        for slot, image in enumerate(refs, start=1)
        if image is not None
    ]
    ref_videos = list(ref_videos or [])[:MAX_VIDEO_REFS]
    ref_video_fps = list(ref_video_fps or [])[:MAX_VIDEO_REFS]
    ref_video_audios = list(ref_video_audios or [])[:MAX_VIDEO_REFS]

    active_videos = []
    for slot, video in enumerate(ref_videos, start=1):
        fps_hint = ref_video_fps[slot - 1] if slot - 1 < len(ref_video_fps) else float(FPS)
        soundtrack = ref_video_audios[slot - 1] if slot - 1 < len(ref_video_audios) else None
        if video is None:
            if soundtrack is not None:
                raise ValueError(
                    f"MiniMax H3 Extender: ref_video_audio_{slot} is connected but ref_video_{slot} is not. "
                    "Pair each video soundtrack with the same-numbered reference video."
                )
            continue
        active_videos.append((slot, video, fps_hint, soundtrack))

    mixed_ref_count = len(active_images) + len(active_videos) + int(standalone_audio_count or 0)
    if mixed_ref_count > MAX_MIXED_REF_ITEMS:
        raise ValueError(
            f"MiniMax H3 Extender: H3 Ref2VA supports at most {MAX_MIXED_REF_ITEMS} mixed reference items; got {mixed_ref_count}."
        )
    if any(soundtrack is not None for _, _, _, soundtrack in active_videos) and audio_vae is None:
        raise ValueError(
            "MiniMax H3 Extender: reference-video soundtrack inputs are connected but audio_vae is not. "
            "Connect the MiniMax H3 Audio VAE to audio_vae."
        )

    ref_items = []
    ref_blocks = []
    active_picture_slots = []
    active_video_slots = []

    # Official Ref2VA presentation order starts with images.
    for image_index, (slot, ref) in enumerate(active_images):
        img = _load_reference_tensor(ref) if isinstance(ref, dict) else ref
        h, w = int(img.shape[1]), int(img.shape[2])
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((int(width) * int(height)) / float(w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))

        tw = max(
            CANVAS_MULTIPLE,
            round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )
        th = max(
            CANVAS_MULTIPLE,
            round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )

        resized = _resize(img[:1], tw, th)
        cached_block = (
            cached_image_blocks[image_index]
            if cached_image_blocks is not None and image_index < len(cached_image_blocks)
            else None
        )
        if (
            isinstance(cached_block, dict)
            and cached_block.get("kind") == "image"
            and cached_block.get("latent") is not None
        ):
            z = cached_block.get("latent")
        else:
            z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        active_picture_slots.append(int(slot))
        ref_blocks.append(
            {
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z,
            }
        )

    # Then reference videos, exactly like ComfyUI's native H3 Ref2VA node.
    target_frames = int(frame_count) if frame_count is not None else MAX_REF_VIDEO_FRAMES
    target_frames = max(5, min(target_frames, MAX_REF_VIDEO_FRAMES))
    total_ref_video_frames = 0
    for video_index, (slot, video_frames, source_fps, soundtrack) in enumerate(active_videos):
        source_fps, full_h3_frames = _ref_video_h3_frame_count(
            video_frames, source_fps, f"ref_video_{slot}"
        )
        if int(full_h3_frames) < int(2 * FPS):
            raise ValueError(
                f"MiniMax H3 Extender: ref_video_{slot} is shorter than MiniMax H3's 2-second minimum at 24 fps."
            )

        vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
        cw, ch = _adapt_ref_video_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

        # Crop to the requested clip duration and H3 17k+5 grid BEFORE any
        # expensive Lanczos resize. This preserves the exact former prefix but
        # never processes frames that will be discarded afterwards.
        n = min(int(full_h3_frames), int(target_frames))
        while n >= 5 and n % 17 != 5:
            n -= 1
        if n < 5:
            raise ValueError(
                f"MiniMax H3 Extender: ref_video_{slot} cannot be aligned to the H3 17k+5 frame grid."
            )
        total_ref_video_frames += n
        if total_ref_video_frames > MAX_REF_VIDEO_FRAMES:
            raise ValueError(
                "MiniMax H3 Extender: total effective reference-video duration exceeds MiniMax H3's 15-second limit."
            )

        cached_block = (
            cached_video_blocks[video_index]
            if cached_video_blocks is not None and video_index < len(cached_video_blocks)
            else None
        )
        latent_cached = (
            isinstance(cached_block, dict)
            and str(cached_block.get("kind") or "").startswith("video")
            and cached_block.get("latent") is not None
        )
        if latent_cached:
            z = cached_block.get("latent")
            # A cached VAE latent only needs the sparse Qwen presentation frames.
            # Resize those samples directly instead of rebuilding all n RGB frames.
            qwen_frames, sample_idx = _resize_ref_video_qwen_frames(
                video_frames, source_fps, n, cw, ch
            )
        else:
            # VAE encoding needs the complete aligned RGB sequence. Build it in
            # bounded chunks directly from the source cadence, avoiding a second
            # full-size 24-fps intermediate batch.
            frames = _resize_ref_video_h3_frames(
                video_frames, source_fps, n, cw, ch, chunk_frames=32
            )
            z = vae.encode(frames)
            sample_step = max(1, FPS // 2)
            sample_idx = list(range(0, n, sample_step))
            qwen_frames = frames[sample_idx]
            del frames

        audio_latent = None
        ref_audio_t = 0
        if soundtrack is not None:
            if (
                isinstance(cached_block, dict)
                and str(cached_block.get("kind") or "") == "video_audio"
                and cached_block.get("audio_latent") is not None
                and int(cached_block.get("ref_audio_t", 0) or 0) > 0
            ):
                audio_latent = cached_block.get("audio_latent")
                ref_audio_t = int(cached_block.get("ref_audio_t", 0) or 0)
            else:
                # The official node encodes the entire soundtrack. The Extender is
                # safer: crop it to the effective aligned reference-video duration
                # before Audio-VAE encoding, which also fixes long Load-Video audio.
                soundtrack = _slice_ref_audio(
                    soundtrack,
                    0.0,
                    n / float(FPS),
                    f"ref_video_audio_{slot}",
                    require_full=False,
                )
                if _audio_duration_seconds(soundtrack) + 1e-6 < MIN_REF_AUDIO_SECONDS:
                    raise ValueError(
                        f"MiniMax H3 Extender: ref_video_audio_{slot} is shorter than {MIN_REF_AUDIO_SECONDS:.0f}s after pairing with ref_video_{slot}."
                    )
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            # The soundtrack gets its own <Audio j> label immediately before its video.
            ref_items.append({"type": "audio"})

        ref_items.append(
            {
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            }
        )
        active_video_slots.append(int(slot))
        ref_blocks.append(
            {
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": int(z.shape[2]),
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": int(ref_audio_t),
                "latent": z,
                "audio_latent": audio_latent,
            }
        )

    return ref_items, ref_blocks, active_picture_slots, active_video_slots


_PICTURE_TAG_RE = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)
_VIDEO_TAG_RE = re.compile(r"<Video\s+(\d+)>", re.IGNORECASE)


def _remap_numbered_tags(
    prompt: str,
    active_picture_slots,
    active_video_slots,
    active_audio_slots=None,
    audio_native_offset: int = 0,
):
    """Map stable Extender logical slots to H3 contiguous native ordinals.

    Picture/Video slots are stable Extender indices. Standalone Audio slots use
    the same rule per clip: if only ref_audio_2 is selected by <Audio 2>, it is
    packed as the first standalone native audio item and the prompt is remapped
    accordingly. Paired video soundtracks already present in ref_items occupy
    the first native Audio ordinals, hence audio_native_offset.
    """
    picture_map = {
        int(slot): ordinal
        for ordinal, slot in enumerate(active_picture_slots or [], start=1)
    }
    video_map = {
        int(slot): ordinal
        for ordinal, slot in enumerate(active_video_slots or [], start=1)
    }
    audio_map = {
        int(slot): int(audio_native_offset) + ordinal
        for ordinal, slot in enumerate(active_audio_slots or [], start=1)
    }

    def replace_picture(match):
        slot = int(match.group(1))
        if slot < 1 or slot > MAX_IMAGE_REFS:
            return match.group(0)
        ordinal = picture_map.get(slot)
        return f"<Picture {ordinal}>" if ordinal is not None else match.group(0)

    def replace_video(match):
        slot = int(match.group(1))
        if slot < 1 or slot > MAX_VIDEO_REFS:
            return match.group(0)
        ordinal = video_map.get(slot)
        return f"<Video {ordinal}>" if ordinal is not None else match.group(0)

    def replace_audio(match):
        slot = int(match.group(1))
        if slot < 1 or slot > MAX_STANDALONE_AUDIO_REFS:
            return match.group(0)
        ordinal = audio_map.get(slot)
        return f"<Audio {ordinal}>" if ordinal is not None else match.group(0)

    text = _PICTURE_TAG_RE.sub(replace_picture, str(prompt))
    text = _VIDEO_TAG_RE.sub(replace_video, text)
    return _AUDIO_TAG_RE.sub(replace_audio, text)


def _make_ref2va_conditioning(
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    frame_count: int,
    ref_items,
    ref_blocks,
    active_picture_slots,
    active_video_slots,
    active_audio_slots=None,
    audio_native_offset: int = 0,
):
    latent = _empty_av_latent(width, height, frame_count)
    resolved_prompt = _remap_numbered_tags(
        prompt,
        active_picture_slots,
        active_video_slots,
        active_audio_slots=active_audio_slots,
        audio_native_offset=audio_native_offset,
    )
    tokens = clip.tokenize(resolved_prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_refs": ref_blocks}
        )
    return cond, latent



class _BasicGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def _sigmas(model, scheduler: str, steps: int, denoise: float):
    steps = max(1, int(steps))
    denoise = float(denoise)
    if denoise <= 0.0:
        return torch.FloatTensor([])
    total_steps = steps
    if denoise < 1.0:
        total_steps = max(steps, int(steps / denoise))
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"),
        str(scheduler),
        total_steps,
    ).cpu()
    return sigmas[-(steps + 1):]


def _sample_h3(model, conditioning, latent, seed: int, sampler_name: str, scheduler: str, steps: int, denoise: float):
    if int(steps) < 1:
        raise ValueError("MiniMax H3 Extender: steps must be >= 1.")

    guider = _BasicGuider(model)
    guider.set_conds(conditioning)
    sampler = comfy.samplers.sampler_object(str(sampler_name))
    sigmas = _sigmas(model, scheduler, steps, denoise)

    latent_out = latent.copy()
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent_out["samples"] = latent_image

    batch_inds = latent_out.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_inds)
    noise_mask = latent_out.get("noise_mask", None)

    x0_output = {}
    callback = latent_preview.prepare_callback(
        model, sigmas.shape[-1] - 1, x0_output
    )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent_out.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _normalize_color_adjustment(value=None):
    raw = value if isinstance(value, dict) else {}

    def _v(name, default=100.0, low=0.0, high=200.0):
        try:
            x = float(raw.get(name, default))
        except Exception:
            x = float(default)
        return max(float(low), min(float(high), x))

    return {
        "saturation": _v("saturation", 100.0, 0.0, 200.0),
        "contrast": _v("contrast", 100.0, 50.0, 150.0),
        "brightness": _v("brightness", 100.0, 50.0, 150.0),
    }


def _normalize_clip_lora(value=None):
    raw = value if isinstance(value, dict) else {}
    name = str(raw.get("name") or "").strip()

    # v14.77 stored `strength_model`; keep accepting it when loading a project
    # created with that test build. Per-clip H3 LoRAs are model-only: the text
    # encoder is deliberately left untouched.
    candidate = raw.get("strength", raw.get("strength_model", 1.0))
    try:
        strength = float(candidate)
    except Exception:
        strength = 1.0
    if not math.isfinite(strength):
        strength = 1.0

    return {
        "name": name,
        "strength": max(-100.0, min(100.0, strength)),
    }


def _normalize_clip_loras(value=None, legacy=None):
    """Normalize the ordered model-only LoRA stack stored on one clip card.

    v14.79 stores `loras: [...]`. v14.77/v14.78 stored one `lora` object, which
    is migrated automatically when loading those test-project states.
    Empty entries are discarded; the UI owns the trailing empty autogrow row.
    """
    raw_items = value if isinstance(value, list) else []
    if not raw_items and isinstance(legacy, dict):
        raw_items = [legacy]

    out = []
    for raw in raw_items:
        cfg = _normalize_clip_lora(raw)
        if cfg["name"]:
            out.append(cfg)
    return out


def _apply_per_clip_loras(owner, model, clip, lora_cfgs, clip_index):
    """Apply an ordered stack of card-local, model-only LoRAs.

    Every card starts from the incoming MODEL. LoRAs selected on that card are
    then patched one after another using ComfyUI's native loader path. The text
    encoder is intentionally untouched, and the patched model is returned only
    for this card so local LoRA state cannot leak into following clips.
    """
    cfgs = _normalize_clip_loras(lora_cfgs)
    if not cfgs:
        return model, clip

    patched_model = model
    for lora_index, cfg in enumerate(cfgs, start=1):
        name = cfg["name"]
        strength = float(cfg["strength"])
        if abs(strength) < 1e-12:
            continue

        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", name)
        except Exception as exc:
            raise ValueError(
                f"MiniMax H3 Extender: LoRA {name!r} selected for clip {int(clip_index) + 1} "
                f"(slot {lora_index}) was not found in ComfyUI's LoRA folders."
            ) from exc

        # Keep the native loader's lightweight single-payload cache semantics.
        # This deliberately avoids retaining an unbounded collection of large
        # LoRA tensors in CPU RAM. Multiple LoRAs are still applied in order.
        loaded = getattr(owner, "_h3_loaded_lora", None)
        lora = None
        lora_metadata = None
        if isinstance(loaded, tuple) and len(loaded) >= 2 and loaded[0] == lora_path:
            lora = loaded[1]
            lora_metadata = loaded[2] if len(loaded) > 2 else None

        if lora is None:
            try:
                lora, lora_metadata = comfy.utils.load_torch_file(
                    lora_path, safe_load=True, return_metadata=True
                )
            except TypeError:
                # Compatibility with older ComfyUI builds lacking return_metadata.
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                lora_metadata = None
            owner._h3_loaded_lora = (lora_path, lora, lora_metadata)

        try:
            try:
                patched_model, _ = comfy.sd.load_lora_for_models(
                    patched_model,
                    None,
                    lora,
                    strength,
                    0.0,
                    lora_metadata=lora_metadata,
                )
            except TypeError:
                # Compatibility with older signatures before LoRA metadata support.
                patched_model, _ = comfy.sd.load_lora_for_models(
                    patched_model, None, lora, strength, 0.0
                )
        except Exception as exc:
            raise RuntimeError(
                f"MiniMax H3 Extender: failed to apply LoRA {name!r} to clip "
                f"{int(clip_index) + 1} (slot {lora_index}): {exc}"
            ) from exc

    return patched_model, clip


def _default_clip(index: int = 0):
    return {
        "id": f"clip_{index + 1}",
        "name": "",
        "prompt": "",
        "seed": int(secrets.randbelow(DEFAULT_SEED_MAX)),
        "seed_mode": "randomize",
        "duration": DEFAULT_DURATION,
        "validated": False,
        "color_adjustment": _normalize_color_adjustment(),
        "loras": [],
        "first_frame": None,
        "last_frame": None,
        "guides": [],
        "first_source": "manual",
    }



def _normalize_fl2va_guides(raw_guides, legacy_frame=None, legacy_idx=0):
    guides = []
    source = raw_guides if isinstance(raw_guides, list) else None
    if source is None:
        legacy = _normalize_ref_descriptor(legacy_frame)
        if legacy is not None:
            source = [{"frame": legacy, "frame_idx": legacy_idx}]
        else:
            source = []

    for raw in source[:MAX_FL2VA_GUIDES]:
        raw = raw if isinstance(raw, dict) else {}
        frame = _normalize_ref_descriptor(raw.get("frame", raw.get("guide_frame")))
        if frame is None:
            continue
        try:
            frame_idx = int(raw.get("frame_idx", raw.get("guide_frame_idx", 0)) or 0)
        except Exception:
            frame_idx = 0
        guides.append({
            "frame": frame,
            "frame_idx": max(-9999, min(9999, frame_idx)),
        })
    return guides

def _parse_clips_json(value: str, generation_mode="ref2va"):
    try:
        payload = json.loads(value or "{}")
    except Exception as exc:
        raise ValueError(f"MiniMax H3 Extender: invalid clips JSON: {exc}") from exc

    if isinstance(payload, list):
        clips = payload
    elif isinstance(payload, dict):
        clips = payload.get("clips", [])
    else:
        clips = []

    if not clips:
        clips = [_default_clip(0)]
    if len(clips) > MAX_CLIPS:
        raise ValueError(
            f"MiniMax H3 Extender: {len(clips)} clips requested; max is {MAX_CLIPS}."
        )

    out = []
    for i, raw in enumerate(clips):
        raw = raw if isinstance(raw, dict) else {}
        try:
            seed = int(raw.get("seed", 0))
        except Exception:
            seed = 0
        seed = max(0, min(DEFAULT_SEED_MAX, seed))

        try:
            duration = float(raw.get("duration", DEFAULT_DURATION))
        except Exception:
            duration = DEFAULT_DURATION
        duration = max(0.25, min(150.0, duration))

        seed_mode = str(raw.get("seed_mode", "randomize"))
        if seed_mode not in {"randomize", "fixed", "increment", "decrement"}:
            seed_mode = "randomize"

        out.append(
            {
                "id": str(raw.get("id") or f"clip_{i + 1}"),
                "name": str(raw.get("name", "")),
                "prompt": str(raw.get("prompt", "")),
                "seed": seed,
                "seed_mode": seed_mode,
                "duration": duration,
                "validated": bool(raw.get("validated", False)),
                "color_adjustment": _normalize_color_adjustment(raw.get("color_adjustment")),
                "loras": _normalize_clip_loras(raw.get("loras"), legacy=raw.get("lora")),
                "first_frame": _normalize_ref_descriptor(raw.get("first_frame")),
                "last_frame": _normalize_ref_descriptor(raw.get("last_frame")),
                "guides": _normalize_fl2va_guides(
                    raw.get("guides"),
                    legacy_frame=raw.get("guide_frame"),
                    legacy_idx=raw.get("guide_frame_idx", 0),
                ),
                "first_source": (
                    "previous_clip"
                    if str(raw.get("first_source") or "manual").lower().strip() == "previous_clip" and i > 0
                    else "manual"
                ),
            }
        )

    # Ref2VA + Motion Context is causal, so validation must remain a continuous
    # prefix. FL2VA plans are independent and deliberately keep per-card
    # validation without forcing downstream cards open.
    if _normalize_generation_mode(generation_mode) == "ref2va":
        found_open = False
        for clip in out:
            if found_open:
                clip["validated"] = False
            elif not clip["validated"]:
                found_open = True

    return out


def _prompt_pack_signature_from_state(value):
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value or "{}"))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        return ""
    signature = str(payload.get("prompt_pack_signature") or "").lower().strip()
    if len(signature) != 64 or any(ch not in "0123456789abcdef" for ch in signature):
        return ""
    return signature


def _state_json(clips, prompt_pack_signature="", generation_mode=None):
    payload = {"version": 1, "clips": clips}
    if generation_mode is not None:
        payload["generation_mode"] = _normalize_generation_mode(generation_mode)
    signature = str(prompt_pack_signature or "").lower().strip()
    if len(signature) == 64 and all(ch in "0123456789abcdef" for ch in signature):
        payload["prompt_pack_signature"] = signature
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_external_prompt_pack(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("MiniMax H3 Extender: external prompt pack is invalid.")

    prompts_raw = value.get("prompts")
    if not isinstance(prompts_raw, (list, tuple)):
        raise ValueError("MiniMax H3 Extender: external prompt pack has no prompt list.")

    prompts = []
    for raw in list(prompts_raw)[:MAX_PROMPTS]:
        text = "" if raw is None else str(raw)
        if not text.strip():
            continue
        prompts.append(text)

    if not prompts:
        raise ValueError("MiniMax H3 Extender: external prompt pack contains no prompts.")

    signature = _prompt_pack_signature(prompts)
    return {
        "type": PROMPT_PACK_TYPE,
        "version": int(value.get("version", 1) or 1),
        "source": str(value.get("source") or "External prompt pack"),
        "count": len(prompts),
        "prompts": prompts,
        "signature": signature,
    }


def _sync_clips_from_prompt_pack(clips, pack, stored_signature=""):
    """Import a changed pack into normal clip prompts and sync card count.

    A pack is an import source, not a second runtime prompt path. Once imported,
    the textarea prompt remains authoritative and can be edited normally. The
    same connected pack is therefore not copied again unless its content changes
    or the user changes the number of Extender cards while it is connected.
    """
    if pack is None:
        return clips, str(stored_signature or ""), False, False

    prompts = list(pack.get("prompts") or [])
    desired_count = len(prompts)
    signature = str(pack.get("signature") or "")
    count_changed = len(clips) != desired_count
    content_changed = signature != str(stored_signature or "")
    should_import = bool(count_changed or content_changed)

    if not should_import:
        return clips, signature, False, False

    synced = [dict(c) for c in list(clips)[:desired_count]]
    while len(synced) < desired_count:
        synced.append(_default_clip(len(synced)))

    for index, prompt in enumerate(prompts):
        synced[index]["prompt"] = str(prompt)

    return synced, signature, True, count_changed


def _manifest_for_extender(owner_id, fps=24.0):
    return _manifest_for_first(f"extender_{owner_id}", fps)


EXTENDER_PROGRESS_EVENT = "h3_extender_progress"
EXTENDER_PROMPT_PACK_EVENT = "h3_extender_prompt_pack_import"
EXTENDER_REF_PACK_EVENT = "h3_extender_ref_pack_import"


def _send_extender_prompt_pack_import(node_id, clips_json, prompt_count, source=""):
    try:
        server = PromptServer.instance
        if server is None:
            return
        server.send_sync(
            EXTENDER_PROMPT_PACK_EVENT,
            {
                "node": str(node_id),
                "clips_json": str(clips_json),
                "prompt_count": int(prompt_count),
                "source": str(source or "External prompt pack"),
            },
            getattr(server, "client_id", None),
        )
    except Exception:
        # Prompt import UI feedback must never be able to break generation.
        pass


def _send_extender_ref_pack_import(node_id, refs_json, imported_slots, ref_count, source=""):
    try:
        server = PromptServer.instance
        if server is None:
            return
        server.send_sync(
            EXTENDER_REF_PACK_EVENT,
            {
                "node": str(node_id),
                "refs_json": str(refs_json),
                "imported_slots": [int(i) for i in imported_slots or []],
                "ref_count": int(ref_count),
                "source": str(source or "External reference pack"),
            },
            getattr(server, "client_id", None),
        )
    except Exception:
        # Reference import UI feedback must never be able to break generation.
        pass


def _send_extender_progress(
    node_id,
    clip_index,
    clip_count,
    phase,
    message="",
):
    """
    Send a tiny live UI event while the single Extender node is still running.

    clip_index is 0-based internally. Use -1 to clear the active card.
    """
    try:
        server = PromptServer.instance
        if server is None:
            return
        payload = {
            "node": str(node_id),
            "clip_index": int(clip_index),
            "clip_count": int(clip_count),
            "phase": str(phase),
            "message": str(message),
        }
        # Target the currently connected execution client when available.
        server.send_sync(
            EXTENDER_PROGRESS_EVENT,
            payload,
            getattr(server, "client_id", None),
        )
    except Exception:
        # Progress UI must never be able to break generation.
        pass


def _project_now_iso():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _project_filename(value):
    stem = _safe_name(Path(str(value or "MiniMax_H3_Project")).stem)
    return f"{stem}.ext"


def _project_temp_root():
    root = _ensure_cache_root() / "_projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cleanup_project_downloads():
    now = time.time()
    stale = []
    for token, info in list(_PROJECT_DOWNLOADS.items()):
        if now - float(info.get("created_at", 0.0)) > PROJECT_DOWNLOAD_TTL_SECONDS:
            stale.append((token, info))
    for token, info in stale:
        _PROJECT_DOWNLOADS.pop(token, None)
        try:
            Path(info.get("path", "")).unlink(missing_ok=True)
        except Exception:
            pass
    # Also clean abandoned downloads left by a previous ComfyUI process where
    # the in-memory token table no longer exists.
    try:
        for path in _project_temp_root().glob("download_*.ext"):
            if now - float(path.stat().st_mtime) > PROJECT_DOWNLOAD_TTL_SECONDS:
                path.unlink(missing_ok=True)
    except Exception:
        pass


def _generation_mode_from_project_payload(project_payload):
    extender = project_payload.get("extender", {}) if isinstance(project_payload, dict) else {}
    if not isinstance(extender, dict):
        return "ref2va"
    value = extender.get("generation_mode")
    if value is None:
        settings = extender.get("settings", {})
        if isinstance(settings, dict):
            value = settings.get("generation_mode")
    # Backward compatibility: every project created before FL2VA existed is Ref2VA.
    return _normalize_generation_mode(value or "ref2va")


def _prompt_pack_signature_from_project_payload(project_payload):
    extender = project_payload.get("extender", {}) if isinstance(project_payload, dict) else {}
    raw = extender.get("clips_json")
    if not isinstance(raw, str) or not raw.strip():
        settings = extender.get("settings", {}) if isinstance(extender, dict) else {}
        raw = settings.get("clips_json") if isinstance(settings, dict) else None
    return _prompt_pack_signature_from_state(raw)


def _clips_from_project_payload(project_payload):
    extender = project_payload.get("extender", {}) if isinstance(project_payload, dict) else {}
    generation_mode = _generation_mode_from_project_payload(project_payload)
    raw = extender.get("clips_json")
    if not isinstance(raw, str) or not raw.strip():
        settings = extender.get("settings", {}) if isinstance(extender, dict) else {}
        raw = settings.get("clips_json") if isinstance(settings, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        clips = extender.get("clips") if isinstance(extender, dict) else None
        if isinstance(clips, list):
            raw = json.dumps({"version": 1, "clips": clips}, ensure_ascii=False)
    if not isinstance(raw, str) or not raw.strip():
        raw = _state_json([_default_clip(0)])
    return _parse_clips_json(raw, generation_mode)


def _write_clips_to_project_payload(project_payload, clips, generation_mode=None):
    mode = _normalize_generation_mode(generation_mode or _generation_mode_from_project_payload(project_payload))
    signature = _prompt_pack_signature_from_project_payload(project_payload)
    raw = _state_json(clips, signature, mode)
    extender = project_payload.setdefault("extender", {})
    extender["generation_mode"] = mode
    extender["clips_json"] = raw
    extender["clips"] = copy.deepcopy(clips)
    settings = extender.setdefault("settings", {})
    if isinstance(settings, dict):
        settings["generation_mode"] = mode
        settings["clips_json"] = raw
    return raw


def _fl2va_frame_entries(project_payload):
    if _generation_mode_from_project_payload(project_payload) != "fl2va":
        return []
    clips = _clips_from_project_payload(project_payload)
    entries = []
    for index, cfg in enumerate(clips, start=1):
        for kind in ("first", "last"):
            ref = cfg.get(f"{kind}_frame")
            if isinstance(ref, dict) and _ref_id_is_safe(ref.get("id")):
                entries.append((index, kind, ref))
        for guide_index, guide in enumerate(cfg.get("guides") or [], start=1):
            ref = guide.get("frame") if isinstance(guide, dict) else None
            if isinstance(ref, dict) and _ref_id_is_safe(ref.get("id")):
                entries.append((index, f"guide_{guide_index}", ref))
    return entries


def _project_cache_snapshot(owner_id, project_payload):
    """Return a coherent, immutable manifest snapshot and byte limit.

    The .h3cache is append-only for a live tail. We snapshot the atomically-written
    manifest first, then copy only through the last referenced segment_end. If a
    generation starts concurrently, extra bytes appended after that boundary are
    intentionally excluded from the project.
    """
    generation_mode = _generation_mode_from_project_payload(project_payload)
    cache_owner = (
        _fl2va_cache_owner_id(owner_id)
        if generation_mode == "fl2va"
        else f"extender_{_safe_name(owner_id)}"
    )
    data_path, manifest_path = _chain_paths(cache_owner)
    if not data_path.exists() or not manifest_path.exists():
        return None

    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        return None
    manifest = copy.deepcopy(manifest)
    segments = [dict(x) for x in manifest.get("segments", [])]

    # The UI state is the authority for explicit validation. Persist it into the
    # project snapshot without mutating the user's live cache manifest.
    try:
        clips = _clips_from_project_payload(project_payload)
    except Exception:
        clips = []
    if generation_mode == "fl2va":
        order_by_id = {str(c.get("id")): i for i, c in enumerate(clips)}
        valid_by_id = {str(c.get("id")): bool(c.get("validated", False)) for c in clips}
        segments = [
            desc for desc in segments
            if str(desc.get("clip_id") or "") in order_by_id
        ]
        segments.sort(key=lambda x: order_by_id[str(x.get("clip_id"))])
        for i, desc in enumerate(segments):
            desc["index"] = i
            desc["trim_frames"] = 0
            desc["validated"] = bool(valid_by_id.get(str(desc.get("clip_id") or ""), False))
        manifest["final_frame_count"] = _final_frame_count(segments)
    else:
        for i, desc in enumerate(segments):
            desc["validated"] = bool(i < len(clips) and clips[i].get("validated", False))
    manifest["segments"] = segments

    if segments:
        # FL2VA random-access replacement is append-only, so the logically last
        # card is not necessarily the latest blob in the file. Snapshot through
        # the largest referenced end offset.
        data_limit = max(int(x.get("segment_end", 0) or 0) for x in segments)
    else:
        data_limit = int(_DATA_START)
    if data_limit < int(_DATA_START):
        raise ValueError("MiniMax H3 Extender Project: invalid cache byte boundary.")
    if int(data_path.stat().st_size) < data_limit:
        raise IOError("MiniMax H3 Extender Project: cache changed while snapshotting; retry Save Project.")

    audio_path = _decoded_audio_cache_path(data_path)
    audio_limit = _decoded_audio_cache_end(segments)
    has_audio_cache = any(
        isinstance(desc.get("decoded_audio"), dict)
        and desc["decoded_audio"].get("storage") == "audio_cache"
        for desc in segments
    )
    if has_audio_cache:
        if not audio_path.exists():
            raise FileNotFoundError(
                "MiniMax H3 Extender Project: decoded audio cache is missing."
            )
        if int(audio_path.stat().st_size) < int(audio_limit):
            raise IOError(
                "MiniMax H3 Extender Project: decoded audio cache changed while snapshotting; retry Save Project."
            )
    else:
        audio_limit = 0

    # Save the exact full preview currently shown by the connected Final Decode
    # when it is known to match this cache snapshot.  Ref2VA Full Batch publishes
    # its neutral preview to ComfyUI temp but historically did not maintain
    # chain.preview.mp4, so portable projects could contain all latents/audio yet
    # have no decoded video that can be restored without running VideoVAE again.
    #
    # The browser only supplies the Final Decode id + the clip/frame counts it
    # actually displayed.  The backend resolves the preview path itself and uses
    # it only when those counts exactly match the manifest snapshot, preventing a
    # stale preview from another render/project being embedded accidentally.
    preview_path = _decoded_preview_cache_path(data_path)
    preview_is_full = False
    final_meta = project_payload.get("final_decode")
    if isinstance(final_meta, dict):
        preview_meta = final_meta.get("preview")
        final_id = str(final_meta.get("node_id") or "").strip()
        if final_id and isinstance(preview_meta, dict) and bool(preview_meta.get("available", False)):
            try:
                shown_clips = int(preview_meta.get("clip_count", 0) or 0)
                shown_frames = int(preview_meta.get("frame_count", 0) or 0)
            except Exception:
                shown_clips = 0
                shown_frames = 0
            expected_frames = int(manifest.get("final_frame_count", 0) or 0)
            if shown_clips == len(segments) and shown_frames == expected_frames and shown_frames > 0:
                candidate = _latest_preview_temp_path(final_id)
                if candidate is not None and candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                    preview_path = candidate
                    preview_is_full = True

    if preview_is_full:
        # This is a snapshot-only manifest mutation; the live cache is untouched.
        # It tells the loader that cache/chain.preview.mp4 represents the complete
        # current timeline, including an unvalidated Clip-by-Clip tail candidate.
        manifest["preview_committed_count"] = int(len(segments))
        manifest["preview_audio_mode"] = PREVIEW_AUDIO_MODE
        manifest["preview_portable_full"] = True
    else:
        manifest.pop("preview_portable_full", None)

    return {
        "data_path": data_path,
        "manifest_path": manifest_path,
        "audio_path": audio_path,
        "preview_path": preview_path,
        "preview_is_full": bool(preview_is_full),
        "manifest": manifest,
        "data_limit": data_limit,
        "audio_limit": int(audio_limit),
    }


def _zip_write_prefix(zf, arcname, source_path, byte_limit):
    source_path = Path(source_path)
    remaining = int(byte_limit)
    info = zipfile.ZipInfo(str(arcname))
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = time.localtime(source_path.stat().st_mtime)[:6]
    with open(source_path, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
        while remaining > 0:
            chunk = src.read(min(PROJECT_COPY_CHUNK, remaining))
            if not chunk:
                raise IOError(
                    f"MiniMax H3 Extender Project: source cache ended {remaining} byte(s) early."
                )
            dst.write(chunk)
            remaining -= len(chunk)


def _build_project_archive(owner_id, requested_name, project_payload, output_path):
    project_payload = copy.deepcopy(project_payload)
    generation_mode = _generation_mode_from_project_payload(project_payload)
    extender_meta = project_payload.setdefault("extender", {})
    extender_meta["generation_mode"] = generation_mode
    settings_meta = extender_meta.setdefault("settings", {})
    if isinstance(settings_meta, dict):
        settings_meta["generation_mode"] = generation_mode
    refs = _refs_from_project_payload(project_payload)
    refs = _write_refs_to_project_payload(project_payload, refs)

    # A portable project is only useful when every referenced image is actually
    # embedded. Fail loudly instead of silently writing a project that depends on
    # a local cache entry which may not exist on the destination machine.
    ref_files = []
    source_ref_files = []
    for index, ref in enumerate(refs, start=1):
        if ref is None:
            continue
        path = _ref_path(ref.get("id"))
        if not path.exists():
            raise FileNotFoundError(
                f"MiniMax H3 Extender Project: reference {index} is missing from the internal store. "
                "Reload that reference image before saving the project."
            )
        width, height = _validate_reference_file(path)
        ref["width"] = int(width)
        ref["height"] = int(height)
        ref["size_bytes"] = int(path.stat().st_size)
        ref_files.append((index, ref, path))

        # If the visible ref is an edited derivative, also embed the immutable
        # source pixels. This keeps Reset meaningful after Save/Load and on
        # another machine.
        source_id = str(ref.get("source_id") or ref.get("id") or "").lower().strip()
        if source_id != str(ref.get("id") or "").lower().strip():
            source_path = _ref_path(source_id)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"MiniMax H3 Extender Project: original source for reference {index} is missing. "
                    "Reload that reference image before saving the project."
                )
            _validate_reference_file(source_path)
            source_ref_files.append((index, source_id, source_path))
    _write_refs_to_project_payload(project_payload, refs)

    frame_files = []
    frame_source_files = []
    for clip_index, kind, ref in _fl2va_frame_entries(project_payload):
        path = _ref_path(ref.get("id"))
        if not path.exists():
            raise FileNotFoundError(
                f"MiniMax H3 Extender Project: FL2VA clip {clip_index} {kind} frame is missing from the internal store."
            )
        width, height = _validate_reference_file(path)
        ref["width"] = int(width)
        ref["height"] = int(height)
        ref["size_bytes"] = int(path.stat().st_size)
        frame_files.append((clip_index, kind, ref, path))

        # FL2VA keyframes use the same non-destructive image editor as internal
        # references. Preserve their immutable source pixels too, so Reset still
        # means the original image after a portable .ext Save/Load.
        source_id = str(ref.get("source_id") or ref.get("id") or "").lower().strip()
        if source_id != str(ref.get("id") or "").lower().strip():
            source_path = _ref_path(source_id)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"MiniMax H3 Extender Project: original source for FL2VA clip {clip_index} {kind} frame is missing."
                )
            _validate_reference_file(source_path)
            frame_source_files.append((clip_index, kind, source_id, source_path))

    # FL2VA random-access editing is append-only for speed. Save Project is the
    # natural point to force compaction so both the live cache and portable .ext
    # contain only currently referenced latent/PCM blobs.
    if generation_mode == "fl2va":
        try:
            compact_fl2va_cache(owner_id, force=True)
        except Exception as exc:
            # Save Project must remain reliable even if Windows temporarily has
            # an mmap handle open on the cache. In that rare case the archive is
            # still correct; it may simply include unreclaimed stale bytes.
            print(f"[WARNING] MiniMax H3 Extender: FL2VA Save Project compaction skipped: {exc}")

    snapshot = _project_cache_snapshot(owner_id, project_payload)
    continuity_files = (
        fl2va_project_continuity_files(
            snapshot["data_path"], snapshot["manifest"].get("segments", [])
        )
        if generation_mode == "fl2va" and snapshot is not None
        else []
    )

    if snapshot is not None:
        cache_resolution = _resolution_from_manifest(snapshot.get("manifest"))
        if cache_resolution is not None:
            extender_payload = project_payload.setdefault("extender", {})
            resolution = extender_payload.setdefault("resolution", {})
            if isinstance(resolution, dict):
                resolution["resolved_width"] = int(cache_resolution["width"])
                resolution["resolved_height"] = int(cache_resolution["height"])
                resolution["source"] = "disk_cache"

    archive_meta = {
        "format": PROJECT_FORMAT,
        "format_version": PROJECT_FORMAT_VERSION,
        "created_at": _project_now_iso(),
        "extender_build": BUILD,
        "cache_version": CACHE_VERSION,
        "source_owner_id": str(owner_id),
        "project_name": Path(_project_filename(requested_name)).stem,
        "generation_mode": generation_mode,
        "project": project_payload,
        "references": {
            "count": int(len(ref_files)),
            "embedded": True,
            "original_sources": int(len(source_ref_files)),
            "fl2va_frames": int(len(frame_files)),
            "fl2va_original_sources": int(len(frame_source_files)),
        },
        "cache": {
            "present": snapshot is not None,
            "clip_count": int(len(snapshot["manifest"].get("segments", []))) if snapshot else 0,
            "frame_count": int(snapshot["manifest"].get("final_frame_count", 0)) if snapshot else 0,
            "has_committed_preview": bool(snapshot and snapshot["preview_path"].exists()),
            "has_portable_full_preview": bool(snapshot and snapshot.get("preview_is_full", False)),
            "has_decoded_audio_cache": bool(snapshot and int(snapshot.get("audio_limit", 0)) > 0),
            "fl2va_continuity": [
                {
                    "clip_id": str(item["clip_id"]),
                    "signature": str(item.get("signature") or ""),
                    "png": f"cache/fl2va_continuity/{index:04d}.png",
                    "meta": f"cache/fl2va_continuity/{index:04d}.json",
                }
                for index, item in enumerate(continuity_files, start=1)
            ],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        project_bytes = json.dumps(
            archive_meta,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        zf.writestr("project.json", project_bytes, compress_type=zipfile.ZIP_DEFLATED)

        for index, ref, path in ref_files:
            zf.write(
                path,
                arcname=f"refs/ref_{index}.png",
                compress_type=zipfile.ZIP_STORED,
            )
        for index, source_id, source_path in source_ref_files:
            zf.write(
                source_path,
                arcname=f"refs/original_ref_{index}.png",
                compress_type=zipfile.ZIP_STORED,
            )
        for clip_index, kind, ref, path in frame_files:
            zf.write(
                path,
                arcname=f"fl2va/clip_{clip_index}_{kind}.png",
                compress_type=zipfile.ZIP_STORED,
            )
        for clip_index, kind, source_id, source_path in frame_source_files:
            zf.write(
                source_path,
                arcname=f"fl2va/original_clip_{clip_index}_{kind}.png",
                compress_type=zipfile.ZIP_STORED,
            )
        for index, item in enumerate(continuity_files, start=1):
            zf.write(
                item["png_path"],
                arcname=f"cache/fl2va_continuity/{index:04d}.png",
                compress_type=zipfile.ZIP_STORED,
            )
            zf.write(
                item["meta_path"],
                arcname=f"cache/fl2va_continuity/{index:04d}.json",
                compress_type=zipfile.ZIP_DEFLATED,
            )

        if snapshot is not None:
            manifest_bytes = json.dumps(
                snapshot["manifest"],
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            zf.writestr("cache/chain.json", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
            _zip_write_prefix(
                zf,
                "cache/chain.h3cache",
                snapshot["data_path"],
                snapshot["data_limit"],
            )
            if int(snapshot.get("audio_limit", 0)) > 0:
                _zip_write_prefix(
                    zf,
                    "cache/chain.audio.h3cache",
                    snapshot["audio_path"],
                    snapshot["audio_limit"],
                )
            if snapshot["preview_path"].exists():
                zf.write(
                    snapshot["preview_path"],
                    arcname="cache/chain.preview.mp4",
                    compress_type=zipfile.ZIP_STORED,
                )
    return archive_meta

def _safe_zip_member(name):
    value = str(name or "").replace("\\", "/")
    if not value or value.startswith("/"):
        return False
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return False
    return True


def _zip_copy_member(zf, member_name, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member_name, "r") as src, open(destination, "wb") as dst:
        while True:
            chunk = src.read(PROJECT_COPY_CHUNK)
            if not chunk:
                break
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())


def _replace_cache_transaction(owner_id, new_data=None, new_manifest=None, new_preview=None, new_audio=None, generation_mode="ref2va"):
    mode = _normalize_generation_mode(generation_mode)
    cache_owner = _fl2va_cache_owner_id(owner_id) if mode == "fl2va" else f"extender_{_safe_name(owner_id)}"
    target_data, target_manifest = _chain_paths(cache_owner)
    target_preview = _decoded_preview_cache_path(target_data)
    target_preview_video = _decoded_preview_video_cache_path(target_data)
    target_audio = _decoded_audio_cache_path(target_data)
    target_fl2va_video_dir = target_data.with_suffix(".fl2va.video")
    # The video-only preview prefix is derived and is intentionally not stored
    # in .ext. The decoded-audio cache is primary cache data and is restored
    # together with the latent chain when present.
    targets = [target_data, target_manifest, target_preview, target_preview_video, target_audio]
    backups = []
    token = uuid.uuid4().hex[:10]

    try:
        # FL2VA per-plan decoded videos are a derived local speed cache, not part
        # of the portable .ext payload. Never let files from the previously
        # loaded project survive into a newly imported FL2VA latent cache.
        if mode == "fl2va" and target_fl2va_video_dir.exists():
            shutil.rmtree(target_fl2va_video_dir, ignore_errors=True)
        for target in targets:
            if target.exists():
                backup = target.with_name(target.name + f".project_backup_{token}")
                os.replace(target, backup)
                backups.append((target, backup))

        if new_data is not None and new_manifest is not None:
            os.replace(str(new_data), target_data)
            os.replace(str(new_manifest), target_manifest)
            if new_preview is not None and Path(new_preview).exists():
                os.replace(str(new_preview), target_preview)
            if new_audio is not None and Path(new_audio).exists():
                os.replace(str(new_audio), target_audio)
        # No imported cache means an intentionally empty project. The old cache
        # remains only in backups until this transaction succeeds.
    except Exception:
        for target in targets:
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _, backup in backups:
            try:
                backup.unlink(missing_ok=True)
            except Exception:
                pass


def _import_project_archive(owner_id, archive_path):
    archive_path = Path(archive_path)
    work_root = _project_temp_root() / f"import_{uuid.uuid4().hex}"
    work_root.mkdir(parents=True, exist_ok=True)
    new_data = work_root / "chain.h3cache"
    new_manifest = work_root / "chain.json"
    new_audio = work_root / "chain.audio.h3cache"
    new_preview = work_root / "chain.preview.mp4"
    continuity_restore = []

    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as zf:
            for info in zf.infolist():
                if not _safe_zip_member(info.filename):
                    raise ValueError(
                        f"MiniMax H3 Extender Project: unsafe ZIP entry '{info.filename}'."
                    )

            names = set(zf.namelist())
            if "project.json" not in names:
                raise ValueError("MiniMax H3 Extender Project: project.json is missing.")
            pinfo = zf.getinfo("project.json")
            if int(pinfo.file_size) > PROJECT_JSON_MAX_BYTES:
                raise ValueError("MiniMax H3 Extender Project: project.json is unexpectedly large.")

            with zf.open("project.json", "r") as f:
                archive_meta = json.loads(f.read().decode("utf-8"))
            if archive_meta.get("format") != PROJECT_FORMAT:
                raise ValueError("MiniMax H3 Extender Project: unsupported project file.")
            format_version = int(archive_meta.get("format_version", -1))
            if format_version not in PROJECT_SUPPORTED_VERSIONS:
                supported = ", ".join(str(v) for v in sorted(PROJECT_SUPPORTED_VERSIONS))
                raise ValueError(
                    "MiniMax H3 Extender Project: incompatible project format "
                    f"{format_version} (supported: {supported})."
                )

            project_payload = archive_meta.get("project", {})
            if not isinstance(project_payload, dict):
                raise ValueError("MiniMax H3 Extender Project: invalid project metadata.")
            generation_mode = _generation_mode_from_project_payload(project_payload)
            # Archives created before FL2VA have no mode marker and are always Ref2VA.
            project_payload.setdefault("extender", {})["generation_mode"] = generation_mode
            project_payload["extender"].setdefault("settings", {})["generation_mode"] = generation_mode
            clips = _clips_from_project_payload(project_payload)
            project_prompt_pack_signature = _prompt_pack_signature_from_project_payload(project_payload)

            # v2 embeds the real reference pixels. Import each image into the
            # Extender's content-addressed store and rewrite the returned project
            # metadata to the local ids. v1 projects remain fully loadable; they
            # simply have no embedded internal references.
            saved_refs = _refs_from_project_payload(project_payload)
            imported_refs = _empty_refs()
            if format_version >= 2:
                for index in range(1, MAX_IMAGE_REFS + 1):
                    member = f"refs/ref_{index}.png"
                    saved = saved_refs[index - 1]
                    if member not in names:
                        if saved is not None:
                            raise ValueError(
                                f"MiniMax H3 Extender Project: embedded reference {index} is missing."
                            )
                        continue
                    ref_info = zf.getinfo(member)
                    if int(ref_info.file_size) > MAX_REF_UPLOAD_BYTES:
                        raise ValueError(
                            f"MiniMax H3 Extender Project: reference {index} exceeds the allowed image size."
                        )
                    extracted = work_root / f"ref_{index}.png"
                    _zip_copy_member(zf, member, extracted)
                    desc = _store_project_reference(
                        extracted,
                        (saved or {}).get("original_name") if isinstance(saved, dict) else f"ref_{index}.png",
                    )
                    if isinstance(saved, dict) and saved.get("id") and desc["id"] != saved.get("id"):
                        raise ValueError(
                            f"MiniMax H3 Extender Project: reference {index} failed its integrity check."
                        )

                    saved_source_id = (
                        str(saved.get("source_id") or saved.get("id") or desc["id"]).lower().strip()
                        if isinstance(saved, dict)
                        else desc["id"]
                    )
                    source_member = f"refs/original_ref_{index}.png"
                    if source_member in names:
                        source_info = zf.getinfo(source_member)
                        if int(source_info.file_size) > MAX_REF_UPLOAD_BYTES:
                            raise ValueError(
                                f"MiniMax H3 Extender Project: original reference {index} exceeds the allowed image size."
                            )
                        source_extracted = work_root / f"original_ref_{index}.png"
                        _zip_copy_member(zf, source_member, source_extracted)
                        source_desc = _store_project_reference(
                            source_extracted,
                            (saved or {}).get("original_name") if isinstance(saved, dict) else f"ref_{index}.png",
                        )
                        if _ref_id_is_safe(saved_source_id) and source_desc["id"] != saved_source_id:
                            raise ValueError(
                                f"MiniMax H3 Extender Project: original reference {index} failed its integrity check."
                            )
                        desc["source_id"] = source_desc["id"]
                    else:
                        # v2 projects created before v14.49 only embedded the
                        # currently used pixels. They remain loadable; that image
                        # becomes their reset baseline because the older archive
                        # contains no recoverable original.
                        desc["source_id"] = desc["id"]

                    if isinstance(saved, dict) and desc["source_id"] != desc["id"]:
                        for key in ("saturation", "contrast", "brightness"):
                            try:
                                desc[key] = max(0.0, min(200.0, float(saved.get(key, 100) or 100)))
                            except Exception:
                                desc[key] = 100.0
                    if isinstance(saved, dict):
                        external_signature = str(saved.get("external_signature") or "").lower().strip()
                        if _ref_id_is_safe(external_signature):
                            desc["external_signature"] = external_signature
                    imported_refs[index - 1] = desc
                imported_refs = _normalize_ref_descriptors(imported_refs)
            _write_refs_to_project_payload(project_payload, imported_refs)

            if generation_mode == "fl2va":
                for clip_index, cfg in enumerate(clips, start=1):
                    targets = [
                        ("first", cfg.get("first_frame"), ("frame", "first_frame")),
                        ("last", cfg.get("last_frame"), ("frame", "last_frame")),
                    ]
                    for guide_index, guide in enumerate(cfg.get("guides") or []):
                        if isinstance(guide, dict):
                            targets.append(
                                (f"guide_{guide_index + 1}", guide.get("frame"), ("guide", guide_index))
                            )

                    for kind, saved, assign_target in targets:
                        if saved is None:
                            continue
                        archive_kind = kind
                        member = f"fl2va/clip_{clip_index}_{archive_kind}.png"
                        # v2.2.x portable projects stored their only AddGuide
                        # image under the legacy unsuffixed guide filename.
                        if (
                            kind == "guide_1"
                            and member not in names
                            and f"fl2va/clip_{clip_index}_guide.png" in names
                        ):
                            archive_kind = "guide"
                            member = f"fl2va/clip_{clip_index}_guide.png"
                        if member not in names:
                            raise ValueError(
                                f"MiniMax H3 Extender Project: embedded FL2VA clip {clip_index} {kind} frame is missing."
                            )
                        info = zf.getinfo(member)
                        if int(info.file_size) > MAX_REF_UPLOAD_BYTES:
                            raise ValueError(
                                f"MiniMax H3 Extender Project: FL2VA clip {clip_index} {kind} frame exceeds the allowed image size."
                            )
                        extracted = work_root / f"fl2va_{clip_index}_{kind}.png"
                        _zip_copy_member(zf, member, extracted)
                        desc = _store_project_reference(
                            extracted,
                            (saved or {}).get("original_name") if isinstance(saved, dict) else f"clip_{clip_index}_{kind}.png",
                        )
                        if isinstance(saved, dict) and saved.get("id") and desc["id"] != saved.get("id"):
                            raise ValueError(
                                f"MiniMax H3 Extender Project: FL2VA clip {clip_index} {kind} frame failed its integrity check."
                            )

                        saved_source_id = (
                            str(saved.get("source_id") or saved.get("id") or desc["id"]).lower().strip()
                            if isinstance(saved, dict)
                            else desc["id"]
                        )
                        source_member = f"fl2va/original_clip_{clip_index}_{archive_kind}.png"
                        if source_member in names:
                            source_info = zf.getinfo(source_member)
                            if int(source_info.file_size) > MAX_REF_UPLOAD_BYTES:
                                raise ValueError(
                                    f"MiniMax H3 Extender Project: original FL2VA clip {clip_index} {kind} frame exceeds the allowed image size."
                                )
                            source_extracted = work_root / f"original_fl2va_{clip_index}_{kind}.png"
                            _zip_copy_member(zf, source_member, source_extracted)
                            source_desc = _store_project_reference(
                                source_extracted,
                                (saved or {}).get("original_name") if isinstance(saved, dict) else f"clip_{clip_index}_{kind}.png",
                            )
                            if _ref_id_is_safe(saved_source_id) and source_desc["id"] != saved_source_id:
                                raise ValueError(
                                    f"MiniMax H3 Extender Project: original FL2VA clip {clip_index} {kind} frame failed its integrity check."
                                )
                            desc["source_id"] = source_desc["id"]
                        else:
                            desc["source_id"] = desc["id"]

                        if isinstance(saved, dict) and desc["source_id"] != desc["id"]:
                            for adjustment_key in ("saturation", "contrast", "brightness"):
                                try:
                                    desc[adjustment_key] = max(
                                        0.0,
                                        min(200.0, float(saved.get(adjustment_key, 100) or 100)),
                                    )
                                except Exception:
                                    desc[adjustment_key] = 100.0

                        target_type, target_value = assign_target
                        if target_type == "frame":
                            cfg[target_value] = desc
                        else:
                            cfg["guides"][int(target_value)]["frame"] = desc
                _write_clips_to_project_payload(project_payload, clips, generation_mode)

            has_data = "cache/chain.h3cache" in names
            has_manifest = "cache/chain.json" in names
            if has_data != has_manifest:
                raise ValueError("MiniMax H3 Extender Project: incomplete cache payload.")

            imported_manifest = None
            if has_data:
                _zip_copy_member(zf, "cache/chain.h3cache", new_data)
                _zip_copy_member(zf, "cache/chain.json", new_manifest)
                if "cache/chain.audio.h3cache" in names:
                    _zip_copy_member(zf, "cache/chain.audio.h3cache", new_audio)
                if "cache/chain.preview.mp4" in names:
                    _zip_copy_member(zf, "cache/chain.preview.mp4", new_preview)

                imported_manifest = _load_manifest_from_paths(new_data, new_manifest)
                if imported_manifest is None:
                    raise ValueError("MiniMax H3 Extender Project: cache manifest is empty.")

                imported_segments = [dict(x) for x in imported_manifest.get("segments", [])]
                needs_audio_cache = any(
                    isinstance(desc.get("decoded_audio"), dict)
                    and desc["decoded_audio"].get("storage") == "audio_cache"
                    for desc in imported_segments
                )
                if needs_audio_cache:
                    if not new_audio.exists():
                        raise ValueError(
                            "MiniMax H3 Extender Project: decoded audio cache is missing."
                        )
                    required_audio = _decoded_audio_cache_end(imported_segments)
                    if int(new_audio.stat().st_size) < int(required_audio):
                        raise ValueError(
                            "MiniMax H3 Extender Project: decoded audio cache is truncated."
                        )

                # Ref2VA projects retain the portable sequential prefix. FL2VA
                # projects use stable clip ids and therefore filter/reorder the
                # manifest without rewriting the append-only latent bytes.
                if generation_mode == "ref2va" and len(imported_manifest.get("segments", [])) > len(clips):
                    imported_manifest = _truncate_chain(
                        new_data, new_manifest, imported_manifest, len(clips)
                    )

                segments = [dict(x) for x in imported_manifest.get("segments", [])]
                if generation_mode == "fl2va":
                    order = {str(c.get("id")): i for i, c in enumerate(clips)}
                    valid = {str(c.get("id")): bool(c.get("validated", False)) for c in clips}
                    segments = [x for x in segments if str(x.get("clip_id") or "") in order]
                    segments.sort(key=lambda x: order[str(x.get("clip_id"))])
                    for i, desc in enumerate(segments):
                        desc["index"] = i
                        desc["trim_frames"] = 0
                        desc["validated"] = bool(valid.get(str(desc.get("clip_id") or ""), False))
                else:
                    for i, desc in enumerate(segments):
                        desc["validated"] = bool(i < len(clips) and clips[i].get("validated", False))
                imported_manifest = dict(imported_manifest)
                imported_manifest["segments"] = segments
                imported_manifest["final_frame_count"] = _final_frame_count(segments)
                imported_manifest["sequence_mode"] = generation_mode
                imported_manifest["owner_id"] = (
                    _fl2va_cache_owner_id(owner_id)
                    if generation_mode == "fl2va"
                    else f"extender_{_safe_name(owner_id)}"
                )
                imported_manifest["imported_at"] = time.time()
                imported_manifest["updated_at"] = time.time()
                _write_json_atomic(new_manifest, imported_manifest)

                imported_resolution = _resolution_from_manifest(imported_manifest)
                if imported_resolution is not None:
                    extender_payload = project_payload.setdefault("extender", {})
                    resolution = extender_payload.setdefault("resolution", {})
                    if isinstance(resolution, dict):
                        resolution["resolved_width"] = int(imported_resolution["width"])
                        resolution["resolved_height"] = int(imported_resolution["height"])
                        resolution["source"] = "disk_cache"

                # v14.97+ projects optionally preserve the tiny lossless FL2VA
                # continuity PNG+JSON pairs. They are derived caches, but restoring
                # them avoids a whole VideoVAE decode the first time Previous is
                # used after loading a portable project.
                if generation_mode == "fl2va":
                    valid_ids = {
                        str(x.get("clip_id") or "")
                        for x in imported_manifest.get("segments", [])
                        if str(x.get("clip_id") or "")
                    }
                    cache_meta = archive_meta.get("cache", {}) if isinstance(archive_meta.get("cache"), dict) else {}
                    continuity_meta = cache_meta.get("fl2va_continuity", [])
                    if isinstance(continuity_meta, list):
                        for entry_index, entry in enumerate(continuity_meta, start=1):
                            if not isinstance(entry, dict):
                                continue
                            clip_id = str(entry.get("clip_id") or "")
                            if not clip_id or clip_id not in valid_ids:
                                continue
                            png_member = str(entry.get("png") or "")
                            json_member = str(entry.get("meta") or "")
                            if png_member not in names or json_member not in names:
                                continue
                            png_info = zf.getinfo(png_member)
                            json_info = zf.getinfo(json_member)
                            if int(png_info.file_size) > MAX_REF_UPLOAD_BYTES or int(json_info.file_size) > 256 * 1024:
                                raise ValueError(
                                    f"MiniMax H3 Extender Project: continuity cache for {clip_id} is unexpectedly large."
                                )
                            png_path = work_root / f"continuity_{entry_index:04d}.png"
                            json_path = work_root / f"continuity_{entry_index:04d}.json"
                            _zip_copy_member(zf, png_member, png_path)
                            _zip_copy_member(zf, json_member, json_path)

                            # Validate the optional derived cache *before* the
                            # live project cache is replaced. Continuity signatures
                            # intentionally keep the historical SHA256-of-PNG-bytes
                            # scheme so v14.96 caches remain portable/compatible.
                            try:
                                continuity_json = json.loads(json_path.read_text(encoding="utf-8"))
                                if (
                                    int(continuity_json.get("version", 0) or 0) != 2
                                    or str(continuity_json.get("algorithm") or "") != "best_of_last_6"
                                ):
                                    raise ValueError("unsupported continuity metadata")
                                continuity_signature = str(continuity_json.get("signature") or "").lower().strip()
                                archive_signature = str(entry.get("signature") or "").lower().strip()
                                if not re.fullmatch(r"[0-9a-f]{64}", continuity_signature):
                                    raise ValueError("invalid continuity signature")
                                if archive_signature and archive_signature != continuity_signature:
                                    raise ValueError("continuity signature mismatch")
                                with Image.open(png_path) as continuity_image:
                                    continuity_image.verify()
                                if _hash_file(png_path) != continuity_signature:
                                    raise ValueError("continuity image integrity mismatch")
                            except Exception as exc:
                                raise ValueError(
                                    f"MiniMax H3 Extender Project: continuity cache for {clip_id} failed its integrity check."
                                ) from exc
                            continuity_restore.append((clip_id, png_path, json_path))

            cached_count = int(len(imported_manifest.get("segments", []))) if imported_manifest else 0
            # A clip can only remain validated when its physical cached segment is present.
            if generation_mode == "fl2va":
                cached_ids = {
                    str(x.get("clip_id")) for x in (imported_manifest or {}).get("segments", [])
                    if str(x.get("clip_id") or "")
                }
                for clip_cfg in clips:
                    if str(clip_cfg.get("id")) not in cached_ids:
                        clip_cfg["validated"] = False
            else:
                for i in range(cached_count, len(clips)):
                    clips[i]["validated"] = False
                found_open = False
                for clip_cfg in clips:
                    if found_open:
                        clip_cfg["validated"] = False
                    elif not clip_cfg["validated"]:
                        found_open = True

            normalized_clips_json = _state_json(clips, project_prompt_pack_signature, generation_mode)
            extender_payload = project_payload.setdefault("extender", {})
            extender_payload["generation_mode"] = generation_mode
            extender_payload["clips_json"] = normalized_clips_json
            extender_payload["clips"] = clips
            settings = extender_payload.setdefault("settings", {})
            if isinstance(settings, dict):
                settings["generation_mode"] = generation_mode
                settings["clips_json"] = normalized_clips_json

            _replace_cache_transaction(
                owner_id,
                new_data if imported_manifest is not None else None,
                new_manifest if imported_manifest is not None else None,
                new_preview if imported_manifest is not None and new_preview.exists() else None,
                new_audio if imported_manifest is not None and new_audio.exists() else None,
                generation_mode=generation_mode,
            )

            # Portable .ext projects are intentionally single-mode. The frontend
            # also resets the inactive card timeline when importing one, so an
            # old cache from the opposite mode must not survive on disk. If it
            # did, a later frontend/default-mode mistake could resurrect a stale
            # preview from a completely different project after F5/restart.
            inactive_mode = "ref2va" if generation_mode == "fl2va" else "fl2va"
            try:
                _replace_cache_transaction(owner_id, generation_mode=inactive_mode)
            except Exception as exc:
                print(
                    f"[WARNING] MiniMax H3 Extender: could not clear stale {inactive_mode} "
                    f"cache while loading project: {exc}"
                )

            if generation_mode == "fl2va" and imported_manifest is not None and continuity_restore:
                target_data, _ = _chain_paths(_fl2va_cache_owner_id(owner_id))
                for clip_id, png_path, json_path in continuity_restore:
                    try:
                        install_fl2va_project_continuity(
                            target_data, clip_id, png_path, json_path
                        )
                    except Exception as exc:
                        # Continuity sidecars are a derived speed cache. Their
                        # failure must never invalidate a project whose primary
                        # latent/PCM cache has already loaded successfully.
                        print(
                            f"[WARNING] MiniMax H3 Extender: could not restore FL2VA continuity cache "
                            f"for {clip_id}: {exc}"
                        )

            loaded_continuity_signatures = {}
            if generation_mode == "fl2va" and imported_manifest is not None:
                target_data, _ = _chain_paths(_fl2va_cache_owner_id(owner_id))
                loaded_continuity_signatures = continuity_signatures_for_segments(
                    target_data, imported_manifest.get("segments", [])
                )

            validated_count = 0
            if imported_manifest is not None:
                if generation_mode == "fl2va":
                    validated_count = sum(bool(x.get("validated", False)) for x in imported_manifest.get("segments", []))
                else:
                    for desc in imported_manifest.get("segments", []):
                        if not bool(desc.get("validated", False)):
                            break
                        validated_count += 1

            loaded_resolution = _resolution_from_manifest(imported_manifest)
            return {
                "project_name": str(archive_meta.get("project_name") or archive_path.stem),
                "project": project_payload,
                "references": {
                    "count": int(_reference_count(imported_refs)),
                    "refs_json": _refs_json(imported_refs),
                },
                "cache": {
                    "present": imported_manifest is not None,
                    "cached_count": cached_count,
                    "validated_count": int(validated_count),
                    "cached_clip_ids": [
                        str(x.get("clip_id")) for x in (imported_manifest or {}).get("segments", [])
                        if str(x.get("clip_id") or "")
                    ],
                    "validated_clip_ids": [
                        str(x.get("clip_id")) for x in (imported_manifest or {}).get("segments", [])
                        if str(x.get("clip_id") or "") and bool(x.get("validated", False))
                    ],
                    "continuity_signatures": loaded_continuity_signatures,
                    "frame_count": int(imported_manifest.get("final_frame_count", 0)) if imported_manifest else 0,
                    "resolved_width": int(loaded_resolution["width"]) if loaded_resolution else 0,
                    "resolved_height": int(loaded_resolution["height"]) if loaded_resolution else 0,
                },
            }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)



class MiniMaxH3Extender:
    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = list(comfy.samplers.SAMPLER_NAMES)
        scheduler_names = list(comfy.samplers.SCHEDULER_NAMES)
        default_sampler = "euler" if "euler" in sampler_names else sampler_names[0]
        default_scheduler = "simple" if "simple" in scheduler_names else scheduler_names[0]

        required = {
            "model": (
                "MODEL",
                {
                    "lazy": True,
                    "tooltip": "MiniMax H3 Ref2VA model. Evaluated only while MODE is REF2VA.",
                },
            ),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "run_mode": (["clip_by_clip", "full_batch"], {"default": "clip_by_clip"}),
            "width": (
                "INT",
                {
                    "default": 896, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Manual resolution width, also used as Auto fallback when no internal image reference is loaded.",
                },
            ),
            "height": (
                "INT",
                {
                    "default": 576, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Manual resolution height, also used as Auto fallback when no internal image reference is loaded.",
                },
            ),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "step": 1}),
            "sampler_name": (sampler_names, {"default": default_sampler}),
            "scheduler": (scheduler_names, {"default": default_scheduler}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            "context_length": (["22", "5", "39", "56"], {"default": "22"}),
            "audio_context_length": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            "clips_json": (
                "STRING",
                {
                    "default": _state_json([_default_clip(0)]),
                    "multiline": True,
                },
            ),
            # Keep these AFTER clips_json so pre-v14.25 workflow widget arrays
            # keep their original positional mapping. The frontend migrates old
            # workflows/projects to Manual mode on load.
            "resolution_mode": (
                ["auto_from_ref", "manual"],
                {
                    "default": "auto_from_ref",
                    "tooltip": "Auto uses internal Ref 1 as the aspect-ratio guide; with no internal image references it falls back to width/height.",
                },
            ),
            "megapixels": (
                "FLOAT",
                {
                    "default": DEFAULT_MEGAPIXELS, "min": 0.01, "max": 16.0, "step": 0.01,
                    "tooltip": "Target total pixels for Auto resolution. Auto and Manual canvases use the MiniMax H3 32-pixel grid; Auto snaps downward without exceeding the requested pixel budget.",
                },
            ),
            # Internal image-reference manager state. Appended after the v14.25
            # widgets so older positional workflow widget arrays keep mapping.
            "refs_json": (
                "STRING",
                {
                    "default": _refs_json(_empty_refs()),
                    "multiline": True,
                },
            ),
            # Appended after every legacy widget so old positional workflow
            # arrays continue to map exactly as before. The frontend presents
            # this as the REF2VA / FL2VA mode button.
            "generation_mode": (
                ["ref2va", "fl2va"],
                {"default": "ref2va"},
            ),
        }

        # Audio and video references remain external sockets. Image refs continue
        # to be owned by the Extender's internal manager; ref_pack is only an
        # optional import path that writes external IMAGEs into those same
        # stable slots.
        optional = {
            "fl2va_model": (
                "MODEL",
                {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Optional MiniMax H3 FL2VA model. Evaluated only while MODE is FL2VA.",
                },
            ),
            "audio_vae": ("VAE", {"forceInput": True}),
            "ref_audio": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Legacy alias of ref_audio_1. Kept for older workflows; new workflows should prefer ref_audio_1..ref_audio_3.",
                },
            ),
            "ref_audio_1": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 standalone reference audio 1. Up to three standalone audio references are supported.",
                },
            ),
            "ref_audio_2": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 standalone reference audio 2.",
                },
            ),
            "ref_audio_3": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 standalone reference audio 3.",
                },
            ),
            "ref_video_1": (
                "IMAGE",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 reference video 1 as an IMAGE frame batch. H3 expects 24 fps; use ref_video_fps_1 when the source batch came from another frame rate. Use <Video 1> in prompts.",
                },
            ),
            "ref_video_fps_1": (
                "FLOAT",
                {
                    "forceInput": True,
                    "tooltip": "Optional source FPS from Get Video Components for ref_video_1. When disconnected the Extender assumes the IMAGE batch is already 24 fps.",
                },
            ),
            "ref_video_audio_1": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional soundtrack of ref_video_1.",
                },
            ),
            "ref_video_2": (
                "IMAGE",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 reference video 2 as an IMAGE frame batch. H3 expects 24 fps; use ref_video_fps_2 when the source batch came from another frame rate. Use <Video 2> in prompts.",
                },
            ),
            "ref_video_fps_2": (
                "FLOAT",
                {
                    "forceInput": True,
                    "tooltip": "Optional source FPS from Get Video Components for ref_video_2. When disconnected the Extender assumes the IMAGE batch is already 24 fps.",
                },
            ),
            "ref_video_audio_2": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional soundtrack of ref_video_2.",
                },
            ),
            "ref_video_3": (
                "IMAGE",
                {
                    "forceInput": True,
                    "tooltip": "Optional MiniMax H3 reference video 3 as an IMAGE frame batch. H3 expects 24 fps; use ref_video_fps_3 when the source batch came from another frame rate. Use <Video 3> in prompts.",
                },
            ),
            "ref_video_fps_3": (
                "FLOAT",
                {
                    "forceInput": True,
                    "tooltip": "Optional source FPS from Get Video Components for ref_video_3. When disconnected the Extender assumes the IMAGE batch is already 24 fps.",
                },
            ),
            "ref_video_audio_3": (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional soundtrack of ref_video_3.",
                },
            ),
            # Keep the two pack sockets visually last. The frontend also
            # preserves this ordering when dynamic AV sockets grow/shrink.
            "ref_pack": (
                REF_PACK_TYPE,
                {
                    "tooltip": "Optional external image-reference pack. Connected Ref N slots are imported into the matching internal Ref N slots on Queue; empty slots leave internal references untouched."
                },
            ),
            "prompt_pack": (
                PROMPT_PACK_TYPE,
                {
                    "tooltip": "Optional external prompt pack. New/changed packs are imported into the normal clip textareas and synchronize the clip count."
                },
            ),
        }

        return {
            "required": required,
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CACHE_TYPE, "INT", "INT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = (
        "cache",
        "clip_count",
        "validated_count",
        "status",
        "cache_size_mb",
        "build",
    )
    FUNCTION = "extend"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = False

    @classmethod
    def check_lazy_status(
        cls,
        generation_mode="ref2va",
        model=_LAZY_UNCONNECTED,
        fl2va_model=_LAZY_UNCONNECTED,
        **_,
    ):
        """Evaluate only the H3 checkpoint required by the active mode.

        Ref2VA and FL2VA are separate ~23 GB checkpoints. Without lazy model
        inputs ComfyUI resolves both upstream loaders before ``extend`` runs,
        even though only one model is sampled. The mode widget is non-lazy, so
        it is already available here and can select the sole dependency.
        """
        mode = _normalize_generation_mode(generation_mode)
        name = "fl2va_model" if mode == "fl2va" else "model"
        value = fl2va_model if mode == "fl2va" else model

        # Optional FL2VA socket may genuinely be empty. In that case do not
        # request the inactive Ref2VA model as a fallback: execution will emit
        # the existing clear "fl2va_model required" error instead.
        if value is _LAZY_UNCONNECTED:
            return []
        return [name] if value is None else []

    def _extend_fl2va(
        self,
        *,
        owner,
        clips,
        active_prompt_pack_signature,
        prompt_pack_imported,
        external_prompt_pack,
        model,
        fl2va_model,
        clip,
        vae,
        run_mode,
        width,
        height,
        steps,
        sampler_name,
        scheduler,
        denoise,
        resolution_mode,
        megapixels,
    ):
        if fl2va_model is None:
            raise ValueError(
                "MiniMax H3 Extender: FL2VA mode requires the fl2va_model input."
            )

        clip_ids = [str(cfg.get("id") or f"clip_{i + 1}") for i, cfg in enumerate(clips)]
        data_path, manifest_path, manifest = sync_fl2va_manifest(owner, FPS, clip_ids)

        # FL2VA Auto resolution uses the first available card keyframe as its
        # aspect-ratio guide. Internal Ref2VA references remain untouched and are
        # simply irrelevant in this mode.
        frame_guides = []
        for cfg in clips:
            if cfg.get("first_frame") is not None:
                frame_guides.append(cfg.get("first_frame"))
            elif cfg.get("last_frame") is not None:
                frame_guides.append(cfg.get("last_frame"))
            else:
                first_guide = next(
                    (g.get("frame") for g in (cfg.get("guides") or []) if isinstance(g, dict) and g.get("frame") is not None),
                    None,
                )
                if first_guide is not None:
                    frame_guides.append(first_guide)
        requested_resolution = _resolve_generation_resolution(
            resolution_mode, megapixels, width, height, frame_guides
        )
        resolution = dict(requested_resolution)
        resolution["requested_width"] = int(requested_resolution["width"])
        resolution["requested_height"] = int(requested_resolution["height"])
        resolution["cache_reset"] = False
        resolved_width = int(resolution["width"])
        resolved_height = int(resolution["height"])

        cache_resolution = _resolution_from_manifest(manifest)
        previous_cache_resolution = None
        if manifest.get("segments") and cache_resolution is not None:
            if (
                int(cache_resolution["width"]) != resolved_width
                or int(cache_resolution["height"]) != resolved_height
            ):
                previous_cache_resolution = dict(cache_resolution)
                manifest = _truncate_chain(data_path, manifest_path, manifest, 0)
                manifest = dict(manifest)
                manifest["sequence_mode"] = "fl2va"
                manifest["updated_at"] = time.time()
                _write_json_atomic(manifest_path, manifest)
                for cfg in clips:
                    cfg["validated"] = False
                resolution["cache_reset"] = True

        if prompt_pack_imported and external_prompt_pack is not None:
            imported_json = _state_json(
                clips, active_prompt_pack_signature, "fl2va"
            )
            _send_extender_prompt_pack_import(
                owner,
                imported_json,
                len(external_prompt_pack.get("prompts") or []),
                external_prompt_pack.get("source") or "External prompt pack",
            )

        # Re-sync after a possible geometry reset. The manifest follows stable
        # card ids, which is what makes insert/remove/reorder independent of the
        # physical append-only latent file.
        data_path, manifest_path, manifest = sync_fl2va_manifest(owner, FPS, clip_ids)
        cached_ids = cached_fl2va_ids(manifest)
        generated = []
        statuses = []
        previous_handle = None

        def _dependent_indices(after_index):
            out = []
            for dep_i in range(int(after_index) + 1, len(clips)):
                if str(clips[dep_i].get("first_source") or "manual") != "previous_clip":
                    break
                out.append(dep_i)
            return out

        def _drop_stale_indices(indices):
            nonlocal data_path, manifest_path, manifest, cached_ids
            unique = sorted({int(x) for x in indices if 0 <= int(x) < len(clips)})
            if not unique:
                return
            stale_ids = []
            for dep_i in unique:
                clips[dep_i]["validated"] = False
                stale_ids.append(clip_ids[dep_i])
            data_path, manifest_path, manifest = drop_fl2va_cached_ids(
                owner, FPS, clip_ids, stale_ids
            )
            cached_ids = cached_fl2va_ids(manifest)

        for i, cfg in enumerate(clips):
            clip_id = clip_ids[i]
            first_source = (
                "previous_clip"
                if str(cfg.get("first_source") or "manual").lower().strip() == "previous_clip" and i > 0
                else "manual"
            )
            cfg["first_source"] = first_source
            current_desc = next(
                (dict(x) for x in manifest.get("segments", []) if str(x.get("clip_id") or "") == clip_id),
                None,
            )
            cached = clip_id in cached_ids

            first_frame = None
            dependency_meta = {"first_source": first_source}
            if first_source == "previous_clip":
                previous_clip_id = clip_ids[i - 1]
                first_frame, previous_signature = resolve_fl2va_previous_frame(
                    owner, FPS, clip_ids, previous_clip_id, vae
                )
                dependency_meta.update({
                    "previous_clip_id": previous_clip_id,
                    "previous_frame_signature": previous_signature,
                })

            # A cached plan linked to the previous final frame is valid only while
            # it still points to the same predecessor and the same generated pixels.
            # This also catches insert/remove/reorder operations across saved workflows.
            stale_dependency = False
            if cached and current_desc is not None:
                stored_source = str(current_desc.get("first_source") or "manual")
                if stored_source != first_source:
                    stale_dependency = True
                elif first_source == "previous_clip":
                    stale_dependency = (
                        str(current_desc.get("previous_clip_id") or "") != str(dependency_meta["previous_clip_id"])
                        or str(current_desc.get("previous_frame_signature") or "") != str(dependency_meta["previous_frame_signature"])
                    )
            if stale_dependency:
                _drop_stale_indices([i] + _dependent_indices(i))
                current_desc = None
                cached = False

            if bool(cfg.get("validated")) and cached:
                if first_frame is not None:
                    del first_frame
                continue
            if bool(cfg.get("validated")) and not cached:
                cfg["validated"] = False

            _send_extender_progress(
                owner, i, len(clips), "preparing",
                f"Preparing FL2VA clip {i + 1}/{len(clips)}",
            )
            frame_count = _duration_to_frames(cfg["duration"])
            first_desc = cfg.get("first_frame")
            last_desc = cfg.get("last_frame")
            if first_source == "manual":
                first_frame = _load_reference_tensor(first_desc) if first_desc is not None else None
            last_frame = _load_reference_tensor(last_desc) if last_desc is not None else None
            guide_frames = []
            for guide_cfg in cfg.get("guides") or []:
                if not isinstance(guide_cfg, dict):
                    continue
                guide_desc = guide_cfg.get("frame")
                if guide_desc is None:
                    continue
                guide_frames.append({
                    "frame": _load_reference_tensor(guide_desc),
                    "frame_idx": int(guide_cfg.get("frame_idx", 0) or 0),
                })

            clip_model, clip_text_encoder = _apply_per_clip_loras(
                self, fl2va_model, clip, cfg.get("loras"), i
            )
            positive, latent = make_fl2va_conditioning(
                clip_text_encoder,
                vae,
                cfg.get("prompt", ""),
                resolved_width,
                resolved_height,
                frame_count,
                first_frame=first_frame,
                last_frame=last_frame,
                guide_frames=guide_frames,
            )

            _send_extender_progress(
                owner, i, len(clips), "sampling",
                f"Rendering FL2VA clip {i + 1}/{len(clips)}",
            )
            sampled = _sample_h3(
                clip_model, positive, latent, cfg["seed"],
                str(sampler_name), str(scheduler), int(steps), float(denoise),
            )
            (
                previous_handle,
                _proxy,
                manifest,
                cache_status,
                _cache_mb,
            ) = store_fl2va_segment(
                owner,
                float(FPS),
                clip_ids,
                i,
                clip_id,
                sampled,
                validated=False,
                run_mode=str(run_mode),
                dependency_meta=dependency_meta,
            )
            statuses.append(cache_status)
            cached_ids.add(clip_id)
            generated.append(i)
            cfg["validated"] = False

            # Rerendering an upstream plan changes the real image used by every
            # consecutive Previous-linked follower. Their latent caches are now
            # stale, while the first following manual plan remains independent.
            dependent = _dependent_indices(i)
            if dependent:
                _drop_stale_indices(dependent)

            _send_extender_progress(
                owner, i, len(clips), "complete",
                f"FL2VA clip {i + 1}/{len(clips)} complete",
            )
            del sampled, positive, latent, clip_model, clip_text_encoder
            if first_frame is not None:
                del first_frame
            if last_frame is not None:
                del last_frame
            for guide_item in guide_frames:
                guide_tensor = guide_item.get("frame") if isinstance(guide_item, dict) else None
                if guide_tensor is not None:
                    del guide_tensor
            guide_frames.clear()

            if str(run_mode) == "clip_by_clip":
                break

        validation_by_id = {
            clip_ids[i]: bool(cfg.get("validated", False))
            for i, cfg in enumerate(clips)
        }
        previous_handle, final_manifest = set_fl2va_validation(
            owner, FPS, clip_ids, validation_by_id, run_mode=str(run_mode)
        )

        # Keep per-plan color metadata aligned by stable clip id.
        color_segments = [dict(x) for x in final_manifest.get("segments", [])]
        clip_by_id = {clip_ids[i]: clips[i] for i in range(len(clips))}
        color_changed = False
        for idx, desc in enumerate(color_segments):
            cfg = clip_by_id.get(str(desc.get("clip_id") or ""))
            if cfg is None:
                continue
            wanted = _normalize_color_adjustment(cfg.get("color_adjustment"))
            if desc.get("color_adjustment") != wanted:
                desc["color_adjustment"] = wanted
                color_segments[idx] = desc
                color_changed = True
        if color_changed:
            final_manifest = dict(final_manifest)
            final_manifest["segments"] = color_segments
            final_manifest["updated_at"] = time.time()
            _write_json_atomic(manifest_path, final_manifest)

        cached_ids = cached_fl2va_ids(final_manifest)
        validated_ids = {
            str(x.get("clip_id"))
            for x in final_manifest.get("segments", [])
            if bool(x.get("validated", False)) and str(x.get("clip_id") or "")
        }
        cached_count = len(cached_ids)
        validated_count = len(validated_ids)
        continuity_signatures = continuity_signatures_for_segments(
            data_path, final_manifest.get("segments", [])
        )
        normalized_json = _state_json(
            clips, active_prompt_pack_signature, "fl2va"
        )

        if resolution.get("mode") == "auto_from_ref" and frame_guides:
            resolution_text = (
                f"{resolved_width}x{resolved_height} from FL keyframe "
                f"@ {float(resolution['megapixels']):.2f}MP"
            )
        elif resolution.get("fallback"):
            resolution_text = f"{resolved_width}x{resolved_height} manual fallback (no FL keyframe)"
        else:
            resolution_text = f"{resolved_width}x{resolved_height} manual"
        if resolution.get("cache_reset") and previous_cache_resolution:
            resolution_text += (
                f" | resolution changed from "
                f"{int(previous_cache_resolution['width'])}x{int(previous_cache_resolution['height'])}: FL cache restarted"
            )

        status = (
            f"FL2VA {str(run_mode)} | {resolution_text} | "
            f"cached {cached_count}/{len(clips)} | validated {validated_count} | "
            + (
                "generated " + ",".join(str(i + 1) for i in generated)
                if generated else "disk only"
            )
        )
        cache_mb = _cache_size_mb(data_path, manifest_path)
        final_cache_resolution = _resolution_from_manifest(final_manifest)

        _send_extender_progress(owner, -1, len(clips), "idle", status)
        ui_state = {
            "generation_mode": "fl2va",
            "clips_json": normalized_json,
            "clip_count": len(clips),
            "cached_count": cached_count,
            "validated_count": validated_count,
            "cached_clip_ids": sorted(cached_ids),
            "validated_clip_ids": sorted(validated_ids),
            "continuity_signatures": continuity_signatures,
            "generated": [i + 1 for i in generated],
            "status": status,
            "resolved_width": resolved_width,
            "resolved_height": resolved_height,
            "resolution_mode": str(resolution.get("mode") or "manual"),
            "resolution_guide": "fl_keyframe" if frame_guides else "",
            "resolution_fallback": bool(resolution.get("fallback", False)),
            "megapixels": float(resolution.get("megapixels", megapixels)),
            "cache_width": int(final_cache_resolution["width"]) if final_cache_resolution else 0,
            "cache_height": int(final_cache_resolution["height"]) if final_cache_resolution else 0,
            "resolution_cache_reset": bool(resolution.get("cache_reset", False)),
            "reference_count": 0,
            "reference_video_count": 0,
            "reference_video_audio_count": 0,
            "reference_audio_count": 0,
            "prompt_pack_connected": external_prompt_pack is not None,
            "prompt_pack_imported": bool(prompt_pack_imported),
            "prompt_pack_count": int(len(external_prompt_pack.get("prompts") or [])) if external_prompt_pack is not None else 0,
            "prompt_pack_signature": str(active_prompt_pack_signature or ""),
            "per_clip_lora_count": int(sum(len(cfg.get("loras") or []) for cfg in clips)),
            "build": BUILD,
        }
        return {
            "ui": {"h3_extender_state": [ui_state]},
            "result": (
                previous_handle,
                int(len(clips)),
                int(validated_count),
                status,
                float(cache_mb),
                BUILD,
            ),
        }

    def extend(
        self,
        model,
        clip,
        vae,
        run_mode,
        width,
        height,
        ref_image_size,
        steps,
        sampler_name,
        scheduler,
        denoise,
        context_length,
        audio_context_length,
        clips_json,
        resolution_mode="auto_from_ref",
        megapixels=DEFAULT_MEGAPIXELS,
        refs_json=None,
        generation_mode="ref2va",
        prompt_pack=None,
        ref_pack=None,
        unique_id=None,
        **kwargs,
    ):
        generation_mode = _normalize_generation_mode(generation_mode)
        stored_prompt_pack_signature = _prompt_pack_signature_from_state(clips_json)
        clips = _parse_clips_json(clips_json, generation_mode)
        external_prompt_pack = _normalize_external_prompt_pack(prompt_pack)
        clips, active_prompt_pack_signature, prompt_pack_imported, _prompt_pack_count_changed = (
            _sync_clips_from_prompt_pack(
                clips,
                external_prompt_pack,
                stored_prompt_pack_signature,
            )
        )
        owner = str(unique_id if unique_id is not None else "h3_extender")
        if external_prompt_pack is None:
            active_prompt_pack_signature = ""

        if generation_mode == "fl2va":
            return self._extend_fl2va(
                owner=owner,
                clips=clips,
                active_prompt_pack_signature=active_prompt_pack_signature,
                prompt_pack_imported=prompt_pack_imported,
                external_prompt_pack=external_prompt_pack,
                model=model,
                fl2va_model=kwargs.get("fl2va_model"),
                clip=clip,
                vae=vae,
                run_mode=run_mode,
                width=width,
                height=height,
                steps=steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                resolution_mode=resolution_mode,
                megapixels=megapixels,
            )

        data_path, manifest_path, manifest = _manifest_for_extender(owner, FPS)

        # If cards were removed, trim the physical cache immediately.
        if len(manifest.get("segments", [])) > len(clips):
            manifest = _truncate_chain(
                data_path, manifest_path, manifest, len(clips)
            )

        segments = manifest.get("segments", [])

        # A TRUE toggle can only validate a clip that actually exists on disk.
        # This also protects old workflow JSON with stale downstream TRUE values.
        if len(segments) < len(clips):
            for i in range(len(segments), len(clips)):
                if clips[i]["validated"]:
                    for j in range(i, len(clips)):
                        clips[j]["validated"] = False
                    break

        refs = _parse_refs_json(refs_json)
        external_ref_pack = _normalize_external_ref_pack(ref_pack)
        refs, ref_pack_imported_slots = _sync_refs_from_ref_pack(refs, external_ref_pack)
        if ref_pack_imported_slots and external_ref_pack is not None:
            _send_extender_ref_pack_import(
                owner,
                _refs_json(refs),
                ref_pack_imported_slots,
                int(external_ref_pack.get("count", 0) or 0),
                external_ref_pack.get("source") or "External reference pack",
            )
        refs_signature = _refs_signature(refs)
        requested_resolution = _resolve_generation_resolution(
            resolution_mode,
            megapixels,
            width,
            height,
            refs,
        )

        cache_resolution = _resolution_from_manifest(manifest)
        cache_has_segments = bool(segments) and cache_resolution is not None

        # Resolution is a live generation setting again. Auto/MP or Manual
        # width/height may be changed at any time, exactly like the old external
        # Scale Image -> Get Image Size workflow. A latent chain cannot mix two
        # geometries, so when the requested size changes we restart only the
        # generated cache while preserving every card/prompt/seed/config.
        #
        # A .ext Load is handled in the frontend by putting the exact archived
        # cache width/height into Manual mode. Therefore the imported project
        # continues at its stored geometry until the user explicitly changes the
        # resolution controls afterwards.
        resolution = dict(requested_resolution)
        resolution["requested_width"] = int(requested_resolution["width"])
        resolution["requested_height"] = int(requested_resolution["height"])
        resolution["cache_reset"] = False

        resolved_width = int(resolution["width"])
        resolved_height = int(resolution["height"])

        requested_mismatch = bool(
            cache_has_segments
            and (
                int(cache_resolution["width"]) != resolved_width
                or int(cache_resolution["height"]) != resolved_height
            )
        )
        previous_cache_resolution = dict(cache_resolution) if requested_mismatch else None

        if requested_mismatch:
            # Resolution is the one unavoidable global invalidation: latent
            # geometry cannot be mixed inside one sequential disk chain.
            manifest = _truncate_chain(data_path, manifest_path, manifest, 0)
            segments = []
            cache_resolution = None
            cache_has_segments = False
            resolution["cache_reset"] = True
            for cfg in clips:
                cfg["validated"] = False
            try:
                preview_path = _decoded_preview_cache_path(data_path)
                if preview_path.exists():
                    preview_path.unlink()
            except Exception:
                pass

        if prompt_pack_imported and external_prompt_pack is not None:
            imported_json = _state_json(clips, active_prompt_pack_signature, "ref2va")
            _send_extender_prompt_pack_import(
                owner,
                imported_json,
                len(external_prompt_pack.get("prompts") or []),
                external_prompt_pack.get("source") or "External prompt pack",
            )

        # References are intentionally user-controlled. The Extender never
        # associates a Ref number with a clip number and never decides which clip
        # becomes obsolete after a reference edit. Keep this fingerprint as
        # informational project/cache metadata, never as an invalidation key.
        manifest = _load_manifest_from_paths(data_path, manifest_path) or manifest
        manifest = dict(manifest)
        manifest["extender_refs_signature"] = refs_signature
        manifest["extender_ref_ids"] = [
            ref.get("id") if isinstance(ref, dict) else None for ref in refs
        ]
        manifest["updated_at"] = time.time()
        _write_json_atomic(manifest_path, manifest)

        resolution_mismatch = False

        audio_vae = kwargs.get("audio_vae")
        legacy_ref_audio = kwargs.get("ref_audio")
        ref_audios = [kwargs.get(f"ref_audio_{index}") for index in range(1, MAX_STANDALONE_AUDIO_REFS + 1)]
        if all(audio is None for audio in ref_audios) and legacy_ref_audio is not None:
            ref_audios[0] = legacy_ref_audio
        ref_videos = [kwargs.get(f"ref_video_{index}") for index in range(1, MAX_VIDEO_REFS + 1)]
        ref_video_fps = [kwargs.get(f"ref_video_fps_{index}", float(FPS)) for index in range(1, MAX_VIDEO_REFS + 1)]
        ref_video_audios = [kwargs.get(f"ref_video_audio_{index}") for index in range(1, MAX_VIDEO_REFS + 1)]
        active_ref_video_count = sum(video is not None for video in ref_videos)
        active_ref_video_audio_count = sum(audio is not None for audio in ref_video_audios)
        active_ref_audio_count = sum(audio is not None for audio in ref_audios)
        # Long standalone audio (>15s) uses independent logical timelines.
        # Precompute the complete plan so cached/validated cards still advance
        # only the Audio slot(s) they actually use. Example: Clip 1 <Audio 1>,
        # Clip 2 <Audio 2> starts both sources at 0; a later <Audio 1> resumes
        # after the duration previously consumed from Audio 1.
        standalone_audio_clip_plan = _build_standalone_audio_clip_plan(clips, ref_audios)
        standalone_audio_cache = {}

        ref_items = None
        ref_blocks = None
        active_picture_slots = None
        active_video_slots = None
        prepared_ref_frame_count = None
        # Keep image VAE blocks once (duration-independent) and at most two
        # duration-specific video/audio latent block sets. Returning to a recent
        # duration therefore avoids expensive VAE re-encoding without retaining
        # the much larger Qwen RGB presentation frames for every duration.
        prepared_image_blocks = None
        prepared_video_blocks_by_frame_count = {}

        disk_join = MiniMaxH3MotionContextDiskJoin()
        motion = MiniMaxH3MotionContextRAM()

        previous_handle = None
        previous_proxy = None
        generated = []
        statuses = []

        # Walk the card list in order. Cached TRUE clips are metadata-only;
        # active clips sample and are written immediately to disk.
        for i, cfg in enumerate(clips):
            # Refresh manifest state every iteration because Disk Join can
            # truncate or append the physical chain.
            current_manifest = _load_manifest_from_paths(data_path, manifest_path)
            existing_count = len(current_manifest.get("segments", [])) if current_manifest else 0
            existing = i < existing_count

            if cfg["validated"] and existing:
                result = disk_join.join(
                    samples=None,
                    trim_frames=None,
                    validated=True,
                    run_mode=str(run_mode),
                    fps=float(FPS),
                    previous_cache=previous_handle,
                    unique_id=f"extender_{owner}",
                )
                previous_handle = result[0]
                previous_proxy = result[1]
                statuses.append(result[4])
                continue

            # Any active clip is unvalidated. Make sure everything after it is
            # false in the serialized state as well.
            _send_extender_progress(
                owner,
                i,
                len(clips),
                "preparing",
                f"Preparing clip {i + 1}/{len(clips)}",
            )
            cfg["validated"] = False
            for j in range(i + 1, len(clips)):
                clips[j]["validated"] = False

            frame_count = _duration_to_frames(cfg["duration"])

            # Standalone Audio selection is deterministic per clip. One connected
            # ref remains the global/timeline ref regardless of prompt tags. With
            # several refs, <Audio N> selects explicitly; without a usable tag,
            # the first connected logical ref is used instead of stacking them.
            selected_ref_audios, selected_audio_slots, selected_audio_offsets = standalone_audio_clip_plan[i]
            selected_ref_audio_count = len(selected_audio_slots)
            clip_mixed_ref_count = (
                _reference_count(refs)
                + active_ref_video_count
                + selected_ref_audio_count
            )
            if clip_mixed_ref_count > MAX_MIXED_REF_ITEMS:
                raise ValueError(
                    f"MiniMax H3 Extender: H3 Ref2VA supports at most {MAX_MIXED_REF_ITEMS} mixed reference items for this clip; "
                    f"got {clip_mixed_ref_count}."
                )

            # Reference-video conditioning is cropped/aligned against the target
            # clip duration. Reuse the prepared payload while duration is the
            # same, but rebuild it if a later card uses a different frame count.
            needs_ref_prepare = (
                ref_items is None
                or ref_blocks is None
                or active_picture_slots is None
                or active_video_slots is None
                or (active_ref_video_count and prepared_ref_frame_count != frame_count)
            )
            if needs_ref_prepare:
                cached_video_blocks = prepared_video_blocks_by_frame_count.get(int(frame_count))
                ref_items, ref_blocks, active_picture_slots, active_video_slots = _prepare_shared_refs(
                    vae,
                    audio_vae,
                    resolved_width,
                    resolved_height,
                    str(ref_image_size),
                    refs,
                    ref_videos=ref_videos,
                    ref_video_fps=ref_video_fps,
                    ref_video_audios=ref_video_audios,
                    standalone_audio_count=0,
                    frame_count=frame_count,
                    cached_image_blocks=prepared_image_blocks,
                    cached_video_blocks=cached_video_blocks,
                )
                image_block_count = len(active_picture_slots or [])
                if prepared_image_blocks is None:
                    prepared_image_blocks = list((ref_blocks or [])[:image_block_count])
                if active_ref_video_count:
                    # Refresh insertion order on a cache hit: this is a real
                    # two-entry LRU, not just a FIFO. The common 5s/10s/5s
                    # pattern therefore keeps both useful duration blocks.
                    duration_key = int(frame_count)
                    prepared_video_blocks_by_frame_count.pop(duration_key, None)
                    prepared_video_blocks_by_frame_count[duration_key] = list((ref_blocks or [])[image_block_count:])
                    while len(prepared_video_blocks_by_frame_count) > 2:
                        oldest_key = next(iter(prepared_video_blocks_by_frame_count))
                        prepared_video_blocks_by_frame_count.pop(oldest_key, None)
                prepared_ref_frame_count = frame_count

            clip_ref_items = list(ref_items or [])
            clip_ref_blocks = list(ref_blocks or [])
            # Paired video soundtracks are already packed before standalone Audio
            # refs and therefore consume the first native <Audio N> ordinals.
            audio_native_offset = sum(
                1
                for item in (ref_items or [])
                if isinstance(item, dict) and item.get("type") == "audio"
            )
            if selected_ref_audio_count:
                if (_reference_count(refs) + active_ref_video_count) < 1:
                    raise ValueError(
                        "MiniMax H3 Extender: standalone reference audio requires at least one image or video reference."
                    )
                audio_items, audio_blocks = _prepare_standalone_audio_refs(
                    audio_vae,
                    selected_ref_audios,
                    selected_audio_offsets,
                    frame_count / float(FPS),
                    cache=standalone_audio_cache,
                )
                clip_ref_items.extend(audio_items)
                clip_ref_blocks.extend(audio_blocks)

            clip_model, clip_text_encoder = _apply_per_clip_loras(
                self, model, clip, cfg.get("loras"), i
            )

            positive, latent = _make_ref2va_conditioning(
                clip_text_encoder,
                vae,
                cfg["prompt"],
                resolved_width,
                resolved_height,
                frame_count,
                clip_ref_items,
                clip_ref_blocks,
                active_picture_slots,
                active_video_slots,
                active_audio_slots=selected_audio_slots,
                audio_native_offset=audio_native_offset,
            )

            trim_frames = None
            if i > 0:
                if previous_proxy is None:
                    raise RuntimeError(
                        "MiniMax H3 Extender: previous cached latent is unavailable."
                    )
                positive, trim_frames, _, _, _ = motion.apply(
                    positive,
                    latent,
                    previous_proxy,
                    str(context_length),
                    int(audio_context_length),
                )

            _send_extender_progress(
                owner,
                i,
                len(clips),
                "sampling",
                f"Rendering clip {i + 1}/{len(clips)}",
            )

            sampled = _sample_h3(
                clip_model,
                positive,
                latent,
                cfg["seed"],
                str(sampler_name),
                str(scheduler),
                int(steps),
                float(denoise),
            )

            result = disk_join.join(
                samples=sampled,
                trim_frames=trim_frames,
                validated=False,
                run_mode=str(run_mode),
                fps=float(FPS),
                previous_cache=previous_handle,
                unique_id=f"extender_{owner}",
            )
            previous_handle = result[0]
            previous_proxy = result[1]
            statuses.append(result[4])
            generated.append(i)

            _send_extender_progress(
                owner,
                i,
                len(clips),
                "complete",
                f"Clip {i + 1}/{len(clips)} complete",
            )

            # Drop full sampled/conditioning and card-local patched MODEL/CLIP
            # references before the next clip. The incoming base model remains
            # untouched, so a LoRA selected on one card cannot leak to another.
            del sampled, positive, latent, clip_model, clip_text_encoder

            if str(run_mode) == "clip_by_clip":
                break

        # A clip_by_clip run may stop before walking all cards; the last handle
        # is still exactly the active cached prefix expected by Final Decode.
        if previous_handle is None:
            # This can only happen if the workflow contains no valid cards,
            # which _parse_clips_json prevents. Keep a defensive error anyway.
            raise RuntimeError("MiniMax H3 Extender: sequence produced no cache handle.")

        final_manifest = _load_manifest_from_paths(data_path, manifest_path)
        # Color grading is montage metadata only. Keep it attached to each cached
        # decoded segment without invalidating latents or validation state.
        if final_manifest is not None:
            color_segments = [dict(x) for x in final_manifest.get("segments", [])]
            color_changed = False
            for color_i, desc in enumerate(color_segments):
                if color_i >= len(clips):
                    break
                wanted = _normalize_color_adjustment(clips[color_i].get("color_adjustment"))
                if desc.get("color_adjustment") != wanted:
                    desc["color_adjustment"] = wanted
                    color_segments[color_i] = desc
                    color_changed = True
            if color_changed:
                final_manifest = dict(final_manifest)
                final_manifest["segments"] = color_segments
                final_manifest["updated_at"] = time.time()
                _write_json_atomic(manifest_path, final_manifest)

        final_cache_resolution = _resolution_from_manifest(final_manifest)
        cached_count = len(final_manifest.get("segments", []))
        validated_count = 0
        for desc in final_manifest.get("segments", []):
            if bool(desc.get("validated", False)):
                validated_count += 1
            else:
                break

        normalized_json = _state_json(clips, active_prompt_pack_signature, generation_mode)
        if resolution.get("mode") == "auto_from_ref" and resolution.get("guide_ref") is not None:
            resolution_text = (
                f"{resolved_width}x{resolved_height} from ref_{int(resolution['guide_ref'])} "
                f"@ {float(resolution['megapixels']):.2f}MP"
            )
        elif resolution.get("fallback"):
            resolution_text = f"{resolved_width}x{resolved_height} manual fallback (no image ref)"
        else:
            resolution_text = f"{resolved_width}x{resolved_height} manual"

        if resolution.get("cache_reset") and previous_cache_resolution:
            resolution_text += (
                f" | resolution changed from "
                f"{int(previous_cache_resolution['width'])}x{int(previous_cache_resolution['height'])}: cache restarted"
            )
        prompt_pack_text = ""
        if external_prompt_pack is not None:
            prompt_pack_text = (
                f" | prompt pack {len(external_prompt_pack.get('prompts') or [])}"
                + (" imported" if prompt_pack_imported else " linked")
            )
        ref_pack_text = ""
        if external_ref_pack is not None:
            connected_ref_count = int(external_ref_pack.get("count", 0) or 0)
            if ref_pack_imported_slots:
                imported_text = ",".join(str(i) for i in ref_pack_imported_slots)
                ref_pack_text = f" | ref pack {connected_ref_count} linked, imported Ref {imported_text}"
            else:
                ref_pack_text = f" | ref pack {connected_ref_count} linked"
        status = (
            f"{str(run_mode)} | {resolution_text} | refs {_reference_count(refs)} | video refs {active_ref_video_count}"
            f" | video audios {active_ref_video_audio_count} | audio refs {active_ref_audio_count} | cached {cached_count}/{len(clips)} | "
            f"validated {validated_count}{prompt_pack_text}{ref_pack_text} | "
            + (
                "generated " + ",".join(str(i + 1) for i in generated)
                if generated
                else "disk only"
            )
        )
        cache_mb = _cache_size_mb(data_path, manifest_path)

        _send_extender_progress(
            owner,
            -1,
            len(clips),
            "idle",
            status,
        )

        ui_state = {
            "generation_mode": "ref2va",
            "clips_json": normalized_json,
            "clip_count": len(clips),
            "cached_count": cached_count,
            "validated_count": validated_count,
            "generated": [i + 1 for i in generated],
            "status": status,
            "resolved_width": resolved_width,
            "resolved_height": resolved_height,
            "resolution_mode": str(resolution.get("mode") or "manual"),
            "resolution_guide": (
                f"ref_{int(resolution['guide_ref'])}"
                if resolution.get("guide_ref") is not None
                else ""
            ),
            "resolution_guide_width": int(resolution.get("guide_src_width", 0) or 0),
            "resolution_guide_height": int(resolution.get("guide_src_height", 0) or 0),
            "resolution_fallback": bool(resolution.get("fallback", False)),
            "megapixels": float(resolution.get("megapixels", megapixels)),
            "cache_width": int(final_cache_resolution["width"]) if final_cache_resolution else 0,
            "cache_height": int(final_cache_resolution["height"]) if final_cache_resolution else 0,
            "resolution_mismatch": False,
            "resolution_cache_locked": False,
            "resolution_cache_reset": bool(resolution.get("cache_reset", False)),
            "reference_cache_reset": False,
            "reference_count": int(_reference_count(refs)),
            "reference_video_count": int(active_ref_video_count),
            "reference_video_audio_count": int(active_ref_video_audio_count),
            "reference_audio_count": int(active_ref_audio_count),
            "refs_json": _refs_json(refs),
            "requested_width": int(resolution.get("requested_width", resolved_width)),
            "requested_height": int(resolution.get("requested_height", resolved_height)),
            "prompt_pack_connected": external_prompt_pack is not None,
            "prompt_pack_imported": bool(prompt_pack_imported),
            "prompt_pack_count": int(len(external_prompt_pack.get("prompts") or [])) if external_prompt_pack is not None else 0,
            "prompt_pack_signature": str(active_prompt_pack_signature or ""),
            "ref_pack_connected": external_ref_pack is not None,
            "ref_pack_count": int(external_ref_pack.get("count", 0) or 0) if external_ref_pack is not None else 0,
            "ref_pack_imported_slots": [int(i) for i in ref_pack_imported_slots],
            "per_clip_lora_count": int(sum(len(cfg.get("loras") or []) for cfg in clips)),
            "build": BUILD,
        }

        return {
            "ui": {"h3_extender_state": [ui_state]},
            "result": (
                previous_handle,
                int(len(clips)),
                int(validated_count),
                status,
                float(cache_mb),
                BUILD,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Extender": MiniMaxH3Extender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Extender": "MiniMax H3 Extender",
}


if getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/h3_extender/loras")
    async def h3_extender_loras(request):
        """Return the current ComfyUI LoRA filename list for card dropdowns."""
        try:
            names = sorted(
                {str(name) for name in folder_paths.get_filename_list("loras") if str(name).strip()},
                key=lambda value: value.lower(),
            )
            return web.json_response({"ok": True, "loras": names})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc), "loras": []}, status=500)

    @PromptServer.instance.routes.post("/h3_extender/ref/upload")
    async def h3_extender_ref_upload(request):
        """Upload one image reference and store the actual pixels internally."""
        temp_path = _project_temp_root() / f"ref_upload_{uuid.uuid4().hex}.bin"
        original_name = "reference.png"
        got_file = False
        size = 0
        try:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name != "ref_file":
                    continue
                original_name = str(part.filename or "reference.png")
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk(size=PROJECT_COPY_CHUNK)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_REF_UPLOAD_BYTES:
                            raise ValueError(
                                f"MiniMax H3 Extender: reference upload exceeds {MAX_REF_UPLOAD_BYTES // (1024 * 1024)} MB."
                            )
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                got_file = True
                break

            if not got_file or not temp_path.exists() or temp_path.stat().st_size <= 0:
                return web.json_response(
                    {"ok": False, "error": "No reference image was uploaded."}, status=400
                )
            try:
                ref = await asyncio.to_thread(
                    _store_uploaded_reference,
                    temp_path,
                    original_name,
                )
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=400)
            return web.json_response({"ok": True, "ref": ref})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @PromptServer.instance.routes.post("/h3_extender/ref/edit")
    async def h3_extender_ref_edit(request):
        """Apply simple photographic adjustments to an internal reference."""
        try:
            body = await request.json()
            source_id = str(body.get("source_id") or body.get("ref_id") or "").lower().strip()
            if not _ref_id_is_safe(source_id):
                return web.json_response(
                    {"ok": False, "error": "Invalid source reference id."}, status=400
                )

            ref = await asyncio.to_thread(
                _edit_internal_reference,
                source_id,
                str(body.get("original_name") or "reference.png"),
                body.get("brightness", 100),
                body.get("contrast", 100),
                body.get("saturation", 100),
                body.get("external_signature", ""),
            )
            return web.json_response({"ok": True, "ref": ref})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/h3_extender/ref/image")
    async def h3_extender_ref_image(request):
        """Serve an internally managed reference thumbnail/full preview."""
        ref_id = str(request.query.get("id", "")).lower().strip()
        if not _ref_id_is_safe(ref_id):
            return web.Response(status=400, text="Invalid reference id.")
        path = _ref_path(ref_id)
        if not path.exists():
            return web.Response(status=404, text="Reference image not found.")
        return web.FileResponse(
            path,
            headers={
                "Content-Type": "image/png",
                "Cache-Control": "public, max-age=31536000, immutable",
            },
        )

    @PromptServer.instance.routes.get("/h3_extender/fl2va/continuity_meta")
    async def h3_extender_fl2va_continuity_meta(request):
        """Return the tiny continuity signature without opening/hash-reading the PNG."""
        owner_id = str(request.query.get("owner_id") or "").strip()
        clip_id = str(request.query.get("clip_id") or "").strip()
        if not owner_id or not clip_id:
            return web.json_response({"found": False, "reason": "missing_id"}, status=400)
        meta = fl2va_continuity_meta(owner_id, clip_id)
        if not meta:
            return web.json_response({"found": False})
        return web.json_response({
            "found": True,
            "signature": str(meta.get("signature") or ""),
            "frame_index": int(meta.get("frame_index", -1)),
            "frame_count": int(meta.get("frame_count", 0)),
        })

    @PromptServer.instance.routes.get("/h3_extender/fl2va/last_frame")
    async def h3_extender_fl2va_last_frame(request):
        """Serve an already decoded FL2VA continuity frame with signature caching."""
        owner_id = str(request.query.get("owner_id") or "").strip()
        clip_id = str(request.query.get("clip_id") or "").strip()
        if not owner_id or not clip_id:
            return web.Response(status=400, text="Missing owner_id or clip_id.")
        path = fl2va_last_frame_path(owner_id, clip_id)
        meta = fl2va_continuity_meta(owner_id, clip_id)
        if not path.exists() or path.stat().st_size <= 0 or not meta:
            return web.Response(
                status=404, text="FL2VA continuity frame not decoded yet.",
                headers={"Cache-Control": "no-store, max-age=0"},
            )
        signature = str(meta.get("signature") or "")
        requested_version = str(request.query.get("v") or "")
        immutable = bool(signature and requested_version == signature)
        headers = {
            "Content-Type": "image/png",
            "Cache-Control": (
                "public, max-age=31536000, immutable"
                if immutable
                else "public, max-age=0, must-revalidate"
            ),
        }
        if signature:
            headers["ETag"] = f'"{signature}"'
        return web.FileResponse(path, headers=headers)

    @PromptServer.instance.routes.post("/h3_extender/project/prepare_save")
    async def h3_extender_project_prepare_save(request):
        """Build a portable .ext archive without buffering the cache in RAM."""
        try:
            body = await request.json()
            owner_id = str(body.get("owner_id", "")).strip()
            if not owner_id:
                return web.json_response(
                    {"ok": False, "error": "Missing Extender node id."}, status=400
                )
            project_payload = body.get("project", {})
            if not isinstance(project_payload, dict):
                return web.json_response(
                    {"ok": False, "error": "Invalid project metadata."}, status=400
                )

            _cleanup_project_downloads()
            filename = _project_filename(body.get("project_name", "MiniMax_H3_Project"))
            token = uuid.uuid4().hex
            temp_path = _project_temp_root() / f"download_{token}.ext"

            try:
                archive_meta = await asyncio.to_thread(
                    _build_project_archive,
                    owner_id,
                    filename,
                    project_payload,
                    temp_path,
                )
            except Exception as exc:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return web.json_response(
                    {"ok": False, "error": str(exc)}, status=500
                )

            _PROJECT_DOWNLOADS[token] = {
                "path": str(temp_path),
                "filename": filename,
                "created_at": time.time(),
            }
            return web.json_response({
                "ok": True,
                "token": token,
                "filename": filename,
                "size_bytes": int(temp_path.stat().st_size),
                "cache": archive_meta.get("cache", {}),
                "references": archive_meta.get("references", {}),
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/h3_extender/project/download")
    async def h3_extender_project_download(request):
        """Stream a prepared .ext directly to the browser, then remove the temp file."""
        _cleanup_project_downloads()
        token = str(request.query.get("token", "")).strip()
        info = _PROJECT_DOWNLOADS.pop(token, None)
        if not info:
            return web.Response(status=404, text="Project download expired or was not found.")

        path = Path(info["path"])
        if not path.exists():
            return web.Response(status=404, text="Project file no longer exists.")

        filename = _project_filename(info.get("filename", "MiniMax_H3_Project.ext"))
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(path.stat().st_size),
                "Cache-Control": "no-store",
            },
        )
        try:
            await response.prepare(request)
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(PROJECT_COPY_CHUNK)
                    if not chunk:
                        break
                    await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    @PromptServer.instance.routes.post("/h3_extender/project/load")
    async def h3_extender_project_load(request):
        """Import a .ext into the cache owned by the Extender node making the request."""
        upload_path = _project_temp_root() / f"upload_{uuid.uuid4().hex}.ext"
        owner_id = ""
        original_name = ""
        got_file = False
        try:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "owner_id":
                    owner_id = (await part.text()).strip()
                elif part.name == "project_file":
                    original_name = str(part.filename or "project.ext")
                    with open(upload_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(size=PROJECT_COPY_CHUNK)
                            if not chunk:
                                break
                            f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                    got_file = True

            if not owner_id:
                return web.json_response(
                    {"ok": False, "error": "Missing Extender node id."}, status=400
                )
            if not got_file or not upload_path.exists():
                return web.json_response(
                    {"ok": False, "error": "No .ext project file was uploaded."}, status=400
                )

            try:
                imported = await asyncio.to_thread(
                    _import_project_archive,
                    owner_id,
                    upload_path,
                )
            except zipfile.BadZipFile:
                return web.json_response(
                    {"ok": False, "error": "The selected .ext file is not a valid project archive."},
                    status=400,
                )
            except Exception as exc:
                return web.json_response(
                    {"ok": False, "error": str(exc)}, status=400
                )

            imported["ok"] = True
            imported["source_filename"] = original_name
            return web.json_response(imported)
        finally:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass
