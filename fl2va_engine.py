"""MiniMax H3 FL2VA branch for the Extender.

This module deliberately keeps FL2VA conditioning/cache behaviour separate from
Ref2VA + Motion Context.  FL2VA cards are independent plans: each card can have
its own optional first/last frame and cached latent, and replacing/inserting one
card does not invalidate the other cached cards.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from PIL import Image

FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32

# FL2VA latents are intentionally append-only during ordinary random-access
# editing. Compact only when stale data is substantial, and force a clean cache
# at Save Project time. This preserves fast rerenders without allowing a long
# editing session to grow the physical cache forever.
_COMPACT_LATENT_DEAD_BYTES = 256 * 1024 * 1024
_COMPACT_LATENT_RATIO_MIN_DEAD = 64 * 1024 * 1024
_COMPACT_AUDIO_DEAD_BYTES = 128 * 1024 * 1024
_COMPACT_AUDIO_RATIO_MIN_DEAD = 32 * 1024 * 1024
_COMPACT_RATIO = 1.5
_COMPACT_COPY_CHUNK = 16 * 1024 * 1024


def normalize_mode(value) -> str:
    value = str(value or "ref2va").strip().lower()
    return "fl2va" if value == "fl2va" else "ref2va"


def cache_owner_id(owner_id) -> str:
    """Use a separate physical cache so switching modes never destroys Ref2VA."""
    from .motion_context_disk import _safe_name
    return f"extender_{_safe_name(owner_id)}_fl2va"


def _plan_video_cache_dir(data_path: Path) -> Path:
    path = Path(data_path).with_suffix(".fl2va.video")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plan_video_cache_path(data_path: Path, clip_id: str) -> Path:
    from .motion_context_disk import _safe_name
    return _plan_video_cache_dir(data_path) / f"{_safe_name(clip_id)}.mp4"


def _plan_last_frame_cache_path(data_path: Path, clip_id: str) -> Path:
    from .motion_context_disk import _safe_name
    return _plan_video_cache_dir(data_path) / f"{_safe_name(clip_id)}.continuity.png"


def _plan_continuity_meta_path(data_path: Path, clip_id: str) -> Path:
    from .motion_context_disk import _safe_name
    return _plan_video_cache_dir(data_path) / f"{_safe_name(clip_id)}.continuity.json"


def _plan_prev_video_cache_path(data_path: Path, clip_id: str) -> Path:
    from .motion_context_disk import _safe_name
    return _plan_video_cache_dir(data_path) / f"{_safe_name(clip_id)}.prev.mp4"


def fl2va_last_frame_path(owner_id, clip_id) -> Path:
    """Return the derived lossless last-frame cache path for one FL2VA plan."""
    from .motion_context_disk import _chain_paths
    data_path, _ = _chain_paths(cache_owner_id(owner_id))
    return _plan_last_frame_cache_path(data_path, str(clip_id))


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _select_best_plan_end_frame(decoded, window: int = 6):
    """Choose the cleanest handoff image from the final six decoded frames.

    FL2VA Previous is a shot-to-shot handoff: the selected image becomes both
    the visible end of the previous plan and the First conditioning image of
    the next plan.  We therefore only inspect the final few frames and prefer
    a sharp, locally stable frame, with a smaller bias toward the latest frame.
    If an earlier candidate wins, Final Decode trims the previous plan exactly
    to that frame, so there is no temporal jump backward at the cut.
    """
    frame = decoded
    # Only the final handful of frames can ever win. Slice *before* any CPU /
    # float32 conversion so a 15 s plan does not sanitize/copy hundreds of full
    # RGB frames just to score the last six. This is especially noticeable when
    # enabling Previous: the continuity PNG becomes available much sooner and
    # the thumbnail can appear without the old full-clip CPU pass.
    if torch.is_tensor(frame):
        if frame.ndim != 4:
            raise ValueError(f"FL2VA continuity-frame selection: invalid decoded shape {tuple(frame.shape)}.")
        count = int(frame.shape[0])
        if count <= 0:
            raise ValueError("FL2VA continuity-frame selection: decoded sequence is empty.")
        candidate_start = max(0, count - max(1, int(window)))
        seq = frame[candidate_start:count].detach().to(device="cpu", dtype=torch.float32)
        seq = torch.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        candidates = np.ascontiguousarray(seq[..., :3].numpy())
    else:
        source = np.asarray(frame)
        if source.ndim != 4:
            raise ValueError(f"FL2VA continuity-frame selection: invalid decoded shape {tuple(source.shape)}.")
        count = int(source.shape[0])
        if count <= 0:
            raise ValueError("FL2VA continuity-frame selection: decoded sequence is empty.")
        candidate_start = max(0, count - max(1, int(window)))
        candidates = np.asarray(source[candidate_start:count], dtype=np.float32)
        if float(np.max(candidates)) > 1.0 or float(np.min(candidates)) < 0.0:
            candidates = np.clip(candidates / 255.0, 0.0, 1.0)
        else:
            candidates = np.clip(candidates, 0.0, 1.0)
        candidates = np.ascontiguousarray(candidates[..., :3])

    if count <= 1:
        return candidates[-1], count - 1
    if int(candidates.shape[0]) == 1:
        return candidates[0], candidate_start

    gray = candidates[..., 0] * 0.299 + candidates[..., 1] * 0.587 + candidates[..., 2] * 0.114
    sharpness = []
    stability = []
    lateness = []
    n = int(candidates.shape[0])
    for local_idx in range(n):
        g = gray[local_idx]
        if g.shape[0] >= 3 and g.shape[1] >= 3:
            lap = (
                -4.0 * g[1:-1, 1:-1]
                + g[:-2, 1:-1] + g[2:, 1:-1]
                + g[1:-1, :-2] + g[1:-1, 2:]
            )
            sharpness.append(float(np.var(lap, dtype=np.float64)))
        else:
            sharpness.append(0.0)

        diffs = []
        if local_idx > 0:
            diffs.append(float(np.mean(np.abs(g - gray[local_idx - 1]), dtype=np.float64)))
        if local_idx + 1 < n:
            diffs.append(float(np.mean(np.abs(g - gray[local_idx + 1]), dtype=np.float64)))
        stability.append(sum(diffs) / max(1, len(diffs)))
        lateness.append(float(local_idx) / max(1.0, float(n - 1)))

    sharpness = np.asarray(sharpness, dtype=np.float64)
    stability = np.asarray(stability, dtype=np.float64)
    lateness = np.asarray(lateness, dtype=np.float64)

    def _norm(values, invert=False):
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi - lo <= 1e-12:
            out = np.ones_like(values, dtype=np.float64)
        else:
            out = (values - lo) / (hi - lo)
        return 1.0 - out if invert else out

    sharp_n = _norm(sharpness)
    stable_n = _norm(stability, invert=True)
    late_n = _norm(lateness)

    # Quality first. Recency is deliberately secondary because Previous is a
    # hard shot change; a clean frame is more valuable than forcing frame -1.
    score = 0.55 * sharp_n + 0.30 * stable_n + 0.15 * late_n
    best_local = int(np.argmax(score))
    best_index = candidate_start + best_local
    return candidates[best_local], best_index


def _normalize_plan_continuity_meta(data):
    if not isinstance(data, dict):
        return None
    if int(data.get("version", 0) or 0) != 2 or str(data.get("algorithm") or "") != "best_of_last_6":
        return None
    try:
        data = dict(data)
        data["frame_index"] = int(data.get("frame_index"))
        data["frame_count"] = int(data.get("frame_count"))
    except Exception:
        return None
    return data


@lru_cache(maxsize=512)
def _read_continuity_meta_cached(path_text: str, mtime_ns: int, size: int):
    # mtime/size are part of the cache key; replacing the tiny JSON sidecar
    # automatically creates a new entry without rereading unchanged metadata.
    try:
        raw = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except Exception:
        return None
    return _normalize_plan_continuity_meta(raw)


def _load_plan_continuity_meta(data_path: Path, clip_id: str):
    path = _plan_continuity_meta_path(data_path, clip_id)
    try:
        st = path.stat()
    except OSError:
        return None
    if int(st.st_size) <= 0:
        return None
    data = _read_continuity_meta_cached(str(path), int(st.st_mtime_ns), int(st.st_size))
    return dict(data) if isinstance(data, dict) else None


def _write_plan_continuity_meta(data_path: Path, clip_id: str, *, frame_index: int, frame_count: int, signature: str):
    path = _plan_continuity_meta_path(data_path, clip_id)
    payload = {
        "version": 2,
        "algorithm": "best_of_last_6",
        "frame_index": int(frame_index),
        "frame_count": int(frame_count),
        "signature": str(signature),
        "updated_at": time.time(),
    }
    temp = path.with_name(path.name + f".tmp_{os.urandom(4).hex()}")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, path)
        _read_continuity_meta_cached.cache_clear()
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
    return path


def _save_plan_last_frame_cache(data_path: Path, clip_id: str, decoded) -> tuple[Path, str]:
    """Persist the automatically selected FL2VA handoff image losslessly."""
    target = _plan_last_frame_cache_path(data_path, clip_id)
    frame, frame_index = _select_best_plan_end_frame(decoded, window=6)

    if torch.is_tensor(frame):
        if frame.ndim != 3:
            raise ValueError(f"FL2VA previous-frame cache: invalid selected shape {tuple(frame.shape)}.")
        frame = frame.detach().to(device="cpu", dtype=torch.float32).numpy()
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[-1] < 3:
        raise ValueError(f"FL2VA previous-frame cache: invalid selected shape {tuple(pixels.shape)}.")
    pixels = np.asarray(pixels[..., :3])
    if np.issubdtype(pixels.dtype, np.floating):
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
        if pixels.size and (float(np.max(pixels)) > 1.0 or float(np.min(pixels)) < 0.0):
            pixels = np.clip(pixels, 0.0, 255.0)
        else:
            pixels = np.clip(pixels, 0.0, 1.0) * 255.0
        pixels = np.rint(pixels).astype(np.uint8)
    elif pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    pixels = np.ascontiguousarray(pixels)

    temp = target.with_name(target.stem + f".tmp_{os.urandom(4).hex()}.png")
    try:
        Image.fromarray(pixels, mode="RGB").save(temp, format="PNG", optimize=False, compress_level=4)
        os.replace(temp, target)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

    # Keep the historical v14.9x signature semantics (SHA256 of the lossless PNG
    # bytes) so existing continuity sidecars/dependencies remain compatible. The
    # hash is computed only once at creation; all later reads trust the JSON
    # sidecar instead of re-hashing the PNG.
    signature = _hash_path(target)
    frame_count = int(decoded.shape[0]) if hasattr(decoded, "shape") else int(np.asarray(decoded).shape[0])
    _write_plan_continuity_meta(
        data_path, clip_id, frame_index=int(frame_index), frame_count=frame_count, signature=signature
    )
    # Any previously derived trimmed H.264 prefix belongs to the old selection.
    try:
        _plan_prev_video_cache_path(data_path, clip_id).unlink(missing_ok=True)
    except OSError:
        pass
    return target, signature


def _load_plan_last_frame_cache(data_path: Path, clip_id: str):
    path = _plan_last_frame_cache_path(data_path, clip_id)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    # Reject continuity images created by older selection algorithms. The JSON
    # sidecar is the source of truth for the dependency signature, so loading a
    # cached thumb no longer hashes the PNG again.
    meta = _load_plan_continuity_meta(data_path, clip_id)
    if meta is None:
        return None
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    signature = str(meta.get("signature") or "")
    if not signature:
        # Backward-compatible recovery for an unusual/incomplete old sidecar.
        signature = _hash_path(path)
    return tensor, signature



def continuity_signatures_for_segments(data_path: Path, segments) -> dict[str, str]:
    """Return cached handoff signatures with at most one JSON read per sidecar."""
    out = {}
    for desc in segments or []:
        clip_id = str(desc.get("clip_id") or "")
        if not clip_id:
            continue
        meta = _load_plan_continuity_meta(Path(data_path), clip_id)
        signature = str((meta or {}).get("signature") or "")
        if signature:
            out[clip_id] = signature
    return out


def fl2va_continuity_meta(owner_id, clip_id):
    from .motion_context_disk import _chain_paths
    data_path, _ = _chain_paths(cache_owner_id(owner_id))
    return _load_plan_continuity_meta(data_path, str(clip_id))


def fl2va_project_continuity_files(data_path: Path, segments):
    """Return portable lossless continuity sidecars for currently live plans."""
    items = []
    for desc in segments or []:
        clip_id = str(desc.get("clip_id") or "")
        if not clip_id:
            continue
        png = _plan_last_frame_cache_path(Path(data_path), clip_id)
        meta_path = _plan_continuity_meta_path(Path(data_path), clip_id)
        meta = _load_plan_continuity_meta(Path(data_path), clip_id)
        if not meta or not png.exists() or png.stat().st_size <= 0 or not meta_path.exists():
            continue
        items.append({
            "clip_id": clip_id,
            "signature": str(meta.get("signature") or ""),
            "png_path": png,
            "meta_path": meta_path,
        })
    return items


def install_fl2va_project_continuity(data_path: Path, clip_id: str, png_source: Path, meta_source: Path):
    """Install one trusted .ext continuity pair into the derived local cache."""
    data_path = Path(data_path)
    png_source = Path(png_source)
    meta_source = Path(meta_source)
    try:
        raw_meta = json.loads(meta_source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"FL2VA project continuity metadata is invalid for {clip_id}.") from exc
    if _normalize_plan_continuity_meta(raw_meta) is None:
        raise ValueError(f"FL2VA project continuity metadata is incompatible for {clip_id}.")
    with Image.open(png_source) as image:
        image.verify()
    target_png = _plan_last_frame_cache_path(data_path, str(clip_id))
    target_meta = _plan_continuity_meta_path(data_path, str(clip_id))
    target_png.parent.mkdir(parents=True, exist_ok=True)
    token = os.urandom(4).hex()
    tmp_png = target_png.with_name(target_png.name + f".tmp_{token}")
    tmp_meta = target_meta.with_name(target_meta.name + f".tmp_{token}")
    try:
        shutil.copy2(png_source, tmp_png)
        shutil.copy2(meta_source, tmp_meta)
        os.replace(tmp_png, target_png)
        os.replace(tmp_meta, target_meta)
        _read_continuity_meta_cached.cache_clear()
        try:
            _plan_prev_video_cache_path(data_path, str(clip_id)).unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        for path in (tmp_png, tmp_meta):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _copy_cache_spec(src, dst, spec):
    remaining = int(spec.get("nbytes", 0) or 0)
    if remaining <= 0:
        raise ValueError("FL2VA cache compaction: invalid zero-sized tensor spec.")
    src.seek(int(spec.get("offset", 0) or 0))
    new_spec = dict(spec)
    new_spec["offset"] = int(dst.tell())
    expected = remaining
    while remaining > 0:
        chunk = src.read(min(remaining, _COMPACT_COPY_CHUNK))
        if not chunk:
            raise IOError("FL2VA cache compaction: source cache is truncated.")
        dst.write(chunk)
        remaining -= len(chunk)
    new_spec["nbytes"] = expected
    return new_spec


def _should_compact(physical: int, live: int, *, force: bool, dead_limit: int, ratio_min_dead: int) -> bool:
    physical = max(0, int(physical))
    live = max(0, int(live))
    dead = max(0, physical - live)
    if dead <= 0:
        return False
    if force:
        return True
    return dead >= int(dead_limit) or (
        dead >= int(ratio_min_dead) and live > 0 and physical >= int(math.ceil(live * _COMPACT_RATIO))
    )


def compact_fl2va_cache(owner_id, *, force=False):
    """Compact stale FL2VA latent/PCM blobs while preserving logical plan ids.

    Normal editing remains append-only. Automatic compaction only runs after a
    meaningful amount of garbage accumulates; Save Project calls this with
    ``force=True`` so portable archives never carry dead rerender blobs.
    """
    from . import motion_context_disk as d

    data_path, manifest_path = d._chain_paths(cache_owner_id(owner_id))
    if not data_path.exists() or not manifest_path.exists():
        return None, {"compacted": False, "reclaimed_bytes": 0}
    manifest = d._load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        return None, {"compacted": False, "reclaimed_bytes": 0}
    segments = [dict(x) for x in manifest.get("segments", [])]

    latent_live = int(d._DATA_START) + sum(
        int(desc.get("video", {}).get("nbytes", 0) or 0)
        + int(desc.get("audio", {}).get("nbytes", 0) or 0)
        for desc in segments
    )
    latent_physical = int(data_path.stat().st_size)
    compact_latent = _should_compact(
        latent_physical, latent_live, force=bool(force),
        dead_limit=_COMPACT_LATENT_DEAD_BYTES,
        ratio_min_dead=_COMPACT_LATENT_RATIO_MIN_DEAD,
    )

    audio_path = d._decoded_audio_cache_path(data_path)
    audio_specs = []
    for desc in segments:
        meta = desc.get("decoded_audio")
        spec = meta.get("waveform") if isinstance(meta, dict) and meta.get("storage") == "audio_cache" else None
        if isinstance(spec, dict):
            audio_specs.append(spec)
    audio_live = int(d._AUDIO_CACHE_START) + sum(int(spec.get("nbytes", 0) or 0) for spec in audio_specs)
    audio_physical = int(audio_path.stat().st_size) if audio_path.exists() else 0
    compact_audio = bool(audio_specs) and audio_path.exists() and _should_compact(
        audio_physical, audio_live, force=bool(force),
        dead_limit=_COMPACT_AUDIO_DEAD_BYTES,
        ratio_min_dead=_COMPACT_AUDIO_RATIO_MIN_DEAD,
    )

    if not compact_latent and not compact_audio:
        return manifest, {"compacted": False, "reclaimed_bytes": 0}

    token = os.urandom(6).hex()
    data_tmp = data_path.with_name(data_path.name + f".compact_{token}.tmp")
    audio_tmp = audio_path.with_name(audio_path.name + f".compact_{token}.tmp")
    manifest_tmp = manifest_path.with_name(manifest_path.name + f".compact_{token}.tmp")
    updated_segments = [dict(x) for x in segments]

    try:
        if compact_latent:
            with open(data_path, "rb", buffering=0) as src, open(data_tmp, "wb", buffering=0) as dst:
                magic = src.read(int(d._DATA_START))
                if magic != d._DATA_MAGIC:
                    raise ValueError("FL2VA cache compaction: invalid latent cache magic.")
                dst.write(d._DATA_MAGIC)
                for i, desc in enumerate(updated_segments):
                    item = dict(desc)
                    item["video"] = _copy_cache_spec(src, dst, item["video"])
                    item["audio"] = _copy_cache_spec(src, dst, item["audio"])
                    item["segment_end"] = int(dst.tell())
                    updated_segments[i] = item
                dst.flush()
                os.fsync(dst.fileno())

        if compact_audio:
            with open(audio_path, "rb", buffering=0) as src, open(audio_tmp, "wb", buffering=0) as dst:
                magic = src.read(int(d._AUDIO_CACHE_START))
                if magic != d._AUDIO_CACHE_MAGIC:
                    raise ValueError("FL2VA cache compaction: invalid decoded-audio cache magic.")
                dst.write(d._AUDIO_CACHE_MAGIC)
                for i, desc in enumerate(updated_segments):
                    item = dict(desc)
                    meta = item.get("decoded_audio")
                    if isinstance(meta, dict) and meta.get("storage") == "audio_cache" and isinstance(meta.get("waveform"), dict):
                        meta = dict(meta)
                        meta["waveform"] = _copy_cache_spec(src, dst, meta["waveform"])
                        item["decoded_audio"] = meta
                        updated_segments[i] = item
                dst.flush()
                os.fsync(dst.fileno())

        updated = dict(manifest)
        updated["segments"] = updated_segments
        updated["final_frame_count"] = d._final_frame_count(updated_segments)
        updated["updated_at"] = time.time()
        updated["last_compacted_at"] = time.time()
        reclaimed = (latent_physical - latent_live if compact_latent else 0) + (
            audio_physical - audio_live if compact_audio else 0
        )
        updated["last_compaction_reclaimed_bytes"] = max(0, int(reclaimed))
        manifest_tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(manifest_tmp, "r+b") as f:
            os.fsync(f.fileno())

        replacements = []
        if compact_latent:
            replacements.append((data_path, data_tmp))
        if compact_audio:
            replacements.append((audio_path, audio_tmp))
        replacements.append((manifest_path, manifest_tmp))
        backups = []
        try:
            for target, _ in replacements:
                if target.exists():
                    backup = target.with_name(target.name + f".compact_backup_{token}")
                    os.replace(target, backup)
                    backups.append((target, backup))
            for target, temp in replacements:
                os.replace(temp, target)
        except Exception:
            for target, _ in replacements:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            for target, backup in reversed(backups):
                if backup.exists():
                    os.replace(backup, target)
            raise
        else:
            for _, backup in backups:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass

        return updated, {"compacted": True, "reclaimed_bytes": max(0, int(reclaimed))}
    finally:
        for path in (data_tmp, audio_tmp, manifest_tmp):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _cleanup_plan_video_cache(data_path: Path, wanted_clip_ids):
    root = Path(data_path).with_suffix(".fl2va.video")
    if not root.exists():
        return
    wanted = {str(x) for x in (wanted_clip_ids or [])}
    from .motion_context_disk import _safe_name
    wanted_names = set()
    for x in wanted:
        safe = _safe_name(x)
        wanted_names.add(f"{safe}.mp4")
        wanted_names.add(f"{safe}.continuity.png")
        wanted_names.add(f"{safe}.continuity.json")
        wanted_names.add(f"{safe}.prev.mp4")
    for pattern in ("*.mp4", "*.continuity.png", "*.continuity.json", "*.last.png"):
        for path in root.glob(pattern):
            if path.name not in wanted_names:
                try:
                    path.unlink()
                except OSError:
                    pass


def _invalidate_plan_video_cache(data_path: Path, clip_id: str):
    for path in (
        _plan_video_cache_path(data_path, clip_id),
        _plan_last_frame_cache_path(data_path, clip_id),
        _plan_continuity_meta_path(data_path, clip_id),
        _plan_prev_video_cache_path(data_path, clip_id),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _ensure_plan_video_cache(data_path, desc, vae, fps, ffmpeg, progress=None):
    """Return one neutral H.264 decoded-plan cache, decoding only if missing."""
    from . import motion_context_disk as d
    clip_id = str(desc.get("clip_id") or f"clip_{int(desc.get('index', 0)) + 1}")
    target = _plan_video_cache_path(data_path, clip_id)
    last_target = _plan_last_frame_cache_path(data_path, clip_id)
    video_ready = target.exists() and target.stat().st_size > 0
    continuity_ready = (
        last_target.exists()
        and last_target.stat().st_size > 0
        and _load_plan_continuity_meta(data_path, clip_id) is not None
    )
    if video_ready and continuity_ready:
        return target, False

    temp = target.with_name(target.stem + f".tmp_{os.urandom(4).hex()}.mp4")
    v = None
    decoded = None
    try:
        v = d._load_segment_video(data_path, desc)
        decoded = vae.decode(v)
        if progress is not None:
            progress.advance()
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        _save_plan_last_frame_cache(data_path, clip_id, decoded)
        wanted = int(desc.get("frames", 0))
        if int(decoded.shape[0]) != wanted:
            raise RuntimeError(
                f"FL2VA preview cache: clip {int(desc.get('index', 0)) + 1} returned "
                f"{decoded.shape[0]} frames, expected {wanted}."
            )
        if not video_ready:
            d._encode_corrected_segment_video_mp4(
                ffmpeg, decoded, fps, temp, f"fl2va_{clip_id}_{os.urandom(3).hex()}"
            )
            os.replace(temp, target)
        return target, not video_ready
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        if decoded is not None:
            del decoded
        if v is not None:
            del v


def _cache_decoded_plan_video(data_path, desc, decoded, fps, ffmpeg):
    """Populate the FL2VA per-plan H.264 cache from an already decoded tensor."""
    clip_id = str(desc.get("clip_id") or f"clip_{int(desc.get('index', 0)) + 1}")
    # The lossless selected end PNG is the continuity source for Previous.
    # Keep the first decode of the current latent stable for dependency signatures.
    last_target = _plan_last_frame_cache_path(data_path, clip_id)
    if (
        not last_target.exists()
        or last_target.stat().st_size <= 0
        or _load_plan_continuity_meta(data_path, clip_id) is None
    ):
        _save_plan_last_frame_cache(data_path, clip_id, decoded)
    target = _plan_video_cache_path(data_path, clip_id)
    if target.exists() and target.stat().st_size > 0:
        return target
    from . import motion_context_disk as d
    temp = target.with_name(target.stem + f".tmp_{os.urandom(4).hex()}.mp4")
    try:
        d._encode_corrected_segment_video_mp4(
            ffmpeg, decoded, fps, temp, f"fl2va_full_{clip_id}_{os.urandom(3).hex()}"
        )
        os.replace(temp, target)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _next_uses_previous(segments, index: int) -> bool:
    i = int(index)
    if i < 0 or i + 1 >= len(segments):
        return False
    current_id = str(segments[i].get("clip_id") or "")
    follower = segments[i + 1]
    return (
        str(follower.get("first_source") or "manual") == "previous_clip"
        and str(follower.get("previous_clip_id") or "") == current_id
    )


def _visible_frames_for_plan(data_path: Path, desc: dict, *, handoff_enabled: bool) -> int:
    full = int(desc.get("frames", 0))
    if not handoff_enabled or full <= 1:
        return full
    clip_id = str(desc.get("clip_id") or "")
    meta = _load_plan_continuity_meta(data_path, clip_id)
    if not meta:
        return full
    if int(meta.get("frame_count", full)) != full:
        return full
    idx = max(0, min(full - 1, int(meta.get("frame_index", full - 1))))
    return idx + 1


def _timeline_segments(data_path: Path, segments):
    """Return descriptor copies using the visible prefix selected by Prev links."""
    out = []
    for i, raw in enumerate(segments):
        desc = dict(raw)
        visible = _visible_frames_for_plan(
            data_path, desc, handoff_enabled=_next_uses_previous(segments, i)
        )
        desc["source_frames"] = int(desc.get("frames", 0))
        desc["frames"] = int(visible)
        desc["trim_frames"] = 0
        out.append(desc)
    return out


def _timeline_signature(data_path: Path, segments) -> str:
    payload = []
    for i, desc in enumerate(segments):
        clip_id = str(desc.get("clip_id") or "")
        handoff = _next_uses_previous(segments, i)
        visible = _visible_frames_for_plan(data_path, desc, handoff_enabled=handoff)
        meta = _load_plan_continuity_meta(data_path, clip_id) if handoff else None
        payload.append({
            "clip_id": clip_id,
            "frames": int(visible),
            "handoff": bool(handoff),
            "signature": str((meta or {}).get("signature") or ""),
        })
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_plan_timeline_video_cache(data_path, desc, full_path, ffmpeg, fps, *, handoff_enabled: bool):
    """Return full plan video or an exact H.264 prefix ending on the Prev frame."""
    full_frames = int(desc.get("frames", 0))
    visible_frames = _visible_frames_for_plan(data_path, desc, handoff_enabled=handoff_enabled)
    if visible_frames <= 0 or visible_frames >= full_frames:
        return Path(full_path)

    target = _plan_prev_video_cache_path(data_path, str(desc.get("clip_id") or ""))
    if target.exists() and target.stat().st_size > 0:
        return target

    temp = target.with_name(target.stem + f".tmp_{os.urandom(4).hex()}.mp4")
    log_path = target.with_name(target.stem + f".log_{os.urandom(3).hex()}.txt")
    from . import motion_context_disk as d
    ffmpeg = d._preferred_h264_ffmpeg(ffmpeg)
    cmd = [
        ffmpeg, "-y",
        "-i", str(full_path),
        "-map", "0:v:0",
        "-vf", f"trim=end_frame={int(visible_frames)},setpts=PTS-STARTPTS",
        "-an",
        *d._h264_encode_args(ffmpeg, 17, "ultrafast"),
        "-movflags", "+faststart",
        str(temp),
    ]
    try:
        with open(log_path, "wb") as log_f:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        if proc.returncode != 0:
            tail = ""
            try:
                tail = log_path.read_bytes()[-12000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"FL2VA Prev timeline trim failed with ffmpeg code {proc.returncode}.\n{tail}"
            )
        os.replace(temp, target)
        return target
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass


def _ensure_plan_audio_cache(data_path, manifest_path, manifest, audio_vae, fps, progress=None):
    """Append only missing decoded PCM entries; unchanged plans are never re-decoded."""
    from . import motion_context_disk as d
    segments = [dict(x) for x in manifest.get("segments", [])]
    audio_path = d._decoded_audio_cache_path(data_path)
    d._ensure_audio_cache_file(audio_path)
    changed = False
    with open(audio_path, "ab", buffering=0) as acf:
        for i, desc in enumerate(segments):
            cached = d._load_cached_decoded_audio(data_path, desc)
            if cached is not None:
                del cached
                continue
            audio = d._decode_single_audio(data_path, desc, audio_vae, fps)
            if progress is not None:
                progress.advance()
            desc["decoded_audio"] = d._decoded_audio_meta_from_waveform(acf, audio)
            segments[i] = desc
            changed = True
            del audio
        if changed:
            acf.flush()
            os.fsync(acf.fileno())

    if changed:
        manifest = dict(manifest)
        manifest["segments"] = segments
        manifest["updated_at"] = time.time()
        d._write_json_atomic(manifest_path, manifest)
    return manifest, segments



def resolve_fl2va_previous_frame(owner_id, fps, clip_ids, previous_clip_id, vae):
    """Return the cached FL2VA chaining frame of a previous plan.

    Final Decode normally creates the tiny lossless PNG for free while decoding.
    If it is absent (for example immediately after importing an .ext project, or
    during one Full Batch execution), decode only the required previous plan once
    and persist the best of its final six frames for subsequent runs.
    """
    from . import motion_context_disk as d

    data_path, manifest_path, manifest = sync_fl2va_manifest(owner_id, fps, clip_ids)
    previous_clip_id = str(previous_clip_id)
    desc = next(
        (dict(x) for x in manifest.get("segments", []) if str(x.get("clip_id") or "") == previous_clip_id),
        None,
    )
    if desc is None:
        raise ValueError(
            "MiniMax H3 Extender: FL2VA 'Previous clip final' requires the previous plan "
            "to be generated and cached first."
        )

    cached = _load_plan_last_frame_cache(data_path, previous_clip_id)
    if cached is not None:
        return cached

    v = None
    decoded = None
    try:
        v = d._load_segment_video(data_path, desc)
        decoded = vae.decode(v)
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        wanted = int(desc.get("frames", 0))
        if wanted > 0 and int(decoded.shape[0]) != wanted:
            raise RuntimeError(
                f"FL2VA previous-frame decode: plan {int(desc.get('index', 0)) + 1} returned "
                f"{decoded.shape[0]} frames, expected {wanted}."
            )
        path, signature = _save_plan_last_frame_cache(data_path, previous_clip_id, decoded)
        cached = _load_plan_last_frame_cache(data_path, previous_clip_id)
        if cached is None:
            raise RuntimeError("FL2VA continuity-frame cache was not created.")
        frame, cached_signature = cached
        return frame, cached_signature
    finally:
        if decoded is not None:
            del decoded
        if v is not None:
            del v


def drop_fl2va_cached_ids(owner_id, fps, clip_ids, drop_ids):
    """Drop stale logical plans without rewriting the append-only latent file."""
    from .motion_context_disk import _final_frame_count, _write_json_atomic

    drop = {str(x) for x in (drop_ids or []) if str(x)}
    data_path, manifest_path, manifest = sync_fl2va_manifest(owner_id, fps, clip_ids)
    if not drop:
        return data_path, manifest_path, manifest

    old_segments = [dict(x) for x in manifest.get("segments", [])]
    segments = [x for x in old_segments if str(x.get("clip_id") or "") not in drop]
    if len(segments) == len(old_segments):
        return data_path, manifest_path, manifest

    for clip_id in drop:
        _invalidate_plan_video_cache(data_path, clip_id)
    manifest = _invalidate_derived_preview(data_path, manifest)
    for idx, desc in enumerate(segments):
        desc["index"] = idx
        desc["trim_frames"] = 0
    manifest["segments"] = segments
    manifest["final_frame_count"] = _final_frame_count(segments)
    manifest["updated_at"] = time.time()
    _write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest

def _align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _empty_av_latent(width: int, height: int, frame_count: int):
    frame_count = _align_frame_count(frame_count)
    latent_t = _video_latent_t(frame_count)
    duration = frame_count / float(FPS)
    audio_t = round(duration * AUDIO_LATENT_FPS)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [1, 24, latent_t, int(height) // 16, int(width) // 16], device=device
    )
    audio = torch.zeros([1, 32, 2, int(audio_t)], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _resize(image, width: int, height: int, crop: str):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", str(crop)
    )
    return samples.movedim(1, -1)


def make_fl2va_conditioning(
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    frame_count: int,
    first_frame=None,
    last_frame=None,
    guide_frames=None,
    guide_frame=None,
    guide_frame_idx: int = 0,
):
    """Mirror native MiniMax H3 FL2VA + chained AddGuide image conditioning.

    First/Last keep MiniMaxH3ImageToVideo semantics. ``guide_frames`` is an
    ordered list of ``{"frame": IMAGE, "frame_idx": int}`` entries. Each one
    mirrors a native MiniMaxH3AddGuide IMAGE invocation: cover-crop to the
    target canvas, VAE encode, then append a keyframe to ``minimax_keyframes``
    without retokenizing the prompt. Negative indices count backward from the
    end. ``guide_frame``/``guide_frame_idx`` remain as a compatibility path for
    callers created before dynamic guides.
    """
    frame_count = _align_frame_count(frame_count)
    latent = _empty_av_latent(width, height, frame_count)
    images = []
    keyframes = []

    if first_frame is not None:
        # Native H3 geometry anchor: plain stretch to the requested canvas.
        img = _resize(first_frame[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})

    if last_frame is not None:
        # Native H3 follower: aspect-preserving cover-crop.
        img = _resize(last_frame[:1], width, height, "center")
        images.append(img)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    tokens = clip.tokenize(str(prompt), images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_keyframes": keyframes}
        )

    normalized_guides = []
    if isinstance(guide_frames, (list, tuple)):
        normalized_guides.extend(guide_frames)
    elif guide_frames is not None:
        normalized_guides.append(guide_frames)
    if not normalized_guides and guide_frame is not None:
        normalized_guides.append({"frame": guide_frame, "frame_idx": guide_frame_idx})

    for guide_number, guide_item in enumerate(normalized_guides, start=1):
        if isinstance(guide_item, dict):
            frame = guide_item.get("frame")
            raw_value = guide_item.get("frame_idx", 0)
        elif isinstance(guide_item, (list, tuple)) and len(guide_item) >= 2:
            frame, raw_value = guide_item[0], guide_item[1]
        else:
            continue
        if frame is None:
            continue
        try:
            raw_idx = int(raw_value)
        except Exception:
            raw_idx = 0
        resolved_idx = raw_idx if raw_idx >= 0 else int(frame_count) + raw_idx
        if resolved_idx < 0 or resolved_idx >= int(frame_count):
            raise ValueError(
                f"FL2VA guide {guide_number} frame_idx {raw_idx} is outside the video's {int(frame_count)} frames."
            )

        # Chaining native AddGuide nodes is equivalent to appending every
        # encoded keyframe to the existing conditioning in order.
        guide = _resize(frame[:1], width, height, "center")
        guide_keyframe = {
            "resolved_frame_index": int(resolved_idx),
            "latent": vae.encode(guide),
        }
        existing = list(cond[0][1].get("minimax_keyframes", []))
        existing.append(guide_keyframe)
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_keyframes": existing}
        )
        del guide

    return cond, latent


def _invalidate_derived_preview(data_path: Path, manifest: dict) -> dict:
    """Drop only derived preview metadata/files after a random-access FL edit."""
    from .motion_context_disk import (
        _decoded_preview_cache_path,
        _decoded_preview_video_cache_path,
    )
    for p in (
        _decoded_preview_cache_path(data_path),
        _decoded_preview_video_cache_path(data_path),
    ):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    updated = dict(manifest)
    for key in (
        "preview_committed_count",
        "preview_audio_mode",
        "preview_fl2va_timeline_signature",
        "preview_updated_at",
    ):
        updated.pop(key, None)
    return updated


def sync_fl2va_manifest(owner_id, fps: float, clip_ids):
    """Reorder/drop cached plans by stable clip id without touching latent bytes."""
    from .motion_context_disk import (
        _manifest_for_first,
        _write_json_atomic,
        _final_frame_count,
    )
    owner = cache_owner_id(owner_id)
    data_path, manifest_path, manifest = _manifest_for_first(owner, fps)
    manifest = dict(manifest)
    manifest["sequence_mode"] = "fl2va"

    wanted = [str(x) for x in (clip_ids or [])]
    _cleanup_plan_video_cache(data_path, wanted)
    by_id = {
        str(desc.get("clip_id")): dict(desc)
        for desc in manifest.get("segments", [])
        if str(desc.get("clip_id") or "")
    }
    ordered = []
    for idx, clip_id in enumerate(wanted):
        desc = by_id.get(clip_id)
        if desc is None:
            continue
        desc["index"] = int(idx)
        desc["trim_frames"] = 0
        ordered.append(desc)

    if ordered != list(manifest.get("segments", [])):
        manifest = _invalidate_derived_preview(data_path, manifest)
    manifest["segments"] = ordered
    manifest["final_frame_count"] = _final_frame_count(ordered)
    manifest["updated_at"] = time.time()
    _write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest


def cached_fl2va_ids(manifest) -> set[str]:
    return {
        str(x.get("clip_id"))
        for x in (manifest or {}).get("segments", [])
        if str(x.get("clip_id") or "")
    }


def store_fl2va_segment(owner_id, fps, clip_ids, clip_index, clip_id, samples, validated=False, run_mode="full_batch", dependency_meta=None):
    """Append a new latent blob and atomically replace/insert one logical plan."""
    from .motion_context_disk import (
        _append_segment,
        _write_json_atomic,
        _final_frame_count,
        _make_handle,
        _proxy_at,
        _cache_size_mb,
    )

    data_path, manifest_path, manifest = sync_fl2va_manifest(owner_id, fps, clip_ids)
    segments = [dict(x) for x in manifest.get("segments", [])]
    clip_index = int(clip_index)
    clip_id = str(clip_id)
    # A rerender keeps the stable logical clip id but must discard its derived
    # decoded-plan preview cache. Other plan caches remain untouched.
    _invalidate_plan_video_cache(data_path, clip_id)

    # _append_segment is append-only on disk, which is ideal here: replacing a
    # middle plan never rewrites or truncates the following cached plans.
    desc, geom = _append_segment(
        data_path,
        samples,
        index=clip_index,
        trim_frames=0,
        validated=bool(validated),
        manifest=manifest,
    )
    desc["clip_id"] = clip_id
    desc["index"] = clip_index
    desc["trim_frames"] = 0
    dependency_meta = dependency_meta if isinstance(dependency_meta, dict) else {}
    first_source = str(dependency_meta.get("first_source") or "manual").lower().strip()
    desc["first_source"] = "previous_clip" if first_source == "previous_clip" else "manual"
    if desc["first_source"] == "previous_clip":
        desc["previous_clip_id"] = str(dependency_meta.get("previous_clip_id") or "")
        desc["previous_frame_signature"] = str(dependency_meta.get("previous_frame_signature") or "")
    else:
        desc.pop("previous_clip_id", None)
        desc.pop("previous_frame_signature", None)

    existing_pos = next(
        (i for i, x in enumerate(segments) if str(x.get("clip_id")) == clip_id),
        None,
    )
    if existing_pos is not None:
        segments[existing_pos] = desc
    else:
        insert_at = max(0, min(clip_index, len(segments)))
        segments.insert(insert_at, desc)

    # Reorder by the current card order after insertion/replacement.
    order = {str(cid): i for i, cid in enumerate(clip_ids or [])}
    segments.sort(key=lambda x: order.get(str(x.get("clip_id")), 10**9))
    for idx, item in enumerate(segments):
        item["index"] = idx
        item["trim_frames"] = 0

    manifest = _invalidate_derived_preview(data_path, manifest)
    manifest["sequence_mode"] = "fl2va"
    manifest["geometry"] = geom if manifest.get("geometry") is None else manifest["geometry"]
    manifest["segments"] = segments
    manifest["final_frame_count"] = _final_frame_count(segments)
    manifest["build"] = "fl2va-independent-cache-v1"
    manifest["updated_at"] = time.time()
    _write_json_atomic(manifest_path, manifest)

    # Keep random-access rerenders fast, but reclaim stale append-only blobs once
    # they become substantial. The returned manifest always matches any rewritten
    # offsets before we expose the lazy proxy downstream.
    try:
        compacted_manifest, _compact_info = compact_fl2va_cache(owner_id, force=False)
    except Exception as exc:
        # Compaction is an opportunistic maintenance optimization. On Windows an
        # unrelated live mmap can briefly prevent file replacement; never turn
        # that into a failed generation. A later rerender/Save Project retries.
        print(f"[WARNING] FL2VA cache compaction skipped: {exc}")
        compacted_manifest = None
    if compacted_manifest is not None:
        manifest = compacted_manifest
        segments = [dict(x) for x in manifest.get("segments", [])]

    pos = next(i for i, x in enumerate(segments) if str(x.get("clip_id")) == clip_id)
    status = f"FL2VA clip {clip_index + 1} {'validated' if validated else 'candidate'} cached"
    handle = _make_handle(
        data_path,
        manifest_path,
        manifest,
        str(run_mode),
        stop=False,
        status=status,
        next_index=len(segments),
    )
    size = _cache_size_mb(data_path, manifest_path)
    return handle, _proxy_at(data_path, manifest, pos), manifest, status, size


def set_fl2va_validation(owner_id, fps, clip_ids, validation_by_id, run_mode="full_batch"):
    from .motion_context_disk import _write_json_atomic, _make_handle
    data_path, manifest_path, manifest = sync_fl2va_manifest(owner_id, fps, clip_ids)
    segments = [dict(x) for x in manifest.get("segments", [])]
    changed = False
    for desc in segments:
        clip_id = str(desc.get("clip_id") or "")
        value = bool(validation_by_id.get(clip_id, False))
        if bool(desc.get("validated", False)) != value:
            desc["validated"] = value
            changed = True
    if changed:
        manifest = dict(manifest)
        manifest["segments"] = segments
        manifest["updated_at"] = time.time()
        _write_json_atomic(manifest_path, manifest)
    handle = _make_handle(
        data_path, manifest_path, manifest, str(run_mode), stop=False,
        status="FL2VA cache synchronized", next_index=len(segments),
    )
    return handle, manifest


def fl2va_cache_state(owner_id):
    from .motion_context_disk import _chain_paths, _load_manifest_from_paths
    owner = cache_owner_id(owner_id)
    data_path, manifest_path = _chain_paths(owner)
    if not data_path.exists() or not manifest_path.exists():
        return None
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        return None
    segments = [dict(x) for x in manifest.get("segments", [])]
    geometry = manifest.get("geometry") if isinstance(manifest.get("geometry"), dict) else {}
    return {
        "manifest": manifest,
        "continuity_signatures": continuity_signatures_for_segments(data_path, segments),
        "cached_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "")],
        "validated_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "") and bool(x.get("validated", False))],
        "cached_count": len(segments),
        "validated_count": sum(bool(x.get("validated", False)) for x in segments),
        "frame_count": int(manifest.get("final_frame_count", 0)),
        "resolved_width": int(geometry.get("video_w", 0) or 0) * 16,
        "resolved_height": int(geometry.get("video_h", 0) or 0) * 16,
    }


def export_fl2va_final(
    cache,
    vae,
    audio_vae,
    fps,
    filename_prefix,
    output_directory,
    codec,
    crf,
    preset,
    audio_bitrate,
    unique_id=None,
    workflow=None,
    prompt=None,
):
    """Decode FL2VA plans as independent hard cuts.

    When a following plan uses Previous, its predecessor ends on the exact
    selected handoff frame (best of the final six). The same lossless frame is
    injected as the following plan's First, eliminating backward jumps.

    Clip-by-clip uses one persistent decoded H.264 cache per logical plan plus
    the shared primary PCM cache. Adding/replacing one plan therefore VAE-decodes
    only that plan; unchanged plans are concatenated from their decoded caches.
    Full Batch still renders the requested final codec directly from latents,
    but opportunistically populates the same per-plan caches during that decode.
    """
    from . import motion_context_disk as d

    data_path, manifest_path, manifest = d._load_manifest(cache)
    segments = [dict(x) for x in manifest.get("segments", [])]
    if not segments:
        raise ValueError("FL2VA Final Decode: empty cache.")
    if abs(float(manifest.get("fps", fps)) - float(fps)) > 1e-6:
        raise ValueError(f"FL2VA Final Decode fps is {manifest.get('fps')}, export requested {fps}.")

    ffmpeg = d._find_ffmpeg()
    if str(output_directory).strip():
        out_dir = Path(str(output_directory).strip()).expanduser().resolve()
    elif d.folder_paths is not None:
        out_dir = Path(d.folder_paths.get_output_directory()).resolve()
    else:
        out_dir = (Path.cwd() / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_mode = str(cache.get("run_mode", "full_batch")) if isinstance(cache, dict) else "full_batch"
    timeline_segments = _timeline_segments(data_path, segments)
    expected_frames = sum(int(x.get("frames", 0)) for x in timeline_segments)
    color_timeline = d._color_timeline(timeline_segments, float(fps))

    # ------------------------------------------------------------------
    # Clip by Clip: decoded plans are independent primary preview caches.
    # ------------------------------------------------------------------
    if effective_mode == "clip_by_clip":
        progress = d._FinalDecodeNativeProgress(unique_id, total=max(4, 2 + len(segments) * 2))

        video_inputs = []
        for i, desc in enumerate(segments):
            path, _ = _ensure_plan_video_cache(
                data_path, desc, vae, float(fps), ffmpeg, progress=progress
            )
            video_inputs.append(
                _ensure_plan_timeline_video_cache(
                    data_path, desc, path, ffmpeg, float(fps),
                    handoff_enabled=_next_uses_previous(segments, i),
                )
            )

        manifest, segments = _ensure_plan_audio_cache(
            data_path, manifest_path, manifest, audio_vae, float(fps), progress=progress
        )
        timeline_segments = _timeline_segments(data_path, segments)
        expected_frames = sum(int(x.get("frames", 0)) for x in timeline_segments)
        color_timeline = d._color_timeline(timeline_segments, float(fps))
        timeline_signature = _timeline_signature(data_path, segments)

        committed_path = d._decoded_preview_cache_path(data_path)
        committed_count = int(manifest.get("preview_committed_count", 0) or 0)
        committed_mode = str(manifest.get("preview_audio_mode") or "")
        committed_signature = str(manifest.get("preview_fl2va_timeline_signature") or "")
        needs_assemble = (
            not committed_path.exists()
            or committed_count != len(segments)
            or committed_mode != d.PREVIEW_AUDIO_MODE
            or committed_signature != timeline_signature
        )
        if needs_assemble:
            token = f"fl2va_live_{os.urandom(5).hex()}"
            d._assemble_progressive_preview(
                ffmpeg,
                video_inputs,
                data_path,
                timeline_segments,
                len(timeline_segments),
                float(fps),
                committed_path,
                token,
            )
            manifest = dict(manifest)
            manifest["preview_committed_count"] = len(segments)
            manifest["preview_audio_mode"] = d.PREVIEW_AUDIO_MODE
            manifest["preview_fl2va_timeline_signature"] = timeline_signature
            manifest["updated_at"] = time.time()
            d._write_json_atomic(manifest_path, manifest)
        progress.advance()

        preview_path = d._publish_full_preview(committed_path, unique_id)
        autosave_path = d._replace_output_from_preview(
            preview_path,
            out_dir,
            filename_prefix,
            ffmpeg=ffmpeg,
            color_timeline=color_timeline,
        )
        d._embed_final_metadata_in_place(autosave_path, workflow=workflow, prompt=prompt)
        progress.advance()

        item = d._comfy_media_item(preview_path, fps, "temp")
        progress.finish()
        return {
            "ui": {
                "h3_video": [item],
                "h3_preview_info": [{
                    "mode": "fl2va_clip_by_clip",
                    "clip": len(segments),
                    "preview_frames": expected_frames,
                    "total_clips": len(segments),
                    "autosave_path": str(autosave_path),
                    "color_timeline": color_timeline,
                    "color_preview_baked": False,
                }],
            },
            "result": (d._video_output_from_path(autosave_path),),
        }

    # ------------------------------------------------------------------
    # Full Batch: final-quality direct decode, while filling plan caches.
    # ------------------------------------------------------------------
    extension = "mkv" if str(codec) == "FFV1 lossless" else "mp4"
    output_path = d._next_output_path(out_dir, filename_prefix, extension)
    progress = d._FinalDecodeNativeProgress(unique_id, total=max(4, 2 + len(segments) * 2))
    temp_root = d._ensure_cache_root()
    token = os.urandom(5).hex()
    temp_video = temp_root / f"_fl2va_{token}_video.{extension}"
    raw_audio = temp_root / f"_fl2va_{token}_audio.f32le"
    video_log = temp_root / f"_fl2va_{token}_video.log"
    mux_log = temp_root / f"_fl2va_{token}_mux.log"
    color_temp = temp_root / f"_fl2va_{token}_color.{extension}"
    video_proc = None
    video_log_f = None

    try:
        written_frames = 0
        plan_video_paths = []
        for i, desc in enumerate(segments):
            v = d._load_segment_video(data_path, desc)
            decoded = vae.decode(v)
            progress.advance()
            if decoded.ndim == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            wanted = int(desc.get("frames", 0))
            if int(decoded.shape[0]) != wanted:
                raise RuntimeError(
                    f"FL2VA Final Decode: clip {i + 1} returned {decoded.shape[0]} frames, expected {wanted}."
                )
            # Select/cache the handoff image from this decode before deciding
            # the visible timeline prefix. This is free: no second VAE pass.
            _save_plan_last_frame_cache(data_path, str(desc.get("clip_id") or f"clip_{i + 1}"), decoded)
            visible_frames = _visible_frames_for_plan(
                data_path, desc, handoff_enabled=_next_uses_previous(segments, i)
            )
            visible_frames = max(1, min(int(decoded.shape[0]), int(visible_frames)))
            if video_proc is None:
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                video_proc, video_log_f = d._start_video_encoder(
                    ffmpeg, temp_video, w, h, fps, codec, crf, preset, video_log
                )
            d._write_image_frames(video_proc, decoded[:visible_frames])
            # Keep the neutral per-plan cache complete; the assembled preview
            # receives a derived exact prefix only when the next plan uses Prev.
            full_plan_path = _cache_decoded_plan_video(data_path, desc, decoded, float(fps), ffmpeg)
            plan_video_paths.append(
                _ensure_plan_timeline_video_cache(
                    data_path, desc, full_plan_path, ffmpeg, float(fps),
                    handoff_enabled=_next_uses_previous(segments, i),
                )
            )
            written_frames += visible_frames
            del decoded, v

        if video_proc is None or video_log_f is None:
            raise RuntimeError("FL2VA Final Decode: encoder never started.")
        d._finish_process(video_proc, video_log_f, video_log, "FL2VA Final Decode encoder")
        video_proc = None
        video_log_f = None
        timeline_segments = _timeline_segments(data_path, segments)
        expected_frames = sum(int(x.get("frames", 0)) for x in timeline_segments)
        color_timeline = d._color_timeline(timeline_segments, float(fps))
        if written_frames != expected_frames:
            raise RuntimeError(f"FL2VA Final Decode wrote {written_frames} frames, expected {expected_frames}.")
        progress.advance()

        # Same primary decoded PCM cache used by Clip by Clip. Only plans whose
        # current descriptor lacks PCM are Audio-VAE decoded.
        manifest, segments = _ensure_plan_audio_cache(
            data_path, manifest_path, manifest, audio_vae, float(fps), progress=progress
        )
        timeline_segments = _timeline_segments(data_path, segments)
        expected_frames = sum(int(x.get("frames", 0)) for x in timeline_segments)
        color_timeline = d._color_timeline(timeline_segments, float(fps))
        timeline_signature = _timeline_signature(data_path, segments)
        sample_rate, channels, _ = d._write_preview_pcm_audio(
            ffmpeg, data_path, timeline_segments, len(timeline_segments), fps, raw_audio, f"{token}_fl_pcm"
        )
        d._mux_final(
            ffmpeg, temp_video, raw_audio, output_path,
            sample_rate, channels, codec, audio_bitrate, mux_log,
        )
        progress.advance()

        # Keep a neutral H.264 assembled preview for instant workflow/project
        # restore. Build it from the per-plan caches instead of copying the final
        # user-selected codec (which may be HEVC or FFV1/MKV). No VAE decode is
        # involved here.
        committed_path = d._decoded_preview_cache_path(data_path)
        d._assemble_progressive_preview(
            ffmpeg,
            plan_video_paths,
            data_path,
            timeline_segments,
            len(timeline_segments),
            float(fps),
            committed_path,
            f"{token}_fl_committed",
        )
        manifest = dict(manifest)
        manifest["preview_committed_count"] = len(segments)
        manifest["preview_audio_mode"] = d.PREVIEW_AUDIO_MODE
        manifest["preview_fl2va_timeline_signature"] = timeline_signature
        manifest["updated_at"] = time.time()
        d._write_json_atomic(manifest_path, manifest)

        preview_path = d._publish_full_preview(output_path, unique_id)
        if d._timeline_has_color(color_timeline):
            d._apply_color_timeline_to_file(
                ffmpeg, output_path, color_temp, color_timeline,
                codec=codec, crf=crf, preset=preset,
            )
            os.replace(color_temp, output_path)

        d._embed_final_metadata_in_place(output_path, workflow=workflow, prompt=prompt)

        item = d._comfy_media_item(preview_path, fps, "temp")
        progress.finish()
        return {
            "ui": {
                "h3_video": [item],
                "h3_preview_info": [{
                    "mode": "fl2va_full_batch",
                    "clip": len(segments),
                    "preview_frames": expected_frames,
                    "total_clips": len(segments),
                    "color_timeline": color_timeline,
                    "color_preview_baked": False,
                }],
            },
            "result": (d._video_output_from_path(output_path),),
        }
    finally:
        if video_proc is not None:
            try:
                if video_proc.stdin is not None:
                    video_proc.stdin.close()
            except Exception:
                pass
            try:
                video_proc.kill()
            except Exception:
                pass
        if video_log_f is not None:
            try:
                video_log_f.close()
            except Exception:
                pass
        for p in (temp_video, raw_audio, video_log, mux_log, color_temp):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

