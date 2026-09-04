"""
MiniMax H3 Motion Context - disk-backed sequential chain (v13 clean).

Two-node design:
  * MiniMax H3 Motion Context Disk Join
  * MiniMax H3 Motion Context Disk Final Decode

Goals:
  * full_batch and clip_by_clip workflows with the SAME graph
  * validated clips never require their sampler branch again
  * one .h3cache + one .json per chain, stored in this custom-node folder/cache
  * rerendering truncates/reuses the cache tail instead of creating files forever
  * cached LATENT output is memory-mapped/lazy; validated prefix does not fill RAM
  * final export decodes one seam pair at a time and streams to ffmpeg

The v10 RAM Motion Context conditioning and its validated seam corrections remain
unchanged. This module only changes persistence/execution and final streaming.
"""

import asyncio
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
import torch
import comfy.nested_tensor
import comfy.utils

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # static tests outside ComfyUI
    web = None
    PromptServer = None

try:
    import folder_paths
except Exception:  # static tests outside ComfyUI
    folder_paths = None

from .motion_context_ram import (
    FPS,
    _audio_exact_frames,
    _audio_t_for_frames,
    _auto_early_seam_shift,
    _frames_from_video_t,
    _luma_map,
    _luma_stats,
    _pixel_frames,
    _steps_for_frames,
    _streams_from_latent,
)

BUILD = "motion-context-disk-v2.5.16"
PREVIEW_AUDIO_MODE = "pcm_single_aac_gain_chain_v3_entry_ramp"
CACHE_VERSION = 12
PREVIEW_ROTATION_SLOTS = 3

# The browser/live-preview cache remains a neutral H.264 convenience cache.
# Full Batch final output uses a separate per-clip cache encoded directly with
# the Final Decode codec/CRF/preset selected when the batch starts. Final Decode
# then performs video stream-copy concat only; it never transcodes an already
# compressed clip to another video profile.
FULL_BATCH_H264_CACHE_CRF = 17
FULL_BATCH_H264_CACHE_PRESET = "fast"
FULL_BATCH_H264_CACHE_PROFILE = "h264_preview_crf17_fast_v2"
FULL_BATCH_FINAL_PROFILE_VERSION = 1
FULL_BATCH_FINAL_CACHE_VERSION = 1


def normalize_full_batch_export_profile(profile=None, *, codec="H.264", crf=17, preset="fast"):
    raw = profile if isinstance(profile, dict) else {}
    wanted_codec = str(raw.get("codec", codec) or codec)
    if wanted_codec not in {"H.264", "H.264 CPU (libx264)", "H.265 / HEVC", "FFV1 lossless"}:
        wanted_codec = "H.264"
    try:
        wanted_crf = max(0, min(51, int(raw.get("crf", crf))))
    except Exception:
        wanted_crf = int(crf)
    wanted_preset = str(raw.get("preset", preset) or preset)
    if wanted_preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
        wanted_preset = "fast"
    return {
        "version": int(FULL_BATCH_FINAL_PROFILE_VERSION),
        "codec": wanted_codec,
        "crf": int(wanted_crf),
        "preset": wanted_preset,
    }


def _full_batch_export_profile_signature(profile):
    normalized = normalize_full_batch_export_profile(profile)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _full_batch_export_profile_extension(profile):
    normalized = normalize_full_batch_export_profile(profile)
    return "mkv" if normalized["codec"] == "FFV1 lossless" else "mp4"


def _final_segment_cache_dir(data_path):
    path = Path(data_path).with_suffix(".final.video")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ref2va_final_segment_cache_path(data_path, index, profile):
    ext = _full_batch_export_profile_extension(profile)
    return _final_segment_cache_dir(data_path) / f"ref2va_{int(index):04d}.{ext}"


def _color_adjustment_signature(value):
    c = _normalize_color_adjustment(value) if "_normalize_color_adjustment" in globals() else (value or {})
    raw = json.dumps(c, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    return hashlib.sha256(raw).hexdigest()


# Full-batch soft-interrupt requests are intentionally process-local. The
# browser asks the currently running Extender node to stop *after* the active
# clip has been safely written to disk; the execution worker polls this tiny
# registry only at clip boundaries. Disk-backed checkpoint state itself lives
# in the manifest, so a later restart does not depend on this in-memory flag.
_FULL_BATCH_INTERRUPT_LOCK = threading.Lock()
_FULL_BATCH_INTERRUPT_REQUESTS = set()


def _full_batch_interrupt_key(owner_id, generation_mode="ref2va"):
    mode = "fl2va" if str(generation_mode or "ref2va").lower() == "fl2va" else "ref2va"
    return str(owner_id), mode


def clear_full_batch_interrupt(owner_id, generation_mode="ref2va"):
    key = _full_batch_interrupt_key(owner_id, generation_mode)
    with _FULL_BATCH_INTERRUPT_LOCK:
        _FULL_BATCH_INTERRUPT_REQUESTS.discard(key)


def request_full_batch_interrupt(owner_id, generation_mode="ref2va"):
    key = _full_batch_interrupt_key(owner_id, generation_mode)
    with _FULL_BATCH_INTERRUPT_LOCK:
        _FULL_BATCH_INTERRUPT_REQUESTS.add(key)


def full_batch_interrupt_requested(owner_id, generation_mode="ref2va", *, consume=False):
    key = _full_batch_interrupt_key(owner_id, generation_mode)
    with _FULL_BATCH_INTERRUPT_LOCK:
        found = key in _FULL_BATCH_INTERRUPT_REQUESTS
        if found and consume:
            _FULL_BATCH_INTERRUPT_REQUESTS.discard(key)
    return bool(found)


def _video_output_from_path(path):
    """Wrap a finished video file as a native ComfyUI VIDEO output.

    The Final Decode already produced the persistent final container on disk.
    This helper simply exposes that file to downstream video-aware nodes
    (upscalers, transcoders, Save Video, etc.) without any second decode or
    frame copy.
    """
    source = str(Path(path).resolve())
    last_error = None
    import_paths = (
        ("comfy_api.latest._input_impl.video_types", "VideoFromFile"),
        ("comfy_api.latest.input_impl.video_types", "VideoFromFile"),
    )
    for module_name, attr in import_paths:
        try:
            module = __import__(module_name, fromlist=[attr])
            factory = getattr(module, attr)
            return factory(source)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "Disk Final Decode: native VIDEO output is unavailable in this ComfyUI "
        f"build ({last_error}). Update ComfyUI core video support to use the "
        "Final Decode VIDEO output."
    )


class _FinalDecodeNativeProgress:
    """Native ComfyUI progress bound to the *currently executing* Final Decode node.

    We deliberately drive both paths used by recent ComfyUI versions:
      1. comfy.utils.ProgressBar -> normal global progress hook / legacy progress event
      2. comfy_execution.progress registry -> native per-node progress_state

    The registry path is optional so the custom node remains import-compatible with
    older ComfyUI versions.  No custom JS or websocket event is involved.
    """

    def __init__(self, unique_id, total):
        self.total = max(1, int(total))
        self.value = 0
        self.registry = None

        # The execution context is the authoritative runtime node id.  UNIQUE_ID is
        # retained as a fallback for older executors / direct tests.
        context_node_id = None
        try:
            from comfy_execution.utils import get_executing_context

            ctx = get_executing_context()
            context_node_id = getattr(ctx, "node_id", None) if ctx is not None else None
        except Exception:
            context_node_id = None

        raw_node_id = context_node_id if context_node_id is not None else unique_id
        self.node_id = None if raw_node_id is None else str(raw_node_id)

        try:
            self.pbar = comfy.utils.ProgressBar(self.total, node_id=self.node_id)
        except TypeError:
            # Compatibility with older ComfyUI ProgressBar signatures.
            self.pbar = comfy.utils.ProgressBar(self.total)

        try:
            from comfy_execution.progress import get_progress_state

            self.registry = get_progress_state()
            if self.node_id is not None:
                self.registry.start_progress(self.node_id)
        except Exception:
            self.registry = None

        # Send a non-zero first state immediately, before the first long VAE decode.
        self.update_absolute(1)

    def update_absolute(self, value):
        value = max(0, min(int(value), self.total))
        if value < self.value:
            return
        self.value = value

        # This is the exact native ProgressBar API sampler callbacks use.
        self.pbar.update_absolute(self.value, self.total)

        # On current ComfyUI, the inline node bar is sourced from progress_state.
        # Drive it explicitly as well; this is harmless if the global hook already
        # updated the same registry entry.
        if self.registry is not None and self.node_id is not None:
            try:
                self.registry.update_progress(
                    self.node_id, self.value, self.total, None
                )
            except Exception:
                pass

    def advance(self, units=1):
        self.update_absolute(self.value + max(0, int(units)))

    def finish(self):
        self.update_absolute(self.total)
        if self.registry is not None and self.node_id is not None:
            try:
                self.registry.finish_progress(self.node_id)
            except Exception:
                pass


CACHE_TYPE = "H3_MOTION_DISK_CACHE"
_LOG = logging.getLogger("minimax_h3_tail_from_latent.motion_context_disk")

_NODE_DIR = Path(__file__).resolve().parent
_CACHE_ROOT = _NODE_DIR / "cache"
_DATA_MAGIC = b"H3MCACHE12\x00"
_DATA_START = len(_DATA_MAGIC)
_AUDIO_CACHE_MAGIC = b"H3MAUDIO1\x00"
_AUDIO_CACHE_START = len(_AUDIO_CACHE_MAGIC)

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
# Float8 names exist only on recent torch builds.
for _name in (
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"
):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _DTYPE_MAP[_name] = _dt


def _safe_name(value):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "h3_chain"


def _ensure_cache_root():
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def _chain_paths(owner_id):
    root = _ensure_cache_root()
    stem = "chain_" + _safe_name(owner_id)
    return root / f"{stem}.h3cache", root / f"{stem}.json"


def _decoded_audio_cache_path(data_path):
    return Path(data_path).with_suffix(".audio.h3cache")


def _new_audio_cache_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_AUDIO_CACHE_MAGIC)
        f.flush()
        os.fsync(f.fileno())


def _ensure_audio_cache_file(path):
    path = Path(path)
    if not path.exists():
        _new_audio_cache_file(path)
        return
    with open(path, "rb") as f:
        magic = f.read(_AUDIO_CACHE_START)
    if magic != _AUDIO_CACHE_MAGIC:
        raise ValueError(f"H3 decoded audio cache: invalid cache data file: {path}")


def _write_json_atomic(path, payload):
    """Durably replace one JSON file without sharing a fixed temp filename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _dtype_name(tensor):
    return str(tensor.dtype).replace("torch.", "")


def _dtype_from_name(name):
    dt = _DTYPE_MAP.get(str(name))
    if dt is None:
        raise ValueError(f"H3 Disk Cache: unsupported tensor dtype '{name}'.")
    return dt


def _new_data_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_DATA_MAGIC)
        f.flush()
        os.fsync(f.fileno())


def _ensure_data_file(path):
    path = Path(path)
    if not path.exists():
        _new_data_file(path)
        return
    with open(path, "rb") as f:
        magic = f.read(_DATA_START)
    if magic != _DATA_MAGIC:
        raise ValueError(f"H3 Disk Cache: invalid cache data file: {path}")


def _geometry(video, audio):
    return {
        "video_batch": int(video.shape[0]),
        "video_channels": int(video.shape[1]),
        "video_h": int(video.shape[3]),
        "video_w": int(video.shape[4]),
        "audio_batch": int(audio.shape[0]),
        "audio_channels": int(audio.shape[1]),
        "audio_planes": int(audio.shape[2]),
    }


def _validate_batch_one(video, audio):
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("MiniMax H3 Disk Join supports batch size 1 only.")


def _validate_geometry(manifest, video, audio):
    current = _geometry(video, audio)
    expected = manifest.get("geometry")
    if expected is not None and current != expected:
        raise ValueError(
            "MiniMax H3 Disk Join: latent geometry changed between clips. "
            f"Expected {expected}, got {current}."
        )


def _final_frame_count(segments):
    if not segments:
        return 0
    total = int(segments[0]["frames"])
    for desc in segments[1:]:
        total += int(desc["frames"]) - int(desc["trim_frames"])
    return int(total)


def _segment_end(desc):
    return int(desc["segment_end"])


def _segment_start(desc):
    return int(desc["video"]["offset"])


def _recover_manifest(data_path, manifest_path, manifest):
    """Recover a safe prefix after an interrupted tail rewrite."""
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    _ensure_data_file(data_path)
    size = int(data_path.stat().st_size)
    good = []
    for desc in manifest.get("segments", []):
        try:
            start = _segment_start(desc)
            end = _segment_end(desc)
            if start < _DATA_START or end < start or end > size:
                break
            good.append(desc)
        except Exception:
            break

    if len(good) != len(manifest.get("segments", [])):
        fixed = dict(manifest)
        fixed["segments"] = [dict(x) for x in good]
        fixed["final_frame_count"] = _final_frame_count(good)
        fixed["updated_at"] = time.time()
        _write_json_atomic(manifest_path, fixed)
        manifest = fixed
        _LOG.warning(
            "H3 Disk Cache recovered %d valid clip(s) after incomplete tail write.",
            len(good),
        )
    return manifest


def _load_manifest_from_paths(data_path, manifest_path):
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"H3 Disk Cache version {manifest.get('version')} is incompatible; "
            f"expected {CACHE_VERSION}."
        )
    return _recover_manifest(data_path, manifest_path, manifest)


def _load_manifest(cache):
    if not isinstance(cache, dict):
        raise ValueError("MiniMax H3 Disk Cache: invalid cache handle.")
    data_path = Path(cache["data_path"])
    manifest_path = Path(cache["manifest_path"])
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"H3 Disk Cache manifest not found: {manifest_path}")
    return data_path, manifest_path, manifest


def _make_handle(
    data_path, manifest_path, manifest, run_mode, stop=False, status="", next_index=None
):
    if next_index is None:
        next_index = len(manifest.get("segments", []))
    return {
        "version": CACHE_VERSION,
        "data_path": str(Path(data_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "run_mode": str(run_mode),
        "stop": bool(stop),
        "next_index": int(next_index),
        "status": str(status),
    }


def _cache_size_mb(data_path, manifest_path):
    total = 0
    data_path = Path(data_path)
    preview_path = data_path.with_suffix(".preview.mp4")
    preview_video_path = data_path.with_suffix(".preview.video.mp4")
    audio_cache_path = _decoded_audio_cache_path(data_path)
    for p in (data_path, Path(manifest_path), preview_path, preview_video_path, audio_cache_path):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    for cache_dir in (data_path.with_suffix(".final.video"), data_path.with_suffix(".fl2va.video")):
        if cache_dir.exists():
            for child in cache_dir.iterdir():
                try:
                    if child.is_file():
                        total += child.stat().st_size
                except OSError:
                    pass
    return float(total / (1024.0 * 1024.0))


def _write_tensor_raw(file_obj, tensor):
    """Write one contiguous tensor without a giant serialization bytes object."""
    x = tensor.detach()
    if x.device.type != "cpu":
        x = x.to(device="cpu")
    if not x.is_contiguous():
        x = x.contiguous()

    offset = int(file_obj.tell())
    shape = [int(v) for v in x.shape]
    dtype_name = _dtype_name(x)
    nbytes = int(x.numel() * x.element_size())

    # Viewing as uint8 makes numpy() work for BF16/FP8 too; memoryview avoids
    # another full-size bytes copy in Python.
    raw = x.view(torch.uint8).numpy()
    written = int(file_obj.write(memoryview(raw)))
    if written != nbytes:
        raise IOError(f"H3 Disk Cache short write: {written}/{nbytes} bytes.")

    return {
        "offset": offset,
        "nbytes": nbytes,
        "shape": shape,
        "dtype": dtype_name,
    }


def _map_tensor(data_path, spec):
    """Memory-map one tensor. Pages are faulted in only when actually touched."""
    data_path = Path(data_path)
    offset = int(spec["offset"])
    nbytes = int(spec["nbytes"])
    shape = tuple(int(v) for v in spec["shape"])
    dtype = _dtype_from_name(spec["dtype"])

    if nbytes <= 0:
        raise ValueError("H3 Disk Cache: invalid zero-sized tensor.")
    mm = np.memmap(
        str(data_path), mode="c", dtype=np.uint8, offset=offset, shape=(nbytes,)
    )
    raw = torch.from_numpy(mm)
    tensor = raw.view(dtype).reshape(shape)
    return tensor


def _load_segment_video(data_path, desc):
    video = _map_tensor(data_path, desc["video"])
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError(f"Invalid cached H3 video shape: {tuple(video.shape)}")
    return video


def _load_segment_audio(data_path, desc):
    audio = _map_tensor(data_path, desc["audio"])
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4 or int(audio.shape[0]) != 1:
        raise ValueError(f"Invalid cached H3 audio shape: {tuple(audio.shape)}")
    return audio


def _append_segment(data_path, latent, index, trim_frames, validated, manifest):
    video, audio = _streams_from_latent(latent, "samples")
    _validate_batch_one(video, audio)
    if manifest.get("geometry") is not None:
        _validate_geometry(manifest, video, audio)

    trim = int(trim_frames)
    frames = _frames_from_video_t(int(video.shape[2]))
    if index == 0:
        trim = 0
    else:
        if trim < 0:
            raise ValueError("MiniMax H3 Disk Join: trim_frames cannot be negative.")
        if trim > 0 and _steps_for_frames(trim) is None:
            raise ValueError(
                f"MiniMax H3 Disk Join: trim_frames={trim} is not on H3 grid."
            )
        if trim >= frames:
            raise ValueError("MiniMax H3 Disk Join: trim removes the entire clip.")

    _ensure_data_file(data_path)
    with open(data_path, "ab", buffering=0) as f:
        video_spec = _write_tensor_raw(f, video)
        audio_spec = _write_tensor_raw(f, audio)
        segment_end = int(f.tell())
        f.flush()
        os.fsync(f.fileno())

    return {
        "index": int(index),
        "validated": bool(validated),
        "frames": int(frames),
        "trim_frames": int(trim),
        "video": video_spec,
        "audio": audio_spec,
        "segment_end": segment_end,
    }, _geometry(video, audio)


def _truncate_chain(data_path, manifest_path, manifest, index):
    """
    Keep clips [0:index), discard index and everything after it.
    Manifest prefix is committed first, so an interruption can only roll back.
    """
    index = max(0, int(index))
    old = [dict(x) for x in manifest.get("segments", [])]
    prefix = old[:index]
    truncate_at = _DATA_START if not prefix else _segment_end(prefix[-1])

    reduced = dict(manifest)
    reduced["segments"] = prefix
    if index == 0:
        reduced["geometry"] = None
    reduced["final_frame_count"] = _final_frame_count(prefix)
    # The assembled preview is derived from the old timeline. A rerun may keep
    # every decoded checkpoint before ``index``, but the joined preview itself
    # must never survive the edit or Final Decode could publish stale pixels.
    for key in (
        "preview_committed_count",
        "preview_audio_mode",
        "preview_fl2va_timeline_signature",
        "preview_updated_at",
        "preview_portable_full",
    ):
        reduced.pop(key, None)
    for preview_file in (
        _decoded_preview_cache_path(data_path),
        _decoded_preview_video_cache_path(data_path),
    ):
        try:
            Path(preview_file).unlink(missing_ok=True)
        except OSError:
            pass
    # Exact Ref2VA final sidecars are independent files. Drop only the suffix
    # whose latent descriptors were truncated; retained clips keep their final
    # bitstreams untouched.
    final_dir = Path(data_path).with_suffix(".final.video")
    if final_dir.exists():
        for sidecar in final_dir.glob("ref2va_*.*"):
            match = re.match(r"ref2va_(\d+)\.", sidecar.name)
            if match and int(match.group(1)) >= int(index):
                try:
                    sidecar.unlink(missing_ok=True)
                except OSError:
                    pass

    reduced["updated_at"] = time.time()
    _write_json_atomic(manifest_path, reduced)

    _ensure_data_file(data_path)
    with open(data_path, "r+b") as f:
        f.truncate(truncate_at)
        f.flush()
        os.fsync(f.fileno())

    # The decoded PCM cache follows the same retained clip prefix. Full-batch
    # and clip-by-clip therefore keep exactly the audio cache that belongs to
    # the surviving segments, with no recovery/redecode path.
    _truncate_decoded_audio_cache(data_path, prefix)
    return reduced


class _LazyDiskLatent(dict):
    """Small LATENT-compatible proxy; tensors stay mmap-backed until consumed."""

    def __init__(self, data_path, desc):
        super().__init__()
        self.data_path = str(data_path)
        self.desc = dict(desc)

    def _samples(self):
        video = _load_segment_video(self.data_path, self.desc)
        audio = _load_segment_audio(self.data_path, self.desc)
        return comfy.nested_tensor.NestedTensor((video, audio))

    def get(self, key, default=None):
        if key == "samples":
            return self._samples()
        if key in ("noise_mask", "batch_index"):
            return default
        return default

    def __getitem__(self, key):
        if key == "samples":
            return self._samples()
        raise KeyError(key)

    def __contains__(self, key):
        return key == "samples"

    def keys(self):
        return ("samples",)

    def copy(self):
        # Compatibility with nodes that copy LATENT dictionaries.
        return {"samples": self._samples()}


def _proxy_at(data_path, manifest, index):
    segments = manifest.get("segments", [])
    index = int(index)
    if index < 0 or index >= len(segments):
        raise ValueError(
            f"MiniMax H3 Disk Join: cached clip {index + 1} is unavailable."
        )
    return _LazyDiskLatent(data_path, segments[index])


def _last_proxy(data_path, manifest):
    segments = manifest.get("segments", [])
    if not segments:
        raise ValueError("MiniMax H3 Disk Join: cache contains no clip.")
    return _proxy_at(data_path, manifest, len(segments) - 1)


def _manifest_for_first(owner_id, fps):
    data_path, manifest_path = _chain_paths(owner_id)
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        _new_data_file(data_path)
        _new_audio_cache_file(_decoded_audio_cache_path(data_path))
        manifest = {
            "version": CACHE_VERSION,
            "build": BUILD,
            "owner_id": str(owner_id),
            "fps": float(fps),
            "geometry": None,
            "segments": [],
            "final_frame_count": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        _write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest


def _effective_state(previous_cache, run_mode, fps, unique_id):
    if previous_cache is not None:
        data_path, manifest_path, manifest = _load_manifest(previous_cache)
        mode = str(previous_cache.get("run_mode", run_mode))
        stop = bool(previous_cache.get("stop", False))
        index = int(previous_cache.get("next_index", len(manifest.get("segments", []))))
    else:
        owner = unique_id if unique_id is not None else "first_join"
        data_path, manifest_path, manifest = _manifest_for_first(owner, fps)
        mode = str(run_mode)
        stop = False
        index = 0

    if mode not in ("full_batch", "clip_by_clip"):
        mode = "full_batch"
    if abs(float(manifest.get("fps", fps)) - float(fps)) > 1e-6:
        raise ValueError(
            f"MiniMax H3 Disk Join: chain fps={manifest.get('fps')} but node fps={fps}."
        )
    return data_path, manifest_path, manifest, mode, stop, index


class MiniMaxH3MotionContextDiskJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"lazy": True}),
                "validated": ("BOOLEAN", {"default": False}),
                "run_mode": (
                    ["full_batch", "clip_by_clip"],
                    {"default": "full_batch"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001},
                ),
            },
            "optional": {
                "previous_cache": (CACHE_TYPE,),
                "trim_frames": ("INT", {"forceInput": True, "lazy": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CACHE_TYPE, "LATENT", "INT", "INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "cache",
        "cached_samples",
        "clip_count",
        "frame_count",
        "status",
        "cache_size_mb",
        "cache_path",
        "build",
    )
    FUNCTION = "join"
    CATEGORY = "MiniMax H3"

    def check_lazy_status(
        self,
        samples=None,
        trim_frames=None,
        validated=False,
        run_mode="full_batch",
        fps=24.0,
        previous_cache=None,
        unique_id=None,
    ):
        try:
            data_path, manifest_path, manifest, mode, stop, index = _effective_state(
                previous_cache, run_mode, fps, unique_id
            )
        except Exception:
            # Let execute surface the real error; request the minimum normal path.
            needed = []
            if samples is None:
                needed.append("samples")
            if previous_cache is not None and trim_frames is None:
                needed.append("trim_frames")
            return needed

        # In clip-by-clip mode, once the first candidate has been generated,
        # every later Disk Join becomes a metadata-only pass-through. Its
        # sampler/RAM branch is therefore never requested in this execution.
        if mode == "clip_by_clip" and stop:
            return []

        segments = manifest.get("segments", [])
        existing = index < len(segments)

        # A validated cached clip is immutable and needs no sampler branch.
        if bool(validated) and existing:
            return []

        needed = []
        if samples is None:
            needed.append("samples")
        if index > 0 and trim_frames is None:
            needed.append("trim_frames")
        return needed

    def join(
        self,
        samples=None,
        trim_frames=None,
        validated=False,
        run_mode="full_batch",
        fps=24.0,
        previous_cache=None,
        unique_id=None,
        reuse_existing=False,
        computed=False,
    ):
        data_path, manifest_path, manifest, mode, stop, index = _effective_state(
            previous_cache, run_mode, fps, unique_id
        )
        segments = [dict(x) for x in manifest.get("segments", [])]

        # Downstream of the first unvalidated clip in incremental mode:
        # preserve the same cache handle and do not touch any sampler input.
        if mode == "clip_by_clip" and stop:
            status = f"skipped after clip {len(segments)} (clip_by_clip)"
            handle = _make_handle(
                data_path, manifest_path, manifest, mode, stop=True, status=status
            )
            size = _cache_size_mb(data_path, manifest_path)
            return (
                handle,
                _last_proxy(data_path, manifest),
                len(segments),
                int(manifest.get("final_frame_count", 0)),
                status,
                size,
                str(data_path.parent),
                BUILD,
            )

        if index < 0 or index > len(segments):
            raise RuntimeError(
                f"MiniMax H3 Disk Join: invalid chain index {index}/{len(segments)}."
            )

        existing = index < len(segments)

        if bool(validated) and index > 0:
            if not all(bool(x.get("validated", False)) for x in segments[:index]):
                raise ValueError(
                    f"MiniMax H3 Disk Join: clip {index + 1} cannot be validated "
                    "before every previous clip is validated."
                )

        if bool(reuse_existing) and existing:
            # Full-batch resume checkpoint: advance through an already computed
            # unvalidated clip without touching its latent bytes or validation
            # state. This is deliberately an internal Extender path; the public
            # Disk Join node keeps its historical inputs unchanged.
            status = f"clip {index + 1} resumed from checkpoint"

        elif bool(validated) and existing:
            # Commit/freeze the existing candidate without evaluating samples.
            if not bool(segments[index].get("validated", False)) or bool(segments[index].get("computed", False)):
                segments[index]["validated"] = True
                segments[index].pop("computed", None)
                manifest = dict(manifest)
                manifest["segments"] = segments
                manifest["build"] = BUILD
                manifest["updated_at"] = time.time()
                _write_json_atomic(manifest_path, manifest)
                status = f"clip {index + 1} validated from disk"
            else:
                status = f"clip {index + 1} validated (disk)"

        else:
            # OFF means candidate: every active execution rewrites this clip and
            # invalidates/truncates the entire downstream tail in one operation.
            if samples is None:
                raise RuntimeError("MiniMax H3 Disk Join: active clip needs samples.")
            trim = 0 if index == 0 else int(trim_frames if trim_frames is not None else 22)

            if existing or len(segments) > index:
                manifest = _truncate_chain(data_path, manifest_path, manifest, index)
                segments = [dict(x) for x in manifest.get("segments", [])]

            desc, geom = _append_segment(
                data_path,
                samples,
                index=index,
                trim_frames=trim,
                validated=bool(validated),
                manifest=manifest,
            )
            if bool(computed) and not bool(validated):
                desc["computed"] = True
            segments = [dict(x) for x in manifest.get("segments", [])] + [desc]
            manifest = dict(manifest)
            manifest["geometry"] = geom if manifest.get("geometry") is None else manifest["geometry"]
            manifest["segments"] = segments
            manifest["final_frame_count"] = _final_frame_count(segments)
            manifest["build"] = BUILD
            manifest["updated_at"] = time.time()
            _write_json_atomic(manifest_path, manifest)
            status = (
                f"clip {index + 1} validated + cached"
                if bool(validated)
                else f"clip {index + 1} candidate cached"
            )

        # The first OFF clip terminates only the current incremental execution.
        stop_out = bool(mode == "clip_by_clip" and not bool(validated))
        handle = _make_handle(
            data_path, manifest_path, manifest, mode, stop=stop_out, status=status,
            next_index=index + 1,
        )
        size = _cache_size_mb(data_path, manifest_path)

        _LOG.info(
            "H3 Disk Join: mode=%s clip=%d validated=%s stop=%s clips=%d frames=%d cache=%.1f MB",
            mode,
            index + 1,
            bool(validated),
            stop_out,
            len(manifest.get("segments", [])),
            int(manifest.get("final_frame_count", 0)),
            size,
        )

        return (
            handle,
            _proxy_at(data_path, manifest, index),
            len(manifest.get("segments", [])),
            int(manifest.get("final_frame_count", 0)),
            status,
            size,
            str(data_path.parent),
            BUILD,
        )


# -----------------------------------------------------------------------------
# Final streaming decode - minimum RAM path: one seam pair at a time.
# -----------------------------------------------------------------------------


def _build_pair_video(data_path, prev_desc, curr_desc):
    prev_v = _load_segment_video(data_path, prev_desc)
    next_v = _load_segment_video(data_path, curr_desc)

    if tuple(prev_v.shape[:2]) != tuple(next_v.shape[:2]) or tuple(prev_v.shape[3:]) != tuple(next_v.shape[3:]):
        raise ValueError("Disk Final Decode: video latent geometry mismatch.")

    previous_frames = _frames_from_video_t(int(prev_v.shape[2]))
    next_frames = _frames_from_video_t(int(next_v.shape[2]))
    trim = int(curr_desc["trim_frames"])
    video_trim_t = 0 if trim == 0 else _steps_for_frames(trim)
    if trim > 0 and video_trim_t is None:
        raise ValueError(f"Disk Final Decode: trim_frames={trim} is not on H3 grid.")
    if int(video_trim_t) >= int(next_v.shape[2]):
        raise ValueError("Disk Final Decode: trim removes entire continuation.")

    warmup_video_t = 5 if int(video_trim_t) >= 7 else 0
    decode_start_t = int(video_trim_t) - int(warmup_video_t)
    warmup_start_frames = _pixel_frames(decode_start_t) if decode_start_t > 0 else 0
    warmup_frames = trim - warmup_start_frames
    if warmup_frames < 0:
        raise RuntimeError("Disk Final Decode: negative VAE warm-up.")

    if (int(prev_v.shape[2]) - decode_start_t) % 5 != 0:
        raise RuntimeError("Disk Final Decode: H3 temporal phase mismatch.")

    chain = torch.cat((prev_v, next_v[:, :, decode_start_t:, :, :]), dim=2)
    decode_frames = _frames_from_video_t(int(chain.shape[2]))
    expected = previous_frames + next_frames - warmup_start_frames
    if decode_frames != expected:
        raise RuntimeError(
            f"Disk Final Decode: pair decode frames {decode_frames} != {expected}."
        )

    meta = {
        "previous_frames": int(previous_frames),
        "next_frames": int(next_frames),
        "trim_frames": int(trim),
        "warmup_frames": int(warmup_frames),
        "continued_frames": int(next_frames - trim),
        "decode_frames": int(decode_frames),
    }
    return chain, meta


def _decode_pair_video(vae, chain, meta):
    decoded = vae.decode(chain)
    if decoded.ndim == 5:
        decoded = decoded.reshape(
            -1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1]
        )
    if int(decoded.shape[0]) != int(meta["decode_frames"]):
        raise RuntimeError(
            f"Disk Final Decode: VAE returned {decoded.shape[0]}, "
            f"expected {meta['decode_frames']}."
        )

    prev_frames = int(meta["previous_frames"])
    warmup = int(meta["warmup_frames"])
    shift = _auto_early_seam_shift(
        decoded,
        previous_frames=prev_frames,
        warmup_frames=warmup,
        max_early=2,
    )
    start = prev_frames + warmup + int(shift)
    end = start + int(meta["continued_frames"])
    if start < 0 or end > int(decoded.shape[0]):
        raise RuntimeError("Disk Final Decode: seam crop lies outside decoded pair.")

    previous_raw = decoded[:prev_frames]
    current_raw = decoded[start:end]
    return decoded, previous_raw, current_raw, int(shift)


def _correct_current_segment(previous_raw, current_raw, chunk_frames=8):
    """Memory-bounded disk Final Decode seam correction.

    This is mathematically equivalent to the disk path's former full-buffer
    photometric correction, but deliberately avoids allocating a second full
    RGB clip.

    ``current_raw`` is a disposable view into the decoded seam-pair buffer, so
    the correction is applied in place.  Large per-frame operations are then
    evaluated in small chunks to keep temporary float32 tensors bounded.  The
    RAM/legacy Join still uses its original helper unchanged.
    """
    tail_n = min(4, int(previous_raw.shape[0]))
    current_n = int(current_raw.shape[0])
    if tail_n < 1 or current_n < 1:
        return current_raw

    ref = previous_raw[-min(4, tail_n):]
    src = current_raw[:min(4, current_n)]

    ref_mean, ref_std, ref_med = _luma_stats(ref)
    src_mean, src_std, src_med = _luma_stats(src)

    if abs(ref_mean - src_mean) < 0.008 and abs(ref_med - src_med) < 0.012:
        return current_raw

    eps = 1e-4
    src_med_c = min(max(src_med, 0.05), 0.95)
    ref_med_c = min(max(ref_med, 0.05), 0.95)

    gamma_full = math.log(ref_med_c) / math.log(src_med_c)
    gamma_full = float(max(0.88, min(1.15, gamma_full)))

    src_y = _luma_map(src).detach().float()
    src_y_gamma = src_y.clamp(eps, 1.0).pow(gamma_full)
    gamma_mean = float(src_y_gamma.mean().item())
    if gamma_mean <= eps:
        del src_y, src_y_gamma
        return current_raw

    gain_full = ref_mean / gamma_mean
    gain_full = float(max(0.88, min(1.12, gain_full)))

    corrected_std = float(
        (src_y_gamma * gain_full).std(unbiased=False).clamp_min(1e-5).item()
    )
    contrast_full = ref_std / corrected_std if corrected_std > eps else 1.0
    contrast_full = float(max(0.92, min(1.08, contrast_full)))
    del src_y, src_y_gamma

    global_strength = 0.55
    seam_full_mean = gamma_mean * gain_full
    mean_offset = ref_mean - seam_full_mean
    step = max(1, int(chunk_frames))

    # Global correction, in place, with bounded temporary tensors.
    for start in range(0, current_n, step):
        end = min(current_n, start + step)
        segment = current_raw[start:end]
        rgb = segment[..., :3].float()
        y = _luma_map(rgb)
        y_full = y.clamp(eps, 1.0).pow(gamma_full)
        y_full = (y_full - ref_mean) * contrast_full + ref_mean
        y_full = y_full * gain_full
        y_full = (y_full + mean_offset).clamp(0.0, 1.0)
        delta = (y_full - y) * global_strength
        segment[..., :3] = (
            rgb + delta.unsqueeze(-1)
        ).clamp(0.0, 1.0).to(segment.dtype)
        del segment, rgb, y, y_full, delta

    # Same local first-three-frame stabilization as the legacy helper.
    local_count = min(3, current_n)
    if local_count > 0:
        stable_start = min(current_n, local_count)
        stable_end = min(current_n, stable_start + 4)
        if stable_end > stable_start:
            stable_mean, _, _ = _luma_stats(current_raw[stable_start:stable_end])
        else:
            stable_mean = ref_mean

        local_mix = (0.10, 0.40, 0.75)
        for j in range(local_count):
            frame = current_raw[j]
            rgb = frame[..., :3].float()
            y = _luma_map(rgb)
            current_mean = float(y.mean().item())
            t = local_mix[j]
            target_mean = ref_mean * (1.0 - t) + stable_mean * t
            offset = max(-0.060, min(0.060, float(target_mean - current_mean)))
            y_local = (y + offset).clamp(0.0, 1.0)
            delta = y_local - y
            frame[..., :3] = (
                rgb + delta.unsqueeze(-1)
            ).clamp(0.0, 1.0).to(frame.dtype)
            del frame, rgb, y, y_local, delta

    # Same +3.5% stable continuation lift, also chunked in place.
    tail_gain = 1.035
    tail_start = 3
    if tail_start < current_n:
        for start in range(tail_start, current_n, step):
            end = min(current_n, start + step)
            tail = current_raw[start:end]
            rgb = tail[..., :3].float()
            y = _luma_map(rgb)
            y_lift = (y * tail_gain).clamp(0.0, 1.0)

            count = int(tail.shape[0])
            strength = torch.ones(
                (count, 1, 1), device=y.device, dtype=y.dtype
            )
            offset0 = start - tail_start
            if offset0 == 0 and count >= 1:
                strength[0] = 1.0 / 3.0
            if offset0 <= 1 < offset0 + count:
                strength[1 - offset0] = 2.0 / 3.0

            delta = (y_lift - y) * strength
            tail[..., :3] = (
                rgb + delta.unsqueeze(-1)
            ).clamp(0.0, 1.0).to(tail.dtype)
            del tail, rgb, y, y_lift, strength, delta

    return current_raw


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("MiniMax H3 Disk Final Decode: ffmpeg executable not found.")


_NVENC_H264_CACHE = {}


def _h264_nvenc_available(ffmpeg):
    """Return True only when this ffmpeg can actually start an NVENC H.264 encode.

    Checking the encoder list is not enough: ffmpeg may have been compiled with
    h264_nvenc while the NVIDIA driver/GPU encoder is unavailable. A tiny 256x256
    one-frame probe catches both cases while staying above NVENC minimum
    H.264 dimensions on recent NVIDIA GPUs. The result is cached per ffmpeg binary.
    """
    key = str(Path(ffmpeg).resolve()) if ffmpeg else str(ffmpeg)
    cached = _NVENC_H264_CACHE.get(key)
    if cached is not None:
        return bool(cached)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1",
        "-frames:v", "1", "-an",
        "-c:v", "h264_nvenc",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8
        )
        available = proc.returncode == 0
    except Exception:
        available = False

    _NVENC_H264_CACHE[key] = bool(available)
    if available:
        _LOG.info("H3 Final Decode: H.264 NVENC available; using hardware encoder by default.")
    else:
        _LOG.info("H3 Final Decode: H.264 NVENC unavailable; falling back to libx264 CPU.")
    return bool(available)


def _preferred_h264_ffmpeg(ffmpeg):
    """Prefer a working NVENC-capable ffmpeg only for automatic H.264.

    imageio-ffmpeg remains the default binary for every other codec/path. If its
    bundled ffmpeg lacks NVENC, a system ffmpeg is tried as a hardware-only
    alternative without changing HEVC, FFV1, muxing, or metadata behavior.
    """
    if _h264_nvenc_available(ffmpeg):
        return ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        try:
            same = Path(system_ffmpeg).resolve() == Path(ffmpeg).resolve()
        except Exception:
            same = str(system_ffmpeg) == str(ffmpeg)
        if not same and _h264_nvenc_available(system_ffmpeg):
            return str(system_ffmpeg)
    return ffmpeg


def _nvenc_preset_from_x264(preset):
    # Preserve the existing speed intent while mapping x264 names to NVENC's p1-p7.
    return {
        "ultrafast": "p1",
        "superfast": "p2",
        "veryfast": "p3",
        "faster": "p4",
        "fast": "p4",
        "medium": "p5",
        "slow": "p6",
    }.get(str(preset), "p4")


def _h264_encode_args(ffmpeg, crf, preset, *, force_cpu=False):
    if not force_cpu and _h264_nvenc_available(ffmpeg):
        return [
            "-c:v", "h264_nvenc",
            "-preset", _nvenc_preset_from_x264(preset),
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(int(crf)),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
    ]


def _next_output_path(output_dir, prefix, extension):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(prefix)
    first = output_dir / f"{stem}.{extension}"
    if not first.exists():
        return first
    for i in range(1, 1000000):
        p = output_dir / f"{stem}_{i:05d}.{extension}"
        if not p.exists():
            return p
    raise RuntimeError("Disk Final Decode: could not allocate output filename.")


def _replace_output_from_preview(
    preview_path, output_dir, filename_prefix, ffmpeg=None, color_timeline=None
):
    """Atomically update the clip-by-clip autosave from the current full preview.

    The rolling browser preview stays neutral/non-destructive. User color
    adjustments are baked only into the persistent autosave copy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{_safe_name(filename_prefix)}.mp4"
    tmp = output_dir / f".{destination.stem}.{uuid.uuid4().hex[:10]}.tmp.mp4"
    try:
        if ffmpeg is not None and _timeline_has_color(color_timeline):
            _apply_color_timeline_to_file(
                ffmpeg, preview_path, tmp, color_timeline,
                codec="H.264", crf=17, preset="fast",
            )
        else:
            # The neutral preview is already the exact file we want to persist.
            # Prefer a same-volume hardlink so long clip-by-clip previews are not
            # recopied in full after every card. Cross-volume/unsupported filesystems
            # transparently fall back to the previous copy2 behaviour.
            try:
                os.link(Path(preview_path), tmp)
            except OSError:
                shutil.copy2(Path(preview_path), tmp)
        os.replace(tmp, destination)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return destination


def _start_video_encoder(
    ffmpeg, temp_video, width, height, fps, codec, crf, preset, log_path,
    video_filter=None,
):
    if str(codec) == "H.265 / HEVC":
        enc = ["-c:v", "libx265", "-preset", str(preset), "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
    elif str(codec) == "FFV1 lossless":
        enc = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]
    else:
        force_cpu = str(codec) == "H.264 CPU (libx264)"
        if not force_cpu:
            ffmpeg = _preferred_h264_ffmpeg(ffmpeg)
        enc = _h264_encode_args(ffmpeg, crf, preset, force_cpu=force_cpu)

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s:v", f"{int(width)}x{int(height)}",
        "-r", f"{float(fps):.9f}",
        "-i", "pipe:0",
    ]
    if video_filter:
        cmd += ["-vf", str(video_filter)]
    cmd += [
        "-an",
        *enc,
        str(temp_video),
    ]
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f
    )
    return proc, log_f


def _write_image_frames(proc, images, batch_frames=8):
    if proc.stdin is None:
        raise RuntimeError("Disk Final Decode: ffmpeg stdin is closed.")
    n = int(images.shape[0])
    for i in range(0, n, int(batch_frames)):
        part = images[i:i + int(batch_frames), ..., :3]
        part = (
            part.detach().float().clamp(0.0, 1.0)
            .mul(255.0).add_(0.5).to(torch.uint8)
            .cpu().contiguous()
        )
        # part is already contiguous uint8. Feed its buffer directly to ffmpeg
        # instead of materialising a second Python bytes object with .tobytes().
        proc.stdin.write(memoryview(part.numpy()).cast("B"))
        del part


def _finish_process(proc, log_f, log_path, label):
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    code = proc.wait()
    log_f.close()
    if code != 0:
        tail = ""
        try:
            tail = Path(log_path).read_bytes()[-12000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"{label} failed with code {code}.\n{tail}")


def _decode_audio_latent(audio_vae, latent, frames, fps):
    waveform = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    sr = int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 32000),
        )
    )
    return _audio_exact_frames(
        {"waveform": waveform, "sample_rate": sr}, int(frames), float(fps)
    )


def _decode_single_audio(data_path, desc, audio_vae, fps):
    latent = _load_segment_audio(data_path, desc)
    return _decode_audio_latent(audio_vae, latent, int(desc["frames"]), fps)


def _decode_pair_audio(data_path, prev_desc, curr_desc, audio_vae, fps, seam_shift):
    prev_a = _load_segment_audio(data_path, prev_desc)
    next_a = _load_segment_audio(data_path, curr_desc)

    previous_frames = int(prev_desc["frames"])
    next_frames = int(curr_desc["frames"])
    trim = int(curr_desc["trim_frames"])
    video_trim_t = 0 if trim == 0 else _steps_for_frames(trim)
    if trim > 0 and video_trim_t is None:
        raise ValueError("Disk Final Decode audio: invalid H3 trim grid.")

    warmup_video_t = 5 if int(video_trim_t) >= 7 else 0
    decode_start_t = int(video_trim_t) - int(warmup_video_t)
    warmup_start_frames = _pixel_frames(decode_start_t) if decode_start_t > 0 else 0
    warmup_frames = trim - warmup_start_frames

    decode_frames = previous_frames + next_frames - warmup_start_frames
    target_audio_t = _audio_t_for_frames(decode_frames)
    audio_start_t = int(prev_a.shape[-1]) + int(next_a.shape[-1]) - int(target_audio_t)
    if audio_start_t < 0 or audio_start_t >= int(next_a.shape[-1]):
        raise RuntimeError("Disk Final Decode audio: invalid warm-up start.")

    chain_audio = torch.cat((prev_a, next_a[..., audio_start_t:]), dim=-1)
    decoded = _decode_audio_latent(audio_vae, chain_audio, decode_frames, fps)
    w = decoded["waveform"]
    sr = int(decoded["sample_rate"])

    prev_n = int(round(float(previous_frames) / float(fps) * sr))
    effective_warm = max(0, int(warmup_frames) + int(seam_shift))
    warm_n = int(round(float(effective_warm) / float(fps) * sr))
    cut_b = prev_n + warm_n
    if cut_b >= int(w.shape[-1]):
        raise RuntimeError("Disk Final Decode audio: warm-up removes continuation.")

    pair = {
        "waveform": torch.cat((w[..., :prev_n], w[..., cut_b:]), dim=-1),
        "sample_rate": sr,
    }
    final_frames = previous_frames + next_frames - trim
    pair = _audio_exact_frames(pair, final_frames, fps)
    return pair, previous_frames, next_frames - trim


def _smooth_segment_entry_level(
    previous_tail,
    current,
    sample_rate,
    level_milliseconds=10.0,
    ramp_milliseconds=300.0,
    max_atten_db=18.0,
):
    """Soften a local upward loudness step without moving the audio timeline.

    H3 can generate a genuinely different musical state for the continuation
    clip. That cannot be repaired by a splice operation, but a sudden onset
    right after a quiet clip tail is especially audible as a bump. Measure the
    *very end* of the accepted previous PCM (10 ms) and the beginning of the
    new clip, then attenuate only the new clip when needed and let it rise
    smoothly to its native level over 300 ms. We deliberately never boost a
    quiet continuation: this is a seam mask, not loudness mastering.
    """
    if previous_tail is None or int(previous_tail.shape[-1]) < 2 or int(current.shape[-1]) < 2:
        return current

    sr = max(1, int(sample_rate))
    level_n = int(round(float(level_milliseconds) * 0.001 * sr))
    level_n = max(2, min(level_n, int(previous_tail.shape[-1]), int(current.shape[-1])))
    ramp_n = int(round(float(ramp_milliseconds) * 0.001 * sr))
    ramp_n = max(2, min(ramp_n, int(current.shape[-1])))

    prev = previous_tail[..., -level_n:].detach().float()
    head = current[..., :level_n].detach().float()
    prev_level = float(torch.sqrt(torch.mean(prev * prev) + 1.0e-12).item())
    curr_level = float(torch.sqrt(torch.mean(head * head) + 1.0e-12).item())
    if not math.isfinite(prev_level) or not math.isfinite(curr_level) or curr_level <= 1.0e-8:
        return current

    start_gain = min(1.0, prev_level / curr_level)
    min_gain = math.pow(10.0, -abs(float(max_atten_db)) / 20.0)
    start_gain = max(min_gain, start_gain)
    # Ignore tiny changes; they are less audible than touching the waveform.
    if start_gain >= math.pow(10.0, -0.5 / 20.0):
        return current

    out = current.clone()
    k = torch.arange(ramp_n, device=out.device, dtype=out.dtype)
    phase = torch.tensor(math.pi, device=out.device, dtype=out.dtype) * k / float(ramp_n - 1)
    rise = 0.5 * (1.0 - torch.cos(phase))
    gain = float(start_gain) + (1.0 - float(start_gain)) * rise
    out[..., :ramp_n] = out[..., :ramp_n] * gain
    return out


def _declick_segment(previous_tail, current, sample_rate, milliseconds=12.0):
    if previous_tail is None or int(previous_tail.shape[-1]) < 2 or int(current.shape[-1]) < 2:
        return current
    n = int(round(float(milliseconds) * 0.001 * int(sample_rate)))
    n = max(2, min(n, int(current.shape[-1])))
    out = current.clone()
    prev2 = previous_tail[..., -2]
    prev1 = previous_tail[..., -1]
    first = out[..., 0]
    target = prev1 + (prev1 - prev2)
    correction = first - target
    k = torch.arange(n, device=out.device, dtype=out.dtype)
    decay = 0.5 * (
        1.0 + torch.cos(
            torch.tensor(math.pi, device=out.device, dtype=out.dtype)
            * k / float(n - 1)
        )
    )
    out[..., :n] = out[..., :n] - correction.unsqueeze(-1) * decay
    return out


def _audio_seam_tail(wave, sample_rate, milliseconds=25.0):
    if wave is None or int(wave.shape[-1]) < 2:
        return None
    n = int(round(float(milliseconds) * 0.001 * max(1, int(sample_rate))))
    n = max(2, min(n, int(wave.shape[-1])))
    return wave[..., -n:].detach().clone()


def _audio_level_for_gain_match(wave, sample_rate):
    """Stable level estimate for two decodes of the same previous clip."""
    if wave is None or int(wave.shape[-1]) < 2:
        return None
    n = int(wave.shape[-1])
    edge = min(int(round(0.100 * int(sample_rate))), max(0, n // 4))
    if n - (2 * edge) >= max(32, int(round(0.250 * int(sample_rate)))):
        x = wave[..., edge:n - edge]
    else:
        x = wave
    level = float(torch.std(x.detach().float()).item())
    if not math.isfinite(level) or level < 1.0e-5:
        return None
    return level


def _match_pair_gain_to_previous(previous_timeline, pair_previous, sample_rate):
    """Recover the gain offset introduced by independent H3 pair decodes.

    The compared tails represent the same previous clip. The resulting scalar is
    applied to the new section of the pair, preserving its internal dynamics.
    """
    if previous_timeline is None or pair_previous is None:
        return 1.0
    common = min(int(previous_timeline.shape[-1]), int(pair_previous.shape[-1]))
    if common < max(32, int(round(0.250 * int(sample_rate)))):
        return 1.0
    ref = previous_timeline[..., -common:]
    cand = pair_previous[..., -common:]
    ref_level = _audio_level_for_gain_match(ref, sample_rate)
    cand_level = _audio_level_for_gain_match(cand, sample_rate)
    if ref_level is None or cand_level is None:
        return 1.0
    gain = float(ref_level / cand_level)
    limit = 10.0 ** (12.0 / 20.0)
    return max(1.0 / limit, min(limit, gain))


def _fit_audio_segment_to_cumulative(wave, target_total, written_total):
    need = max(0, int(target_total) - int(written_total))
    if int(wave.shape[-1]) > need:
        wave = wave[..., :need]
    elif int(wave.shape[-1]) < need:
        wave = torch.nn.functional.pad(wave, (0, need - int(wave.shape[-1])))
    return wave


def _write_audio_raw(file_obj, wave):
    x = wave[0].detach().float().transpose(0, 1).cpu().contiguous()
    raw = x.numpy().astype("float32", copy=False)
    file_obj.write(memoryview(raw).cast("B"))


def _mux_final(ffmpeg, temp_video, raw_audio, output_path, sr, channels, codec, audio_bitrate, log_path):
    audio_args = ["-c:a", "flac"] if str(codec) == "FFV1 lossless" else ["-c:a", "aac", "-b:a", str(audio_bitrate)]
    cmd = [
        ffmpeg, "-y",
        "-i", str(temp_video),
        "-f", "f32le",
        "-ar", str(int(sr)),
        "-ac", str(int(channels)),
        "-i", str(raw_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        *audio_args,
    ]
    if str(output_path).lower().endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))

    with open(log_path, "wb") as log_f:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
    if p.returncode != 0:
        tail = ""
        try:
            tail = Path(log_path).read_bytes()[-12000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Disk Final Decode mux failed with code {p.returncode}.\n{tail}")


def _comfy_media_item(path, fps, media_type):
    """
    Build a normal ComfyUI /view media descriptor.
    media_type must be 'temp' or 'output'.
    """
    path = Path(path).resolve()

    if folder_paths is None:
        return {
            "filename": path.name,
            "subfolder": "",
            "type": str(media_type),
            "format": "video/mp4",
            "frame_rate": float(fps),
        }

    if str(media_type) == "temp":
        root = Path(folder_paths.get_temp_directory()).resolve()
    else:
        root = Path(folder_paths.get_output_directory()).resolve()

    try:
        rel = path.relative_to(root)
        subfolder = "" if str(rel.parent) == "." else str(rel.parent).replace("\\", "/")
        filename = rel.name
    except Exception:
        # Custom output directories are not served by /view. The caller should
        # publish a temp preview copy first.
        subfolder = ""
        filename = path.name

    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": str(media_type),
        "format": "video/mp4",
        "frame_rate": float(fps),
    }


def _preview_temp_root():
    if folder_paths is not None:
        root = Path(folder_paths.get_temp_directory()).resolve()
    else:
        root = _ensure_cache_root() / "_preview"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _preview_temp_path(unique_id, slot=0):
    root = _preview_temp_root()
    return root / f"h3_motion_preview_{_safe_name(unique_id)}_{int(slot)}.mp4"


def _preview_temp_legacy_path(unique_id):
    return _preview_temp_root() / f"h3_motion_preview_{_safe_name(unique_id)}.mp4"


def _preview_rotation_order(unique_id):
    slots = [
        _preview_temp_path(unique_id, i) for i in range(int(PREVIEW_ROTATION_SLOTS))
    ]
    existing = [(i, p.stat().st_mtime) for i, p in enumerate(slots) if p.exists()]
    if not existing:
        return list(range(int(PREVIEW_ROTATION_SLOTS)))
    latest_idx = max(existing, key=lambda x: x[1])[0]
    return [
        (latest_idx + step) % int(PREVIEW_ROTATION_SLOTS)
        for step in range(1, int(PREVIEW_ROTATION_SLOTS) + 1)
    ]


def _reserve_preview_temp_path(unique_id):
    """Pick the next preview slot in a 3-file rotation.

    We never touch the slot that was most recently published first, because on
    Windows the browser/video player may keep it locked for a long time. We try
    the other two slots first and only reuse a slot when it can actually be
    deleted/replaced.
    """
    # Best-effort cleanup of the pre-v14.50 single preview file.
    legacy = _preview_temp_legacy_path(unique_id)
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass

    last_error = None
    for idx in _preview_rotation_order(unique_id):
        candidate = _preview_temp_path(unique_id, idx)
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError as e:
                last_error = e
                continue
        return candidate

    raise PermissionError(
        "H3 preview rotation: all preview slots are currently locked by another "
        f"process for node {unique_id!r}."
    ) from last_error


def _latest_preview_temp_path(unique_id):
    candidates = [
        _preview_temp_path(unique_id, i) for i in range(int(PREVIEW_ROTATION_SLOTS))
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        legacy = _preview_temp_legacy_path(unique_id)
        if legacy.exists():
            return legacy
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _normalize_color_adjustment(value=None):
    raw = value if isinstance(value, dict) else {}

    def _v(name, default, low, high):
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


def _color_is_neutral(value):
    c = _normalize_color_adjustment(value)
    return all(abs(float(c[k]) - 100.0) < 1e-6 for k in ("saturation", "contrast", "brightness"))


def _color_timeline(segments, fps):
    fps = float(fps or FPS)
    cursor = 0
    out = []
    for i, desc in enumerate(segments or []):
        contribution = int(desc.get("frames", 0))
        if i > 0:
            contribution -= int(desc.get("trim_frames", 0))
        contribution = max(0, contribution)
        start = float(cursor / fps)
        cursor += contribution
        end = float(cursor / fps)
        adjustment = _normalize_color_adjustment(desc.get("color_adjustment"))
        out.append({
            "index": int(i),
            "start": start,
            "end": end,
            "adjustment": adjustment,
            "modified": not _color_is_neutral(adjustment),
        })
    return out


def _timeline_has_color(timeline):
    return any(bool(item.get("modified")) for item in (timeline or []))


def _ffmpeg_color_filter(timeline):
    """Build filters that closely mirror browser CSS saturate/contrast/brightness.

    Keeping the live editor and the baked FFmpeg result on the same transform
    model makes the adjustment effectively WYSIWYG while preserving a neutral
    decoded source in cache.
    """
    filters = []
    for item in timeline or []:
        if not bool(item.get("modified")):
            continue
        c = _normalize_color_adjustment(item.get("adjustment"))
        sat = float(c["saturation"]) / 100.0
        contrast = float(c["contrast"]) / 100.0
        brightness = float(c["brightness"]) / 100.0
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        enable = f"gte(t\\,{start:.6f})*lt(t\\,{end:.6f})"

        # CSS saturate() matrix (Filter Effects spec luminance coefficients).
        rr = 0.213 + 0.787 * sat
        rg = 0.715 - 0.715 * sat
        rb = 0.072 - 0.072 * sat
        gr = 0.213 - 0.213 * sat
        gg = 0.715 + 0.285 * sat
        gb = 0.072 - 0.072 * sat
        br = 0.213 - 0.213 * sat
        bg = 0.715 - 0.715 * sat
        bb = 0.072 + 0.928 * sat
        filters.append(
            "colorchannelmixer="
            f"rr={rr:.8f}:rg={rg:.8f}:rb={rb:.8f}:"
            f"gr={gr:.8f}:gg={gg:.8f}:gb={gb:.8f}:"
            f"br={br:.8f}:bg={bg:.8f}:bb={bb:.8f}:"
            f"enable='{enable}'"
        )

        # CSS contrast() followed by brightness(), combined as one affine RGB LUT.
        gain = contrast * brightness
        offset = 255.0 * (0.5 * (1.0 - contrast) * brightness)
        expr = f"clip(val*{gain:.8f}{offset:+.8f},0,255)"
        filters.append(
            "lutrgb="
            f"r='{expr}':g='{expr}':b='{expr}':enable='{enable}'"
        )
    return ",".join(filters)


def _video_reencode_args(ffmpeg, codec, crf, preset):
    if str(codec) == "H.265 / HEVC":
        return ["-c:v", "libx265", "-preset", str(preset), "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
    if str(codec) == "FFV1 lossless":
        return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]
    return _h264_encode_args(
        ffmpeg, crf, preset,
        force_cpu=(str(codec) == "H.264 CPU (libx264)"),
    )


def _apply_color_timeline_to_file(
    ffmpeg,
    source_path,
    destination_path,
    timeline,
    codec="H.264",
    crf=17,
    preset="fast",
):
    source = Path(source_path)
    destination = Path(destination_path)
    vf = _ffmpeg_color_filter(timeline)
    if not vf:
        shutil.copy2(source, destination)
        return destination

    log_path = _ensure_cache_root() / f"_color_{uuid.uuid4().hex[:10]}.log"
    if str(codec) == "H.264":
        ffmpeg = _preferred_h264_ffmpeg(ffmpeg)
    cmd = [
        ffmpeg, "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", vf,
        *_video_reencode_args(ffmpeg, codec, crf, preset),
        "-c:a", "copy",
    ]
    if str(destination).lower().endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(destination))
    try:
        with open(log_path, "wb") as log_f:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        if proc.returncode != 0:
            tail = ""
            try:
                tail = log_path.read_bytes()[-12000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"H3 color correction failed with ffmpeg code {proc.returncode}.\n{tail}")
        return destination
    finally:
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass


def _publish_full_preview(output_path, unique_id):
    """
    Put the final file under ComfyUI temp so the in-node browser player always
    has a /view-compatible URL, including when output_directory is custom.
    Uses a 3-file rotation so the UI never tries to replace the MP4 that is
    currently opened by the browser player on Windows.
    """
    src = Path(output_path).resolve()
    dst = _reserve_preview_temp_path(unique_id)

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)
    return dst



def _saved_preview_output_path():
    """Return a unique human-readable MP4 path in the normal ComfyUI output dir."""
    if folder_paths is not None:
        root = Path(folder_paths.get_output_directory()).resolve()
    else:
        root = _ensure_cache_root() / "_saved_previews"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / f"MiniMax_H3_preview_{stamp}_{uuid.uuid4().hex[:4]}.mp4"


def _ffmetadata_escape(value):
    """Escape one single-line ffmetadata value without putting JSON on argv."""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("=", "\\=")
    text = text.replace(";", "\\;")
    text = text.replace("#", "\\#")
    text = text.replace("\r", "")
    text = text.replace("\n", "\\\n")
    return text


def _workflow_from_extra_pnginfo(extra_pnginfo):
    """Return the serialized ComfyUI workflow carried by EXTRA_PNGINFO."""
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if workflow is not None:
            return workflow
    return None


def _embed_final_metadata_in_place(source_path, workflow=None, prompt=None):
    """Embed ComfyUI workflow/prompt metadata without re-encoding media streams.

    Final H.264/H.265 exports are MP4 files.  The metadata is written through a
    tiny ffmetadata sidecar and ffmpeg stream-copy remux, then atomically replaces
    the original output.  Preview/cache files remain untouched.
    """
    source = Path(source_path).resolve()
    if source.suffix.lower() != ".mp4":
        return source
    if not source.exists() or not source.is_file():
        raise FileNotFoundError("H3 Final Decode: final MP4 was not found for metadata embedding.")

    metadata = {}
    if workflow is not None:
        metadata["workflow"] = workflow
    if prompt is not None:
        metadata["prompt"] = prompt
    if not metadata:
        return source

    ffmpeg = _find_ffmpeg()
    root = _ensure_cache_root()
    token = f"final_metadata_{uuid.uuid4().hex[:10]}"
    metadata_path = root / f"_{token}.ffmeta"
    log_path = root / f"_{token}.log"
    temp_path = source.with_name(source.stem + f".metadata_{uuid.uuid4().hex[:8]}" + source.suffix)

    try:
        lines = [";FFMETADATA1"]
        for key, value in metadata.items():
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{key}={_ffmetadata_escape(encoded)}")
        metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg,
            "-y",
            "-i", str(source),
            "-f", "ffmetadata",
            "-i", str(metadata_path),
            "-map", "0",
            # Keep ordinary source metadata and overlay the ComfyUI tags.
            "-map_metadata", "0",
            "-map_metadata", "1",
            "-c", "copy",
            "-movflags", "use_metadata_tags+faststart",
            str(temp_path),
        ]
        with open(log_path, "wb") as log_f:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        if proc.returncode != 0:
            tail = ""
            try:
                tail = log_path.read_bytes()[-12000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"H3 Final Decode metadata embedding failed with ffmpeg code {proc.returncode}.\n{tail}"
            )
        os.replace(temp_path, source)
        return source
    finally:
        for path in (metadata_path, log_path, temp_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def _save_preview_with_metadata(source_path, workflow=None, prompt=None, color_timeline=None):
    """Save the assembled preview to output with ComfyUI-compatible MP4 metadata.

    The rolling preview itself stays non-destructive/neutral. If per-clip color
    adjustments exist, bake them only into the saved copy, then attach the same
    `workflow` / `prompt` tags used by ComfyUI SaveVideo.
    """
    source = Path(source_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError("H3 Save Preview: current preview file was not found.")

    ffmpeg = _find_ffmpeg()
    output = _saved_preview_output_path()
    root = _ensure_cache_root()
    token = f"save_preview_{uuid.uuid4().hex[:10]}"
    metadata_path = root / f"_{token}.ffmeta"
    log_path = root / f"_{token}.log"
    corrected_path = root / f"_{token}_color.mp4"

    metadata = {}
    if workflow is not None:
        metadata["workflow"] = workflow
    if prompt is not None:
        metadata["prompt"] = prompt

    try:
        source_for_metadata = source
        if _timeline_has_color(color_timeline):
            _apply_color_timeline_to_file(
                ffmpeg, source, corrected_path, color_timeline,
                codec="H.264", crf=17, preset="fast",
            )
            source_for_metadata = corrected_path

        lines = [";FFMETADATA1"]
        for key, value in metadata.items():
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{key}={_ffmetadata_escape(encoded)}")
        metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg,
            "-y",
            "-i", str(source_for_metadata),
            "-f", "ffmetadata",
            "-i", str(metadata_path),
            "-map", "0",
            "-map_metadata", "1",
            "-c", "copy",
            "-movflags", "use_metadata_tags+faststart",
            str(output),
        ]
        with open(log_path, "wb") as log_f:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        if proc.returncode != 0:
            tail = ""
            try:
                tail = log_path.read_bytes()[-12000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            try:
                output.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"H3 Save Preview failed with ffmpeg code {proc.returncode}.\n{tail}"
            )
        return output
    finally:
        for path in (metadata_path, log_path, corrected_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass



# -----------------------------------------------------------------------------
# v12.5 - progressive FULL decoded preview cache
# -----------------------------------------------------------------------------

def _decoded_preview_cache_path(data_path):
    return Path(data_path).with_suffix(".preview.mp4")


def _decoded_preview_video_cache_path(data_path):
    return Path(data_path).with_suffix(".preview.video.mp4")


def _validated_prefix_count(segments):
    n = 0
    for desc in segments:
        if not bool(desc.get("validated", False)):
            break
        n += 1
    return n


def _latent_payload_end(desc):
    a = desc["audio"]
    return int(a["offset"]) + int(a["nbytes"])


def _write_blob_raw(file_obj, source_path):
    source_path = Path(source_path)
    spec = {
        "offset": int(file_obj.tell()),
        "nbytes": int(source_path.stat().st_size),
    }
    with open(source_path, "rb") as src:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)
    return spec


def _copy_blob_to_file(data_path, spec, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remaining = int(spec["nbytes"])
    with open(data_path, "rb") as src, open(destination, "wb") as dst:
        src.seek(int(spec["offset"]))
        while remaining > 0:
            chunk = src.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise IOError("H3 decoded-render blob is truncated.")
            dst.write(chunk)
            remaining -= len(chunk)


def _decoded_audio_meta_from_waveform(file_obj, rendered_audio):
    wave = rendered_audio["waveform"]
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    if wave.ndim != 3 or int(wave.shape[0]) != 1:
        raise ValueError(
            f"H3 progressive preview: invalid decoded audio shape {tuple(wave.shape)}."
        )
    wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()
    spec = _write_tensor_raw(file_obj, wave)
    return {
        "storage": "audio_cache",
        "waveform": spec,
        "sample_rate": int(rendered_audio["sample_rate"]),
        "timeline_gain": 1.0,
    }


def _decoded_audio_cache_end(segments):
    end = int(_AUDIO_CACHE_START)
    for desc in segments:
        meta = desc.get("decoded_audio")
        if not isinstance(meta, dict) or meta.get("storage") != "audio_cache":
            continue
        spec = meta.get("waveform")
        if not isinstance(spec, dict):
            continue
        end = max(end, int(spec.get("offset", 0)) + int(spec.get("nbytes", 0)))
    return int(end)


def _truncate_decoded_audio_cache(data_path, segments):
    audio_path = _decoded_audio_cache_path(data_path)
    if not audio_path.exists():
        return
    _ensure_audio_cache_file(audio_path)
    truncate_at = _decoded_audio_cache_end(segments)
    with open(audio_path, "r+b", buffering=0) as f:
        f.truncate(truncate_at)
        f.flush()
        os.fsync(f.fileno())


def _cache_candidate_render(
    data_path,
    manifest_path,
    manifest,
    clip_index,
    rendered_mp4=None,
    rendered_audio=None,
    seam_shift=None,
    video_profile=None,
):
    """Persist decoded data for the current tail segment.

    The same primitive is used by Clip-by-Clip and resumable Full Batch. Video
    stays beside the latent payload in the main ``.h3cache`` while decoded PCM
    lives in the dedicated primary audio cache. Either part may be omitted, so
    Full Batch can secure the expensive VideoVAE result even when ``audio_vae``
    is not connected to the Extender.
    """
    segments = [dict(x) for x in manifest.get("segments", [])]
    idx = int(clip_index)
    if idx != len(segments) - 1:
        raise RuntimeError(
            "H3 progressive preview can cache only the current tail candidate."
        )

    desc = dict(segments[idx])
    latent_end = int(desc.get("latent_end", _latent_payload_end(desc)))

    if rendered_mp4 is not None:
        # Main cache: latent payload + video-only decoded candidate. A rerender
        # of the tail replaces any older derived blob without touching latents.
        with open(data_path, "r+b", buffering=0) as f:
            f.truncate(latent_end)
            f.seek(latent_end)
            render_spec = _write_blob_raw(f, rendered_mp4)
            segment_end = int(f.tell())
            f.flush()
            os.fsync(f.fileno())
        desc["latent_end"] = latent_end
        desc["decoded_mp4_blob"] = render_spec
        desc["segment_end"] = segment_end
        # Overwriting the decoded video also overwrites its profile identity.
        # Clip-by-Clip preview blobs intentionally have no Full-Batch-final
        # profile, while progressive Full Batch records the exact default one.
        desc.pop("decoded_video_profile", None)
        if video_profile is not None:
            desc["decoded_video_profile"] = str(video_profile)

    if rendered_audio is not None:
        # Primary decoded-audio cache: one lossless PCM tensor per clip.
        audio_path = _decoded_audio_cache_path(data_path)
        _ensure_audio_cache_file(audio_path)
        with open(audio_path, "ab", buffering=0) as af:
            audio_meta = _decoded_audio_meta_from_waveform(af, rendered_audio)
            af.flush()
            os.fsync(af.fileno())
        desc["decoded_audio"] = audio_meta

    if seam_shift is not None:
        desc["decoded_seam_shift"] = int(seam_shift)

    segments[idx] = desc
    updated = dict(manifest)
    updated["segments"] = segments
    updated["build"] = BUILD
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated, desc

def _encode_corrected_segment_video_mp4(
    ffmpeg,
    video,
    fps,
    target_path,
    token,
    *,
    crf=17,
    preset="ultrafast",
):
    """Encode one corrected preview segment as VIDEO ONLY.

    Keeping AAC out of the per-clip cache avoids both audio encoder priming and
    MP4 audio edit-list timestamps from ever participating in an internal join.
    The matching lossless PCM is stored in the dedicated primary audio cache.
    """
    root = _ensure_cache_root()
    video_log = root / f"_{token}_video.log"
    proc = None
    log_f = None
    try:
        h, w = int(video.shape[1]), int(video.shape[2])
        proc, log_f = _start_video_encoder(
            ffmpeg,
            target_path,
            w,
            h,
            fps,
            "H.264",
            int(crf),
            str(preset),
            video_log,
        )
        _write_image_frames(proc, video)
        _finish_process(proc, log_f, video_log, "H3 progressive cache video encoder")
        proc = None
        log_f = None
    finally:
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        try:
            if video_log.exists():
                video_log.unlink()
        except OSError:
            pass


def _encode_final_segment_video(
    ffmpeg,
    video,
    fps,
    target_path,
    token,
    export_profile,
    color_adjustment=None,
):
    """Encode one decoded clip directly to its immutable Full-Batch final profile.

    This is the only lossy video encode for that clip. Final assembly later uses
    concat demuxer + ``-c:v copy`` and therefore cannot introduce another video
    generation. Color is baked here from the same decoded RGB tensor.
    """
    profile = normalize_full_batch_export_profile(export_profile)
    root = _ensure_cache_root()
    video_log = root / f"_{token}_final_segment.log"
    proc = None
    log_f = None
    try:
        h, w = int(video.shape[1]), int(video.shape[2])
        adjustment = _normalize_color_adjustment(color_adjustment)
        video_filter = None
        if not _color_is_neutral(adjustment):
            duration = max(1.0 / float(fps), float(video.shape[0]) / float(fps))
            video_filter = _ffmpeg_color_filter([{
                "index": 0,
                "start": 0.0,
                "end": duration + (1.0 / float(fps)),
                "adjustment": adjustment,
                "modified": True,
            }])
        proc, log_f = _start_video_encoder(
            ffmpeg,
            target_path,
            w,
            h,
            fps,
            profile["codec"],
            int(profile["crf"]),
            str(profile["preset"]),
            video_log,
            video_filter=video_filter,
        )
        _write_image_frames(proc, video)
        _finish_process(proc, log_f, video_log, "H3 Full Batch final segment encoder")
        proc = None
        log_f = None
    finally:
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        try:
            video_log.unlink(missing_ok=True)
        except OSError:
            pass


def _final_segment_cache_meta_matches(desc, export_profile, color_adjustment, path, *, visible_frames=None):
    profile_sig = _full_batch_export_profile_signature(export_profile)
    color_sig = _color_adjustment_signature(color_adjustment)
    if not Path(path).exists() or Path(path).stat().st_size <= 0:
        return False
    if bool(desc.get("final_video_dirty", False)):
        return False
    if int(desc.get("final_video_cache_version", 0) or 0) != int(FULL_BATCH_FINAL_CACHE_VERSION):
        return False
    if str(desc.get("final_video_profile_signature") or "") != profile_sig:
        return False
    if str(desc.get("final_video_color_signature") or "") != color_sig:
        return False
    if visible_frames is not None and int(desc.get("final_video_visible_frames", -1)) != int(visible_frames):
        return False
    return True


def _tag_ref2va_final_segment_cache(
    manifest_path, manifest, index, export_profile, color_adjustment, *, visible_frames=None
):
    segments = [dict(x) for x in manifest.get("segments", [])]
    idx = int(index)
    if idx < 0 or idx >= len(segments):
        raise IndexError(f"H3 final segment tag: invalid clip index {idx}.")
    desc = dict(segments[idx])
    desc["final_video_cache_version"] = int(FULL_BATCH_FINAL_CACHE_VERSION)
    desc["final_video_profile_signature"] = _full_batch_export_profile_signature(export_profile)
    desc["final_video_color_signature"] = _color_adjustment_signature(color_adjustment)
    desc["final_video_codec"] = normalize_full_batch_export_profile(export_profile)["codec"]
    desc["color_adjustment"] = _normalize_color_adjustment(color_adjustment)
    if visible_frames is not None:
        desc["final_video_visible_frames"] = int(visible_frames)
    desc.pop("final_video_dirty", None)
    segments[idx] = desc
    updated = dict(manifest)
    updated["segments"] = segments
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated, desc


def _ensure_ref2va_final_segment_cache(
    data_path,
    manifest_path,
    manifest,
    index,
    vae,
    fps,
    ffmpeg,
    export_profile,
    *,
    progress=None,
    decoded_video=None,
    color_adjustment=None,
):
    """Ensure exactly one Ref2VA clip has a final-profile sidecar.

    Existing matching sidecars are never decoded or re-encoded. A missing/dirty
    sidecar causes VideoVAE work for this clip only.
    """
    profile = normalize_full_batch_export_profile(export_profile)
    segments = [dict(x) for x in manifest.get("segments", [])]
    idx = int(index)
    if idx < 0 or idx >= len(segments):
        raise IndexError(f"H3 final segment cache: invalid clip index {idx}.")
    desc = dict(segments[idx])
    adjustment = _normalize_color_adjustment(
        color_adjustment if color_adjustment is not None else desc.get("color_adjustment")
    )
    path = _ref2va_final_segment_cache_path(data_path, idx, profile)
    if _final_segment_cache_meta_matches(desc, profile, adjustment, path):
        return manifest, path, False

    own_decode = decoded_video is None
    video = decoded_video
    if own_decode:
        _LOG.info(
            "H3 final cache repair: decoding changed Ref2VA clip %d only", idx + 1
        )
        video, _ = _render_one_final_video_segment(
            data_path, segments, idx, vae, progress=progress
        )
    elif progress is not None:
        # The caller already paid the decode cost while producing the preview
        # checkpoint, so there is no extra VAE progress step here.
        pass

    temp = path.with_name(path.stem + f".tmp_{uuid.uuid4().hex[:8]}" + path.suffix)
    try:
        _encode_final_segment_video(
            ffmpeg,
            video,
            float(fps),
            temp,
            f"ref2va_final_{idx}_{uuid.uuid4().hex[:6]}",
            profile,
            adjustment,
        )
        os.replace(temp, path)
        manifest, _ = _tag_ref2va_final_segment_cache(
            manifest_path, manifest, idx, profile, adjustment
        )
        return manifest, path, bool(own_decode)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        if own_decode and video is not None:
            del video


def _load_cached_decoded_audio(data_path, desc):
    meta = desc.get("decoded_audio")
    if not isinstance(meta, dict):
        return None
    spec = meta.get("waveform")
    if not isinstance(spec, dict):
        return None

    if meta.get("storage") == "audio_cache":
        source_path = _decoded_audio_cache_path(data_path)
        if not source_path.exists():
            return None
        _ensure_audio_cache_file(source_path)
    else:
        # Embedded PCM from pre-primary-cache live chains remains readable.
        # There is no decode/recovery fallback: newly generated caches always
        # use the dedicated primary audio cache.
        source_path = Path(data_path)

    wave = _map_tensor(source_path, spec)
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    if wave.ndim != 3 or int(wave.shape[0]) != 1:
        raise ValueError(
            f"H3 progressive preview: invalid cached decoded audio shape {tuple(wave.shape)}."
        )
    gain = float(meta.get("timeline_gain", 1.0))
    if not math.isfinite(gain) or gain <= 0.0:
        gain = 1.0
    if abs(gain - 1.0) > 1.0e-8:
        wave = wave * gain
    return {
        "waveform": wave,
        "sample_rate": int(meta.get("sample_rate", 32000)),
    }

def _upgrade_cached_audio_gain_chain(
    data_path,
    manifest_path,
    manifest,
    audio_vae,
    fps,
):
    """Non-destructive v14.42 PCM migration to the v14.43 gain chain."""
    segments = [dict(x) for x in manifest.get("segments", [])]
    if not segments:
        return manifest

    changed = False
    first_meta = segments[0].get("decoded_audio")
    if isinstance(first_meta, dict) and "timeline_gain" not in first_meta:
        first_meta = dict(first_meta)
        first_meta["timeline_gain"] = 1.0
        segments[0]["decoded_audio"] = first_meta
        changed = True

    for i in range(1, len(segments)):
        meta = segments[i].get("decoded_audio")
        if not isinstance(meta, dict) or "timeline_gain" in meta:
            continue
        previous_cached = _load_cached_decoded_audio(data_path, segments[i - 1])
        if previous_cached is None:
            continue
        pair, prev_frames, _ = _decode_pair_audio(
            data_path, segments[i - 1], segments[i], audio_vae, fps, 0
        )
        sr = int(pair["sample_rate"])
        if sr != int(previous_cached["sample_rate"]):
            del pair
            continue
        prev_n = int(round(float(prev_frames) / float(fps) * sr))
        pair_previous = pair["waveform"][..., :prev_n]
        gain = _match_pair_gain_to_previous(
            previous_cached["waveform"], pair_previous, sr
        )
        meta = dict(meta)
        meta["timeline_gain"] = float(gain)
        segments[i]["decoded_audio"] = meta
        changed = True
        del pair, pair_previous

    if not changed:
        return manifest

    updated = dict(manifest)
    updated["segments"] = segments
    updated.pop("preview_audio_mode", None)
    updated["build"] = BUILD
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated


def _write_preview_pcm_audio(
    ffmpeg,
    data_path,
    segments,
    count,
    fps,
    raw_audio_path,
    token,
):
    """Write exact timeline PCM for a progressive preview.

    Each clip contributes only its final corrected audio duration.  Segment
    boundaries are therefore sample-exact before ONE AAC encode is performed
    by _mux_final.  No AAC packet/padding is ever concatenated internally.
    """
    count = max(0, min(int(count), len(segments)))
    if count <= 0:
        raise ValueError("H3 progressive preview PCM builder has no segments.")
    sample_rate = None
    channels = None
    for i in range(count):
        cached = _load_cached_decoded_audio(data_path, segments[i])
        if cached is not None:
            sample_rate = int(cached["sample_rate"])
            wave = cached["waveform"]
            channels = int(wave.shape[1])
            break

    if sample_rate is None:
        # Native H3 audio output is 32 kHz stereo.  This branch exists only for
        # legacy caches where every decoded waveform predates v14.42.
        sample_rate = 32000
        channels = 2

    written_samples = 0
    cumulative_frames = 0
    previous_tail = None
    with open(raw_audio_path, "wb") as af:
        for i in range(count):
            desc = segments[i]
            audio = _load_cached_decoded_audio(data_path, desc)

            if audio is not None:
                sr = int(audio["sample_rate"])
                wave = audio["waveform"]
                if int(wave.shape[1]) != int(channels):
                    raise RuntimeError("H3 progressive preview: cached audio channel count changed.")
                if sr != int(sample_rate):
                    raise RuntimeError("H3 progressive preview: cached audio sample rate changed.")
            else:
                raise RuntimeError(
                    "H3 progressive preview: decoded audio cache is missing."
                )

            if i > 0:
                wave = _smooth_segment_entry_level(previous_tail, wave, sample_rate)
                wave = _declick_segment(previous_tail, wave, sample_rate, 12.0)

            out_frames = int(desc["frames"])
            if i > 0:
                out_frames -= int(desc.get("trim_frames", 0))
            cumulative_frames += int(out_frames)
            target = int(round(float(cumulative_frames) / float(fps) * int(sample_rate)))
            wave = _fit_audio_segment_to_cumulative(
                wave, target, written_samples
            )
            _write_audio_raw(af, wave)
            written_samples += int(wave.shape[-1])
            previous_tail = _audio_seam_tail(wave, sample_rate)
            del wave

    return int(sample_rate), int(channels), int(written_samples)


def _concat_video_stream_copy(ffmpeg, inputs, output_path, log_path):
    """Concatenate homogeneous video-only segments with zero video re-encode."""
    inputs = [Path(p) for p in inputs if p is not None and Path(p).exists()]
    if not inputs:
        raise ValueError("H3 video concat has no input.")

    if len(inputs) == 1:
        if Path(output_path).resolve() != inputs[0].resolve():
            shutil.copy2(inputs[0], output_path)
        return

    list_path = Path(log_path).with_suffix(".concat.txt")
    try:
        lines = []
        for p in inputs:
            escaped = str(p.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
        ]
        if str(output_path).lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(output_path))
        with open(log_path, "wb") as log_f:
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        if p.returncode != 0:
            tail = ""
            try:
                tail = Path(log_path).read_bytes()[-12000:].decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                pass
            raise RuntimeError(
                f"H3 video stream-copy concat failed with code {p.returncode}.\n{tail}"
            )
    finally:
        try:
            if list_path.exists():
                list_path.unlink()
        except OSError:
            pass


def _concat_mp4_video_stream_copy(ffmpeg, inputs, output_path, log_path):
    """Backward-compatible preview wrapper for H.264 MP4 checkpoints."""
    return _concat_video_stream_copy(ffmpeg, inputs, output_path, log_path)


def _assemble_progressive_preview(
    ffmpeg,
    video_inputs,
    data_path,
    segments,
    count,
    fps,
    output_path,
    token,
):
    """Assemble a full preview with video stream-copy + one AAC encode."""
    root = _ensure_cache_root()
    temp_video = root / f"_{token}_joined_video.mp4"
    raw_audio = root / f"_{token}_joined_audio.f32le"
    video_log = root / f"_{token}_video_concat.log"
    mux_log = root / f"_{token}_audio_mux.log"
    try:
        _concat_mp4_video_stream_copy(
            ffmpeg, video_inputs, temp_video, video_log
        )
        sr, channels, _ = _write_preview_pcm_audio(
            ffmpeg,
            data_path,
            segments,
            count,
            fps,
            raw_audio,
            token,
        )
        if Path(output_path).exists():
            Path(output_path).unlink()
        _mux_final(
            ffmpeg,
            temp_video,
            raw_audio,
            output_path,
            sr,
            channels,
            "H.264",
            "192k",
            mux_log,
        )
    finally:
        for p in (temp_video, raw_audio, video_log, mux_log):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except OSError:
                pass


def _render_one_final_video_segment(
    data_path,
    segments,
    index,
    vae,
    progress=None,
):
    i = int(index)
    curr = segments[i]

    if i == 0:
        v = _load_segment_video(data_path, curr)
        video = vae.decode(v)
        if progress is not None:
            progress.advance()
        if video.ndim == 5:
            video = video.reshape(
                -1, video.shape[-3], video.shape[-2], video.shape[-1]
            )
        expected = int(curr["frames"])
        if int(video.shape[0]) != expected:
            raise RuntimeError(
                f"H3 progressive preview: clip 1 decoded {video.shape[0]}, "
                f"expected {expected}."
            )
        del v
        return video, 0

    prev = segments[i - 1]
    chain, meta = _build_pair_video(data_path, prev, curr)
    decoded, previous_raw, current_raw, shift = _decode_pair_video(
        vae, chain, meta
    )
    # The latent seam pair is no longer needed once VAE decode returned.  Drop
    # it before photometric work so latent + large RGB temporaries do not overlap.
    del chain
    if progress is not None:
        progress.advance()
    current_video = _correct_current_segment(previous_raw, current_raw)
    del decoded, previous_raw, current_raw
    return current_video, int(shift)


def _render_one_final_audio_segment(
    data_path,
    segments,
    index,
    audio_vae,
    fps,
    seam_shift,
    progress=None,
):
    """Decode exactly one final audio segment for a known video seam shift."""
    i = int(index)
    curr = segments[i]

    if i == 0:
        audio = _decode_single_audio(data_path, curr, audio_vae, fps)
        if progress is not None:
            progress.advance()
        return audio

    prev = segments[i - 1]
    pair_audio, prev_frames, curr_frames = _decode_pair_audio(
        data_path,
        prev,
        curr,
        audio_vae,
        fps,
        int(seam_shift),
    )
    if progress is not None:
        progress.advance()
    sr = int(pair_audio["sample_rate"])
    wave = pair_audio["waveform"]

    prev_samples = int(round(float(prev_frames) / float(fps) * sr))
    previous_audio = wave[..., :prev_samples]
    current_audio = wave[..., prev_samples:]

    previous_cached = _load_cached_decoded_audio(data_path, prev)
    if previous_cached is not None and int(previous_cached["sample_rate"]) == sr:
        gain = _match_pair_gain_to_previous(
            previous_cached["waveform"], previous_audio, sr
        )
        if abs(gain - 1.0) > 1.0e-8:
            current_audio = current_audio * gain

    # The click repair is deliberately deferred to full PCM assembly, where
    # the exact previous timeline tail is known.
    wanted = int(round(float(curr_frames) / float(fps) * sr))
    current_audio = _fit_audio_segment_to_cumulative(
        current_audio, wanted, 0
    )

    audio = {
        "waveform": current_audio,
        "sample_rate": sr,
    }

    del pair_audio, wave, previous_audio
    return audio


def _render_one_final_segment(
    data_path,
    segments,
    index,
    vae,
    audio_vae,
    fps,
    progress=None,
):
    """Compatibility helper returning one video+audio segment.

    Memory-sensitive progressive-preview code calls the video/audio helpers
    separately so the large VideoVAE pair can be encoded and released before
    AudioVAE work starts.
    """
    i = int(index)
    video, shift = _render_one_final_video_segment(
        data_path, segments, i, vae, progress=progress
    )
    audio = _render_one_final_audio_segment(
        data_path,
        segments,
        i,
        audio_vae,
        fps,
        int(shift),
        progress=progress,
    )
    return video, audio, int(shift)

def _export_final_from_exact_segment_caches(
    ffmpeg,
    segment_paths,
    data_path,
    segments,
    fps,
    output_path,
    export_profile,
    audio_bitrate,
    token,
):
    """Mux a Full-Batch final from already-final video segments.

    Video operations are strictly packet-copy: per-clip exact-profile caches are
    concat-demuxed with ``-c:v copy`` and the resulting stream is copied again
    while the lossless PCM cache is encoded to the requested audio format.
    """
    profile = normalize_full_batch_export_profile(export_profile)
    root = _ensure_cache_root()
    extension = _full_batch_export_profile_extension(profile)
    joined_video = root / f"_{token}_joined_exact.{extension}"
    raw_audio = root / f"_{token}_final_audio.f32le"
    concat_log = root / f"_{token}_exact_concat.log"
    mux_log = root / f"_{token}_final_mux.log"
    try:
        _concat_video_stream_copy(ffmpeg, segment_paths, joined_video, concat_log)
        sr, channels, _ = _write_preview_pcm_audio(
            ffmpeg,
            data_path,
            segments,
            len(segments),
            float(fps),
            raw_audio,
            token,
        )
        _mux_final(
            ffmpeg,
            joined_video,
            raw_audio,
            output_path,
            sr,
            channels,
            profile["codec"],
            audio_bitrate,
            mux_log,
        )
        return "exact_segment_stream_copy"
    finally:
        for item in (joined_video, raw_audio, concat_log, mux_log):
            try:
                Path(item).unlink(missing_ok=True)
            except OSError:
                pass


def _full_batch_manifest_export_profile(manifest, requested_profile=None):
    stored = manifest.get("full_batch_export_profile") if isinstance(manifest, dict) else None
    if isinstance(stored, dict):
        return normalize_full_batch_export_profile(stored)
    if requested_profile is not None:
        return normalize_full_batch_export_profile(requested_profile)
    return normalize_full_batch_export_profile()


def _resolve_full_batch_export_profile(manifest_path, manifest, requested_profile, *, context="H3 Full Batch"):
    """Resolve the active Full-Batch export profile.

    An interrupted/in-progress checkpoint must keep the profile it already
    started with so the resumed batch remains internally consistent. Outside an
    active checkpoint, a new CRF/preset/codec selection is treated as the start
    of a fresh Full Batch: the manifest adopts the requested profile and exact
    final sidecars are rebuilt clip-by-clip on demand.
    """
    requested = normalize_full_batch_export_profile(requested_profile)
    manifest = dict(manifest or {})
    stored_raw = manifest.get("full_batch_export_profile")
    if not isinstance(stored_raw, dict):
        manifest["full_batch_export_profile"] = requested
        manifest["updated_at"] = time.time()
        _write_json_atomic(manifest_path, manifest)
        return manifest, requested

    stored = normalize_full_batch_export_profile(stored_raw)
    stored_sig = _full_batch_export_profile_signature(stored)
    requested_sig = _full_batch_export_profile_signature(requested)
    if stored_sig == requested_sig:
        return manifest, stored

    checkpoint_active = bool(
        manifest.get("batch_in_progress", False)
        or manifest.get("batch_interrupted", False)
    )
    if checkpoint_active:
        _LOG.warning(
            "%s: keeping active checkpoint export profile %s CRF %s %s; "
            "ignoring requested %s CRF %s %s until a fresh Full Batch starts.",
            str(context),
            stored["codec"], int(stored["crf"]), stored["preset"],
            requested["codec"], int(requested["crf"]), requested["preset"],
        )
        return manifest, stored

    manifest["full_batch_export_profile"] = requested
    manifest["updated_at"] = time.time()
    _write_json_atomic(manifest_path, manifest)
    _LOG.info(
        "%s: adopting new Full Batch export profile %s CRF %s %s "
        "(previous %s CRF %s %s). Exact-final segment caches will be rebuilt clip-by-clip as needed.",
        str(context),
        requested["codec"], int(requested["crf"]), requested["preset"],
        stored["codec"], int(stored["crf"]), stored["preset"],
    )
    return manifest, requested


def cache_full_batch_ref2va_segment(
    data_path,
    manifest_path,
    clip_index,
    vae,
    audio_vae,
    fps,
    export_profile=None,
    color_adjustment=None,
):
    """Decode/cache one Ref2VA Full-Batch clip exactly once.

    The decoded RGB result feeds two independent caches while it is still in
    memory: a neutral H.264 browser-preview checkpoint and, when a Full-Batch
    export profile is available, the exact final-profile sidecar. The latter is
    what Final Decode concatenates with ``-c:v copy``.
    """
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        raise FileNotFoundError("H3 Full Batch cache: manifest disappeared.")

    segments = [dict(x) for x in manifest.get("segments", [])]
    idx = int(clip_index)
    if idx < 0 or idx >= len(segments):
        raise IndexError(f"H3 Full Batch cache: invalid clip index {idx}.")

    desc = dict(segments[idx])
    adjustment = _normalize_color_adjustment(
        color_adjustment if color_adjustment is not None else desc.get("color_adjustment")
    )
    profile = (
        normalize_full_batch_export_profile(export_profile)
        if export_profile is not None
        else None
    )
    is_tail = idx == len(segments) - 1
    video_ready = isinstance(desc.get("decoded_mp4_blob"), dict)
    shift_ready = idx == 0 or "decoded_seam_shift" in desc
    cached_audio = _load_cached_decoded_audio(data_path, desc)
    audio_ready = cached_audio is not None
    if cached_audio is not None:
        del cached_audio

    # Ref2VA's embedded preview blob can only be rewritten safely at the physical
    # tail. The exact-final sidecar is external, so even a legacy middle clip can
    # be repaired independently without touching any following latent payload.
    if not is_tail and not (video_ready and shift_ready):
        if profile is not None:
            ffmpeg = _find_ffmpeg()
            manifest, _, _ = _ensure_ref2va_final_segment_cache(
                data_path, manifest_path, manifest, idx, vae, float(fps), ffmpeg,
                profile, color_adjustment=adjustment,
            )
        return manifest, {
            "video_cached": bool(video_ready),
            "audio_cached": bool(audio_ready),
            "seam_shift": int(desc.get("decoded_seam_shift", 0) or 0),
            "final_cached": bool(profile is not None),
            "deferred": True,
        }

    rendered_video = None
    rendered_audio = None
    seam_shift = int(desc.get("decoded_seam_shift", 0) or 0)
    temp_root = _ensure_cache_root()
    token = f"fullbatch_ref_{idx}_{uuid.uuid4().hex[:8]}"
    temp_mp4 = temp_root / f"_{token}.mp4"
    ffmpeg = None

    try:
        if not video_ready or not shift_ready:
            rendered_video, seam_shift = _render_one_final_video_segment(
                data_path, segments, idx, vae
            )
            rendered_mp4 = None
            video_profile = None
            if not video_ready:
                ffmpeg = _find_ffmpeg()
                _encode_corrected_segment_video_mp4(
                    ffmpeg,
                    rendered_video,
                    float(fps),
                    temp_mp4,
                    token,
                    crf=FULL_BATCH_H264_CACHE_CRF,
                    preset=FULL_BATCH_H264_CACHE_PRESET,
                )
                rendered_mp4 = temp_mp4
                video_profile = FULL_BATCH_H264_CACHE_PROFILE

            # Commit the neutral browser checkpoint first. This is the only
            # write that touches the append-only .h3cache body.
            manifest, desc = _cache_candidate_render(
                data_path,
                manifest_path,
                manifest,
                idx,
                rendered_mp4=rendered_mp4,
                seam_shift=int(seam_shift),
                video_profile=video_profile,
            )
            segments = [dict(x) for x in manifest.get("segments", [])]
            desc = dict(segments[idx])
            video_ready = isinstance(desc.get("decoded_mp4_blob"), dict)
            shift_ready = idx == 0 or "decoded_seam_shift" in desc

        # While the decoded RGB tensor is still resident, encode the exact final
        # segment directly. If the neutral preview was already cached, this helper
        # decodes only this one clip when its final sidecar is missing/dirty.
        if profile is not None:
            if ffmpeg is None:
                ffmpeg = _find_ffmpeg()
            manifest, _final_path, _final_decoded = _ensure_ref2va_final_segment_cache(
                data_path,
                manifest_path,
                manifest,
                idx,
                vae,
                float(fps),
                ffmpeg,
                profile,
                decoded_video=rendered_video,
                color_adjustment=adjustment,
            )
            segments = [dict(x) for x in manifest.get("segments", [])]
            desc = dict(segments[idx])

        # Release the large VideoVAE RGB tensor before AudioVAE work starts.
        if rendered_video is not None:
            del rendered_video
            rendered_video = None

        can_cache_audio = bool(audio_vae is not None)
        if can_cache_audio and idx > 0:
            previous_audio = _load_cached_decoded_audio(data_path, segments[idx - 1])
            can_cache_audio = previous_audio is not None
            if previous_audio is not None:
                del previous_audio
        if can_cache_audio and not audio_ready:
            rendered_audio = _render_one_final_audio_segment(
                data_path,
                segments,
                idx,
                audio_vae,
                float(fps),
                int(seam_shift),
            )
            manifest, desc = _cache_candidate_render(
                data_path,
                manifest_path,
                manifest,
                idx,
                rendered_audio=rendered_audio,
                seam_shift=int(seam_shift),
            )
            cached_audio = _load_cached_decoded_audio(data_path, desc)
            audio_ready = cached_audio is not None
            if cached_audio is not None:
                del cached_audio

        final_ready = False
        if profile is not None:
            latest_segments = [dict(x) for x in manifest.get("segments", [])]
            latest_desc = latest_segments[idx]
            final_path = _ref2va_final_segment_cache_path(data_path, idx, profile)
            final_ready = _final_segment_cache_meta_matches(
                latest_desc, profile, adjustment, final_path
            )

        return manifest, {
            "video_cached": bool(video_ready),
            "audio_cached": bool(audio_ready),
            "final_cached": bool(final_ready),
            "seam_shift": int(seam_shift),
            "deferred": False,
        }
    finally:
        if rendered_video is not None:
            del rendered_video
        if rendered_audio is not None:
            del rendered_audio
        try:
            temp_mp4.unlink(missing_ok=True)
        except OSError:
            pass


def _ensure_ref2va_audio_cache(
    data_path,
    manifest_path,
    manifest,
    vae,
    audio_vae,
    fps,
    count=None,
    progress=None,
):
    """Append only missing Ref2VA decoded PCM entries in timeline order.

    Fresh resumable Full Batch runs normally create these entries while each
    clip is completed. If ``audio_vae`` was not connected to the Extender, Final
    Decode fills only the missing PCM here. Persisted ``decoded_seam_shift``
    metadata means this path does not need VideoVAE again. Legacy clips without
    that metadata may require a one-time video seam decode to recover the exact
    shift.
    """
    if audio_vae is None:
        raise ValueError("H3 Final Decode: audio_vae is required to build the audio cache.")

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    full_segments = [dict(x) for x in manifest.get("segments", [])]
    target = len(full_segments) if count is None else max(0, min(int(count), len(full_segments)))
    if target <= 0:
        return manifest, []

    audio_path = _decoded_audio_cache_path(data_path)
    _ensure_audio_cache_file(audio_path)
    changed = False

    with open(audio_path, "ab", buffering=0) as acf:
        for i in range(target):
            desc = dict(full_segments[i])
            cached = _load_cached_decoded_audio(data_path, desc)
            if cached is not None:
                del cached
                continue

            if i == 0:
                seam_shift = 0
            elif "decoded_seam_shift" in desc:
                seam_shift = int(desc.get("decoded_seam_shift", 0) or 0)
            else:
                # Compatibility for old caches created before per-clip seam
                # shifts were persisted. Repair this clip only; never replay the
                # complete project through VideoVAE.
                _LOG.info(
                    "H3 incremental cache repair: recovering seam shift for Ref2VA clip %d only",
                    i + 1,
                )
                video, seam_shift = _render_one_final_video_segment(
                    data_path, full_segments, i, vae
                )
                del video
                desc["decoded_seam_shift"] = int(seam_shift)

            audio = _render_one_final_audio_segment(
                data_path,
                full_segments,
                i,
                audio_vae,
                float(fps),
                int(seam_shift),
                progress=progress,
            )
            desc["decoded_audio"] = _decoded_audio_meta_from_waveform(acf, audio)
            full_segments[i] = desc
            changed = True
            del audio

        if changed:
            acf.flush()
            os.fsync(acf.fileno())

    if changed:
        manifest = dict(manifest)
        manifest["segments"] = full_segments
        manifest["build"] = BUILD
        manifest["updated_at"] = time.time()
        _write_json_atomic(manifest_path, manifest)
    return manifest, [dict(x) for x in full_segments[:target]]

def _sync_committed_preview(
    data_path,
    manifest_path,
    manifest,
    target_count,
    vae,
    audio_vae,
    fps,
    ffmpeg,
    token,
):
    target = max(0, int(target_count))
    segments = [dict(x) for x in manifest.get("segments", [])]
    committed_path = _decoded_preview_cache_path(data_path)
    committed_video_path = _decoded_preview_video_cache_path(data_path)
    current_count = int(manifest.get("preview_committed_count", 0))
    current_audio_mode = str(manifest.get("preview_audio_mode", ""))

    if target <= 0:
        for p in (committed_path, committed_video_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        updated = dict(manifest)
        updated["preview_committed_count"] = 0
        updated.pop("preview_audio_mode", None)
        updated["build"] = BUILD
        updated["updated_at"] = time.time()
        _write_json_atomic(manifest_path, updated)
        return updated, committed_path, committed_video_path

    # Both persistent files are needed: the normal muxed preview for immediate
    # display, and a video-only prefix so future candidates can be appended
    # without any AAC/edit-list timestamps taking part in the video join.
    if (
        current_count == target
        and current_audio_mode == PREVIEW_AUDIO_MODE
        and committed_path.exists()
        and committed_video_path.exists()
    ):
        return manifest, committed_path, committed_video_path

    rebuild = (
        current_audio_mode != PREVIEW_AUDIO_MODE
        or current_count <= 0
        or current_count > target
        or not committed_video_path.exists()
    )
    start_i = 0 if rebuild else current_count
    video_inputs = [] if rebuild else [committed_video_path]
    temp_segments = []
    root = _ensure_cache_root()
    joined_video_tmp = committed_video_path.with_name(
        committed_video_path.stem + f"_{token}.tmp.mp4"
    )
    committed_tmp = committed_path.with_name(
        committed_path.stem + f"_{token}.tmp.mp4"
    )
    raw_audio = root / f"_{token}_commit_audio.f32le"
    video_log = root / f"_{token}_commit_video.log"
    mux_log = root / f"_{token}_commit_mux.log"

    try:
        for i in range(start_i, target):
            desc = segments[i]
            segment_tmp = root / f"_{token}_commit_{i:04d}.mp4"
            blob = desc.get("decoded_mp4_blob")
            if blob is not None:
                _copy_blob_to_file(data_path, blob, segment_tmp)
            else:
                # Full Batch does not need to keep duplicate per-clip decoded
                # video files. When entering Clip by Clip, bootstrap only the
                # required video prefix from latent cache. Decoded audio is
                # already present in the primary per-clip audio cache and is
                # never decoded again here.
                if _load_cached_decoded_audio(data_path, desc) is None:
                    raise RuntimeError(
                        "H3 progressive preview: decoded audio cache is missing."
                    )
                _LOG.info(
                    "H3 incremental cache repair: decoding missing Ref2VA clip %d only",
                    i + 1,
                )
                video, _ = _render_one_final_video_segment(
                    data_path,
                    segments,
                    i,
                    vae,
                )
                _encode_corrected_segment_video_mp4(
                    ffmpeg,
                    video,
                    fps,
                    segment_tmp,
                    f"{token}_bootstrap_{i}",
                )
                del video

            video_inputs.append(segment_tmp)
            temp_segments.append(segment_tmp)

        _concat_mp4_video_stream_copy(
            ffmpeg,
            video_inputs,
            joined_video_tmp,
            video_log,
        )

        sr, channels, _ = _write_preview_pcm_audio(
            ffmpeg,
            data_path,
            segments,
            target,
            fps,
            raw_audio,
            f"{token}_commit_pcm",
        )
        _mux_final(
            ffmpeg,
            joined_video_tmp,
            raw_audio,
            committed_tmp,
            sr,
            channels,
            "H.264",
            "192k",
            mux_log,
        )

        os.replace(joined_video_tmp, committed_video_path)
        os.replace(committed_tmp, committed_path)
    finally:
        for p in temp_segments:
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except OSError:
                pass
        for p in (joined_video_tmp, committed_tmp, raw_audio, video_log, mux_log):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except OSError:
                pass

    updated = dict(manifest)
    updated["preview_committed_count"] = target
    updated["preview_audio_mode"] = PREVIEW_AUDIO_MODE
    updated["build"] = BUILD
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated, committed_path, committed_video_path



def _export_live_candidate_preview(
    data_path,
    manifest_path,
    manifest,
    segments,
    vae,
    audio_vae,
    fps,
    ffmpeg,
    unique_id,
    progress=None,
    export_profile=None,
):
    segments = [dict(x) for x in segments]
    if not segments:
        raise ValueError("H3 progressive preview: empty chain.")

    validated_count = _validated_prefix_count(segments)
    token = f"v125_{_safe_name(unique_id)}_{uuid.uuid4().hex[:8]}"
    preview_path = None
    root = _ensure_cache_root()

    # Upgrade v14.42 lossless cached PCM once. This changes only small gain
    # metadata in the manifest; latents and validation states stay untouched.
    manifest = _upgrade_cached_audio_gain_chain(
        data_path, manifest_path, manifest, audio_vae, fps
    )
    segments = [dict(x) for x in manifest.get("segments", [])]

    # Commit already validated clips into ONE persistent full preview cache.
    manifest, committed_path, committed_video_path = _sync_committed_preview(
        data_path,
        manifest_path,
        manifest,
        validated_count,
        vae,
        audio_vae,
        fps,
        ffmpeg,
        token,
    )
    segments = [dict(x) for x in manifest.get("segments", [])]

    # Everything currently present is validated: show cache directly.
    if validated_count >= len(segments):
        if not committed_path.exists():
            raise RuntimeError(
                "H3 progressive preview: validated preview cache is missing."
            )
        preview_path = _reserve_preview_temp_path(unique_id)
        try:
            os.link(committed_path, preview_path)
        except Exception:
            shutil.copy2(committed_path, preview_path)

        # The Final Decode node has no tensor outputs. Once every clip is
        # validated the complete committed preview is already authoritative, so
        # decoding the last clip/pair merely to manufacture an unused last_frame
        # would waste a full VideoVAE pass and several GB of temporary RGB memory.
        return (
            preview_path,
            int(manifest.get("final_frame_count", 0)),
            0,
            None,
            "committed_full_cache",
        )

    candidate_index = validated_count
    if candidate_index != len(segments) - 1:
        raise RuntimeError(
            "H3 progressive preview expects one unvalidated tail candidate."
        )

    # Normal VAE cost:
    #   clip 1 alone, otherwise ONLY previous + current pair.
    # Decode/correct video first, encode it immediately, then release the
    # VideoVAE pair storage BEFORE AudioVAE starts. This prevents the two large
    # native workloads from
    # overlapping in Clip by Clip mode.
    current_video, seam_shift = _render_one_final_video_segment(
        data_path,
        segments,
        candidate_index,
        vae,
        progress=progress,
    )

    candidate_mp4 = root / f"_{token}_candidate.mp4"

    try:
        _encode_corrected_segment_video_mp4(
            ffmpeg,
            current_video,
            fps,
            candidate_mp4,
            token,
        )

        # Clip-by-Clip must leave behind the same exact-final sidecar that a
        # Full Batch would create. Encode it NOW from the already resident RGB
        # tensor, before releasing VideoVAE output. Switching a validated prefix
        # to Full Batch can then reuse those clips with zero VideoVAE work.
        if export_profile is not None:
            adjustment = _normalize_color_adjustment(
                segments[candidate_index].get("color_adjustment")
            )
            manifest, _final_path, _decoded_now = _ensure_ref2va_final_segment_cache(
                data_path,
                manifest_path,
                manifest,
                candidate_index,
                vae,
                float(fps),
                ffmpeg,
                export_profile,
                progress=None,
                decoded_video=current_video,
                color_adjustment=adjustment,
            )
            segments = [dict(x) for x in manifest.get("segments", [])]

        del current_video

        current_audio = _render_one_final_audio_segment(
            data_path,
            segments,
            candidate_index,
            audio_vae,
            fps,
            int(seam_shift),
            progress=progress,
        )

        # Save candidate's already decoded/corrected render in .h3cache.
        manifest, _ = _cache_candidate_render(
            data_path,
            manifest_path,
            manifest,
            candidate_index,
            candidate_mp4,
            current_audio,
            seam_shift=int(seam_shift),
        )

        preview_path = _reserve_preview_temp_path(unique_id)

        # FULL VIDEO preview: stream-copy only the H.264 video.  Audio is
        # rebuilt from lossless per-clip PCM and encoded ONCE for the complete
        # preview, so AAC priming/padding can never sit on an internal seam.
        video_inputs = (
            [committed_video_path, candidate_mp4]
            if committed_video_path.exists() and validated_count > 0
            else [candidate_mp4]
        )
        segments = [dict(x) for x in manifest.get("segments", [])]
        _assemble_progressive_preview(
            ffmpeg,
            video_inputs,
            data_path,
            segments,
            len(segments),
            fps,
            preview_path,
            f"{token}_candidate_full",
        )

        total_frames = int(manifest.get("final_frame_count", 0))

        _LOG.info(
            "H3 v12.5 FULL preview: cached validated clips=%d, candidate=%d, "
            "shift=%d; VAE decoded only %s",
            validated_count,
            candidate_index + 1,
            int(seam_shift),
            (
                "clip 1"
                if candidate_index == 0
                else f"pair {candidate_index}->{candidate_index + 1}"
            ),
        )

        return (
            preview_path,
            total_frames,
            int(seam_shift),
            None if candidate_index == 0 else int(candidate_index),
            "full_cached_prefix_plus_candidate",
        )

    finally:
        for p in (candidate_mp4,):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass



def _restore_cached_preview_without_decode(owner_id, final_id, generation_mode="ref2va"):
    """
    Rebuild the current full preview using ONLY already cached decoded MP4 blobs.

    This is used when ComfyUI starts and the workflow is restored. No sampler,
    video VAE or audio VAE is executed. The cache owner is derived from the
    upstream MiniMax H3 Extender node id.
    """
    owner = _safe_name(owner_id)
    final = _safe_name(final_id)

    if str(generation_mode or "ref2va").lower() == "fl2va":
        from .fl2va_engine import cache_owner_id
        cache_owner = cache_owner_id(owner_id)
    else:
        cache_owner = f"extender_{owner}"
    data_path, manifest_path = _chain_paths(cache_owner)
    if not data_path.exists() or not manifest_path.exists():
        return None

    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        return None

    requested_mode = "fl2va" if str(generation_mode or "ref2va").lower() == "fl2va" else "ref2va"
    cached_mode = "fl2va" if str(manifest.get("sequence_mode") or "ref2va").lower() == "fl2va" else "ref2va"
    if cached_mode != requested_mode:
        _LOG.warning(
            "H3 restore preview refused mode mismatch: requested=%s cached=%s owner=%s",
            requested_mode, cached_mode, owner_id,
        )
        return None

    segments = [dict(x) for x in manifest.get("segments", [])]
    if not segments:
        return None
    project_total_clips = int(manifest.get("batch_total_clips", len(segments)) or len(segments))

    # A cooperative Full Batch stop owns an immutable preview snapshot.  FL2VA
    # may still have older cached plans after that prefix, so startup/project
    # restore must publish exactly the prefix decoded at Stop rather than
    # silently rebuilding a longer preview from unrelated old plan caches.
    interrupted_snapshot = bool(manifest.get("batch_interrupted", False))
    if interrupted_snapshot:
        snapshot_count = int(manifest.get("batch_snapshot_count", len(segments)) or 0)
        snapshot_count = max(1, min(len(segments), snapshot_count))
        segments = segments[:snapshot_count]

    preview_path = None
    committed_path = _decoded_preview_cache_path(data_path)
    committed_video_path = _decoded_preview_video_cache_path(data_path)
    committed_count = int(manifest.get("preview_committed_count", 0))

    # Fastest path: the persistent committed preview already contains every
    # cached clip. Just republish it to ComfyUI temp for /view.
    if (
        committed_count >= len(segments)
        and committed_path.exists()
        and str(manifest.get("preview_audio_mode", "")) == PREVIEW_AUDIO_MODE
    ):
        preview_path = _reserve_preview_temp_path(final_id)
        try:
            os.link(committed_path, preview_path)
        except Exception:
            shutil.copy2(committed_path, preview_path)

        return {
            "path": preview_path,
            "clip_count": int(len(segments)),
            "frame_count": int(_final_frame_count(segments)),
            "fps": float(manifest.get("fps", FPS)),
            "cache_mode": "committed_preview",
            "interrupted": bool(interrupted_snapshot),
            "project_total_clips": int(project_total_clips),
        }

    # Otherwise rebuild the full current preview from the per-clip decoded video
    # blobs plus cached PCM from the primary decoded-audio cache. Video is stream-copied
    # and audio gets one cheap AAC encode; no VAE/sampler runs on application load.
    # It remains available even if the last clip was only a candidate when the
    # application was closed.
    root = _ensure_cache_root()
    token = f"restore_{final}_{uuid.uuid4().hex[:8]}"
    segment_files = []
    temp_preview = root / f"_{token}.mp4"

    try:
        video_inputs = []
        start_i = 0
        if (
            committed_count > 0
            and committed_count < len(segments)
            and committed_video_path.exists()
        ):
            video_inputs.append(committed_video_path)
            start_i = committed_count

        for i in range(start_i, len(segments)):
            desc = segments[i]
            blob = desc.get("decoded_mp4_blob")
            if blob is None:
                # Old cache created before decoded candidate blobs existed:
                # restoring it would require a VAE decode, which must never
                # happen automatically just by opening ComfyUI.
                return None

            segment_path = root / f"_{token}_{i:04d}.mp4"
            _copy_blob_to_file(data_path, blob, segment_path)
            segment_files.append(segment_path)
            video_inputs.append(segment_path)

        ffmpeg = _find_ffmpeg()
        _assemble_progressive_preview(
            ffmpeg,
            video_inputs,
            data_path,
            segments,
            len(segments),
            float(manifest.get("fps", FPS)),
            temp_preview,
            token,
        )

        preview_path = _reserve_preview_temp_path(final_id)
        os.replace(temp_preview, preview_path)

        return {
            "path": preview_path,
            "clip_count": int(len(segments)),
            "frame_count": int(_final_frame_count(segments)),
            "fps": float(manifest.get("fps", FPS)),
            "cache_mode": "decoded_blobs",
            "interrupted": bool(interrupted_snapshot),
            "project_total_clips": int(project_total_clips),
        }
    finally:
        for tmp in segment_files:
            try:
                if Path(tmp).exists():
                    Path(tmp).unlink()
            except OSError:
                pass
        for tmp in (temp_preview,):
            try:
                if Path(tmp).exists():
                    Path(tmp).unlink()
            except OSError:
                pass


if web is not None and PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.post("/h3_extender/full_batch_interrupt")
    async def h3_extender_full_batch_interrupt(request):
        """Request a cooperative stop after the currently rendering clip."""
        try:
            body = await request.json()
            owner_id = str(body.get("owner_id") or "").strip()
            generation_mode = str(body.get("generation_mode") or "ref2va").lower()
            if not owner_id:
                return web.json_response({"ok": False, "error": "Missing owner id."}, status=400)
            request_full_batch_interrupt(owner_id, generation_mode)
            return web.json_response({"ok": True, "pending": True})
        except Exception as exc:
            _LOG.exception("H3 full-batch interrupt request failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/h3_extender/discard_computed")
    async def h3_extender_discard_computed(request):
        """Discard one resumable Full-Batch checkpoint without touching preview.

        Ref2VA is causal, so discarding clip N truncates N and every following
        cached clip. FL2VA is random-access except for explicit Previous links:
        discarding a plan immediately discards each consecutive cached follower
        whose First frame depends on the preceding plan. This keeps COMPUTED UI
        state truthful before the next execution instead of invalidating a hidden
        dependency only after its predecessor has been rerendered.
        """
        try:
            body = await request.json()
            owner_id = str(body.get("owner_id") or "").strip()
            generation_mode = str(body.get("generation_mode") or "ref2va").lower()
            if not owner_id:
                return web.json_response({"ok": False, "error": "Missing owner id."}, status=400)

            if generation_mode == "fl2va":
                clip_id = str(body.get("clip_id") or "").strip()
                clip_ids = [str(x) for x in (body.get("clip_ids") or []) if str(x)]
                if not clip_id or not clip_ids:
                    return web.json_response({"ok": False, "error": "Missing FL2VA clip id/order."}, status=400)
                from .fl2va_engine import cache_owner_id, drop_fl2va_cached_ids
                cache_owner = cache_owner_id(owner_id)
                data_path, manifest_path = _chain_paths(cache_owner)
                manifest = _load_manifest_from_paths(data_path, manifest_path)
                if manifest is None:
                    return web.json_response({"ok": False, "error": "No FL2VA cache found."}, status=404)
                target = next(
                    (dict(x) for x in manifest.get("segments", []) if str(x.get("clip_id") or "") == clip_id),
                    None,
                )
                if target is None or not bool(target.get("computed", False)) or bool(target.get("validated", False)):
                    return web.json_response({"ok": False, "error": "This FL2VA clip is not a discardable computed checkpoint."}, status=400)
                # FL2VA plans are independent unless a follower explicitly
                # starts from Previous. If clip N is going to change, every
                # consecutive cached Previous-linked follower already has stale
                # conditioning and must stop advertising itself as COMPUTED now,
                # not only after N has been rerendered.
                segments_before = [dict(x) for x in manifest.get("segments", [])]
                by_id = {
                    str(x.get("clip_id") or ""): x
                    for x in segments_before
                    if str(x.get("clip_id") or "")
                }
                drop_ids = [clip_id]
                try:
                    start_pos = clip_ids.index(clip_id)
                except ValueError:
                    start_pos = -1
                previous_id = clip_id
                if start_pos >= 0:
                    for follower_id in clip_ids[start_pos + 1:]:
                        follower = by_id.get(str(follower_id))
                        if follower is None:
                            break
                        if (
                            str(follower.get("first_source") or "manual") == "previous_clip"
                            and str(follower.get("previous_clip_id") or "") == str(previous_id)
                        ):
                            drop_ids.append(str(follower_id))
                            previous_id = str(follower_id)
                            continue
                        break

                data_path, manifest_path, manifest = drop_fl2va_cached_ids(
                    owner_id, float(manifest.get("fps", FPS)), clip_ids, drop_ids,
                    preserve_preview=True,
                )
                segments = [dict(x) for x in manifest.get("segments", [])]
                computed_clip_ids = [
                    str(x.get("clip_id")) for x in segments
                    if str(x.get("clip_id") or "") and bool(x.get("computed", False)) and not bool(x.get("validated", False))
                ]
                validated_clip_ids = [
                    str(x.get("clip_id")) for x in segments
                    if str(x.get("clip_id") or "") and bool(x.get("validated", False))
                ]
                return web.json_response({
                    "ok": True,
                    "generation_mode": "fl2va",
                    "cached_count": len(segments),
                    "validated_count": len(validated_clip_ids),
                    "cached_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "")],
                    "validated_clip_ids": validated_clip_ids,
                    "computed_clip_ids": computed_clip_ids,
                    "computed_indices": [],
                    "discarded_clip_ids": [str(x) for x in drop_ids],
                    "checkpoint_active": bool(manifest.get("batch_in_progress", False) or manifest.get("batch_interrupted", False)),
                    "checkpoint_interrupted": bool(manifest.get("batch_interrupted", False)),
                    "checkpoint_snapshot_count": int(manifest.get("batch_snapshot_count", 0) or 0),
                })

            # Ref2VA: physical cache is a causal prefix.
            idx = int(body.get("clip_index"))
            cache_owner = f"extender_{_safe_name(owner_id)}"
            data_path, manifest_path = _chain_paths(cache_owner)
            manifest = _load_manifest_from_paths(data_path, manifest_path)
            if manifest is None:
                return web.json_response({"ok": False, "error": "No Ref2VA cache found."}, status=404)
            segments = [dict(x) for x in manifest.get("segments", [])]
            if idx < 0 or idx >= len(segments):
                return web.json_response({"ok": False, "error": "This clip is not cached."}, status=400)
            target = segments[idx]
            if not bool(target.get("computed", False)) or bool(target.get("validated", False)):
                return web.json_response({"ok": False, "error": "This Ref2VA clip is not a discardable computed checkpoint."}, status=400)
            manifest = _truncate_chain(data_path, manifest_path, manifest, idx)
            segments = [dict(x) for x in manifest.get("segments", [])]
            validated_count = _validated_prefix_count(segments)
            computed_indices = [
                i for i, x in enumerate(segments)
                if bool(x.get("computed", False)) and not bool(x.get("validated", False))
            ]
            return web.json_response({
                "ok": True,
                "generation_mode": "ref2va",
                "cached_count": len(segments),
                "validated_count": int(validated_count),
                "cached_clip_ids": [],
                "validated_clip_ids": [],
                "computed_clip_ids": [],
                "computed_indices": computed_indices,
                "checkpoint_active": bool(manifest.get("batch_in_progress", False) or manifest.get("batch_interrupted", False)),
                "checkpoint_interrupted": bool(manifest.get("batch_interrupted", False)),
                "checkpoint_snapshot_count": int(manifest.get("batch_snapshot_count", 0) or 0),
            })
        except Exception as exc:
            _LOG.exception("H3 discard computed checkpoint failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/h3_extender/color_editor_info")
    async def h3_extender_color_editor_info(request):
        owner_id = request.query.get("owner_id", "")
        final_id = request.query.get("final_id", "")
        clip_index = request.query.get("clip_index", "")
        if not owner_id or not final_id or clip_index == "":
            return web.json_response({"ok": False, "error": "Missing owner/final/clip id."}, status=400)
        try:
            idx = int(clip_index)
            generation_mode = str(request.query.get("mode") or "ref2va").lower()
            if generation_mode == "fl2va":
                from .fl2va_engine import cache_owner_id
                cache_owner = cache_owner_id(owner_id)
            else:
                cache_owner = f"extender_{_safe_name(owner_id)}"
            data_path, manifest_path = _chain_paths(cache_owner)
            manifest = _load_manifest_from_paths(data_path, manifest_path)
            if manifest is None:
                return web.json_response({"ok": False, "error": "No cached H3 sequence found."}, status=404)
            segments = [dict(x) for x in manifest.get("segments", [])]
            if idx < 0 or idx >= len(segments):
                return web.json_response({"ok": False, "error": "This clip has not been rendered yet."}, status=400)
            preview_path = _latest_preview_temp_path(final_id)
            if preview_path is None or not preview_path.exists():
                return web.json_response({
                    "ok": False,
                    "error": "No decoded preview is available yet. Run Final Decode/Preview once first.",
                }, status=404)
            timeline = _color_timeline(segments, float(manifest.get("fps", FPS)))
            return web.json_response({
                "ok": True,
                "video": _comfy_media_item(preview_path, float(manifest.get("fps", FPS)), "temp"),
                "timeline": timeline,
                "clip_index": idx,
                "total_clips": len(segments),
            })
        except Exception as exc:
            _LOG.exception("H3 color editor info failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/h3_extender/color_adjust")
    async def h3_extender_color_adjust(request):
        try:
            body = await request.json()
            owner_id = str(body.get("owner_id") or "")
            idx = int(body.get("clip_index"))
            if not owner_id:
                return web.json_response({"ok": False, "error": "Missing owner id."}, status=400)
            generation_mode = str(body.get("generation_mode") or "ref2va").lower()
            if generation_mode == "fl2va":
                from .fl2va_engine import cache_owner_id
                cache_owner = cache_owner_id(owner_id)
            else:
                cache_owner = f"extender_{_safe_name(owner_id)}"
            data_path, manifest_path = _chain_paths(cache_owner)
            manifest = _load_manifest_from_paths(data_path, manifest_path)
            if manifest is None:
                return web.json_response({"ok": False, "error": "No cached H3 sequence found."}, status=404)
            segments = [dict(x) for x in manifest.get("segments", [])]
            if idx < 0 or idx >= len(segments):
                return web.json_response({"ok": False, "error": "This clip has not been rendered yet."}, status=400)
            adjustment = _normalize_color_adjustment(body.get("adjustment"))
            desc = dict(segments[idx])
            desc["color_adjustment"] = adjustment
            desc["final_video_dirty"] = True
            segments[idx] = desc
            manifest = dict(manifest)
            manifest["segments"] = segments
            manifest["updated_at"] = time.time()
            _write_json_atomic(manifest_path, manifest)
            timeline = _color_timeline(segments, float(manifest.get("fps", FPS)))
            return web.json_response({
                "ok": True,
                "adjustment": adjustment,
                "modified": not _color_is_neutral(adjustment),
                "timeline": timeline,
            })
        except Exception as exc:
            _LOG.exception("H3 color adjustment failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/h3_extender/save_preview")
    async def h3_extender_save_preview(request):
        """Save only the currently assembled Final Decode preview to output."""
        try:
            body = await request.json()
            owner_id = str(body.get("owner_id") or "").strip()
            filename = str(body.get("filename") or "").strip()
            media_type = str(body.get("type") or "temp").strip()
            subfolder = str(body.get("subfolder") or "").strip()

            if media_type != "temp" or subfolder not in ("", "."):
                return web.json_response(
                    {"ok": False, "error": "Save Preview only accepts the Extender temp preview."},
                    status=400,
                )
            if Path(filename).name != filename or not re.fullmatch(
                r"h3_motion_preview_[A-Za-z0-9._-]+_[0-2]\.mp4", filename
            ):
                return web.json_response(
                    {"ok": False, "error": "Invalid H3 preview filename."}, status=400
                )

            source = (_preview_temp_root() / filename).resolve()
            if source.parent != _preview_temp_root().resolve():
                return web.json_response(
                    {"ok": False, "error": "Invalid H3 preview path."}, status=400
                )
            if not source.exists():
                return web.json_response(
                    {"ok": False, "error": "The currently displayed preview no longer exists."},
                    status=404,
                )

            workflow = body.get("workflow")
            prompt = body.get("prompt")
            color_timeline = None
            if owner_id:
                try:
                    generation_mode = str(body.get("generation_mode") or "ref2va").lower()
                    if generation_mode == "fl2va":
                        from .fl2va_engine import cache_owner_id
                        cache_owner = cache_owner_id(owner_id)
                    else:
                        cache_owner = f"extender_{_safe_name(owner_id)}"
                    data_path, manifest_path = _chain_paths(cache_owner)
                    manifest = _load_manifest_from_paths(data_path, manifest_path)
                    if manifest is not None:
                        color_timeline = _color_timeline(
                            manifest.get("segments", []), float(manifest.get("fps", FPS))
                        )
                except Exception:
                    color_timeline = None
            output = await asyncio.to_thread(
                _save_preview_with_metadata,
                source,
                workflow,
                prompt,
                color_timeline,
            )
            return web.json_response({
                "ok": True,
                "video": _comfy_media_item(output, float(body.get("fps") or FPS), "output"),
                "filename": output.name,
            })
        except Exception as exc:
            _LOG.exception("H3 Save Preview failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/h3_extender/cache_state")
    async def h3_extender_cache_state(request):
        """Restore Extender card cache/validation UI state without execution."""
        owner_id = request.query.get("owner_id", "")
        if not owner_id:
            return web.json_response({"found": False, "reason": "missing_id"})

        try:
            generation_mode = str(request.query.get("mode") or "ref2va").lower()
            if generation_mode == "fl2va":
                from .fl2va_engine import cache_owner_id
                cache_owner = cache_owner_id(owner_id)
            else:
                cache_owner = f"extender_{_safe_name(owner_id)}"
            data_path, manifest_path = _chain_paths(cache_owner)
            if not data_path.exists() or not manifest_path.exists():
                return web.json_response({"found": False})

            manifest = _load_manifest_from_paths(data_path, manifest_path)
            if manifest is None:
                return web.json_response({"found": False})

            segments = [dict(x) for x in manifest.get("segments", [])]
            is_fl2va = str(manifest.get("sequence_mode") or generation_mode).lower() == "fl2va"
            validated_count = (
                sum(bool(x.get("validated", False)) for x in segments)
                if is_fl2va else _validated_prefix_count(segments)
            )
            geometry = manifest.get("geometry") if isinstance(manifest.get("geometry"), dict) else {}
            resolved_width = int(geometry.get("video_w", 0) or 0) * 16
            resolved_height = int(geometry.get("video_h", 0) or 0) * 16
            continuity_signatures = {}
            if is_fl2va:
                from .fl2va_engine import continuity_signatures_for_segments
                continuity_signatures = continuity_signatures_for_segments(data_path, segments)
            computed_indices = [
                i for i, x in enumerate(segments)
                if bool(x.get("computed", False)) and not bool(x.get("validated", False))
            ]
            computed_clip_ids = [
                str(x.get("clip_id")) for x in segments
                if str(x.get("clip_id") or "") and bool(x.get("computed", False)) and not bool(x.get("validated", False))
            ]
            return web.json_response({
                "found": True,
                "generation_mode": "fl2va" if is_fl2va else "ref2va",
                "cached_count": int(len(segments)),
                "validated_count": int(validated_count),
                "cached_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "")],
                "validated_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "") and bool(x.get("validated", False))],
                "continuity_signatures": continuity_signatures,
                "computed_indices": computed_indices,
                "computed_clip_ids": computed_clip_ids,
                "checkpoint_active": bool(
                    manifest.get("batch_in_progress", False)
                    or manifest.get("batch_interrupted", False)
                ),
                "checkpoint_interrupted": bool(manifest.get("batch_interrupted", False)),
                "checkpoint_snapshot_count": int(manifest.get("batch_snapshot_count", 0) or 0),
                "frame_count": int(manifest.get("final_frame_count", 0)),
                "resolved_width": int(resolved_width),
                "resolved_height": int(resolved_height),
            })
        except Exception as exc:
            _LOG.warning("H3 restore Extender cache state failed: %s", exc)
            return web.json_response({
                "found": False,
                "reason": "restore_failed",
            })

    @PromptServer.instance.routes.get("/h3_extender/restored_preview")
    async def h3_extender_restored_preview(request):
        """
        Frontend startup helper. It exposes only the deterministic cache belonging
        to an Extender node id; it never accepts an arbitrary filesystem path.
        """
        owner_id = request.query.get("owner_id", "")
        final_id = request.query.get("final_id", "")
        if not owner_id or not final_id:
            return web.json_response({"found": False, "reason": "missing_id"})

        try:
            generation_mode = str(request.query.get("mode") or "ref2va").lower()
            restored = _restore_cached_preview_without_decode(owner_id, final_id, generation_mode)
            if restored is None:
                return web.json_response({"found": False})

            item = _comfy_media_item(
                restored["path"],
                restored["fps"],
                "temp",
            )
            if generation_mode == "fl2va":
                from .fl2va_engine import cache_owner_id
                cache_owner = cache_owner_id(owner_id)
            else:
                cache_owner = f"extender_{_safe_name(owner_id)}"
            data_path, manifest_path = _chain_paths(cache_owner)
            manifest = _load_manifest_from_paths(data_path, manifest_path)
            restored_segments = list(manifest.get("segments", []) if manifest else [])[:int(restored["clip_count"])]
            color_timeline = _color_timeline(
                restored_segments,
                float(manifest.get("fps", FPS)) if manifest else FPS,
            )
            return web.json_response({
                "found": True,
                "video": item,
                "clip_count": restored["clip_count"],
                "frame_count": restored["frame_count"],
                "cache_mode": restored["cache_mode"],
                "interrupted": bool(restored.get("interrupted", False)),
                "project_total_clips": int(restored.get("project_total_clips", restored["clip_count"])),
                "color_timeline": color_timeline,
            })
        except Exception as exc:
            _LOG.warning("H3 restore preview on load failed: %s", exc)
            return web.json_response({
                "found": False,
                "reason": "restore_failed",
            })



class MiniMaxH3MotionContextDiskFinalDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": (CACHE_TYPE,),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                # Compatibility-only input. The frontend hides it and export()
                # ignores its value; keeping its original position prevents old
                # workflow widgets_values arrays from shifting every later field.
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "filename_prefix": ("STRING", {"default": "MiniMax_H3_cached"}),
                "output_directory": ("STRING", {"default": ""}),
                "codec": (["H.264", "H.264 CPU (libx264)", "H.265 / HEVC", "FFV1 lossless"], {
                    "default": "H.264",
                    "tooltip": "H.264 automatically uses NVIDIA NVENC hardware encoding when available, with transparent libx264 CPU fallback. Choose H.264 CPU (libx264) to force software encoding.",
                }),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"], {"default": "fast"}),
                "audio_bitrate": (["128k", "192k", "256k", "320k"], {"default": "192k"}),
                "autoplay": ("BOOLEAN", {"default": True, "tooltip": "Auto-play the video preview when generating finishes or the node is loaded."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "export"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = True

    def export(
        self,
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
        autoplay=True,
        unique_id=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        data_path, manifest_path, manifest = _load_manifest(cache)
        # FPS is cache metadata, never a user choice. The compatibility widget
        # value above is deliberately ignored so old workflows keep their widget
        # positions without being able to alter H3 timing.
        fps = float(manifest.get("fps", FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"Disk Final Decode: invalid cached fps {fps!r}.")
        workflow = _workflow_from_extra_pnginfo(extra_pnginfo)
        segments = [dict(x) for x in manifest.get("segments", [])]
        if not segments:
            raise ValueError("Disk Final Decode: empty cache.")
        interrupted = bool(cache.get("interrupted", False)) if isinstance(cache, dict) else False
        project_total_clips = int(cache.get("project_total_clips", len(segments))) if isinstance(cache, dict) else len(segments)
        snapshot_count = int(cache.get("snapshot_count", len(segments))) if isinstance(cache, dict) else len(segments)
        if interrupted:
            snapshot_count = max(1, min(len(segments), snapshot_count))
            segments = segments[:snapshot_count]
        if str(manifest.get("sequence_mode") or "ref2va").lower() == "fl2va":
            from .fl2va_engine import export_fl2va_final
            return export_fl2va_final(
                cache=cache, vae=vae, audio_vae=audio_vae, fps=fps,
                filename_prefix=filename_prefix, output_directory=output_directory,
                codec=codec, crf=crf, preset=preset, audio_bitrate=audio_bitrate,
                unique_id=unique_id, workflow=workflow, prompt=prompt,
            )
        color_timeline = _color_timeline(segments, float(fps))

        ffmpeg = _find_ffmpeg()

        if str(output_directory).strip():
            out_dir = Path(str(output_directory).strip()).expanduser().resolve()
        elif folder_paths is not None:
            out_dir = Path(folder_paths.get_output_directory()).resolve()
        else:
            out_dir = (Path.cwd() / "output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # clip_by_clip keeps its fast progressive-preview path, but now also
        # persists that complete current sequence after every rendered clip.
        # The already encoded preview is reused directly: no extra sampling and
        # no second VAE decode are performed for the autosave.
        effective_mode = str(cache.get("run_mode", "full_batch")) if isinstance(cache, dict) else "full_batch"
        if effective_mode == "clip_by_clip":
            progress = _FinalDecodeNativeProgress(unique_id, total=6)
            clip_by_clip_export_profile = normalize_full_batch_export_profile({
                "codec": codec, "crf": crf, "preset": preset,
            })
            (
                preview_path,
                preview_frames,
                seam_shift,
                previous_clip,
                preview_cache_mode,
            ) = _export_live_candidate_preview(
                data_path=data_path,
                manifest_path=manifest_path,
                manifest=manifest,
                segments=segments,
                vae=vae,
                audio_vae=audio_vae,
                fps=float(fps),
                ffmpeg=ffmpeg,
                unique_id=unique_id,
                progress=progress,
                export_profile=clip_by_clip_export_profile,
            )
            progress.advance()  # preview encode/cache/concat completed

            # Keep one continuously updated real file in the requested output
            # directory.  Re-rendering the same candidate replaces it atomically,
            # so clip-by-clip testing never creates a pile of numbered files.
            autosave_path = _replace_output_from_preview(
                preview_path, out_dir, filename_prefix,
                ffmpeg=ffmpeg, color_timeline=color_timeline,
            )
            _embed_final_metadata_in_place(autosave_path, workflow=workflow, prompt=prompt)
            progress.advance()

            total_frames = int(manifest.get("final_frame_count", 0))
            total_duration = float(total_frames / float(fps))
            size = _cache_size_mb(data_path, manifest_path)
            item = _comfy_media_item(preview_path, fps, "temp")
            progress.finish()
            status_shift = (
                f"full_preview_cached_{len(segments)}_clips_shift_{int(seam_shift)}"
            )

            return {
                "ui": {
                    "h3_video": [item],
                    "h3_preview_info": [{
                        "mode": "clip_by_clip",
                        "clip": int(len(segments)),
                        "previous_clip": (
                            None if previous_clip is None else int(previous_clip)
                        ),
                        "seam_shift": int(seam_shift),
                        "cache_mode": str(preview_cache_mode),
                        "preview_frames": int(preview_frames),
                        "total_clips": int(len(segments)),
                        "autosave_path": str(autosave_path),
                        "color_timeline": color_timeline,
                        "color_preview_baked": False,
                    }],
                },
                "result": (_video_output_from_path(autosave_path),),
            }

        # Full Batch is strictly incremental and keeps a separate exact-final
        # video sidecar per clip. CRF/preset/codec are frozen when the Extender
        # starts the batch; Final Decode never transcodes an already compressed
        # cache and never has a project-wide VideoVAE path.
        progress = _FinalDecodeNativeProgress(
            unique_id, total=max(8, 5 + (2 * len(segments)))
        )
        requested_profile = normalize_full_batch_export_profile({
            "codec": codec, "crf": crf, "preset": preset,
        })
        manifest, export_profile = _resolve_full_batch_export_profile(
            manifest_path, manifest, requested_profile, context="H3 Final Decode"
        )

        manifest, segments = _ensure_ref2va_audio_cache(
            data_path,
            manifest_path,
            manifest,
            vae,
            audio_vae,
            float(fps),
            count=len(segments),
            progress=progress,
        )
        if interrupted:
            segments = [dict(x) for x in segments[:snapshot_count]]
        color_timeline = _color_timeline(segments, float(fps))
        token = f"full_exact_{_safe_name(unique_id)}_{uuid.uuid4().hex[:8]}"

        # Ensure only missing/dirty exact-final clips. Fresh v2.5.8 Full Batch
        # runs already created all these files upstream from the first VAE decode,
        # so this loop normally performs zero VideoVAE work.
        exact_segment_paths = []
        for i in range(len(segments)):
            manifest, final_segment_path, _decoded_now = _ensure_ref2va_final_segment_cache(
                data_path,
                manifest_path,
                manifest,
                i,
                vae,
                float(fps),
                ffmpeg,
                export_profile,
                progress=progress,
            )
            exact_segment_paths.append(final_segment_path)
        all_manifest_segments = [dict(x) for x in manifest.get("segments", [])]
        segments = all_manifest_segments[:len(segments)]
        color_timeline = _color_timeline(segments, float(fps))

        # Browser preview remains neutral H.264 and independent of final quality.
        # Its assembly is also video stream-copy from the preview checkpoints.
        manifest, committed_path, _committed_video_path = _sync_committed_preview(
            data_path,
            manifest_path,
            manifest,
            len(segments),
            vae,
            audio_vae,
            float(fps),
            ffmpeg,
            token,
        )
        progress.advance()
        preview_path = _publish_full_preview(committed_path, unique_id)
        expected_frames = int(_final_frame_count(segments))

        extension = _full_batch_export_profile_extension(export_profile)
        output_path = _next_output_path(out_dir, filename_prefix, extension)
        final_video_mode = _export_final_from_exact_segment_caches(
            ffmpeg=ffmpeg,
            segment_paths=exact_segment_paths,
            data_path=data_path,
            segments=segments,
            fps=float(fps),
            output_path=output_path,
            export_profile=export_profile,
            audio_bitrate=audio_bitrate,
            token=token,
        )

        _embed_final_metadata_in_place(output_path, workflow=workflow, prompt=prompt)
        progress.advance()

        _LOG.info(
            "H3 incremental Full Decode: clips=%d frames=%d interrupted=%s video=%s output=%s",
            len(segments), expected_frames, bool(interrupted), final_video_mode, output_path,
        )
        item = _comfy_media_item(preview_path, fps, "temp")
        progress.finish()
        return {
            "ui": {
                "h3_video": [item],
                "h3_preview_info": [{
                    "mode": "full_batch_incremental",
                    "clip": int(len(segments)),
                    "preview_frames": int(expected_frames),
                    "total_clips": int(project_total_clips if interrupted else len(segments)),
                    "preview_clips": int(len(segments)),
                    "interrupted": bool(interrupted),
                    "cache_mode": "decoded_segments_incremental",
                    "final_video_mode": str(final_video_mode),
                    "color_timeline": color_timeline,
                    "color_preview_baked": False,
                }],
            },
            "result": (_video_output_from_path(output_path),),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContextDiskJoin": MiniMaxH3MotionContextDiskJoin,
    "MiniMaxH3MotionContextDiskFinalDecode": MiniMaxH3MotionContextDiskFinalDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContextDiskJoin": "MiniMax H3 Motion Context Disk Join",
    "MiniMaxH3MotionContextDiskFinalDecode": "MiniMax H3 Extender Final Decode / Preview",
}
