"""Optional RelayVSR x2 post-process for a COMPLETED H3 Final Decode video.

This module uses ComfyUI core's RelayVSR model, KSampler and Flash VAE.
No H3 latents, temporal context or Final Decode caches are touched.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

_LOG = logging.getLogger(__name__)
_REPO = "Comfy-Org/RelayVSR"
_FILES = {
    "diffusion_models": "relayvsr_x2_lightx2v_int8_convrot.safetensors",
    "vae": "relayvsr_flash_vae_x2_fp32.safetensors",
}
_CACHE_FORMAT = 1


def free_vram_for_relay():
    """Free H3/VideoVAE GPU allocations BEFORE loading the RelayVSR network."""
    import comfy.model_management as mm

    # Comfy's model manager owns dynamically loaded model weights; never move
    # models manually out from underneath it or destroy live project caches.
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _find_or_download(kind):
    import folder_paths

    name = _FILES[kind]
    for base in folder_paths.get_folder_paths(kind):
        path = Path(base) / name
        if path.is_file() and path.stat().st_size:
            return path
    # Also reuse a model installed under a registered subfolder (often
    # `RelayVSR/`), not just ComfyUI's root models/diffusion_models or vae.
    for registered in folder_paths.get_filename_list(kind):
        if Path(registered).name.casefold() != name.casefold():
            continue
        found = folder_paths.get_full_path(kind, registered)
        if found and Path(found).is_file() and Path(found).stat().st_size:
            return Path(found)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "RelayVSR requires huggingface-hub to auto-download missing weights. "
            "Install the extension requirements, then try again."
        ) from exc

    roots = folder_paths.get_folder_paths(kind)
    if not roots:
        raise RuntimeError(f"RelayVSR: no registered ComfyUI model folder for {kind}.")
    target = Path(roots[0]) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    _LOG.info("RelayVSR x2: first-use download %s from %s", name, _REPO)
    # Hugging Face uses incomplete files and atomic moves internally. It also
    # reuses an existing completed download rather than downloading it twice.
    downloaded = Path(hf_hub_download(repo_id=_REPO, filename=f"{kind}/{name}", local_dir=str(target.parent.parent)))
    if downloaded != target:
        staged = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
        try:
            shutil.copy2(downloaded, staged)
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"RelayVSR: downloaded model is missing or empty: {target}")
    return target


def _ffprobe_media(ffmpeg, video):
    # ComfyUI normally has OpenCV; use it for metadata so Windows portable does
    # not need an extra ffprobe binary just for RelayVSR.
    try:
        import cv2
        capture = cv2.VideoCapture(str(video))
        try:
            if capture.isOpened():
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if min(width, height, frames) > 0 and fps > 0:
                    return width, height, frames, fps
        finally:
            capture.release()
    except ImportError:
        pass

    probe = str(Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else "")))
    if not Path(probe).is_file():
        probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError(
            "RelayVSR could not read video dimensions via OpenCV; install ffprobe or OpenCV."
        )
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames,nb_frames,r_frame_rate",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if min(width, height, count) < 1:
        raise RuntimeError("RelayVSR could not determine the source geometry/frame count.")
    numerator, denominator = (stream.get("r_frame_rate", "24/1").split("/") + ["1"])[:2]
    return width, height, count, float(numerator) / float(denominator)


def _read_video_frames(ffmpeg, source, width, height, count):
    """Decode the final movie ONCE, using the normal native IMAGE batch layout."""
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    frame_nbytes = width * height * 3
    frames = np.empty((count, height, width, 3), dtype=np.uint8)
    try:
        for i in range(count):
            view = memoryview(frames[i]).cast("B")
            offset = 0
            while offset < frame_nbytes:
                got = proc.stdout.readinto(view[offset:])
                if not got:
                    raise RuntimeError(f"RelayVSR video decode ended at frame {i}/{count}.")
                offset += got
        proc.stdout.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError("RelayVSR ffmpeg input decode failed: " + err[-2000:].decode(errors="replace"))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()
    return frames


def _encode_upscaled(ffmpeg, image_batch, source, target, fps):
    from . import motion_context_disk as d
    import torch

    if image_batch.ndim != 4 or image_batch.shape[-1] < 3:
        raise RuntimeError(f"RelayVSR returned an invalid IMAGE tensor: {tuple(image_batch.shape)}")
    height, width = [int(x) for x in image_batch.shape[1:3]]
    encoded_ffmpeg = d._preferred_h264_ffmpeg(ffmpeg)
    encoder = d._h264_encode_args(encoded_ffmpeg, 17, "fast")
    # Both input streams are explicitly mapped; copy the original soundtrack.
    command = [
        encoded_ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(float(fps)), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", *encoder, "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", str(target),
    ]
    # Avoid stdio deadlocks on Windows during large batches.
    log = Path(tempfile.gettempdir()) / f"h3_relay_encode_{uuid.uuid4().hex}.log"
    try:
        with log.open("wb") as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
            try:
                for frame in image_batch:
                    # ComfyUI VSR Decode returns [0,1] float RGB IMAGE frames.
                    rgb = (frame[..., :3].detach().to("cpu", dtype=torch.float32).clamp(0, 1) * 255.0).round().byte()
                    proc.stdin.write(rgb.contiguous().numpy().tobytes())
                proc.stdin.close()
                code = proc.wait()
                if code:
                    raise RuntimeError("RelayVSR ffmpeg encode failed: " + log.read_text(errors="replace")[-3000:])
            except Exception:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                raise
    finally:
        log.unlink(missing_ok=True)


def _source_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _h3_semantic_signature(manifest):
    """Stable visible-clip identity across FL2VA/independent cache compaction.

    The compactor may move byte offsets but cannot change card order, content
    configuration, reference state, frame counts, or color corrections.
    """
    segment_keys = (
        "clip_id", "frames", "trim_frames", "generation_seed",
        "visible_offset", "source_frames", "continued_from_source",
        "decoded_seam_shift", "color_adjustment", "first_frame_correction",
    )
    desc = {
        "fps": manifest.get("fps"),
        "sequence_mode": manifest.get("sequence_mode", "ref2va"),
        "geometry": manifest.get("geometry"),
        "final_frame_count": manifest.get("final_frame_count"),
        "source_video": manifest.get("source_video"),
        "full_batch_export_profile": manifest.get("full_batch_export_profile"),
        "segments": [
            {key: part[key] for key in segment_keys if key in part}
            for part in manifest.get("segments", [])
        ],
    }
    encoded = json.dumps(desc, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _h3_cache_identity(data_path, manifest=None):
    """Bind the upscale to the *current* H3 cache, not just an old output MP4.

    Only content-affecting metadata is included. Validation changes, cache
    compaction timestamps, project node IDs and import timestamps do not affect
    the decoded frames. A latent rewrite still changes the H3 data file mtime.
    """
    data_path = Path(data_path)
    stat = data_path.stat()
    if manifest is None:
        manifest = json.loads(data_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("RelayVSR: H3 cache manifest is invalid")
    segment_keys = (
        "clip_id", "frames", "trim_frames", "generation_seed", "video", "audio",
        "segment_end", "visible_offset", "source_frames", "continued_from_source",
        "decoded_seam_shift", "color_adjustment", "first_frame_correction",
    )
    descriptor = {
        "fps": manifest.get("fps"),
        "sequence_mode": manifest.get("sequence_mode", "ref2va"),
        "geometry": manifest.get("geometry"),
        "final_frame_count": manifest.get("final_frame_count"),
        "source_video": manifest.get("source_video"),
        "full_batch_export_profile": manifest.get("full_batch_export_profile"),
        "segments": [
            {key: segment[key] for key in segment_keys if key in segment}
            for segment in manifest.get("segments", [])
        ],
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {
        "cache_bytes": int(stat.st_size),
        "cache_mtime_ns": int(stat.st_mtime_ns),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def relay_cache_is_current(data_path, manifest=None, *, source_check=True, expected_detail=None):
    """Check a Relay cache before publishing or embedding it into portable .ext.

    Old cache files without an H3 binding are intentionally excluded from new
    archives. They can be regenerated on the next Relay Final Decode.
    """
    video = Path(data_path).with_suffix(".relayvsr_x2.mp4")
    sidecar = Path(data_path).with_suffix(".relayvsr_x2.json")
    if not video.is_file() or video.stat().st_size <= 0 or not sidecar.is_file():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or metadata.get("format") != _CACHE_FORMAT
                or not isinstance(metadata.get("h3_state"), dict)
                or metadata["h3_state"] != _h3_cache_identity(data_path, manifest)
                or not isinstance(metadata.get("source_sha256"), str)
                or len(metadata["source_sha256"]) != 64):
            return False
        if expected_detail is not None:
            if abs(float(metadata.get("detail", -1)) - float(expected_detail)) > 1e-8:
                return False
        # If the original exported MP4 still exists locally, ensure that it
        # has not been overwritten since RelayVSR generated this upscale.
        source_path = metadata.get("h3_original_output")
        if source_check and isinstance(source_path, str) and source_path:
            original = Path(source_path)
            if original.is_file() and _source_digest(original) != metadata["source_sha256"]:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _metadata_signature(source, models, detail=0.5, *, h3_state=None):
    signature = {
        "format": _CACHE_FORMAT,
        "detail": float(detail),
        "source_sha256": _source_digest(source),
        "model_files": {
            kind: {"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for kind, path in models.items()
        },
    }
    if h3_state is not None:
        signature["h3_state"] = h3_state
    return signature


def _make_relay_video(source, cached, models, ffmpeg, detail=0.5):
    """Native ComfyUI RelayVSR conditioning -> 1-step KSampler -> VSR Decode."""
    import torch
    import nodes
    import comfy.sd
    import comfy.model_management as mm
    import comfy.ldm.wan.model_vsr as vsr

    width, height, frames, fps = _ffprobe_media(ffmpeg, source)
    _LOG.info("RelayVSR x2: processing final video %s (%d frames @ %dx%d)", source, frames, width, height)
    pixels = _read_video_frames(ffmpeg, source, width, height, frames)
    lq_images = torch.from_numpy(pixels).float().div_(255.0)
    del pixels

    # ComfyUI native model loader and scheduler, not a separate Relay backend.
    model = comfy.sd.load_diffusion_model(str(models["diffusion_models"]))
    # Exactly as ComfyUI's native VAELoader: FlashVAE requires the safetensors
    # metadata (notably RelayVSR's upscale_factor) as well as its weights.
    import comfy.utils
    vae_sd, vae_metadata = comfy.utils.load_torch_file(str(models["vae"]), return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
    vae.throw_exception_if_invalid()
    del vae_sd
    backbone = model.model.diffusion_model
    keyframes = vsr.relay_keyframes(lq_images.shape[0])
    positive = [[backbone.positive_conditioning.unsqueeze(0).clone(), {"lq_images": lq_images[keyframes]}]]
    shape = vsr.relay_latent_size(lq_images.shape[1], lq_images.shape[2], backbone.latent_stride)
    latent = {"samples": torch.zeros(
        (1, model.model.latent_format.latent_channels, len(keyframes), *shape),
        device=mm.intermediate_device(),
    )}
    # cfg=1 means the negative branch is not evaluated. Using the positive as
    # both inputs is the official 1-step workflow convention, not a text prompt.
    sampled = nodes.KSampler().sample(
        model=model, seed=0, steps=1, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=positive, negative=positive, latent_image=latent, denoise=1.0,
    )[0]
    del positive, latent
    mm.unload_all_models()
    mm.soft_empty_cache()
    decoded = vae.decode(sampled["samples"], vae_options={"lq_images": lq_images, "detail": float(detail)})
    del sampled, lq_images, model, vae, backbone
    images = decoded.reshape(-1, *decoded.shape[-3:])
    if len(images) != frames:
        raise RuntimeError(f"RelayVSR decoded {len(images)} frames, expected {frames}; output not written.")
    if tuple(images.shape[1:3]) != (height * 2, width * 2):
        raise RuntimeError(f"RelayVSR returned {tuple(images.shape[1:3])}, expected {(height * 2, width * 2)}.")
    _encode_upscaled(ffmpeg, images, source, cached, fps)
    del decoded
    free_vram_for_relay()


def upscale_finished_video(source, data_path, ffmpeg, *, workflow=None, prompt=None, detail=0.5, h3_state=None):
    """Upscale once, then preserve a cache and a separate user-facing MP4."""
    from . import motion_context_disk as d
    source = Path(source).resolve()
    detail = float(detail)
    if not 0.0 <= detail <= 1.0:
        raise ValueError(f"RelayVSR detail must be between 0 and 1, got {detail!r}")
    models = {kind: _find_or_download(kind) for kind in _FILES}
    signature = _metadata_signature(source, models, detail, h3_state=h3_state)
    cache = Path(data_path).with_suffix(".relayvsr_x2.mp4")
    sidecar = Path(data_path).with_suffix(".relayvsr_x2.json")
    reused = False
    if cache.is_file() and cache.stat().st_size and sidecar.is_file():
        try:
            cached_signature = json.loads(sidecar.read_text(encoding="utf-8"))
            # Cache the *exact* finished input movie, not a physical offset in
            # the H3 cache. Preview/cache bookkeeping may move the H3 .h3cache
            # while producing byte-identical finished video. In that case the
            # existing upscale is still valid and its H3 binding is refreshed.
            # The original output's SHA256, model files and Detail MUST match.
            stable_keys = tuple(key for key in signature if key != "h3_state")
            reused = (isinstance(cached_signature, dict)
                      and all(cached_signature.get(key) == signature[key] for key in stable_keys))
        except (ValueError, OSError):
            reused = False
    if not reused:
        # Leave the old output untouched if the new Relay execution fails.
        temp = cache.with_name(f"{cache.stem}.{uuid.uuid4().hex}.tmp.mp4")
        try:
            free_vram_for_relay()
            _make_relay_video(source, temp, models, ffmpeg, detail=detail)
            if not temp.is_file() or not temp.stat().st_size:
                raise RuntimeError("RelayVSR produced an empty video file.")
            os.replace(temp, cache)
            sidecar.write_text(json.dumps({**signature, "h3_original_output": str(source)}, indent=2), encoding="utf-8")
        finally:
            temp.unlink(missing_ok=True)
    else:
        _LOG.info("RelayVSR x2: existing output matches the final source; reusing cache")
        if h3_state is not None:
            sidecar.write_text(json.dumps({**signature, "h3_original_output": str(source)}, indent=2), encoding="utf-8")

    final_output = source.with_name(source.stem + "_RelayVSR_x2.mp4")
    # Avoid accidental hardlinks: metadata gets embedded only into the exported
    # copy, never into the reusable upscaled cache.
    staged = final_output.with_name(final_output.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(cache, staged)
        os.replace(staged, final_output)
    finally:
        staged.unlink(missing_ok=True)
    d._embed_final_metadata_in_place(final_output, workflow=workflow, prompt=prompt)
    return final_output, reused
