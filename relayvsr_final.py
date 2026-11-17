"""Optional RelayVSR x2 post-process for a COMPLETED H3 Final Decode video.

This module uses ComfyUI core's RelayVSR model, KSampler and Flash VAE.
No H3 latents, temporal context or Final Decode caches are touched.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import re
import sys
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
# The Flash decoder is window-local; keep at most two keyframe intervals
# (roughly 31 input/output frames) live at once. No change to the sampler.
_DECODE_WINDOWS_PER_CHUNK = 2


class RelayEncodeError(RuntimeError):
    """Only ffmpeg/encoder errors should trigger the CPU retry."""


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
    registered_roots = [Path(root) for root in folder_paths.get_folder_paths(kind)]
    for base in registered_roots:
        candidate = base / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    for registered in folder_paths.get_filename_list(kind):
        if Path(registered).name.casefold() != name.casefold():
            continue
        found = folder_paths.get_full_path(kind, registered)
        if found and Path(found).is_file() and Path(found).stat().st_size > 0:
            return Path(found)

    if not registered_roots:
        raise RuntimeError(f"RelayVSR: no registered model directory for {kind}.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("RelayVSR requires huggingface-hub for first-use model downloads.") from exc

    # On standard ComfyUI, models/unet is listed before
    # models/diffusion_models. Prefer the latter: otherwise the download to
    # models/diffusion_models is copied into models/unet, wasting ~5 GB.
    chosen = next((root for root in registered_roots if root.name.casefold() == kind.casefold()),
                  registered_roots[0])
    chosen.mkdir(parents=True, exist_ok=True)
    target = chosen / name
    _LOG.info("RelayVSR: downloading missing model %s", name)
    if chosen.name.casefold() == kind.casefold():
        # hf_hub_download writes <local_dir>/<kind>/<name> directly at target.
        downloaded = Path(hf_hub_download(
            repo_id=_REPO, filename=f"{kind}/{name}", local_dir=str(chosen.parent),
        ))
        if downloaded.resolve() == target.resolve():
            return downloaded
        if any((base / name).resolve() == downloaded.resolve() for base in registered_roots):
            return downloaded
        # Filesystem safety net for unusual registered folder layouts.
        try:
            os.replace(downloaded, target)
        except OSError:
            shutil.copy2(downloaded, target)
    else:
        # External custom model directory: use a short-lived local staging
        # folder, never infer the model root from target.parent.parent.
        with tempfile.TemporaryDirectory(prefix="h3_relay_download_", dir=str(chosen)) as stage:
            downloaded = Path(hf_hub_download(
                repo_id=_REPO, filename=f"{kind}/{name}", local_dir=stage,
            ))
            os.replace(downloaded, target)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"RelayVSR: downloaded model missing: {target}")
    return target


def _ffprobe_media(ffmpeg, video):
    """Read geometry/FPS; actual frame count is taken from the decoded pipe."""
    try:
        import cv2
        capture = cv2.VideoCapture(str(video))
        try:
            if capture.isOpened():
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if min(width, height) > 0 and fps > 0:
                    return width, height, max(0, frames), fps
        finally:
            capture.release()
    except ImportError:
        pass

    probe = str(Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else "")))
    if not Path(probe).is_file():
        probe = shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("RelayVSR requires OpenCV or ffprobe to read video geometry.")
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    frames = int(stream.get("nb_frames") or 0)
    if min(width, height) < 1:
        raise RuntimeError("RelayVSR could not determine the source geometry.")
    numerator, denominator = (stream.get("r_frame_rate", "24/1").split("/") + ["1"])[:2]
    return width, height, frames, float(numerator) / float(denominator)


@contextmanager
def _video_frames_memmap(ffmpeg, source, width, height, *, scratch_dir=None):
    """Decode H3 to a temporary uint8 disk array, not a multi-GB RAM tensor.

    Counting the *actual* frames from ffmpeg avoids MKV/FFV1 frame-count
    estimates; the scratch file is removed on success, OOM and interruption.
    """
    frame_bytes = int(width) * int(height) * 3
    temp = tempfile.NamedTemporaryFile(
        prefix="h3_relay_frames_", suffix=".rgb", dir=scratch_dir, delete=False,
    )
    temp_path = Path(temp.name)
    log = temp_path.with_suffix(".decode.log")
    proc = None
    pixels = None
    count = 0
    try:
        with temp, log.open("wb") as stderr:
            proc = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "pipe:1"],
                stdout=subprocess.PIPE, stderr=stderr,
            )
            buf = bytearray(frame_bytes)
            while True:
                view = memoryview(buf)
                got = proc.stdout.readinto(view)
                if not got:
                    break
                offset = got
                while offset < frame_bytes:
                    n = proc.stdout.readinto(view[offset:])
                    if not n:
                        raise RuntimeError("RelayVSR received a truncated source frame.")
                    offset += n
                temp.write(buf)
                count += 1
            proc.stdout.close()
            code = proc.wait()
            if code:
                raise RuntimeError("RelayVSR source decode failed: " + log.read_text(errors="replace")[-2000:])
        if count < 1:
            raise RuntimeError("RelayVSR source video contains no decoded frames.")
        pixels = np.memmap(temp_path, mode="r", dtype=np.uint8,
                           shape=(count, height, width, 3))
        yield pixels, count
    finally:
        # Explicitly close the mapping BEFORE unlinking on Windows.
        if pixels is not None:
            mapping = getattr(pixels, "_mmap", None)
            if mapping is not None:
                mapping.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        temp_path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)


def _relay_audio_encode_args(ffmpeg, source, audio_bitrate="192k"):
    """Keep compatible AAC untouched; encode other audio at the selected rate.

    imageio-ffmpeg ships ffmpeg but usually not ffprobe on Windows Portable.
    Its normal `ffmpeg -i` stderr still identifies audio streams reliably.
    The source's audio track is optional in the Relay output.
    """
    bitrate = str(audio_bitrate)
    if not re.fullmatch(r"(?:128|192|256|320)k", bitrate):
        raise ValueError(f"RelayVSR: unsupported audio bitrate {bitrate!r}")
    probe = str(Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else "")))
    if not Path(probe).is_file():
        probe = shutil.which("ffprobe")
    codec = None
    if probe:
        try:
            result = subprocess.run(
                [probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=codec_name", "-of", "json", str(source)],
                capture_output=True, text=True, check=True,
            )
            streams = json.loads(result.stdout).get("streams") or []
            codec = str(streams[0].get("codec_name", "")).lower() if streams else "none"
        except (OSError, ValueError, subprocess.CalledProcessError, IndexError):
            codec = None
    if codec is None:
        # Fallback for imageio-ffmpeg without a matching ffprobe executable.
        try:
            result = subprocess.run(
                [str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(source)],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stderr.splitlines():
                if re.search(r"Stream\s+#\d+:\d+", line) and "Audio:" in line:
                    codec = line.split("Audio:", 1)[1].strip().split()[0].rstrip(",").lower()
                    break
            else:
                # No audio stream shown: -map 1:a? is optional in the encoder.
                codec = "none" if "Input #" in result.stderr else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    if codec in ("aac", "none"):
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", bitrate]


@contextmanager
def _relay_ffmpeg_encoder(ffmpeg, source, target, width, height, fps, *, force_cpu=False,
                          audio_bitrate="192k"):
    from . import motion_context_disk as d

    # NVENC H.264 rejects some very large 2x canvases. Select CPU proactively;
    # the caller can also retry CPU after a smaller NVENC encode failure without
    # recomputing the sampled RelayVSR latent.
    use_cpu = force_cpu or max(width, height) > 4096
    encoder_ffmpeg = ffmpeg if use_cpu else d._preferred_h264_ffmpeg(ffmpeg)
    encoder = (["-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p"]
               if use_cpu else d._h264_encode_args(encoder_ffmpeg, 17, "fast"))
    command = [
        encoder_ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(float(fps)), "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", *encoder,
        *_relay_audio_encode_args(ffmpeg, source, audio_bitrate),
        "-shortest", "-movflags", "+faststart", str(target),
    ]
    log = Path(tempfile.gettempdir()) / f"h3_relay_encode_{uuid.uuid4().hex}.log"
    proc = None
    try:
        with log.open("wb") as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=stderr)
            yield proc
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                raise RelayEncodeError("RelayVSR ffmpeg closed before the stream finished") from exc
            code = proc.wait()
            if code:
                raise RelayEncodeError("RelayVSR ffmpeg encode failed: " + log.read_text(errors="replace")[-3000:])
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            # A broken pipe may be caused by a failed encoder. Close/kill and
            # read its diagnostics BEFORE deleting the temporary log.
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            proc.kill()
            proc.wait()
        if isinstance(exc, RelayEncodeError):
            try:
                tail = log.read_text(errors="replace")[-3000:].strip()
            except OSError:
                tail = ""
            if tail and tail not in str(exc):
                raise RelayEncodeError(f"{exc}\nffmpeg: {tail}") from exc
        raise
    finally:
        log.unlink(missing_ok=True)


def _write_relay_frames(proc, images):
    import torch
    if images.ndim != 4 or images.shape[-1] < 3:
        raise RuntimeError(f"RelayVSR returned invalid IMAGE batch {tuple(images.shape)}")
    for frame in images:
        rgb = (frame[..., :3].detach().to("cpu", dtype=torch.float32)
               .clamp(0, 1).mul(255.0).round().byte())
        try:
            proc.stdin.write(rgb.contiguous().numpy().tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RelayEncodeError("RelayVSR ffmpeg closed its input before encoding finished") from exc


def _iter_relay_decoded_chunks(vae, sampled_latents, pixels, keyframes, detail):
    """Decode bounded keyframe windows, sharing one frame at each boundary."""
    import torch
    keyframes = [int(k) for k in keyframes]
    if len(keyframes) < 2:
        # A one-frame input cannot form a temporal Relay interval. Handle it
        # deliberately instead of attempting a zero-window Flash decode.
        import torch.nn.functional as F
        one = torch.from_numpy(np.array(pixels[0:1], copy=True)).float().div_(255.0)
        yield F.interpolate(one.permute(0, 3, 1, 2), scale_factor=2,
                            mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        return
    for a in range(0, len(keyframes) - 1, _DECODE_WINDOWS_PER_CHUNK):
        b = min(a + _DECODE_WINDOWS_PER_CHUNK, len(keyframes) - 1)
        start, end = keyframes[a], keyframes[b]
        lq = torch.from_numpy(np.array(pixels[start:end + 1], copy=True)).float().div_(255.0)
        part = vae.decode(sampled_latents[:, :, a:b + 1],
                          vae_options={"lq_images": lq, "detail": float(detail)})
        part = part.reshape(-1, *part.shape[-3:])
        wanted = end - start + 1
        if len(part) != wanted:
            raise RuntimeError(f"RelayVSR Flash decoded {len(part)} frames; expected {wanted} in [{start},{end}].")
        if a != 0:
            part = part[1:]  # frame shared with previous keyframe interval
        yield part
        del part, lq


def _encode_relay_chunks(ffmpeg, source, target, width, height, fps,
                         vae, sampled_latents, pixels, keyframes, detail, *, force_cpu=False,
                         audio_bitrate="192k"):
    count = 0
    with _relay_ffmpeg_encoder(ffmpeg, source, target, width * 2, height * 2,
                               fps, force_cpu=force_cpu, audio_bitrate=audio_bitrate) as proc:
        for batch in _iter_relay_decoded_chunks(vae, sampled_latents, pixels, keyframes, detail):
            if tuple(batch.shape[1:3]) != (height * 2, width * 2):
                raise RuntimeError(f"RelayVSR returned {tuple(batch.shape[1:3])}; expected {(height * 2, width * 2)}")
            _write_relay_frames(proc, batch)
            count += int(len(batch))
    if count != len(pixels):
        raise RuntimeError(f"RelayVSR encoded {count} frames; expected {len(pixels)}")


def _relay_ffmpeg_path():
    """Use the same bundled ffmpeg for cache checks as for Final Decode."""
    try:
        from . import motion_context_disk as d
        return d._find_ffmpeg()
    except (ImportError, AttributeError, OSError, RuntimeError):
        return shutil.which("ffmpeg")


def _source_digest(path, ffmpeg=None):
    """SHA256 of the actual video/audio streams, independent of MP4 metadata.

    streamhash with stream-copy hashes compressed packets without decoding or
    expensive re-encoding. If unavailable (including isolated unit tests),
    conservatively fall back to the original whole-file checksum.
    """
    ffmpeg = ffmpeg or _relay_ffmpeg_path()
    if ffmpeg:
        try:
            result = subprocess.run(
                [str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:v:0",
                 "-map", "0:a?", "-c", "copy", "-f", "streamhash",
                 "-hash", "SHA256", "pipe:1"],
                capture_output=True, text=True, timeout=180,
            )
            stream_hashes = sorted(
                (int(parts[0]), parts[1], parts[2])
                for line in result.stdout.splitlines()
                if len(parts := line.strip().split(",", 2)) == 3
                and parts[0].isdigit() and parts[1] in ("v", "a")
                and parts[2].startswith("SHA256=")
                and len(parts[2]) == len("SHA256=") + 64
            )
            if result.returncode == 0 and any(kind == "v" for _, kind, _ in stream_hashes):
                stable = "\n".join(f"{kind}:{digest}" for _, kind, digest in stream_hashes)
                return hashlib.sha256(stable.encode("ascii")).hexdigest()
        except (OSError, subprocess.TimeoutExpired):
            pass
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# Single list of content-affecting H3 fields shared by the semantic and
# physical cache checks. The physical signature adds cache storage descriptors.
_H3_VISIBLE_SEGMENT_KEYS = (
    "clip_id", "frames", "trim_frames", "generation_seed",
    "visible_offset", "source_frames", "continued_from_source",
    "decoded_seam_shift", "color_adjustment", "first_frame_correction",
)
_H3_STORAGE_SEGMENT_KEYS = _H3_VISIBLE_SEGMENT_KEYS + (
    "video", "audio", "segment_end",
)


def _h3_semantic_signature(manifest):
    """Stable visible-clip identity across FL2VA/independent cache compaction.

    The compactor may move byte offsets but cannot change card order, content
    configuration, reference state, frame counts, or color corrections.
    """
    segment_keys = _H3_VISIBLE_SEGMENT_KEYS
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
    segment_keys = _H3_STORAGE_SEGMENT_KEYS
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


def _metadata_signature(source, models, detail=0.5, *, h3_state=None, ffmpeg=None,
                        audio_bitrate="192k"):
    signature = {
        "format": _CACHE_FORMAT,
        "detail": float(detail),
        "source_sha256": _source_digest(source, ffmpeg=ffmpeg),
        # FLAC/PCM require AAC encoding. Different selected bitrates must not
        # reuse a cached output with the previous bitrate. AAC is copied.
        "audio_encode_args": _relay_audio_encode_args(ffmpeg or _relay_ffmpeg_path(), source, audio_bitrate),
        "model_files": {
            kind: {"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for kind, path in models.items()
        },
    }
    if h3_state is not None:
        signature["h3_state"] = h3_state
    return signature


def _make_relay_video(source, cached, models, ffmpeg, detail=0.5, fps_override=None,
                      audio_bitrate="192k"):
    """Native Relay 1-step sampling once; Flash decode is disk/RAM bounded."""
    import torch
    import nodes
    import comfy.sd
    import comfy.model_management as mm
    import comfy.ldm.wan.model_vsr as vsr
    import comfy.utils

    width, height, estimated_count, probed_fps = _ffprobe_media(ffmpeg, source)
    fps = float(fps_override) if fps_override is not None else probed_fps
    with _video_frames_memmap(ffmpeg, source, width, height,
                              scratch_dir=str(Path(cached).parent)) as (pixels, frames):
        if estimated_count and frames != estimated_count:
            _LOG.info("RelayVSR source has %d actual frames (container reported %d)", frames, estimated_count)
        _LOG.info("RelayVSR x2: %s, %d frames, %dx%d @ %.3f fps", source, frames, width, height, fps)
        keyframes = list(vsr.relay_keyframes(frames))
        if not keyframes or keyframes[0] != 0 or keyframes[-1] != frames - 1:
            raise RuntimeError("RelayVSR keyframe indexing does not cover the full source.")
        if frames == 1:
            # Single-frame videos cannot use temporal Relay; bicubic fallback.
            _LOG.warning("RelayVSR: one-frame video; using bicubic ×2 output")
            _encode_relay_chunks(ffmpeg, source, cached, width, height, fps,
                                 None, None, pixels, [0], detail, force_cpu=True,
                                 audio_bitrate=audio_bitrate)
            return

        model = comfy.sd.load_diffusion_model(str(models["diffusion_models"]))
        vae_sd, vae_metadata = comfy.utils.load_torch_file(str(models["vae"]), return_metadata=True)
        vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
        vae.throw_exception_if_invalid()
        del vae_sd
        backbone = model.model.diffusion_model
        key_pixels = torch.from_numpy(np.array(pixels[keyframes], copy=True)).float().div_(255.0)
        positive = [[backbone.positive_conditioning.unsqueeze(0).clone(), {"lq_images": key_pixels}]]
        shape = vsr.relay_latent_size(height, width, backbone.latent_stride)
        latent = {"samples": torch.zeros(
            (1, model.model.latent_format.latent_channels, len(keyframes), *shape),
            device=mm.intermediate_device(),
        )}
        sampled = nodes.KSampler().sample(
            model=model, seed=0, steps=1, cfg=1.0, sampler_name="euler", scheduler="simple",
            positive=positive, negative=positive, latent_image=latent, denoise=1.0,
        )[0]
        del positive, latent, model, backbone, key_pixels
        mm.unload_all_models()
        mm.soft_empty_cache()
        sampled_latents = sampled["samples"]
        del sampled
        # If NVENC fails, retry encoding without rerunning Relay sampling.
        try:
            _encode_relay_chunks(ffmpeg, source, cached, width, height, fps,
                                 vae, sampled_latents, pixels, keyframes, detail,
                                 audio_bitrate=audio_bitrate)
        except RelayEncodeError as exc:
            if max(width * 2, height * 2) > 4096:
                raise  # already using CPU for large canvases
            _LOG.warning("RelayVSR NVENC/encode attempt failed; retrying CPU without sampling again: %s", exc)
            Path(cached).unlink(missing_ok=True)
            _encode_relay_chunks(ffmpeg, source, cached, width, height, fps,
                                 vae, sampled_latents, pixels, keyframes, detail, force_cpu=True,
                                 audio_bitrate=audio_bitrate)
        del sampled_latents, vae


def _write_relay_sidecar_atomic(path, metadata):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def upscale_finished_video(source, data_path, ffmpeg, *, workflow=None, prompt=None, detail=0.5,
                           h3_state=None, fps_override=None, audio_bitrate="192k"):
    """Upscale once, then preserve a cache and a separate user-facing MP4."""
    from . import motion_context_disk as d
    source = Path(source).resolve()
    detail = float(detail)
    if not 0.0 <= detail <= 1.0:
        raise ValueError(f"RelayVSR detail must be between 0 and 1, got {detail!r}")
    models = {kind: _find_or_download(kind) for kind in _FILES}
    signature = _metadata_signature(source, models, detail, h3_state=h3_state, ffmpeg=ffmpeg,
                                    audio_bitrate=audio_bitrate)
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
            try:
                free_vram_for_relay()
                if fps_override is None:
                    _make_relay_video(source, temp, models, ffmpeg, detail=detail,
                                      audio_bitrate=audio_bitrate)
                else:
                    _make_relay_video(source, temp, models, ffmpeg, detail=detail,
                                      fps_override=fps_override, audio_bitrate=audio_bitrate)
                if not temp.is_file() or not temp.stat().st_size:
                    raise RuntimeError("RelayVSR produced an empty video file.")
                # Invalidate the old sidecar BEFORE replacing the video. A crash
                # between these writes must never validate the wrong pair.
                sidecar.unlink(missing_ok=True)
                os.replace(temp, cache)
                _write_relay_sidecar_atomic(sidecar, {**signature, "h3_original_output": str(source)})
            finally:
                # Cleanup must never replace an OOM / interruption / sampling
                # error with a secondary VRAM-manager exception.
                active_error = sys.exc_info()[0] is not None
                try:
                    free_vram_for_relay()  # even on OOM/interruption
                except Exception as cleanup_exc:
                    if active_error:
                        _LOG.warning("RelayVSR VRAM cleanup failed after original error: %s", cleanup_exc)
                    else:
                        raise
        finally:
            temp.unlink(missing_ok=True)
    else:
        _LOG.info("RelayVSR x2: existing output matches the final source; reusing cache")
        if h3_state is not None:
            _write_relay_sidecar_atomic(sidecar, {**signature, "h3_original_output": str(source)})

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
