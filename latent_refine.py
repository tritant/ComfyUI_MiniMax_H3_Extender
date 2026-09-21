"""
Non-destructive H3 latent refine pass for the Extender.

Draft cache stays intact. Refine writes ``*.refine.h3cache`` / ``*.refine.json``.
Pipeline per clip: load draft video → neural latent upscale → sample with
custom denoise → keep draft audio unchanged.

Supports the same causal ``clip_by_clip`` / ``full_batch`` + validated-prefix
rules as the draft Extender pass.
"""

from __future__ import annotations

import hashlib
import logging
import time

import comfy.nested_tensor
import comfy.model_management
import torch

from .latent_upscaler import (
    offload_upscale_models,
    target_latent_hw,
    upscale_video_latent,
)
from .motion_context_disk import (
    CACHE_VERSION,
    _append_segment,
    _clear_refine_sidecar,
    _final_frame_count,
    _invalidate_layer_preview_cache,
    _load_refine_sidecar,
    _load_segment_audio,
    _load_segment_video,
    _new_data_file,
    _proxy_at,
    _refine_paths_from_draft,
    _truncate_chain,
    _validated_prefix_count,
    _write_json_atomic,
)
from .motion_context_ram import _streams_from_latent

_LOG = logging.getLogger("minimax_h3_extender.latent_refine")
VAE_DOWNSAMPLE = 16


def _enforce_refine_validated_prefix(clips):
    """REF2VA-style continuous prefix for refine_validated flags."""
    found_open = False
    for clip in clips:
        if found_open:
            clip["refine_validated"] = False
        elif not clip.get("refine_validated"):
            found_open = True
    return clips


def _sigmas_cache_signature(sigmas) -> str:
    """Compact fingerprint of an external SIGMAS grid for refine cache matching."""
    if sigmas is None:
        return ""
    if not torch.is_tensor(sigmas):
        return ""
    flat = sigmas.detach().float().reshape(-1).cpu().tolist()
    if not flat:
        return ""
    payload = f"{len(flat)}:" + ",".join(f"{float(v):.8g}" for v in flat)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _refine_settings_match(
    manifest,
    *,
    upscale_model,
    megapixels,
    denoise,
    steps,
    precision,
    width,
    height,
    external_sigmas=False,
    sigmas_signature="",
):
    meta = (manifest or {}).get("refine") if isinstance(manifest, dict) else None
    if not isinstance(meta, dict):
        return False
    try:
        return (
            str(meta.get("upscale_model") or "") == str(upscale_model)
            and abs(float(meta.get("megapixels", -1)) - float(megapixels)) < 1e-6
            and abs(float(meta.get("denoise", -1)) - float(denoise)) < 1e-6
            and int(meta.get("steps", -1)) == int(steps)
            and str(meta.get("precision") or "") == str(precision)
            and int(meta.get("width", -1)) == int(width)
            and int(meta.get("height", -1)) == int(height)
            and bool(meta.get("external_sigmas", False)) == bool(external_sigmas)
            and str(meta.get("sigmas_signature") or "") == str(sigmas_signature or "")
        )
    except (TypeError, ValueError):
        return False


def _new_refine_manifest(draft_manifest, fps, refine_meta):
    return {
        "version": int(CACHE_VERSION),
        "build": "latent-refine-v1",
        "owner_id": draft_manifest.get("owner_id"),
        "fps": float(draft_manifest.get("fps", fps)),
        "geometry": None,
        "segments": [],
        "final_frame_count": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "refine": dict(refine_meta),
    }


def run_refine_pass(
    *,
    owner,
    data_path,
    draft_manifest,
    clips,
    model,
    clip,
    vae,
    audio_vae,
    refs,
    ref_videos,
    ref_video_fps,
    ref_video_audios,
    standalone_audio_clip_plan,
    active_ref_video_count,
    prepared_image_blocks,
    prepared_video_blocks_by_frame_count,
    ref_image_size,
    context_length,
    audio_context_length,
    sampler_name,
    scheduler,
    upscale_model,
    refine_megapixels,
    refine_denoise,
    refine_steps,
    upscale_precision,
    make_ref2va_conditioning,
    prepare_shared_refs,
    prepare_standalone_audio_refs,
    apply_per_clip_loras,
    sample_h3,
    sample_sigmas=None,
    motion,
    duration_to_frames,
    reference_count,
    max_mixed_ref_items,
    fps,
    send_progress,
    extender_self,
    run_mode="full_batch",
):
    """Build / extend the refine sidecar from an existing draft cache."""
    if str(upscale_model or "None") in ("", "None"):
        raise ValueError(
            "MiniMax H3 Extender: run_refine requires latent_upscale_model "
            "(place weights in models/latent_upscale_models/)."
        )

    mode = str(run_mode or "full_batch")
    if mode not in ("full_batch", "clip_by_clip"):
        mode = "full_batch"

    draft_segments = list(draft_manifest.get("segments", []))
    if not draft_segments:
        raise ValueError(
            "MiniMax H3 Extender: refine needs at least one draft clip in cache."
        )
    # clip_by_clip testing: refine only the drafted prefix. Cards beyond the
    # current draft chain wait until more draft clips exist.
    draft_limit = min(len(draft_segments), len(clips))
    if draft_limit < len(clips):
        _LOG.info(
            "Refine limited to drafted prefix %d/%d cards (run_mode=%s)",
            draft_limit,
            len(clips),
            mode,
        )

    geom = draft_manifest.get("geometry") or {}
    h_in = int(geom.get("video_h") or 0)
    w_in = int(geom.get("video_w") or 0)
    if h_in < 1 or w_in < 1:
        raise ValueError("MiniMax H3 Extender: draft cache has no video geometry.")

    h_out, w_out, effective_scale = target_latent_hw(
        h_in, w_in, megapixels=float(refine_megapixels), align=32
    )
    resolved_width = int(w_out * VAE_DOWNSAMPLE)
    resolved_height = int(h_out * VAE_DOWNSAMPLE)
    if sample_sigmas is not None:
        sample_step_count = max(1, int(sample_sigmas.shape[-1]) - 1)
        steps_label = f"{sample_step_count} (external SIGMAS, denoise={float(refine_denoise):.3f})"
    else:
        sample_step_count = int(refine_steps)
        steps_label = f"{sample_step_count} (scheduler)"
    used_external_sigmas = bool(sample_sigmas is not None)
    sigmas_signature = _sigmas_cache_signature(sample_sigmas)
    refine_meta = {
        "upscale_model": str(upscale_model),
        "megapixels": float(refine_megapixels),
        "denoise": float(refine_denoise),
        # Persist the step count that actually drives sampling so cache matches
        # stay consistent when external SIGMAS replace refine_steps.
        "steps": int(sample_step_count),
        "precision": str(upscale_precision),
        "width": resolved_width,
        "height": resolved_height,
        "external_sigmas": used_external_sigmas,
        "sigmas_signature": sigmas_signature,
    }
    _LOG.info(
        "Refine pass %sx%s -> %sx%s (scale=%.3f, denoise=%.3f, steps=%s, mode=%s)",
        w_in * VAE_DOWNSAMPLE,
        h_in * VAE_DOWNSAMPLE,
        resolved_width,
        resolved_height,
        effective_scale,
        float(refine_denoise),
        steps_label,
        mode,
    )

    _enforce_refine_validated_prefix(clips)

    refine_data, refine_manifest_path = _refine_paths_from_draft(data_path)
    existing = _load_refine_sidecar(data_path, draft_manifest, require_complete=False)
    if existing is not None and _refine_settings_match(
        existing[2],
        upscale_model=upscale_model,
        megapixels=refine_megapixels,
        denoise=refine_denoise,
        steps=sample_step_count,
        precision=upscale_precision,
        width=resolved_width,
        height=resolved_height,
        external_sigmas=used_external_sigmas,
        sigmas_signature=sigmas_signature,
    ):
        refine_data, refine_manifest_path, refine_manifest = existing
        refine_manifest = dict(refine_manifest)
        refine_manifest["segments"] = [dict(x) for x in refine_manifest.get("segments", [])]
        refine_manifest["refine"] = dict(refine_meta)
    else:
        _clear_refine_sidecar(data_path)
        for cfg in clips:
            cfg["refine_validated"] = False
        _new_data_file(refine_data)
        refine_manifest = _new_refine_manifest(draft_manifest, fps, refine_meta)
        _write_json_atomic(refine_manifest_path, refine_manifest)

    motion_ctx = motion
    previous_proxy = None
    prepared_ref_frame_count = None
    ref_items = ref_blocks = active_picture_slots = active_video_slots = None
    image_blocks = prepared_image_blocks
    video_blocks_by_frame = dict(prepared_video_blocks_by_frame_count or {})
    standalone_audio_cache = {}
    generated = []
    statuses = []

    try:
        for i, cfg in enumerate(clips):
            if i >= draft_limit:
                statuses.append(
                    f"refine stopped before clip {i + 1}: draft cache only has "
                    f"{len(draft_segments)} clip(s)"
                )
                break

            segments = list(refine_manifest.get("segments", []))
            existing_seg = i < len(segments)

            if bool(cfg.get("refine_validated")) and existing_seg:
                if i > 0 and not all(
                    bool(x.get("validated", False)) for x in segments[:i]
                ):
                    raise ValueError(
                        f"MiniMax H3 Extender: refine clip {i + 1} cannot be "
                        "validated before every previous refine clip is validated."
                    )
                if not bool(segments[i].get("validated", False)):
                    segments[i] = dict(segments[i])
                    segments[i]["validated"] = True
                    refine_manifest = dict(refine_manifest)
                    refine_manifest["segments"] = segments
                    refine_manifest["updated_at"] = time.time()
                    _write_json_atomic(refine_manifest_path, refine_manifest)
                    statuses.append(f"refine clip {i + 1} validated from disk")
                else:
                    statuses.append(f"refine clip {i + 1} validated (disk)")
                previous_proxy = _proxy_at(refine_data, refine_manifest, i)
                continue

            # Active refine candidate: invalidate downstream refine_validated.
            cfg["refine_validated"] = False
            for j in range(i + 1, len(clips)):
                clips[j]["refine_validated"] = False

            send_progress(
                owner,
                i,
                len(clips),
                "refining",
                f"Refining clip {i + 1}/{len(clips)}",
            )

            if existing_seg or len(segments) > i:
                refine_manifest = _truncate_chain(
                    refine_data, refine_manifest_path, refine_manifest, i
                )
                refine_manifest = _invalidate_layer_preview_cache(
                    refine_data, refine_manifest_path, refine_manifest
                )
                segments = list(refine_manifest.get("segments", []))
                previous_proxy = (
                    _proxy_at(refine_data, refine_manifest, i - 1) if i > 0 else None
                )

            draft_desc = draft_segments[i]
            frame_count = int(draft_desc.get("frames") or duration_to_frames(cfg["duration"]))
            trim_from_draft = int(draft_desc.get("trim_frames") or 0)

            draft_video = _load_segment_video(data_path, draft_desc)
            draft_audio = _load_segment_audio(data_path, draft_desc)
            _LOG.info(
                "Refine clip %s/%s: latent upscale %s -> target ~%.2f MP (can take a while, no step bar)",
                i + 1,
                len(clips),
                str(upscale_model),
                float(refine_megapixels),
            )
            send_progress(
                owner,
                i,
                len(clips),
                "upscaling",
                f"Latent upscale clip {i + 1}/{len(clips)}",
            )
            up_video = upscale_video_latent(
                draft_video,
                str(upscale_model),
                megapixels=float(refine_megapixels),
                align=32,
                device="cuda",
                precision=str(upscale_precision),
                enable_temporal_chunking=True,
                force_unload=True,
            )
            del draft_video
            _LOG.info(
                "Refine clip %s/%s: upscale done, sampling %s",
                i + 1,
                len(clips),
                steps_label,
            )
            send_progress(
                owner,
                i,
                len(clips),
                "refining",
                f"Refine sample clip {i + 1}/{len(clips)} ({steps_label})",
            )

            selected_ref_audios, selected_audio_slots, selected_audio_offsets = (
                standalone_audio_clip_plan[i]
            )
            selected_ref_audio_count = len(selected_audio_slots)
            clip_mixed_ref_count = (
                reference_count(refs)
                + active_ref_video_count
                + selected_ref_audio_count
            )
            if clip_mixed_ref_count > max_mixed_ref_items:
                raise ValueError(
                    f"MiniMax H3 Extender: H3 Ref2VA supports at most {max_mixed_ref_items} "
                    f"mixed reference items for this clip; got {clip_mixed_ref_count}."
                )

            needs_ref_prepare = (
                ref_items is None
                or ref_blocks is None
                or active_picture_slots is None
                or active_video_slots is None
                or (active_ref_video_count and prepared_ref_frame_count != frame_count)
            )
            if needs_ref_prepare:
                cached_video_blocks = video_blocks_by_frame.get(int(frame_count))
                ref_items, ref_blocks, active_picture_slots, active_video_slots = (
                    prepare_shared_refs(
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
                        cached_image_blocks=image_blocks,
                        cached_video_blocks=cached_video_blocks,
                    )
                )
                image_block_count = len(active_picture_slots or [])
                if image_blocks is None:
                    image_blocks = list((ref_blocks or [])[:image_block_count])
                if active_ref_video_count:
                    duration_key = int(frame_count)
                    video_blocks_by_frame.pop(duration_key, None)
                    video_blocks_by_frame[duration_key] = list(
                        (ref_blocks or [])[image_block_count:]
                    )
                prepared_ref_frame_count = frame_count

            clip_ref_items = list(ref_items or [])
            clip_ref_blocks = list(ref_blocks or [])
            audio_native_offset = sum(
                1
                for item in (ref_items or [])
                if isinstance(item, dict) and item.get("type") == "audio"
            )
            if selected_ref_audio_count:
                if (reference_count(refs) + active_ref_video_count) < 1:
                    raise ValueError(
                        "MiniMax H3 Extender: standalone reference audio requires "
                        "at least one image or video reference."
                    )
                audio_items, audio_blocks = prepare_standalone_audio_refs(
                    audio_vae,
                    selected_ref_audios,
                    selected_audio_offsets,
                    frame_count / float(fps),
                    cache=standalone_audio_cache,
                )
                clip_ref_items.extend(audio_items)
                clip_ref_blocks.extend(audio_blocks)

            clip_model, clip_text_encoder = apply_per_clip_loras(
                extender_self, model, clip, cfg.get("loras"), i
            )

            positive, _empty = make_ref2va_conditioning(
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

            # Init from upscaled draft video; keep draft audio channels.
            draft_audio = draft_audio.to(
                device=comfy.model_management.intermediate_device()
            )
            up_video = up_video.to(device=draft_audio.device)
            latent = {
                "samples": comfy.nested_tensor.NestedTensor((up_video, draft_audio))
            }

            trim_frames = trim_from_draft if i > 0 else 0
            if i > 0:
                if previous_proxy is None:
                    raise RuntimeError(
                        "MiniMax H3 Extender: previous refine latent missing."
                    )
                positive, trim_frames, _, _, _ = motion_ctx.apply(
                    positive,
                    latent,
                    previous_proxy,
                    str(context_length),
                    int(audio_context_length),
                )

            sampled = sample_h3(
                clip_model,
                positive,
                latent,
                cfg["seed"],
                str(sampler_name),
                str(scheduler),
                int(refine_steps),
                float(refine_denoise),
                sigmas=sample_sigmas,
            )

            # Preserve draft audio exactly; refine only the video stream.
            out_video, _out_audio = _streams_from_latent(sampled, "samples")
            sampled = {
                "samples": comfy.nested_tensor.NestedTensor(
                    (out_video, draft_audio.to(device=out_video.device))
                )
            }

            desc, geom = _append_segment(
                refine_data,
                sampled,
                i,
                int(trim_frames or 0),
                False,
                refine_manifest,
            )
            if refine_manifest.get("geometry") is None:
                refine_manifest["geometry"] = geom
            if isinstance(draft_desc.get("color_adjustment"), dict):
                desc["color_adjustment"] = dict(draft_desc["color_adjustment"])
            desc["clip_id"] = str(cfg.get("id") or f"clip_{i + 1}")
            refine_manifest["segments"].append(desc)
            refine_manifest["final_frame_count"] = _final_frame_count(
                refine_manifest["segments"]
            )
            refine_manifest["updated_at"] = time.time()
            refine_manifest = _invalidate_layer_preview_cache(
                refine_data, refine_manifest_path, refine_manifest
            )
            _write_json_atomic(refine_manifest_path, refine_manifest)

            previous_proxy = sampled
            generated.append(i)
            statuses.append(f"refine clip {i + 1} candidate cached")
            del sampled, positive, latent, clip_model, clip_text_encoder, up_video

            send_progress(
                owner,
                i,
                len(clips),
                "complete",
                f"Refine clip {i + 1}/{len(clips)} complete",
            )

            if mode == "clip_by_clip":
                break
    finally:
        offload_upscale_models()

    # Mirror refine validated/cached prefix back onto clip cards.
    segments = list(refine_manifest.get("segments", []))
    for i, cfg in enumerate(clips):
        if i < len(segments):
            cfg["refine_validated"] = bool(segments[i].get("validated", False))
        else:
            cfg["refine_validated"] = False
    _enforce_refine_validated_prefix(clips)

    return {
        "manifest": refine_manifest,
        "generated": generated,
        "statuses": statuses,
        "cached_count": len(segments),
        "validated_count": int(_validated_prefix_count(segments)),
        "run_mode": mode,
    }
