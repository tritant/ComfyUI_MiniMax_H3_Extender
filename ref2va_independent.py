"""Independent Ref2VA execution/cache support.

This module intentionally keeps the non-causal Ref2VA path separate from the
existing Ref2VA + Motion Context implementation.  It reuses the proven FL2VA
random-access cache primitives (stable clip ids, per-plan decoded sidecars,
stream-copy final assembly) without any First/Last/Previous conditioning.

The important invariant is simple: changing, inserting, deleting or rerunning
one clip never invalidates any other clip merely because of timeline position.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SEQUENCE_MODE = "ref2va_independent"
INTERRUPT_MODE = "ref2va"
BUILD = "ref2va-independent-v2.7.0"


def cache_owner_id(owner_id) -> str:
    from .motion_context_disk import _safe_name
    return f"extender_{_safe_name(owner_id)}_ref2va_independent"


def sync_manifest(owner_id, fps: float, clip_ids):
    """Reorder/drop cached clips by stable id without touching latent blobs."""
    from . import motion_context_disk as d
    from . import fl2va_engine as f

    data_path, manifest_path, manifest = d._manifest_for_first(cache_owner_id(owner_id), fps)
    manifest = dict(manifest)
    manifest["sequence_mode"] = SEQUENCE_MODE
    wanted = [str(x) for x in (clip_ids or [])]

    # Reuse FL2VA's per-plan decoded sidecar layout. The physical latent owner is
    # different, so the two modes can never collide despite sharing the helper.
    f._cleanup_plan_video_cache(data_path, wanted)
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
        desc.pop("first_source", None)
        desc.pop("previous_clip_id", None)
        desc.pop("previous_frame_signature", None)
        ordered.append(desc)

    if ordered != list(manifest.get("segments", [])):
        manifest = f._invalidate_derived_preview(data_path, manifest)
    manifest["segments"] = ordered
    manifest["final_frame_count"] = d._final_frame_count(ordered)
    raw_audio_offsets = manifest.get("independent_audio_offsets")
    if isinstance(raw_audio_offsets, dict):
        wanted_set = set(wanted)
        manifest["independent_audio_offsets"] = {
            str(k): v for k, v in raw_audio_offsets.items() if str(k) in wanted_set
        }
    manifest["build"] = BUILD
    manifest["updated_at"] = time.time()
    d._write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest



def compact_cache(owner_id, *, force=False):
    """Compact stale independent Ref2VA latent/PCM blobs by stable clip id.

    This is the same proven append-only maintenance strategy as FL2VA, but it
    operates on the physically separate independent Ref2VA cache owner.
    """
    from . import motion_context_disk as d
    from . import fl2va_engine as f

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
    compact_latent = f._should_compact(
        latent_physical, latent_live, force=bool(force),
        dead_limit=f._COMPACT_LATENT_DEAD_BYTES,
        ratio_min_dead=f._COMPACT_LATENT_RATIO_MIN_DEAD,
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
    compact_audio = bool(audio_specs) and audio_path.exists() and f._should_compact(
        audio_physical, audio_live, force=bool(force),
        dead_limit=f._COMPACT_AUDIO_DEAD_BYTES,
        ratio_min_dead=f._COMPACT_AUDIO_RATIO_MIN_DEAD,
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
                if src.read(int(d._DATA_START)) != d._DATA_MAGIC:
                    raise ValueError("Independent Ref2VA cache compaction: invalid latent cache magic.")
                dst.write(d._DATA_MAGIC)
                for i, desc in enumerate(updated_segments):
                    item = dict(desc)
                    item["video"] = f._copy_cache_spec(src, dst, item["video"])
                    item["audio"] = f._copy_cache_spec(src, dst, item["audio"])
                    item["segment_end"] = int(dst.tell())
                    updated_segments[i] = item
                dst.flush(); os.fsync(dst.fileno())

        if compact_audio:
            with open(audio_path, "rb", buffering=0) as src, open(audio_tmp, "wb", buffering=0) as dst:
                if src.read(int(d._AUDIO_CACHE_START)) != d._AUDIO_CACHE_MAGIC:
                    raise ValueError("Independent Ref2VA cache compaction: invalid decoded-audio cache magic.")
                dst.write(d._AUDIO_CACHE_MAGIC)
                for i, desc in enumerate(updated_segments):
                    item = dict(desc)
                    meta = item.get("decoded_audio")
                    if isinstance(meta, dict) and meta.get("storage") == "audio_cache" and isinstance(meta.get("waveform"), dict):
                        meta = dict(meta)
                        meta["waveform"] = f._copy_cache_spec(src, dst, meta["waveform"])
                        item["decoded_audio"] = meta
                        updated_segments[i] = item
                dst.flush(); os.fsync(dst.fileno())

        updated = dict(manifest)
        updated["segments"] = updated_segments
        updated["final_frame_count"] = d._final_frame_count(updated_segments)
        updated["updated_at"] = time.time()
        updated["last_compacted_at"] = time.time()
        reclaimed = (latent_physical - latent_live if compact_latent else 0) + (audio_physical - audio_live if compact_audio else 0)
        updated["last_compaction_reclaimed_bytes"] = max(0, int(reclaimed))
        manifest_tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(manifest_tmp, "r+b") as fh:
            os.fsync(fh.fileno())

        replacements = []
        if compact_latent: replacements.append((data_path, data_tmp))
        if compact_audio: replacements.append((audio_path, audio_tmp))
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
                try: target.unlink(missing_ok=True)
                except OSError: pass
            for target, backup in reversed(backups):
                if backup.exists(): os.replace(backup, target)
            raise
        else:
            for _, backup in backups:
                try: backup.unlink(missing_ok=True)
                except OSError: pass
        return updated, {"compacted": True, "reclaimed_bytes": max(0, int(reclaimed))}
    finally:
        for path in (data_tmp, audio_tmp, manifest_tmp):
            try: path.unlink(missing_ok=True)
            except OSError: pass

def cached_ids(manifest) -> set[str]:
    return {
        str(x.get("clip_id"))
        for x in (manifest or {}).get("segments", [])
        if str(x.get("clip_id") or "")
    }


def computed_ids(manifest) -> set[str]:
    return {
        str(x.get("clip_id"))
        for x in (manifest or {}).get("segments", [])
        if str(x.get("clip_id") or "")
        and bool(x.get("computed", False))
        and not bool(x.get("validated", False))
    }


def store_segment(
    owner_id,
    fps,
    clip_ids,
    clip_index,
    clip_id,
    samples,
    *,
    validated=False,
    run_mode="full_batch",
    computed=False,
):
    """Append latent bytes and replace/insert one logical clip by stable id."""
    from . import motion_context_disk as d
    from . import fl2va_engine as f

    data_path, manifest_path, manifest = sync_manifest(owner_id, fps, clip_ids)
    segments = [dict(x) for x in manifest.get("segments", [])]
    clip_index = int(clip_index)
    clip_id = str(clip_id)

    # Rerender only this clip's derived video caches.
    f._invalidate_plan_video_cache(data_path, clip_id)

    desc, geom = d._append_segment(
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
    if bool(computed) and not bool(validated):
        desc["computed"] = True
    else:
        desc.pop("computed", None)

    existing_pos = next(
        (i for i, x in enumerate(segments) if str(x.get("clip_id") or "") == clip_id),
        None,
    )
    if existing_pos is not None:
        segments[existing_pos] = desc
    else:
        segments.insert(max(0, min(clip_index, len(segments))), desc)

    order = {str(cid): i for i, cid in enumerate(clip_ids or [])}
    segments.sort(key=lambda x: order.get(str(x.get("clip_id") or ""), 10**9))
    for idx, item in enumerate(segments):
        item["index"] = idx
        item["trim_frames"] = 0

    manifest = f._invalidate_derived_preview(data_path, manifest)
    manifest["sequence_mode"] = SEQUENCE_MODE
    manifest["geometry"] = geom if manifest.get("geometry") is None else manifest["geometry"]
    manifest["segments"] = segments
    manifest["final_frame_count"] = d._final_frame_count(segments)
    manifest["build"] = BUILD
    manifest["updated_at"] = time.time()
    d._write_json_atomic(manifest_path, manifest)

    try:
        compacted_manifest, _compact_info = compact_cache(owner_id, force=False)
    except Exception as exc:
        print(f"[WARNING] Independent Ref2VA cache compaction skipped: {exc}")
        compacted_manifest = None
    if compacted_manifest is not None:
        manifest = compacted_manifest
        segments = [dict(x) for x in manifest.get("segments", [])]

    pos = next(i for i, x in enumerate(segments) if str(x.get("clip_id") or "") == clip_id)
    status = f"Ref2VA independent clip {clip_index + 1} {'validated' if validated else 'candidate'} cached"
    handle = d._make_handle(
        data_path,
        manifest_path,
        manifest,
        str(run_mode),
        stop=False,
        status=status,
        next_index=len(segments),
    )
    size = d._cache_size_mb(data_path, manifest_path)
    return handle, d._proxy_at(data_path, manifest, pos), manifest, status, size


def set_validation(owner_id, fps, clip_ids, validation_by_id, run_mode="full_batch"):
    from . import motion_context_disk as d

    data_path, manifest_path, manifest = sync_manifest(owner_id, fps, clip_ids)
    segments = [dict(x) for x in manifest.get("segments", [])]
    changed = False
    for desc in segments:
        clip_id = str(desc.get("clip_id") or "")
        value = bool(validation_by_id.get(clip_id, False))
        if bool(desc.get("validated", False)) != value:
            desc["validated"] = value
            changed = True
        if value and bool(desc.get("computed", False)):
            desc.pop("computed", None)
            changed = True
    if changed:
        manifest = dict(manifest)
        manifest["segments"] = segments
        manifest["updated_at"] = time.time()
        d._write_json_atomic(manifest_path, manifest)
    handle = d._make_handle(
        data_path,
        manifest_path,
        manifest,
        str(run_mode),
        stop=False,
        status="Ref2VA independent cache synchronized",
        next_index=len(segments),
    )
    return handle, manifest


def drop_cached_ids(owner_id, fps, clip_ids, drop_ids, preserve_preview=False):
    from . import motion_context_disk as d
    from . import fl2va_engine as f

    drop = {str(x) for x in (drop_ids or []) if str(x)}
    data_path, manifest_path, manifest = sync_manifest(owner_id, fps, clip_ids)
    if not drop:
        return data_path, manifest_path, manifest

    old_segments = [dict(x) for x in manifest.get("segments", [])]
    segments = [x for x in old_segments if str(x.get("clip_id") or "") not in drop]
    if len(segments) == len(old_segments):
        return data_path, manifest_path, manifest

    for clip_id in drop:
        f._invalidate_plan_video_cache(data_path, clip_id)
    if not bool(preserve_preview):
        manifest = f._invalidate_derived_preview(data_path, manifest)
    for idx, desc in enumerate(segments):
        desc["index"] = idx
        desc["trim_frames"] = 0
    manifest["segments"] = segments
    manifest["final_frame_count"] = d._final_frame_count(segments)
    manifest["updated_at"] = time.time()
    d._write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest


def cache_full_batch_plan(
    owner_id,
    fps,
    clip_ids,
    clip_id,
    vae,
    *,
    audio_vae=None,
    export_profile=None,
    color_adjustment=None,
):
    """Secure one completed independent Ref2VA clip as preview/final caches."""
    from . import motion_context_disk as d
    from . import fl2va_engine as f

    data_path, manifest_path, manifest = sync_manifest(owner_id, fps, clip_ids)
    wanted_id = str(clip_id)
    desc = next(
        (dict(x) for x in manifest.get("segments", []) if str(x.get("clip_id") or "") == wanted_id),
        None,
    )
    if desc is None:
        raise ValueError(f"Ref2VA independent Full Batch cache: clip {wanted_id!r} is missing.")

    adjustment = d._normalize_color_adjustment(
        color_adjustment if color_adjustment is not None else desc.get("color_adjustment")
    )
    profile = d.normalize_full_batch_export_profile(export_profile) if export_profile is not None else None
    cache_desc = dict(desc)
    cache_desc["color_adjustment"] = adjustment

    ffmpeg = d._find_ffmpeg()
    video_path, decoded_now = f._ensure_plan_video_cache(
        data_path,
        cache_desc,
        vae,
        float(fps),
        ffmpeg,
        encode_crf=d.FULL_BATCH_H264_CACHE_CRF,
        encode_preset=d.FULL_BATCH_H264_CACHE_PRESET,
        final_export_profile=profile,
        final_color_adjustment=adjustment,
        final_handoff_enabled=False,
        require_continuity=False,
    )

    manifest = d._load_manifest_from_paths(data_path, manifest_path) or manifest
    tagged_segments = [dict(x) for x in manifest.get("segments", [])]
    for item in tagged_segments:
        if str(item.get("clip_id") or "") == wanted_id:
            item["decoded_video_profile"] = d.FULL_BATCH_H264_CACHE_PROFILE
            item["color_adjustment"] = adjustment
            break
    manifest = dict(manifest)
    manifest["segments"] = tagged_segments
    manifest["updated_at"] = time.time()
    d._write_json_atomic(manifest_path, manifest)

    final_cached = False
    if profile is not None:
        manifest = d._load_manifest_from_paths(data_path, manifest_path) or manifest
        current_desc = next(
            dict(x) for x in manifest.get("segments", [])
            if str(x.get("clip_id") or "") == wanted_id
        )
        visible_frames = int(current_desc.get("frames", 0))
        final_path = f._plan_final_video_cache_path(data_path, wanted_id, profile)
        if final_path.exists() and final_path.stat().st_size > 0:
            manifest = f._tag_fl2va_final_segment_cache(
                manifest_path, manifest, wanted_id, profile, adjustment, visible_frames
            )
            final_cached = True

    audio_cached = False
    if audio_vae is not None:
        manifest = d._load_manifest_from_paths(data_path, manifest_path) or manifest
        manifest, selected = f._ensure_plan_audio_cache(
            data_path,
            manifest_path,
            manifest,
            audio_vae,
            float(fps),
            selected_clip_ids=[wanted_id],
        )
        if selected:
            cached_audio = d._load_cached_decoded_audio(data_path, selected[0])
            audio_cached = cached_audio is not None
            if cached_audio is not None:
                del cached_audio

    return manifest, {
        "video_cached": bool(Path(video_path).exists() and Path(video_path).stat().st_size > 0),
        "video_decoded_now": bool(decoded_now),
        "final_cached": bool(final_cached),
        "audio_cached": bool(audio_cached),
    }


def cache_state(owner_id):
    from . import motion_context_disk as d

    data_path, manifest_path = d._chain_paths(cache_owner_id(owner_id))
    if not data_path.exists() or not manifest_path.exists():
        return None
    manifest = d._load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        return None
    segments = [dict(x) for x in manifest.get("segments", [])]
    geometry = manifest.get("geometry") if isinstance(manifest.get("geometry"), dict) else {}
    return {
        "manifest": manifest,
        "cached_clip_ids": [str(x.get("clip_id")) for x in segments if str(x.get("clip_id") or "")],
        "validated_clip_ids": [
            str(x.get("clip_id")) for x in segments
            if str(x.get("clip_id") or "") and bool(x.get("validated", False))
        ],
        "cached_count": len(segments),
        "validated_count": sum(bool(x.get("validated", False)) for x in segments),
        "frame_count": int(manifest.get("final_frame_count", 0)),
        "resolved_width": int(geometry.get("video_w", 0) or 0) * 16,
        "resolved_height": int(geometry.get("video_h", 0) or 0) * 16,
    }


def export_final(**kwargs):
    """Use FL2VA's proven independent hard-cut decoder/assembler.

    Independent Ref2VA has no Previous links, so the FL2VA exporter naturally
    takes the simple full-plan path for every clip.  The cache owner is already
    separate; only user-facing mode labels are rewritten here.
    """
    from .fl2va_engine import export_fl2va_final

    # Independent Ref2VA never consumes a Previous/continuity handoff between
    # clips. Requiring FL2VA continuity sidecars here would force a useless
    # VideoVAE re-decode of every cached COMPUTED clip after Interrupt/Save/Load.
    kwargs["require_continuity"] = False
    result = export_fl2va_final(**kwargs)
    try:
        infos = result.get("ui", {}).get("h3_preview_info", [])
        for info in infos:
            if isinstance(info, dict):
                mode = str(info.get("mode") or "")
                if mode.startswith("fl2va_"):
                    info["mode"] = "ref2va_independent_" + mode[len("fl2va_"):]
    except Exception:
        pass
    return result



def _stable_standalone_audio_clip_plan(clips, ref_audios, manifest):
    """Return Ref2VA standalone-audio selection with offsets pinned by clip id.

    Independent Ref2VA allows insert/delete/rerun without changing neighbours.
    The normal Ref2VA audio planner advances long global audio through timeline
    position, which would silently move an old clip's source slice after an
    insertion.  Pin each selected slot's offset to the stable clip id the first
    time this independent timeline sees it. Existing pins are never shifted by
    reordering/insertion; newly introduced clips use the normal current-timeline
    offset as their initial value.
    """
    from . import extender as e

    base_plan = e._build_standalone_audio_clip_plan(clips, ref_audios)
    raw_pins = manifest.get("independent_audio_offsets") if isinstance(manifest, dict) else None
    pins = raw_pins if isinstance(raw_pins, dict) else {}
    pins = {
        str(clip_id): {
            str(slot): float(offset)
            for slot, offset in (slot_map.items() if isinstance(slot_map, dict) else [])
            if str(slot).isdigit()
        }
        for clip_id, slot_map in pins.items()
        if str(clip_id)
    }

    wanted_ids = {str(cfg.get("id") or f"clip_{i + 1}") for i, cfg in enumerate(clips)}
    pins = {clip_id: slot_map for clip_id, slot_map in pins.items() if clip_id in wanted_ids}

    out = []
    changed = pins != (raw_pins if isinstance(raw_pins, dict) else {})
    for i, (selected, slots, base_offsets) in enumerate(base_plan):
        clip_id = str(clips[i].get("id") or f"clip_{i + 1}")
        slot_map = dict(pins.get(clip_id) or {})
        resolved = {}
        for slot in slots:
            key = str(int(slot))
            if key in slot_map:
                resolved[int(slot)] = float(slot_map[key])
            else:
                value = float(base_offsets.get(int(slot), 0.0))
                slot_map[key] = value
                resolved[int(slot)] = value
                changed = True
        # Keep only slots still relevant to this clip's prompt/connected refs.
        keep = {str(int(slot)) for slot in slots}
        trimmed = {k: v for k, v in slot_map.items() if k in keep}
        if trimmed != slot_map:
            changed = True
        if trimmed:
            pins[clip_id] = trimmed
        elif clip_id in pins:
            pins.pop(clip_id, None)
            changed = True
        out.append((selected, list(slots), resolved))
    return out, pins, changed

def run(
    extender_instance,
    *,
    owner,
    clips,
    active_prompt_pack_signature,
    prompt_pack_imported,
    external_prompt_pack,
    model,
    clip,
    vae,
    audio_vae,
    run_mode,
    width,
    height,
    ref_image_size,
    steps,
    sampler_name,
    scheduler,
    denoise,
    resolution_mode,
    megapixels,
    refs_json,
    ref_pack,
    export_profile,
    kwargs,
):
    """Execute Ref2VA with no Motion Context and random-access clip caches."""
    from . import extender as e
    from . import motion_context_disk as d

    clip_ids = [str(cfg.get("id") or f"clip_{i + 1}") for i, cfg in enumerate(clips)]
    data_path, manifest_path, manifest = sync_manifest(owner, e.FPS, clip_ids)

    refs = e._parse_refs_json(refs_json)
    external_ref_pack = e._normalize_external_ref_pack(ref_pack)
    local_picture_slots = e._local_picture_slot_reservations(clips)
    refs, ref_pack_imported_slots, ref_pack_skipped_slots = e._sync_refs_from_ref_pack(
        refs, external_ref_pack, local_picture_slots
    )
    if (ref_pack_imported_slots or ref_pack_skipped_slots) and external_ref_pack is not None:
        e._send_extender_ref_pack_import(
            owner,
            e._refs_json(refs),
            ref_pack_imported_slots,
            int(external_ref_pack.get("count", 0) or 0),
            external_ref_pack.get("source") or "External reference pack",
            skipped_slots=ref_pack_skipped_slots,
        )

    refs_signature = e._refs_signature(refs)
    requested_resolution = e._resolve_generation_resolution(
        resolution_mode, megapixels, width, height, refs
    )
    resolution = dict(requested_resolution)
    resolution["requested_width"] = int(requested_resolution["width"])
    resolution["requested_height"] = int(requested_resolution["height"])
    resolution["cache_reset"] = False
    resolved_width = int(resolution["width"])
    resolved_height = int(resolution["height"])

    cache_resolution = e._resolution_from_manifest(manifest)
    previous_cache_resolution = None
    if manifest.get("segments") and cache_resolution is not None:
        if (
            int(cache_resolution["width"]) != resolved_width
            or int(cache_resolution["height"]) != resolved_height
        ):
            previous_cache_resolution = dict(cache_resolution)
            # Geometry is the only global invalidation: a random-access cache still
            # cannot mix incompatible latent tensor sizes.
            for cid in list(cached_ids(manifest)):
                from . import fl2va_engine as f
                f._invalidate_plan_video_cache(data_path, cid)
            manifest = d._truncate_chain(data_path, manifest_path, manifest, 0)
            manifest = dict(manifest)
            manifest["sequence_mode"] = SEQUENCE_MODE
            manifest["updated_at"] = time.time()
            d._write_json_atomic(manifest_path, manifest)
            for cfg in clips:
                cfg["validated"] = False
            resolution["cache_reset"] = True

    if prompt_pack_imported and external_prompt_pack is not None:
        imported_json = e._state_json(clips, active_prompt_pack_signature, "ref2va", motion_context=False)
        e._send_extender_prompt_pack_import(
            owner,
            imported_json,
            len(external_prompt_pack.get("prompts") or []),
            external_prompt_pack.get("source") or "External prompt pack",
        )

    data_path, manifest_path, manifest = sync_manifest(owner, e.FPS, clip_ids)
    manifest = dict(manifest)
    manifest["extender_refs_signature"] = refs_signature
    manifest["extender_ref_ids"] = [ref.get("id") if isinstance(ref, dict) else None for ref in refs]
    manifest["updated_at"] = time.time()
    d._write_json_atomic(manifest_path, manifest)

    manifest, active_export_profile = e._pin_full_batch_export_profile(
        manifest_path, manifest, run_mode, export_profile
    )
    d.clear_full_batch_interrupt(owner, INTERRUPT_MODE)
    manifest, _resume_checkpoint = e._begin_batch_checkpoint(manifest_path, manifest, run_mode)
    cached = cached_ids(manifest)

    legacy_ref_audio = kwargs.get("ref_audio")
    ref_audios = [kwargs.get(f"ref_audio_{index}") for index in range(1, e.MAX_STANDALONE_AUDIO_REFS + 1)]
    if all(audio is None for audio in ref_audios) and legacy_ref_audio is not None:
        ref_audios[0] = legacy_ref_audio
    ref_videos = [kwargs.get(f"ref_video_{index}") for index in range(1, e.MAX_VIDEO_REFS + 1)]
    ref_video_fps = [kwargs.get(f"ref_video_fps_{index}", float(e.FPS)) for index in range(1, e.MAX_VIDEO_REFS + 1)]
    ref_video_audios = [kwargs.get(f"ref_video_audio_{index}") for index in range(1, e.MAX_VIDEO_REFS + 1)]
    active_ref_video_count = sum(video is not None for video in ref_videos)
    active_ref_video_audio_count = sum(audio is not None for audio in ref_video_audios)
    active_ref_audio_count = sum(audio is not None for audio in ref_audios)
    standalone_audio_clip_plan, independent_audio_offsets, audio_offsets_changed = (
        _stable_standalone_audio_clip_plan(clips, ref_audios, manifest)
    )
    if audio_offsets_changed or manifest.get("independent_audio_offsets") != independent_audio_offsets:
        manifest = dict(manifest)
        manifest["independent_audio_offsets"] = independent_audio_offsets
        manifest["updated_at"] = time.time()
        d._write_json_atomic(manifest_path, manifest)
    standalone_audio_cache = {}
    local_media_decode_cache = {}

    ref_items = None
    ref_blocks = None
    active_picture_slots = None
    active_video_slots = None
    prepared_ref_frame_count = None
    prepared_image_blocks = None
    prepared_video_blocks_by_frame_count = {}

    generated = []
    statuses = []
    interrupted = False
    interrupted_after = 0

    if str(run_mode) == "full_batch":
        e._LOG.info(
            "H3 Full Batch Ref2VA independent start: validated=%s cached_ids=%s",
            [bool(c.get("validated", False)) for c in clips],
            sorted(cached),
        )

    for i, cfg in enumerate(clips):
        if (
            str(run_mode) == "full_batch"
            and i > 0
            and d.full_batch_interrupt_requested(owner, INTERRUPT_MODE, consume=True)
        ):
            interrupted = True
            interrupted_after = i
            break

        clip_id = clip_ids[i]
        current_manifest = d._load_manifest_from_paths(data_path, manifest_path) or manifest
        current_desc = next(
            (dict(x) for x in current_manifest.get("segments", []) if str(x.get("clip_id") or "") == clip_id),
            None,
        )
        is_cached = current_desc is not None and clip_id in cached_ids(current_manifest)

        if bool(cfg.get("validated", False)):
            if not is_cached:
                raise RuntimeError(
                    f"MiniMax H3 Extender: Ref2VA independent clip {i + 1} is marked Validated "
                    "but its cache is missing. Refusing to rerender a validated clip. "
                    "Uncheck Validated explicitly if you want this clip rendered again."
                )
            e._LOG.info(
                "H3 validated hard-skip: Ref2VA independent clip %d exists on disk; sampler forbidden",
                i + 1,
            )
            continue

        if (
            str(run_mode) == "full_batch"
            and is_cached
            and bool(current_desc.get("computed", False))
            and not bool(current_desc.get("validated", False))
        ):
            e._LOG.info(
                "H3 COMPUTED hard-reuse: Ref2VA independent clip %d exists on disk; sampler forbidden",
                i + 1,
            )
            e._send_extender_progress(
                owner, i, len(clips), "sampling",
                f"Checking Ref2VA independent checkpoint {i + 1}/{len(clips)}",
            )
            manifest, _cache_info = cache_full_batch_plan(
                owner,
                float(e.FPS),
                clip_ids,
                clip_id,
                vae,
                audio_vae=audio_vae,
                export_profile=active_export_profile,
                color_adjustment=cfg.get("color_adjustment"),
            )
            statuses.append(f"Ref2VA independent clip {i + 1} resumed from checkpoint")
            continue

        e._send_extender_progress(
            owner, i, len(clips), "preparing",
            f"Preparing Ref2VA independent clip {i + 1}/{len(clips)}",
        )
        cfg["validated"] = False
        frame_count = e._duration_to_frames(cfg["duration"])

        local_refs = e._normalize_local_refs(cfg.get("local_refs"))
        clip_refs = list(refs)
        clip_ref_videos = list(ref_videos)
        clip_ref_video_fps = list(ref_video_fps)
        clip_ref_video_audios = list(ref_video_audios)

        local_visual = False
        for item in local_refs.get("images", []):
            slot = int(item["slot"])
            if clip_refs[slot - 1] is not None:
                e._LOG.warning(
                    "H3 Extender: Clip %d local Picture %d overrides a conflicting global Picture %d for this clip.",
                    i + 1, slot, slot,
                )
            clip_refs[slot - 1] = item["ref"]
            local_visual = True

        for item in local_refs.get("videos", []):
            slot = int(item["slot"])
            if clip_ref_videos[slot - 1] is not None:
                e._LOG.warning(
                    "H3 Extender: Clip %d local Video %d overrides a conflicting global Video %d for this clip.",
                    i + 1, slot, slot,
                )
            clip_ref_videos[slot - 1] = e._load_local_video_media(
                item["media"], frame_count, cache=local_media_decode_cache
            )
            clip_ref_video_fps[slot - 1] = float(e.FPS)
            clip_ref_video_audios[slot - 1] = None
            local_visual = True

        selected_ref_audios, selected_audio_slots, selected_audio_offsets = standalone_audio_clip_plan[i]
        selected_ref_audios = list(selected_ref_audios)
        selected_audio_slots = list(selected_audio_slots)
        selected_audio_offsets = dict(selected_audio_offsets)
        for item in local_refs.get("audios", []):
            slot = int(item["slot"])
            if ref_audios[slot - 1] is not None:
                e._LOG.warning(
                    "H3 Extender: Clip %d local Audio %d overrides a conflicting global Audio %d for this clip.",
                    i + 1, slot, slot,
                )
            selected_ref_audios[slot - 1] = e._load_local_audio_media(
                item["media"], cache=local_media_decode_cache
            )
            if slot not in selected_audio_slots:
                selected_audio_slots.append(slot)
            selected_audio_offsets[slot] = 0.0
        selected_audio_slots = sorted(set(int(x) for x in selected_audio_slots))
        selected_ref_audio_count = len(selected_audio_slots)

        active_clip_video_count = sum(video is not None for video in clip_ref_videos)
        clip_mixed_ref_count = e._reference_count(clip_refs) + active_clip_video_count + selected_ref_audio_count
        if clip_mixed_ref_count > e.MAX_MIXED_REF_ITEMS:
            raise ValueError(
                f"MiniMax H3 Extender: Clip {i + 1} has {clip_mixed_ref_count} mixed references; "
                f"MiniMax H3 supports at most {e.MAX_MIXED_REF_ITEMS}."
            )

        if local_visual:
            clip_base_items, clip_base_blocks, clip_picture_slots, clip_video_slots = e._prepare_shared_refs(
                vae,
                audio_vae,
                resolved_width,
                resolved_height,
                str(ref_image_size),
                clip_refs,
                ref_videos=clip_ref_videos,
                ref_video_fps=clip_ref_video_fps,
                ref_video_audios=clip_ref_video_audios,
                standalone_audio_count=0,
                frame_count=frame_count,
            )
        else:
            needs_ref_prepare = (
                ref_items is None
                or ref_blocks is None
                or active_picture_slots is None
                or active_video_slots is None
                or (active_ref_video_count and prepared_ref_frame_count != frame_count)
            )
            if needs_ref_prepare:
                cached_video_blocks = prepared_video_blocks_by_frame_count.get(int(frame_count))
                ref_items, ref_blocks, active_picture_slots, active_video_slots = e._prepare_shared_refs(
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
                    duration_key = int(frame_count)
                    prepared_video_blocks_by_frame_count.pop(duration_key, None)
                    prepared_video_blocks_by_frame_count[duration_key] = list((ref_blocks or [])[image_block_count:])
                    while len(prepared_video_blocks_by_frame_count) > 2:
                        oldest_key = next(iter(prepared_video_blocks_by_frame_count))
                        prepared_video_blocks_by_frame_count.pop(oldest_key, None)
                prepared_ref_frame_count = frame_count
            clip_base_items = ref_items or []
            clip_base_blocks = ref_blocks or []
            clip_picture_slots = active_picture_slots or []
            clip_video_slots = active_video_slots or []

        clip_ref_items = list(clip_base_items or [])
        clip_ref_blocks = list(clip_base_blocks or [])
        audio_native_offset = sum(
            1 for item in (clip_base_items or [])
            if isinstance(item, dict) and item.get("type") == "audio"
        )
        if selected_ref_audio_count:
            if (e._reference_count(clip_refs) + active_clip_video_count) < 1:
                raise ValueError(
                    "MiniMax H3 Extender: standalone reference audio requires at least one image or video reference."
                )
            audio_items, audio_blocks = e._prepare_standalone_audio_refs(
                audio_vae,
                selected_ref_audios,
                selected_audio_offsets,
                frame_count / float(e.FPS),
                cache=standalone_audio_cache,
            )
            clip_ref_items.extend(audio_items)
            clip_ref_blocks.extend(audio_blocks)

        clip_model, clip_text_encoder = e._apply_per_clip_loras(
            extender_instance, model, clip, cfg.get("loras"), i
        )
        positive, latent = e._make_ref2va_conditioning(
            clip_text_encoder,
            vae,
            cfg["prompt"],
            resolved_width,
            resolved_height,
            frame_count,
            clip_ref_items,
            clip_ref_blocks,
            clip_picture_slots,
            clip_video_slots,
            active_audio_slots=selected_audio_slots,
            audio_native_offset=audio_native_offset,
        )

        e._send_extender_progress(
            owner, i, len(clips), "sampling",
            f"Rendering Ref2VA independent clip {i + 1}/{len(clips)}",
        )
        sampled = e._sample_h3(
            clip_model,
            positive,
            latent,
            cfg["seed"],
            str(sampler_name),
            str(scheduler),
            int(steps),
            float(denoise),
        )

        _handle, _proxy, manifest, cache_status, _cache_mb = store_segment(
            owner,
            float(e.FPS),
            clip_ids,
            i,
            clip_id,
            sampled,
            validated=False,
            run_mode=str(run_mode),
            computed=(str(run_mode) == "full_batch"),
        )
        statuses.append(cache_status)
        generated.append(i)
        cached.add(clip_id)

        del sampled, positive, latent, clip_model, clip_text_encoder

        if str(run_mode) == "full_batch":
            e._send_extender_progress(
                owner, i, len(clips), "sampling",
                f"Caching Ref2VA independent clip {i + 1}/{len(clips)}",
            )
            manifest, _cache_info = cache_full_batch_plan(
                owner,
                float(e.FPS),
                clip_ids,
                clip_id,
                vae,
                audio_vae=audio_vae,
                export_profile=active_export_profile,
                color_adjustment=cfg.get("color_adjustment"),
            )

        e._send_extender_progress(
            owner, i, len(clips), "complete",
            f"Ref2VA independent clip {i + 1}/{len(clips)} complete",
        )

        if (
            str(run_mode) == "full_batch"
            and d.full_batch_interrupt_requested(owner, INTERRUPT_MODE, consume=True)
        ):
            interrupted = True
            interrupted_after = i + 1
            break

        if str(run_mode) == "clip_by_clip":
            break

    validation_by_id = {
        clip_ids[i]: bool(cfg.get("validated", False))
        for i, cfg in enumerate(clips)
    }
    previous_handle, final_manifest = set_validation(
        owner, e.FPS, clip_ids, validation_by_id, run_mode=str(run_mode)
    )

    if str(run_mode) == "full_batch":
        if interrupted:
            final_manifest = e._interrupt_batch_checkpoint(
                manifest_path, final_manifest, interrupted_after, len(clips)
            )
            previous_handle = dict(previous_handle)
            previous_handle["interrupted"] = True
            previous_handle["snapshot_count"] = int(interrupted_after)
            previous_handle["project_total_clips"] = int(len(clips))
            previous_handle["status"] = f"interrupted after clip {int(interrupted_after)}"
        else:
            final_manifest = e._finish_batch_checkpoint(manifest_path, final_manifest)

    # Per-clip color metadata follows stable clip ids.
    color_segments = [dict(x) for x in final_manifest.get("segments", [])]
    clip_by_id = {clip_ids[i]: clips[i] for i in range(len(clips))}
    color_changed = False
    for idx, desc in enumerate(color_segments):
        cfg = clip_by_id.get(str(desc.get("clip_id") or ""))
        if cfg is None:
            continue
        wanted = e._normalize_color_adjustment(cfg.get("color_adjustment"))
        if desc.get("color_adjustment") != wanted:
            desc["color_adjustment"] = wanted
            desc["final_video_dirty"] = True
            color_segments[idx] = desc
            color_changed = True
    if color_changed:
        final_manifest = dict(final_manifest)
        final_manifest["segments"] = color_segments
        final_manifest["updated_at"] = time.time()
        d._write_json_atomic(manifest_path, final_manifest)

    cached_now = cached_ids(final_manifest)
    validated_ids = {
        str(x.get("clip_id"))
        for x in final_manifest.get("segments", [])
        if bool(x.get("validated", False)) and str(x.get("clip_id") or "")
    }
    computed_now = computed_ids(final_manifest)
    cached_count = len(cached_now)
    validated_count = len(validated_ids)
    normalized_json = e._state_json(
        clips, active_prompt_pack_signature, "ref2va", motion_context=False
    )

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
            f" | resolution changed from {int(previous_cache_resolution['width'])}x"
            f"{int(previous_cache_resolution['height'])}: independent cache restarted"
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
        details = []
        if ref_pack_imported_slots:
            details.append("imported Ref " + ",".join(str(x) for x in ref_pack_imported_slots))
        if ref_pack_skipped_slots:
            details.append("ignored local-reserved Ref " + ",".join(str(x) for x in ref_pack_skipped_slots))
        ref_pack_text = f" | ref pack {connected_ref_count} linked"
        if details:
            ref_pack_text += ", " + "; ".join(details)

    status = (
        f"Ref2VA independent {str(run_mode)} | {resolution_text} | refs {e._reference_count(refs)} | "
        f"video refs {active_ref_video_count} | video audios {active_ref_video_audio_count} | "
        f"audio refs {active_ref_audio_count} | cached {cached_count}/{len(clips)} | "
        f"validated {validated_count}{prompt_pack_text}{ref_pack_text} | "
        + ("generated " + ",".join(str(i + 1) for i in generated) if generated else "disk only")
    )
    if interrupted:
        status += f" | INTERRUPTED after clip {int(interrupted_after)} | checkpoint saved"

    cache_mb = d._cache_size_mb(data_path, manifest_path)
    final_cache_resolution = e._resolution_from_manifest(final_manifest)
    e._send_extender_progress(owner, -1, len(clips), "idle", status)

    ui_state = {
        "generation_mode": "ref2va",
        "motion_context": False,
        "clips_json": normalized_json,
        "clip_count": len(clips),
        "cached_count": cached_count,
        "validated_count": validated_count,
        "cached_clip_ids": sorted(cached_now),
        "validated_clip_ids": sorted(validated_ids),
        "computed_clip_ids": sorted(computed_now),
        "computed_indices": [],
        "checkpoint_active": bool(interrupted),
        "checkpoint_interrupted": bool(interrupted),
        "checkpoint_snapshot_count": int(interrupted_after if interrupted else 0),
        "generated": [i + 1 for i in generated],
        "status": status,
        "resolved_width": resolved_width,
        "resolved_height": resolved_height,
        "resolution_mode": str(resolution.get("mode") or "manual"),
        "resolution_guide": (
            f"ref_{int(resolution['guide_ref'])}" if resolution.get("guide_ref") is not None else ""
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
        "reference_count": int(e._reference_count(refs)),
        "reference_video_count": int(active_ref_video_count),
        "reference_video_audio_count": int(active_ref_video_audio_count),
        "reference_audio_count": int(active_ref_audio_count),
        "prompt_pack_connected": external_prompt_pack is not None,
        "prompt_pack_imported": bool(prompt_pack_imported),
        "prompt_pack_count": int(len(external_prompt_pack.get("prompts") or [])) if external_prompt_pack is not None else 0,
        "prompt_pack_signature": str(active_prompt_pack_signature or ""),
        "per_clip_lora_count": int(sum(len(cfg.get("loras") or []) for cfg in clips)),
        "build": e.BUILD,
    }
    return {
        "ui": {"h3_extender_state": [ui_state]},
        "result": (
            previous_handle,
            int(len(clips)),
            int(validated_count),
            status,
            float(cache_mb),
            e.BUILD,
        ),
    }
