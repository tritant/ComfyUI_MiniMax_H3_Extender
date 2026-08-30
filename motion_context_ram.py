
"""
Direct in-RAM MiniMax H3 motion-context chaining.

The previous clip's *sampled H3 AV latent* is not copied into the next
sampling latent. Its tail is attached as never-denoised temporal
conditioning rows, which is the mechanism H3 already uses for FL2VA
keyframes.

The implementation follows the public H3 Motion Context research by
NikoDemon80, but removes disk Save/Load because this workflow keeps clip A
and clip B in the same ComfyUI DAG.
"""
import inspect
import logging
import math

import torch
import node_helpers
import comfy.nested_tensor
import comfy.ldm.minimax.model as minimax_model

from .patch_motion_layout import (
    MC_KEY,
    MC_AUDIO_KEY,
    apply_patch as _apply_layout_patch,
    is_applied as _layout_patch_applied,
)
from .patch_motion_payload import (
    apply_patch as _apply_payload_patch,
    is_applied as _payload_patch_applied,
)

FPS = 24
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

BUILD = "motion-context-ram-v2.1.1-perf-pass"
_LOG = logging.getLogger("minimax_h3_tail_from_latent.motion_context")


def _native_guide_api_supported():
    """Detect ComfyUI's native arbitrary MiniMax H3 guide API.

    Older builds expose PackedLayout(..., frame_count=...) and need the
    compatibility patches below. Newer builds removed frame_count and accept
    arbitrary resolved_frame_index values directly.
    """
    cls = getattr(minimax_model, "PackedLayout", None)
    if cls is None:
        return False
    try:
        params = inspect.signature(cls.__init__).parameters
    except Exception:
        return False
    return "frame_count" not in params


def _ensure_patches():
    if _native_guide_api_supported():
        return "native"

    if not _layout_patch_applied():
        if not _apply_layout_patch():
            raise RuntimeError(
                "MiniMax H3 Motion Context RAM: could not enable interior H3 "
                "keyframe anchors. Check the ComfyUI log for a conflicting H3 "
                "custom-node patch."
            )
    if not _payload_patch_applied():
        if not _apply_payload_patch():
            raise RuntimeError(
                "MiniMax H3 Motion Context RAM: could not enable Ref2VA "
                "keyframe/reference coexistence. Check the ComfyUI log."
            )
    return "compat"


def _streams_from_latent(latent, name):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(
            f"{name} must be a ComfyUI LATENT dict containing 'samples'."
        )

    samples = latent["samples"]

    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            f"{name} must be a MiniMax H3 joint AV latent, got {type(samples)!r}."
        )

    if len(parts) < 2:
        raise ValueError(f"{name} contains no H3 audio stream.")

    video, audio = parts[0], parts[1]

    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)

    if video.ndim != 5:
        raise ValueError(
            f"{name} video latent must be [B,C,T,H,W], got {tuple(video.shape)}."
        )
    if audio.ndim != 4:
        raise ValueError(
            f"{name} audio latent must be [B,C,2,T], got {tuple(audio.shape)}."
        )

    return video, audio


def _pixel_frames(latent_t):
    return sum(
        FRAME_PER_TOKEN[k % 5]
        for k in range(int(latent_t))
    )


def _step_offsets(latent_t):
    offsets = []
    acc = 0
    for k in range(int(latent_t)):
        offsets.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return offsets


def _steps_for_frames(frame_count):
    n = int(frame_count)
    steps = 0
    covered = 0
    while covered < n:
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == n else None



def _frames_from_video_t(video_t):
    """
    Exact H3 17k+5 inverse grid:
      video T 2,7,12,17,... -> frames 5,22,39,56,...
    """
    t = int(video_t)
    if t < 2 or (t - 2) % 5 != 0:
        raise ValueError(
            f"Invalid MiniMax H3 video latent T={t}; expected 2 mod 5."
        )
    return 5 + 17 * ((t - 2) // 5)


def _audio_t_for_frames(frame_count):
    # H3 audio latent runs at 40 Hz while target video runs at 24 fps.
    return int(round(float(frame_count) / float(FPS) * AUDIO_HZ))


def _video_tail_from_latent(context_latent, frame_count):
    video, _ = _streams_from_latent(
        context_latent, "context_latent"
    )
    total_t = int(video.shape[2])
    steps = _steps_for_frames(frame_count)

    if steps is None:
        raise ValueError(
            "MiniMax H3 Motion Context RAM: context length is not a whole "
            "number of H3 video latent steps."
        )
    if steps > total_t:
        raise ValueError(
            f"MiniMax H3 Motion Context RAM: need {steps} video latent steps, "
            f"previous clip only has {total_t}."
        )

    start = total_t - steps

    # H3 17k+5 clips and the offered context lengths are chosen so this is
    # cycle position 0. If not, the 1/4/4/4/4 frame-span phase is wrong.
    if start % 5 != 0:
        raise RuntimeError(
            "MiniMax H3 Motion Context RAM: previous latent/context length "
            f"phase mismatch (tail starts at cycle {start % 5})."
        )

    covered = _pixel_frames(steps)
    if covered != int(frame_count):
        raise RuntimeError(
            f"MiniMax H3 Motion Context RAM: {steps} latent steps cover "
            f"{covered} frames, expected {frame_count}."
        )

    blocks = [
        video[:1, :, start + k:start + k + 1].clone()
        for k in range(steps)
    ]
    offsets = _step_offsets(steps)
    return blocks, offsets, covered


def _audio_tail_from_latent(context_latent, audio_frames):
    video, audio = _streams_from_latent(
        context_latent, "context_latent"
    )

    total_t = int(audio.shape[-1])
    source_frames = _pixel_frames(int(video.shape[2]))

    # H3 rounds its 40 Hz audio grid to the nearest latent step. The
    # residual is therefore signed (normally -1/3, 0 or +1/3 step on the
    # valid 17n+5 video grid). Preserve that phase so the carried audio END
    # remains aligned with the exact source timeline.
    ideal_t = FRAME_RESCALE * source_frames
    expected_t = int(round(ideal_t))
    overhang = total_t - ideal_t
    if total_t != expected_t or not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "MiniMax H3 Motion Context RAM: unexpected audio grid "
            "(%d audio steps for %d frames; expected %d); ignoring phase offset.",
            total_t, source_frames, expected_t,
        )
        overhang = 0.0

    rt = int(round(
        int(audio_frames) / float(FPS) * AUDIO_HZ
    ))
    rt = min(rt, total_t)
    if rt < 1:
        raise ValueError(
            "MiniMax H3 Motion Context RAM: audio context window is empty."
        )

    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


def _decode_video(vae, latent):
    video, _ = _streams_from_latent(latent, "samples")
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(
            -1,
            images.shape[-3],
            images.shape[-2],
            images.shape[-1],
        )
    return images


def _decode_audio(audio_vae, latent):
    _, audio_latent = _streams_from_latent(latent, "samples")
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)

    # Mirror ComfyUI's native H3 audio decode normalization.
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std

    sr = int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 32000),
        )
    )
    return {"waveform": audio, "sample_rate": sr}


def _audio_exact_frames(audio, frames, fps=24.0):
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    wanted = max(0, int(round(
        float(frames) / float(fps) * sr
    )))

    if waveform.shape[-1] > wanted:
        waveform = waveform[..., :wanted]
    elif waveform.shape[-1] < wanted:
        # H3 normally rounds long, so this is only a defensive fallback.
        pad = wanted - waveform.shape[-1]
        waveform = torch.nn.functional.pad(
            waveform, (0, pad)
        )

    return {"waveform": waveform, "sample_rate": sr}






def _luma_map(frames):
    return (
        frames[..., 0] * 0.299
        + frames[..., 1] * 0.587
        + frames[..., 2] * 0.114
    )


def _luma_stats(frames):
    y = _luma_map(frames).detach().float().reshape(-1)
    if int(y.numel()) == 0:
        return 0.5, 0.1, 0.5
    return (
        float(y.mean().item()),
        float(y.std(unbiased=False).clamp_min(1e-5).item()),
        float(y.median().item()),
    )






def _pad_motion_context_block_to_target(block, target_video):
    """
    MiniMax H3 pads the TARGET video latent internally to the DiT patch grid
    (patch size 1x2x2), but condition latents are patchified directly.

    If an input resolution is not a multiple of 32, the VAE latent can have
    an odd H or W (e.g. 525x799 -> latent 32x49). The target is internally
    padded to 32x50, while an unpadded motion-context block would make
    patchify_video() reshape fail.

    Pad only the copied conditioning block, on the bottom/right, to the exact
    spatial grid the target will use. The sampled/output latent itself is NOT
    resized and keeps its original geometry.
    """
    if block.ndim != 5 or target_video.ndim != 5:
        raise ValueError(
            "MiniMax H3 Motion Context RAM: expected 5D video latents."
        )

    # H3 diffusion patch size is spatial 2x2.
    target_h = int(target_video.shape[3])
    target_w = int(target_video.shape[4])
    padded_h = ((target_h + 1) // 2) * 2
    padded_w = ((target_w + 1) // 2) * 2

    h = int(block.shape[3])
    w = int(block.shape[4])

    if h > padded_h or w > padded_w:
        raise RuntimeError(
            "MiniMax H3 Motion Context RAM: context block is larger than "
            f"target patch grid ({w}x{h} vs {padded_w}x{padded_h})."
        )

    pad_h = padded_h - h
    pad_w = padded_w - w

    if pad_h == 0 and pad_w == 0:
        return block

    # torch.nn.functional.pad for [B,C,T,H,W]:
    # (W_left, W_right, H_top, H_bottom)
    padded = torch.nn.functional.pad(
        block,
        (0, pad_w, 0, pad_h),
        mode="constant",
        value=0.0,
    )

    _LOG.info(
        "MiniMax H3 Motion Context RAM: padded context condition "
        "%dx%d -> %dx%d latent pixels for DiT 2x2 patch grid",
        w, h, int(padded.shape[4]), int(padded.shape[3]),
    )
    return padded



class MiniMaxH3MotionContextRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context_latent": ("LATENT",),
                "context_length": (
                    ["22", "5", "39", "56"],
                    {"default": "22"},
                ),
                "audio_context_length": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 240,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "conditioning",
        "trim_frames",
        "video_context_tokens",
        "audio_context_tokens",
        "build",
    )
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3"

    def apply(
        self,
        conditioning,
        latent,
        context_latent,
        context_length,
        audio_context_length,
    ):
        guide_api = _ensure_patches()

        target_video, _ = _streams_from_latent(
            latent, "latent"
        )
        source_video, _ = _streams_from_latent(
            context_latent, "context_latent"
        )

        if target_video.shape[0] != source_video.shape[0]:
            raise ValueError(
                "MiniMax H3 Motion Context RAM: batch size differs between "
                "previous and next clip."
            )
        if target_video.shape[1] != source_video.shape[1]:
            raise ValueError(
                "MiniMax H3 Motion Context RAM: video latent channels differ."
            )
        if target_video.shape[3:] != source_video.shape[3:]:
            sw = int(source_video.shape[4]) * 16
            sh = int(source_video.shape[3]) * 16
            tw = int(target_video.shape[4]) * 16
            th = int(target_video.shape[3]) * 16
            raise ValueError(
                "MiniMax H3 Motion Context RAM: resolution mismatch "
                f"{sw}x{sh} -> {tw}x{th}. Latent motion context cannot resize."
            )

        context_frames = int(context_length)
        target_frame_count = _pixel_frames(
            int(target_video.shape[2])
        )

        if context_frames >= target_frame_count:
            raise ValueError(
                "MiniMax H3 Motion Context RAM: context window must be "
                "shorter than the next clip."
            )

        blocks, offsets, span = _video_tail_from_latent(
            context_latent, context_frames
        )

        keyframes = []
        for pixel_index, block in zip(offsets, blocks):
            block = _pad_motion_context_block_to_target(
                block,
                target_video,
            )
            if guide_api == "native":
                # Same one-token block, same pixel offset, same order as v14.21.
                # New ComfyUI can express the interior anchor directly.
                keyframes.append(
                    {
                        "resolved_frame_index": int(pixel_index),
                        "latent": block,
                    }
                )
            else:
                keyframes.append(
                    {
                        # Old stock H3 only accepts first/last here. The exact
                        # temporal coordinate rides in MC_KEY and is rewritten
                        # after stock PackedLayout construction.
                        "resolved_frame_index": 0,
                        MC_KEY: int(pixel_index),
                        "latent": block,
                    }
                )

        # Carry audio from the SAME previous sampled AV latent.
        a_frames = int(audio_context_length) or span
        audio_latent, audio_t, overhang = _audio_tail_from_latent(
            context_latent, a_frames
        )

        # EXACT v14.21 end-alignment calculation.
        end_frame = float(span) + overhang / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE

        if guide_api == "native":
            # New PackedLayout starts native guide audio at:
            # target_origin + FRAME_RESCALE * resolved_frame_index.
            # Choose the start index so its END is exactly the same position
            # as v14.21's _fixup_audio():
            # target_origin + FRAME_RESCALE * end_frame - audio_t.
            audio_start_frame = (
                float(end_frame)
                - float(audio_t) / FRAME_RESCALE
            )
            keyframes.append(
                {
                    "resolved_frame_index": float(audio_start_frame),
                    "audio_latent": audio_latent,
                }
            )
            out = node_helpers.conditioning_set_values(
                conditioning,
                {"minimax_keyframes": keyframes},
            )
        else:
            values = {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": int(target_frame_count),
            }

            audio_ref = {
                "kind": "audio",
                "ref_audio_t": int(audio_t),
                "audio_latent": audio_latent,
            }
            audio_ref[MC_AUDIO_KEY] = float(end_frame)

            out = node_helpers.conditioning_set_values(
                conditioning, values
            )
            out = node_helpers.conditioning_set_values(
                out,
                {"minimax_refs": [audio_ref]},
                append=True,
            )

        _LOG.info(
            "MiniMax H3 Motion Context RAM: %d video frames -> %d latent "
            "conditioning blocks, %d audio latent steps, trim=%d",
            span, len(blocks), audio_t, span,
        )

        return (
            out,
            int(span),
            int(len(blocks)),
            int(audio_t),
            BUILD,
        )


def _temporal_gray(frame, out_h=96):
    """
    Small exposure-normalized grayscale descriptor used only to decide whether
    B should start 0, 1 or 2 decoded frames earlier.
    """
    x = frame[..., :3].detach().float().permute(2, 0, 1).unsqueeze(0).cpu()
    y = (
        x[:, 0:1] * 0.299
        + x[:, 1:2] * 0.587
        + x[:, 2:3] * 0.114
    )
    h0 = int(frame.shape[0])
    w0 = int(frame.shape[1])
    oh = min(int(out_h), h0)
    ow = max(8, int(round(float(w0) * float(oh) / max(1.0, float(h0)))))
    y = torch.nn.functional.interpolate(
        y, size=(oh, ow), mode="bilinear", align_corners=False
    )[0, 0]
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-4)
    return y


def _temporal_history_descriptor(decoded, previous_frames):
    """Precompute the invariant A-side descriptor for seam candidates."""
    seam = int(previous_frames)
    if seam < 3:
        return None

    a0 = _temporal_gray(decoded[seam - 3])
    a1 = _temporal_gray(decoded[seam - 2])
    a2 = _temporal_gray(decoded[seam - 1])

    v0 = a1 - a0
    v1 = a2 - a1
    velocity = v1 * 0.65 + v0 * 0.35
    predicted = a2 + velocity

    motion = v1.abs() + v0.abs() * 0.5
    motion_mean = motion.mean().clamp_min(1e-5)
    weight = 0.25 + (motion / motion_mean).clamp(max=4.0)
    expected_norm = float(velocity.norm().item())
    return a2, velocity, predicted, weight, expected_norm


def _temporal_candidate_score(decoded, previous_frames, candidate_index, history=None):
    """Score one B-start candidate against a precomputed A-side history."""
    seam = int(previous_frames)
    ci = int(candidate_index)
    if seam < 3 or ci < seam or ci >= int(decoded.shape[0]):
        return float("inf"), float("inf")

    if history is None:
        history = _temporal_history_descriptor(decoded, seam)
    if history is None:
        return float("inf"), float("inf")

    a2, velocity, predicted, weight, expected_norm = history
    c = _temporal_gray(decoded[ci])

    mse = float((((c - predicted) ** 2) * weight).mean().item())

    cross = c - a2
    cross_norm = float(cross.norm().item())
    jump_ratio = cross_norm / max(expected_norm, 1e-6)

    # Penalize both a forward jump and a near-duplicate/replay.
    mag_penalty = abs(math.log(max(jump_ratio, 1e-6)))

    denom = max(expected_norm * cross_norm, 1e-6)
    cosine = float(
        (velocity.flatten() * cross.flatten()).sum().item() / denom
    )
    cosine = max(-1.0, min(1.0, cosine))
    direction_penalty = 1.0 - cosine

    score = mse + 0.02 * mag_penalty + 0.02 * direction_penalty
    return float(score), float(jump_ratio)

def _auto_early_seam_shift(decoded, previous_frames, warmup_frames, max_early=2):
    """
    Detect the small 'person jumps forward' seam seen after Motion Context.

    The nominal B start is unchanged unless:
      * the nominal seam step is clearly larger than recent A motion, AND
      * starting B 1 or 2 already-decoded warm-up frames earlier gives a
        meaningfully better temporal continuation.

    Returns 0, -1 or -2. Negative means B starts earlier.
    """
    warm = int(warmup_frames)
    seam = int(previous_frames)

    if warm <= 0 or seam < 3:
        return 0

    nominal = seam + warm
    max_early = min(int(max_early), warm)
    offsets = [0] + [-i for i in range(1, max_early + 1)]

    history = _temporal_history_descriptor(decoded, seam)
    if history is None:
        return 0

    results = {}
    for off in offsets:
        idx = nominal + off
        if idx < seam or idx >= int(decoded.shape[0]):
            continue
        results[off] = _temporal_candidate_score(
            decoded, seam, idx, history=history
        )

    if 0 not in results:
        return 0

    nominal_score, nominal_jump = results[0]

    # No obvious forward jump: preserve the exact nominal 22-frame cut.
    if not math.isfinite(nominal_score) or nominal_jump <= 1.35:
        return 0

    valid = [
        (off, vals[0])
        for off, vals in results.items()
        if math.isfinite(vals[0])
    ]
    if not valid:
        return 0

    best_off, best_score = min(valid, key=lambda x: x[1])

    # Require at least a 6% score improvement before touching the seam.
    if best_off == 0 or best_score >= nominal_score * 0.94:
        return 0

    # If -1 is almost as good as -2, prefer -1: smallest possible correction.
    near_best = best_score * 1.03
    for preferred in (-1, -2):
        if preferred in results and results[preferred][0] <= near_best:
            chosen = preferred
            break
    else:
        chosen = best_off

    _LOG.info(
        "MiniMax H3 Motion Context Join: auto seam shift=%d frame(s); "
        "nominal jump ratio=%.3f, scores=%s",
        chosen,
        nominal_jump,
        {
            k: round(v[0], 5)
            for k, v in sorted(results.items())
        },
    )
    return int(chosen)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContextRAM": MiniMaxH3MotionContextRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContextRAM": "MiniMax H3 Motion Context RAM",
}
