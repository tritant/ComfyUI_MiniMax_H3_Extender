"""MiniMax H3 external prompt bridge.

Packs numbered STRING prompt outputs from story/prompt generator nodes into one
H3_PROMPT_PACK socket that the MiniMax H3 Extender can import into its normal
per-clip prompt cards.

The frontend presents these sockets as an autogrowing list.  The backend keeps
a generous fixed declaration so the legacy V1 node API can still validate and
execute dynamically-created prompt_N sockets on both classic nodes and Nodes 2.0.
"""

from __future__ import annotations

import hashlib
import json

# V1 custom nodes do not have native backend autogrow inputs.  The frontend
# therefore shows only the sockets that are useful while this declaration keeps
# enough valid prompt_N names available for execution/validation.
MAX_PROMPTS = 128
PROMPT_PACK_TYPE = "H3_PROMPT_PACK"
PROMPT_PACK_VERSION = 1


def _pack_prompts(values):
    """Return prompts as one compact ordered list.

    Empty/unconnected inputs are ignored.  This makes the bridge behave like a
    normal ordered list: if a middle source is disconnected, every later prompt
    moves up one position in the pack instead of leaving a hole or raising an
    error.  The frontend mirrors the same behavior by compacting/renaming its
    visible prompt_N sockets while preserving the connected cables.
    """
    prompts = []
    for value in list(values)[:MAX_PROMPTS]:
        text = "" if value is None else str(value)
        if not text.strip():
            continue
        prompts.append(text)

    if not prompts:
        raise ValueError(
            "MiniMax H3 Prompt Pack Bridge: no non-empty prompts were received."
        )
    return prompts


def _prompt_pack_signature(prompts):
    payload = json.dumps(
        list(prompts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MiniMaxH3PromptPackBridge:
    """Pack a dynamic ordered series of STRING inputs into one Extender input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the first STRING prompt. More prompt inputs appear automatically.",
                    },
                ),
            },
            "optional": {
                **{
                    f"prompt_{i}": (
                        "STRING",
                        {
                            "forceInput": True,
                            "tooltip": f"Dynamic STRING prompt {i}.",
                        },
                    )
                    for i in range(2, MAX_PROMPTS + 1)
                },
            },
        }

    RETURN_TYPES = (PROMPT_PACK_TYPE, "INT")
    RETURN_NAMES = ("prompt_pack", "prompt_count")
    FUNCTION = "pack"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = False

    def pack(self, prompt_1, **kwargs):
        values = [prompt_1]
        values.extend(kwargs.get(f"prompt_{i}") for i in range(2, MAX_PROMPTS + 1))
        prompts = _pack_prompts(values)
        signature = _prompt_pack_signature(prompts)
        pack = {
            "type": PROMPT_PACK_TYPE,
            "version": PROMPT_PACK_VERSION,
            "source": "MiniMax H3 Prompt Pack Bridge",
            "count": len(prompts),
            "prompts": prompts,
            "signature": signature,
        }
        return (pack, int(len(prompts)))


def _normalize_pack_for_merge(value, label):
    """Accept a full H3_PROMPT_PACK or a bare prompts list → (prompts, source)."""
    if value is None:
        return [], label
    if isinstance(value, (list, tuple)):
        prompts_raw = value
        source = label
    elif isinstance(value, dict):
        pack_type = str(value.get("type") or "").strip()
        if pack_type and pack_type != PROMPT_PACK_TYPE:
            raise ValueError(
                f"MiniMax H3 Prompt Pack Merge: {label} type must be {PROMPT_PACK_TYPE}."
            )
        prompts_raw = value.get("prompts")
        if not isinstance(prompts_raw, (list, tuple)):
            raise ValueError(
                f"MiniMax H3 Prompt Pack Merge: {label} has no prompt list."
            )
        source = str(value.get("source") or label)
    else:
        raise ValueError(
            f"MiniMax H3 Prompt Pack Merge: {label} is not a prompt pack."
        )

    out = []
    for raw in list(prompts_raw)[:MAX_PROMPTS]:
        text = "" if raw is None else str(raw)
        if text.strip():
            out.append(text)
    return out, source


class MiniMaxH3PromptPackMerge:
    """Concatenate several H3_PROMPT_PACK outputs into one Extender pack.

    Typical MVP wiring: one Minimax Prompt Director per Extender clip → Merge →
    Extender.prompt_pack. Empty / unconnected pack sockets are skipped; order is
    visual cable order (same compact list behavior as Prompt Pack Bridge).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pack_1": (
                    PROMPT_PACK_TYPE,
                    {
                        "tooltip": "First prompt pack (e.g. Director clip 1). More pack inputs appear automatically.",
                    },
                ),
            },
            "optional": {
                **{
                    f"pack_{i}": (
                        PROMPT_PACK_TYPE,
                        {
                            "tooltip": f"Additional prompt pack {i} (e.g. Director clip {i}).",
                        },
                    )
                    for i in range(2, MAX_PROMPTS + 1)
                },
            },
        }

    RETURN_TYPES = (PROMPT_PACK_TYPE, "INT")
    RETURN_NAMES = ("prompt_pack", "prompt_count")
    FUNCTION = "merge"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = False

    def merge(self, pack_1, **kwargs):
        merged = []
        sources = []
        packs = [pack_1]
        packs.extend(kwargs.get(f"pack_{i}") for i in range(2, MAX_PROMPTS + 1))
        for i, pack in enumerate(packs, start=1):
            if pack is None:
                continue
            prompts, source = _normalize_pack_for_merge(pack, f"pack_{i}")
            if not prompts:
                continue
            merged.extend(prompts)
            if source and source not in sources:
                sources.append(source)
            if len(merged) >= MAX_PROMPTS:
                merged = merged[:MAX_PROMPTS]
                break

        if not merged:
            raise ValueError(
                "MiniMax H3 Prompt Pack Merge: no non-empty prompts were received."
            )

        signature = _prompt_pack_signature(merged)
        source_label = " + ".join(sources) if sources else "MiniMax H3 Prompt Pack Merge"
        if len(source_label) > 160:
            source_label = "MiniMax H3 Prompt Pack Merge"
        pack = {
            "type": PROMPT_PACK_TYPE,
            "version": PROMPT_PACK_VERSION,
            "source": source_label,
            "count": len(merged),
            "prompts": merged,
            "signature": signature,
        }
        return (pack, int(len(merged)))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptPackBridge": MiniMaxH3PromptPackBridge,
    "MiniMaxH3PromptPackMerge": MiniMaxH3PromptPackMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptPackBridge": "MiniMax H3 Prompt Pack Bridge",
    "MiniMaxH3PromptPackMerge": "MiniMax H3 Prompt Pack Merge",
}
