import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3Extender";
const FINAL_TARGET = "MiniMaxH3MotionContextDiskFinalDecode";
const PROGRESS_EVENT = "h3_extender_progress";
const PROMPT_PACK_EVENT = "h3_extender_prompt_pack_import";
const REF_PACK_EVENT = "h3_extender_ref_pack_import";
const CARD_WIDTH = 318;
const UI_MIN_HEIGHT = 650;
const NODES2_MIN_HEIGHT = 700;
// Keep a real visual gap between the native Nodes 2.0 widgets and the CLIP
// panel. This is internal padding only: we deliberately do NOT rewrite Vue
// grid tracks or absolutely position the DOM widget.
const NODES2_TOP_GAP = 28;
const NODE_MIN_WIDTH = 520;
const BOTTOM_PAD = 16;
// Leave an empty gutter under each card so an overlay horizontal scrollbar
// never covers the Validated/footer row.
const CARD_SCROLLBAR_SPACE = 24;
// Cards keep the same compact structure in both modes. FL2VA keyframes live
// in the shared media strip above the cards; FL2VA also exposes one compact
// per-card dynamic image guides with exact frame indices.
const CARD_MIN_HEIGHT_REF2VA = 455;
const CARD_MIN_HEIGHT_FL2VA = 560;
const REF_SLOT_WIDTH = 96;
const FL2VA_FRAME_SLOT_WIDTH = 145;
const REF_THUMB_HEIGHT = 96;
// Reserve the scrollbar inside the existing reference section only.
// Do not grow the DOM widget or alter card sizing/layout for this.
const REF_SCROLLBAR_SPACE = 14;
const REF_SECTION_HEIGHT = 160;
const MAX_IMAGE_REFS = 9;
const MAX_FL2VA_GUIDES = 3;
const MAX_RESOLUTION = 4096;
const DEFAULT_MEGAPIXELS = 0.40;

// Nodes 2.0 lifecycle guard. Workflow loading is bracketed by the official
// beforeConfigureGraph/afterConfigureGraph extension hooks; while this flag is
// set, custom DOM code may render but must not publish graph mutations or
// restore cache state from temporary schema defaults.
let h3GraphConfiguring = false;

function isH3GraphConfiguring() {
    return h3GraphConfiguring;
}

function maxSelectedLoraRows(state) {
    let maxRows = 0;
    for (const clip of state?.clips || []) {
        const rows = Array.isArray(clip?.loras)
            ? clip.loras.filter((cfg) => String(cfg?.name || "").trim()).length
            : (String(clip?.lora?.name || "").trim() ? 1 : 0);
        maxRows = Math.max(maxRows, rows);
    }
    return maxRows;
}

function cardMinHeightForState(state) {
    const fl2va = String(state?.generation_mode || "ref2va") === "fl2va";
    const base = fl2va ? CARD_MIN_HEIGHT_FL2VA : CARD_MIN_HEIGHT_REF2VA;
    // One empty Add-LoRA selector is already included in the base height.
    // Each selected LoRA adds one full row above it.
    return base + maxSelectedLoraRows(state) * 46;
}

function uiMinHeightForState(state) {
    // Both modes own the same media strip above the cards: Ref2VA shows the
    // nine internal references, FL2VA shows each plan's First/Last frames.
    // Keeping one fixed strip height also prevents mode switches from pulling
    // the DOM widget upward into the native widgets in Nodes 2.0.
    return Math.max(UI_MIN_HEIGHT, 55 + REF_SECTION_HEIGHT + cardMinHeightForState(state) + CARD_SCROLLBAR_SPACE);
}

function nodes2MinHeightForState(state) {
    return Math.max(NODES2_MIN_HEIGHT, uiMinHeightForState(state) + NODES2_TOP_GAP);
}

const PROJECT_WIDGETS = [
    "run_mode",
    "width",
    "height",
    "ref_image_size",
    "steps",
    "sampler_name",
    "scheduler",
    "denoise",
    "context_length",
    "audio_context_length",
    "clips_json",
    "resolution_mode",
    "megapixels",
    "refs_json",
    "generation_mode",
];

const FINAL_PROJECT_WIDGETS = [
    "filename_prefix",
    "output_directory",
    "codec",
    "crf",
    "preset",
    "audio_bitrate",
];

// Validation and reference semantics are user-controlled. The Extender never
// associates Ref N with Clip N and never decides which clip a reference edit
// invalidates. Existing validation flags stay exactly as the user left them.
// The one unavoidable global exception is RESOLUTION: cached latents cannot be
// reused at another geometry, so an effective width/height change immediately
// clears validation for the whole chain.


function emptyRefsState() {
    return { version: 2, refs: Array(MAX_IMAGE_REFS).fill(null) };
}

function normalizeRefDescriptor(value) {
    if (!value || typeof value !== "object") return null;
    const id = String(value.id || value.ref_id || "").toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(id)) return null;
    const sourceCandidate = String(value.source_id || value.original_id || id).toLowerCase();
    const source_id = /^[0-9a-f]{64}$/.test(sourceCandidate) ? sourceCandidate : id;
    const adjustment = (name) => {
        const n = Number(value[name] ?? 100);
        return Number.isFinite(n) ? Math.min(200, Math.max(0, n)) : 100;
    };
    const descriptor = {
        id,
        source_id,
        original_name: String(value.original_name || value.name || "reference.png"),
        width: Math.max(0, Number(value.width || 0)),
        height: Math.max(0, Number(value.height || 0)),
        size_bytes: Math.max(0, Number(value.size_bytes || 0)),
        saturation: adjustment("saturation"),
        contrast: adjustment("contrast"),
        brightness: adjustment("brightness"),
    };
    const externalSignature = String(value.external_signature || "").toLowerCase();
    if (/^[0-9a-f]{64}$/.test(externalSignature)) {
        descriptor.external_signature = externalSignature;
    }
    return descriptor;
}

function normalizeRefsArray(values) {
    // Ref slots are stable logical identities. Never compact holes: moving Ref 3
    // into Ref 2 would silently break prompts that intentionally use <Picture 3>.
    const refs = Array(MAX_IMAGE_REFS).fill(null);
    const source = Array.isArray(values) ? values : [];
    for (let i = 0; i < Math.min(MAX_IMAGE_REFS, source.length); i++) {
        refs[i] = normalizeRefDescriptor(source[i]);
    }
    return refs;
}

function parseRefsState(raw) {
    try {
        const parsed = typeof raw === "string" ? JSON.parse(raw || "{}") : raw;
        const refs = Array.isArray(parsed) ? parsed : parsed?.refs;
        return { version: 2, refs: normalizeRefsArray(Array.isArray(refs) ? refs : []) };
    } catch (_) {
        return emptyRefsState();
    }
}

function serializeRefsState(state) {
    return JSON.stringify({ version: 2, refs: normalizeRefsArray(state?.refs || []) });
}

function refCount(runtime) {
    return (runtime?.refsState?.refs || []).filter(Boolean).length;
}

function refImageUrl(ref) {
    if (!ref?.id) return "";
    return api.apiURL("/h3_extender/ref/image?id=" + encodeURIComponent(String(ref.id)));
}

function sameRefContent(a, b) {
    return String(a?.id || "") === String(b?.id || "");
}

function removeLegacyImageRefInputs(node) {
    if (!node?.inputs) return false;
    let removed = false;
    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const name = String(node.inputs[index]?.name || "");
        if (/^ref_[1-9]$/.test(name)) {
            try {
                node.removeInput(index);
                removed = true;
            } catch (_) {}
        }
    }
    if (removed) node.graph?.setDirtyCanvas(true, true);
    return removed;
}


const MAX_VIDEO_REFS = 3;
const MAX_STANDALONE_AUDIO_REFS = 3;
const REF_VIDEO_RE = /^ref_video_([1-3])$/;
const REF_VIDEO_AUDIO_RE = /^ref_video_audio_([1-3])$/;
const REF_VIDEO_FPS_RE = /^ref_video_fps_([1-3])$/;
const REF_AUDIO_RE = /^ref_audio_([1-3])$/;

function inputConnected(input) {
    return input?.link !== null && input?.link !== undefined;
}

function findInputEntry(node, name) {
    const inputs = node?.inputs || [];
    for (let slot = 0; slot < inputs.length; slot++) {
        if (String(inputs[slot]?.name || "") === String(name)) {
            return { input: inputs[slot], slot };
        }
    }
    return null;
}

function addDynamicRefInput(node, name, type, tooltip = "") {
    if (!node || findInputEntry(node, name)) return false;
    try {
        const input = node.addInput(name, type, tooltip ? { tooltip } : undefined);
        // LiteGraph versions differ on whether addInput returns the slot object.
        // If it does, keep the tooltip there too; otherwise the socket still works.
        if (input && tooltip && !input.tooltip) input.tooltip = tooltip;
        return true;
    } catch (_) {
        return false;
    }
}

function removeDynamicRefInput(node, name) {
    const entry = findInputEntry(node, name);
    if (!entry || inputConnected(entry.input)) return false;
    try {
        node.removeInput(entry.slot);
        return true;
    } catch (_) {
        return false;
    }
}

function graphLinkById(graph, linkId) {
    if (!graph || linkId === null || linkId === undefined) return null;
    try {
        if (graph.links instanceof Map) return graph.links.get(linkId) || null;
        if (graph.links && graph.links[linkId] !== undefined) return graph.links[linkId];
        if (graph._links instanceof Map) return graph._links.get(linkId) || null;
        if (graph._links && graph._links[linkId] !== undefined) return graph._links[linkId];
    } catch (_) {}
    return null;
}

function normalizeDynamicReferenceInputOrder(node) {
    // addInput() always appends sockets, so an autogrown ref_audio_3 could end up
    // below the already-visible video sockets. Rebuild only the visual socket
    // order after each sync while preserving the exact input objects and cables.
    // Desired order:
    //   static model inputs
    //   ref_audio_1..3
    //   ref_video_1 / fps_1 / video_audio_1
    //   ref_video_2 / fps_2 / video_audio_2
    //   ref_video_3 / fps_3 / video_audio_3
    //   ref_pack / prompt_pack
    if (!node?.inputs?.length) return false;

    const packOrder = ["ref_pack", "prompt_pack"];
    const dynamicNames = new Set(packOrder);
    for (let i = 1; i <= MAX_STANDALONE_AUDIO_REFS; i++) {
        dynamicNames.add(`ref_audio_${i}`);
    }
    for (let i = 1; i <= MAX_VIDEO_REFS; i++) {
        dynamicNames.add(`ref_video_${i}`);
        dynamicNames.add(`ref_video_fps_${i}`);
        dynamicNames.add(`ref_video_audio_${i}`);
    }

    const byName = new Map();
    const staticInputs = [];
    for (const input of node.inputs) {
        const name = String(input?.name || "");
        if (dynamicNames.has(name)) byName.set(name, input);
        else staticInputs.push(input);
    }

    const desired = [...staticInputs];
    for (let i = 1; i <= MAX_STANDALONE_AUDIO_REFS; i++) {
        const input = byName.get(`ref_audio_${i}`);
        if (input) desired.push(input);
    }
    for (let i = 1; i <= MAX_VIDEO_REFS; i++) {
        for (const name of [
            `ref_video_${i}`,
            `ref_video_fps_${i}`,
            `ref_video_audio_${i}`,
        ]) {
            const input = byName.get(name);
            if (input) desired.push(input);
        }
    }
    for (const name of packOrder) {
        const input = byName.get(name);
        if (input) desired.push(input);
    }

    const alreadyOrdered = desired.length === node.inputs.length
        && desired.every((input, slot) => node.inputs[slot] === input);
    if (alreadyOrdered) return false;

    node.inputs.splice(0, node.inputs.length, ...desired);

    // LiteGraph stores target sockets as numeric indices. Re-point every linked
    // input after the visual reorder so existing workflows keep all cables.
    for (let slot = 0; slot < node.inputs.length; slot++) {
        const input = node.inputs[slot];
        if (!inputConnected(input)) continue;
        const link = graphLinkById(node.graph, input.link);
        if (link && String(link.target_id) === String(node.id)) {
            link.target_slot = slot;
        }
    }
    return true;
}

function renameInputPreservingLink(input, name) {
    if (!input || !name || String(input.name || "") === name) return false;
    input.name = name;
    if (typeof input.label === "string" && /^ref_audio(?:_[1-3])?$/.test(input.label)) {
        input.label = name;
    }
    return true;
}

function migrateLegacyStandaloneAudio(node) {
    // v14.64/14.65 kept the old single `ref_audio` socket as a backend alias.
    // New nodes should not show it. When loading an older workflow with a cable
    // on that socket, rename the socket in place to ref_audio_1 so the cable is
    // preserved and the workflow joins the new dynamic group cleanly.
    const legacy = findInputEntry(node, "ref_audio");
    if (!legacy) return false;

    const canonical = findInputEntry(node, "ref_audio_1");
    if (!inputConnected(legacy.input)) {
        try {
            node.removeInput(legacy.slot);
            return true;
        } catch (_) {
            return false;
        }
    }

    if (canonical && !inputConnected(canonical.input)) {
        try {
            node.removeInput(canonical.slot);
        } catch (_) {
            return false;
        }
    } else if (canonical && inputConnected(canonical.input)) {
        // Extremely unusual transitional workflow with both sockets connected:
        // keep both rather than destroying either cable. Backend compatibility
        // remains authoritative for this one legacy edge case.
        return false;
    }

    return renameInputPreservingLink(legacy.input, "ref_audio_1");
}

function highestConnectedIndex(node, regex, maxIndex) {
    let highest = 0;
    for (const input of node?.inputs || []) {
        const match = String(input?.name || "").match(regex);
        if (!match || !inputConnected(input)) continue;
        const index = Number(match[1]);
        if (Number.isInteger(index) && index >= 1 && index <= maxIndex) {
            highest = Math.max(highest, index);
        }
    }
    return highest;
}

function syncDynamicAVReferenceInputs(node) {
    if (!node || node.__h3AVRefSyncing) return;
    node.__h3AVRefSyncing = true;
    let changed = false;
    try {
        changed = migrateLegacyStandaloneAudio(node) || changed;

        // ---- Standalone audio refs -------------------------------------------------
        // Classic autogrow: always show ref_audio_1, then one free socket after
        // the highest connected audio, up to H3's three-audio limit. Connected
        // higher slots are never removed, so loading sparse/older workflows does
        // not destroy cables.
        const highestAudio = highestConnectedIndex(node, REF_AUDIO_RE, MAX_STANDALONE_AUDIO_REFS);
        const visibleAudioMax = Math.min(
            MAX_STANDALONE_AUDIO_REFS,
            Math.max(1, highestAudio + 1),
        );
        for (let i = 1; i <= MAX_STANDALONE_AUDIO_REFS; i++) {
            const name = `ref_audio_${i}`;
            if (i <= visibleAudioMax) {
                changed = addDynamicRefInput(
                    node,
                    name,
                    "AUDIO",
                    `Optional MiniMax H3 standalone reference audio ${i}.`,
                ) || changed;
            } else {
                changed = removeDynamicRefInput(node, name) || changed;
            }
        }

        // ---- Paired video + soundtrack refs ---------------------------------------
        // Stable logical Video slots: never compact or rename a connected Video N.
        // Connecting Video N reveals its same-numbered optional soundtrack and the
        // next video socket. A soundtrack with an existing cable is also preserved
        // even if its video is temporarily disconnected, allowing the user to fix
        // the pair instead of silently losing the cable.
        const highestVideo = highestConnectedIndex(node, REF_VIDEO_RE, MAX_VIDEO_REFS);
        const visibleVideoMax = Math.min(MAX_VIDEO_REFS, Math.max(1, highestVideo + 1));

        for (let i = 1; i <= MAX_VIDEO_REFS; i++) {
            const videoName = `ref_video_${i}`;
            const fpsName = `ref_video_fps_${i}`;
            const audioName = `ref_video_audio_${i}`;
            const videoEntry = findInputEntry(node, videoName);
            const fpsEntry = findInputEntry(node, fpsName);
            const audioEntry = findInputEntry(node, audioName);
            const videoIsConnected = inputConnected(videoEntry?.input);
            const fpsIsConnected = inputConnected(fpsEntry?.input);
            const audioIsConnected = inputConnected(audioEntry?.input);
            const companionConnected = fpsIsConnected || audioIsConnected;

            // Preserve the numbered video socket if one of its companion cables
            // is still connected, so dynamic cleanup never strands an FPS/audio
            // cable without a matching Video N socket.
            if (i <= visibleVideoMax || videoIsConnected || companionConnected) {
                changed = addDynamicRefInput(
                    node,
                    videoName,
                    "IMAGE",
                    `Optional MiniMax H3 reference video ${i} as an IMAGE frame batch. Connect the matching fps output from Get Video Components when the source is not already 24 fps. Use <Video ${i}> in prompts.`,
                ) || changed;
            } else {
                changed = removeDynamicRefInput(node, videoName) || changed;
            }

            // Re-read after potential video insertion/removal.
            const liveVideo = findInputEntry(node, videoName);
            const liveFps = findInputEntry(node, fpsName);
            const liveAudio = findInputEntry(node, audioName);
            const liveVideoConnected = inputConnected(liveVideo?.input);
            const liveFpsConnected = inputConnected(liveFps?.input);
            const liveAudioConnected = inputConnected(liveAudio?.input);

            // Once Video N is connected, expose both companion inputs directly:
            // FLOAT fps from Get Video Components + optional matching soundtrack.
            // Connected companion sockets are preserved during temporary rewiring.
            if (liveVideoConnected || liveFpsConnected) {
                changed = addDynamicRefInput(
                    node,
                    fpsName,
                    "FLOAT",
                    `Source FPS of ref_video_${i}. Connect Get Video Components → fps. Leave disconnected only when the IMAGE batch is already 24 fps.`,
                ) || changed;
            } else {
                changed = removeDynamicRefInput(node, fpsName) || changed;
            }

            if (liveVideoConnected || liveAudioConnected) {
                changed = addDynamicRefInput(
                    node,
                    audioName,
                    "AUDIO",
                    `Optional soundtrack of ref_video_${i}.`,
                ) || changed;
            } else {
                changed = removeDynamicRefInput(node, audioName) || changed;
            }
        }

        changed = normalizeDynamicReferenceInputOrder(node) || changed;
        if (changed) node.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__h3AVRefSyncing = false;
    }
}

function deferDynamicAVReferenceSync(node) {
    if (!node || node.__h3AVRefSyncQueued) return;
    node.__h3AVRefSyncQueued = true;
    requestAnimationFrame(() => {
        node.__h3AVRefSyncQueued = false;
        syncDynamicAVReferenceInputs(node);
    });
}

function randomSeed() {
    try {
        const a = new Uint32Array(2);
        crypto.getRandomValues(a);
        // stay inside JS exact-integer range
        return Number((BigInt(a[0]) << 21n) ^ BigInt(a[1] & 0x1fffff));
    } catch (_) {
        return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    }
}

function normalizeColorAdjustment(value) {
    const c = value && typeof value === "object" ? value : {};
    const clamp = (v, lo, hi, fallback) => {
        const n = Number(v);
        return Math.max(lo, Math.min(hi, Number.isFinite(n) ? n : fallback));
    };
    return {
        saturation: clamp(c.saturation, 0, 200, 100),
        contrast: clamp(c.contrast, 50, 150, 100),
        brightness: clamp(c.brightness, 50, 150, 100),
    };
}

function colorAdjustmentIsNeutral(value) {
    const c = normalizeColorAdjustment(value);
    return [c.saturation, c.contrast, c.brightness].every((v) => Math.abs(v - 100) < 1e-6);
}

function cssColorFilter(value) {
    const c = normalizeColorAdjustment(value);
    return `saturate(${c.saturation}%) contrast(${c.contrast}%) brightness(${c.brightness}%)`;
}

function normalizeClipLora(value) {
    const raw = value && typeof value === "object" ? value : {};
    const candidate = raw.strength ?? raw.strength_model ?? 1.0;
    const n = Number(candidate);
    const strength = Number.isFinite(n) ? Math.max(-100, Math.min(100, n)) : 1.0;
    return {
        name: String(raw.name || "").trim(),
        strength,
    };
}

function normalizeClipLoras(value, legacy = null) {
    let source = Array.isArray(value) ? value : [];
    if (!source.length && legacy && typeof legacy === "object") source = [legacy];
    return source
        .map((entry) => normalizeClipLora(entry))
        .filter((entry) => Boolean(entry.name));
}

function h3FrameCountForDuration(duration) {
    const rawFrames = Math.max(5, Math.round(Math.max(0.25, Number(duration || 10)) * 24));
    let aligned = rawFrames;
    while (aligned % 17 !== 5) aligned++;
    return aligned;
}

function normalizeGuideFrameIdx(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(-9999, Math.min(9999, Math.trunc(n)));
}

function normalizeGuideList(clip) {
    const raw = Array.isArray(clip?.guides)
        ? clip.guides
        : (normalizeRefDescriptor(clip?.guide_frame)
            ? [{ frame: clip.guide_frame, frame_idx: clip?.guide_frame_idx ?? 0 }]
            : []);
    const out = [];
    for (const item of raw.slice(0, MAX_FL2VA_GUIDES)) {
        const frame = normalizeRefDescriptor(item?.frame ?? item?.guide_frame);
        if (!frame) continue;
        out.push({
            frame,
            frame_idx: normalizeGuideFrameIdx(item?.frame_idx ?? item?.guide_frame_idx ?? 0),
        });
    }
    return out;
}

function newClip(index) {
    return {
        id: `clip_${index + 1}_${Date.now().toString(36)}`,
        name: "",
        prompt: "",
        seed: randomSeed(),
        seed_mode: "randomize",
        duration: 10.0,
        validated: false,
        color_adjustment: normalizeColorAdjustment(),
        loras: [],
        first_frame: null,
        last_frame: null,
        guides: [],
        first_source: "manual",
    };
}

function normalizeClipList(rawClips) {
    const clips = Array.isArray(rawClips) && rawClips.length ? rawClips : [newClip(0)];
    return clips.map((c, i) => ({
        id: String(c?.id || `clip_${i + 1}`),
        name: String(c?.name || ""),
        prompt: String(c?.prompt || ""),
        seed: Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Number(c?.seed || 0))),
        seed_mode: ["randomize", "fixed", "increment", "decrement"].includes(String(c?.seed_mode))
            ? String(c.seed_mode)
            : "randomize",
        duration: Math.max(0.25, Math.min(150, Number(c?.duration || 10))),
        validated: Boolean(c?.validated),
        color_adjustment: normalizeColorAdjustment(c?.color_adjustment),
        loras: normalizeClipLoras(c?.loras, c?.lora),
        first_frame: normalizeRefDescriptor(c?.first_frame),
        last_frame: normalizeRefDescriptor(c?.last_frame),
        guides: normalizeGuideList(c),
        first_source: (i > 0 && String(c?.first_source || "manual") === "previous_clip")
            ? "previous_clip"
            : "manual",
    }));
}

function blankModeClips() {
    return [newClip(0)];
}

function ensureModeClipState(state) {
    if (!state || typeof state !== "object") return state;
    const activeMode = String(state.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    if (!state.mode_clips || typeof state.mode_clips !== "object") state.mode_clips = {};
    if (!Array.isArray(state.mode_clips.ref2va) || !state.mode_clips.ref2va.length) {
        state.mode_clips.ref2va = activeMode === "ref2va" && Array.isArray(state.clips) && state.clips.length
            ? state.clips
            : blankModeClips();
    }
    if (!Array.isArray(state.mode_clips.fl2va) || !state.mode_clips.fl2va.length) {
        state.mode_clips.fl2va = activeMode === "fl2va" && Array.isArray(state.clips) && state.clips.length
            ? state.clips
            : blankModeClips();
    }
    state.mode_clips[activeMode] = Array.isArray(state.clips) && state.clips.length
        ? state.clips
        : state.mode_clips[activeMode];
    state.clips = state.mode_clips[activeMode];
    return state;
}

function activateModeState(state, mode) {
    ensureModeClipState(state);
    const current = String(state.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    const next = String(mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    state.mode_clips[current] = state.clips;
    state.generation_mode = next;
    if (!Array.isArray(state.mode_clips[next]) || !state.mode_clips[next].length) {
        state.mode_clips[next] = blankModeClips();
    }
    state.clips = state.mode_clips[next];
    return state;
}

function parseState(raw) {
    try {
        const p = JSON.parse(raw || "{}");
        const legacyArray = Array.isArray(p);
        const payload = legacyArray ? { clips: p } : (p && typeof p === "object" ? p : {});
        const generationMode = String(payload?.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
        const rawActive = Array.isArray(payload?.clips) && payload.clips.length ? payload.clips : null;
        const savedModes = payload?.mode_clips && typeof payload.mode_clips === "object"
            ? payload.mode_clips
            : {};

        // `clips` is always the authoritative active-mode payload sent to the
        // backend. `mode_clips` is frontend/workflow state only and keeps the
        // inactive mode completely independent while switching REF2VA <-> FL2VA.
        const ref2vaClips = generationMode === "ref2va" && rawActive
            ? normalizeClipList(rawActive)
            : (Array.isArray(savedModes?.ref2va) && savedModes.ref2va.length
                ? normalizeClipList(savedModes.ref2va)
                : blankModeClips());
        const fl2vaClips = generationMode === "fl2va" && rawActive
            ? normalizeClipList(rawActive)
            : (Array.isArray(savedModes?.fl2va) && savedModes.fl2va.length
                ? normalizeClipList(savedModes.fl2va)
                : blankModeClips());
        const activeClips = generationMode === "fl2va" ? fl2vaClips : ref2vaClips;

        return {
            version: 2,
            generation_mode: generationMode,
            load_token: String(payload?.project_load_token || ""),
            prompt_pack_signature: String(payload?.prompt_pack_signature || ""),
            clips: activeClips,
            mode_clips: {
                ref2va: ref2vaClips,
                fl2va: fl2vaClips,
            },
        };
    } catch (_) {}
    const ref2vaClips = blankModeClips();
    return {
        version: 2,
        generation_mode: "ref2va",
        load_token: "",
        prompt_pack_signature: "",
        clips: ref2vaClips,
        mode_clips: { ref2va: ref2vaClips, fl2va: blankModeClips() },
    };
}

function serializeState(state) {
    ensureModeClipState(state);
    const mode = state?.generation_mode === "fl2va" ? "fl2va" : "ref2va";
    state.mode_clips[mode] = state.clips;
    const payload = {
        version: 2,
        generation_mode: mode,
        clips: state.clips,
        mode_clips: {
            ref2va: state.mode_clips.ref2va,
            fl2va: state.mode_clips.fl2va,
        },
    };
    if (state?.load_token) payload.project_load_token = String(state.load_token);
    if (state?.prompt_pack_signature) payload.prompt_pack_signature = String(state.prompt_pack_signature);
    return JSON.stringify(payload);
}

function serializeProjectState(state) {
    // Portable .ext projects intentionally remain single-mode. This preserves
    // the existing archive/cache contract and avoids embedding inactive FL2VA
    // frames in a Ref2VA project (or vice versa). Old projects without a mode
    // marker are still interpreted as Ref2VA by the backend.
    ensureModeClipState(state);
    const mode = state?.generation_mode === "fl2va" ? "fl2va" : "ref2va";
    const payload = { version: 2, generation_mode: mode, clips: state.clips };
    if (state?.load_token) payload.project_load_token = String(state.load_token);
    if (state?.prompt_pack_signature) payload.prompt_pack_signature = String(state.prompt_pack_signature);
    return JSON.stringify(payload);
}

function mergeActiveStateJson(runtime, raw, explicitMode = null) {
    const incoming = parseState(raw);
    if (!runtime?.state) return incoming;
    ensureModeClipState(runtime.state);
    const mode = String(explicitMode || incoming.generation_mode || runtime.state.generation_mode || "ref2va") === "fl2va"
        ? "fl2va"
        : "ref2va";
    const incomingClips = incoming.generation_mode === mode
        ? incoming.clips
        : incoming.mode_clips?.[mode];
    runtime.state.mode_clips[mode] = Array.isArray(incomingClips) && incomingClips.length
        ? incomingClips
        : blankModeClips();
    runtime.state.generation_mode = mode;
    runtime.state.clips = runtime.state.mode_clips[mode];
    runtime.state.load_token = incoming.load_token || runtime.state.load_token || "";
    runtime.state.prompt_pack_signature = incoming.prompt_pack_signature || "";
    return runtime.state;
}

async function refreshLoraNames(node, runtime) {
    if (!node || !runtime || runtime.loraListLoading) return;
    runtime.loraListLoading = true;
    try {
        const response = await fetch(api.apiURL("/h3_extender/loras"), { cache: "no-store" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `LoRA list failed (${response.status})`);
        }
        runtime.loraNames = Array.isArray(payload?.loras)
            ? payload.loras.map((name) => String(name)).filter(Boolean)
            : [];
        runtime.loraListError = "";
    } catch (error) {
        runtime.loraNames = [];
        runtime.loraListError = String(error?.message || error);
    } finally {
        runtime.loraListLoading = false;
        runtime.loraListLoaded = true;
        render(node, runtime);
    }
}

function validatedPrefixFromState(state) {
    if (state?.generation_mode === "fl2va") {
        return (state?.clips || []).filter((clip) => Boolean(clip?.validated)).length;
    }
    let count = 0;
    for (const clip of state?.clips || []) {
        if (!clip?.validated) break;
        count += 1;
    }
    return count;
}

async function restoreCacheState(node, runtime) {
    if (!node || !runtime || runtime.hydrating || runtime.cacheStateRequestRunning) return;

    runtime.cacheStateRequestRunning = true;
    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(node.id));
        params.set("mode", String(runtime.state?.generation_mode || getWidget(node, "generation_mode")?.value || "ref2va"));
        const response = await fetch(
            api.apiURL("/h3_extender/cache_state?" + params.toString())
        );
        if (!response.ok) return;

        const payload = await response.json();
        if (!payload?.found) return;

        // Do not overwrite live execution information if generation started
        // while the startup request was in flight.
        if (["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""))) {
            return;
        }

        runtime.cachedCount = Number(payload.cached_count || 0);
        runtime.validatedCount = Number(payload.validated_count || 0);
        runtime.cachedClipIds = new Set(Array.isArray(payload.cached_clip_ids) ? payload.cached_clip_ids.map(String) : []);
        runtime.validatedClipIds = new Set(Array.isArray(payload.validated_clip_ids) ? payload.validated_clip_ids.map(String) : []);
        runtime.continuitySignatures = new Map(
            Object.entries(payload?.continuity_signatures || {}).map(([key, value]) => [String(key), String(value || "")]).filter(([, value]) => Boolean(value))
        );
        const activeMode = String(runtime.state?.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
        if (activeMode === "fl2va") {
            for (const clip of runtime.state?.clips || []) {
                clip.validated = runtime.validatedClipIds.has(String(clip.id));
            }
        } else {
            for (let i = 0; i < (runtime.state?.clips || []).length; i++) {
                runtime.state.clips[i].validated = i < runtime.validatedCount;
            }
        }
        snapshotModeValidation(runtime, activeMode);
        runtime.jsonWidget.value = serializeState(runtime.state);
        const restoredW = Number(payload.resolved_width || 0);
        const restoredH = Number(payload.resolved_height || 0);
        if (restoredW > 0 && restoredH > 0) {
            // Cache restore is informational only. Do not overwrite live
            // resolution controls: outside an explicit .ext Load the user is
            // free to change Auto/MP or Manual width/height at any time.
            runtime.expectedResolution = { width: restoredW, height: restoredH };
        }
        runtime.cacheStateRestored = true;
        const resolutionText = restoredW > 0 && restoredH > 0
            ? ` | project ${restoredW}x${restoredH}`
            : "";
        runtime.statusText =
            `Restored cache${resolutionText} | cached ${runtime.cachedCount}/${runtime.state.clips.length} | ` +
            `validated ${runtime.validatedCount}`;
        syncResolutionAndInvalidate(node, runtime);
        render(node, runtime);
        node.graph?.setDirtyCanvas(true, true);
    } catch (_) {
        // Cache-state restoration is visual convenience only. Never block UI load.
    } finally {
        runtime.cacheStateRequestRunning = false;
    }
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function effectiveManualResolution(width, height) {
    const step = 32;
    const w = Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(Number(width || 0) / step) * step));
    const h = Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(Number(height || 0) / step) * step));
    return { width: w, height: h };
}

function pythonRound(value) {
    // Python round() uses bankers rounding for exact .5 ties; match the
    // backend so the visible mirror can never disagree by a latent-grid step.
    const x = Number(value);
    if (!Number.isFinite(x)) return 0;
    const floor = Math.floor(x);
    const frac = x - floor;
    if (Math.abs(frac - 0.5) < 1e-12) return (floor % 2 === 0) ? floor : floor + 1;
    return Math.round(x);
}

function autoResolutionFromDimensions(srcWidth, srcHeight, megapixels) {
    const srcW = Number(srcWidth || 0);
    const srcH = Number(srcHeight || 0);
    if (!(srcW > 0) || !(srcH > 0)) return null;

    const mp = Math.max(0.01, Math.min(16.0, Number(megapixels ?? DEFAULT_MEGAPIXELS)));
    const total = mp * 1024.0 * 1024.0;
    const scale = Math.sqrt(total / (srcW * srcH));
    let scaledW = srcW * scale;
    let scaledH = srcH * scale;

    if (scaledW > MAX_RESOLUTION || scaledH > MAX_RESOLUTION) {
        const shrink = Math.min(MAX_RESOLUTION / scaledW, MAX_RESOLUTION / scaledH);
        scaledW *= shrink;
        scaledH *= shrink;
    }

    // H3 32-pixel canvas. Auto snaps downward so the resolved canvas never
    // exceeds the requested megapixel budget; Manual uses the same 32px grid.
    const step = 32;
    return {
        width: Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(scaledW / step) * step)),
        height: Math.max(step, Math.min(MAX_RESOLUTION, Math.floor(scaledH / step) * step)),
    };
}

function currentGuideRefNumber(runtime) {
    const refs = runtime?.refsState?.refs || [];
    if (refs[0]) return 1;
    for (let i = 0; i < Math.min(MAX_IMAGE_REFS, refs.length); i++) {
        if (refs[i]) return i + 1;
    }
    return null;
}

function dimensionsFromFl2vaKeyframe(runtime) {
    for (const clip of runtime?.state?.clips || []) {
        for (const key of ["first_frame", "last_frame"]) {
            const ref = normalizeRefDescriptor(clip?.[key]);
            const width = Number(ref?.width || 0);
            const height = Number(ref?.height || 0);
            if (width > 0 && height > 0) return { width, height };
        }
        for (const guide of normalizeGuideList(clip)) {
            const ref = normalizeRefDescriptor(guide?.frame);
            const width = Number(ref?.width || 0);
            const height = Number(ref?.height || 0);
            if (width > 0 && height > 0) return { width, height };
        }
    }
    return null;
}

function hasAutoResolutionGuide(runtime) {
    return runtime?.state?.generation_mode === "fl2va"
        ? Boolean(dimensionsFromFl2vaKeyframe(runtime))
        : currentGuideRefNumber(runtime) != null;
}

function dimensionsFromInternalRef(runtime, refNumber) {
    const index = Number(refNumber) - 1;
    if (!runtime || !Number.isInteger(index) || index < 0 || index >= MAX_IMAGE_REFS) return null;
    const ref = runtime.refsState?.refs?.[index];
    const width = Number(ref?.width || 0);
    const height = Number(ref?.height || 0);
    return width > 0 && height > 0 ? { width, height } : null;
}

function setResolutionMirrorValues(node, runtime, width, height) {
    if (!runtime || !(width > 0) || !(height > 0)) return;
    runtime.applyingResolutionMirror = true;
    try {
        setWidgetValue(node, "width", Number(width));
        setWidgetValue(node, "height", Number(height));
    } finally {
        runtime.applyingResolutionMirror = false;
    }
}

function rememberManualResolution(node, runtime, width, height) {
    if (!runtime) return;
    if (Number(width) > 0) runtime.manualWidth = Number(width);
    if (Number(height) > 0) runtime.manualHeight = Number(height);
    if (node) {
        node.properties = node.properties || {};
        if (runtime.manualWidth > 0) node.properties.h3_manual_width = runtime.manualWidth;
        if (runtime.manualHeight > 0) node.properties.h3_manual_height = runtime.manualHeight;
    }
}

function syncResolutionMirror(node, runtime) {
    if (!node || !runtime) return;

    const mode = String(getWidget(node, "resolution_mode")?.value || "auto_from_ref");
    const widthWidget = getWidget(node, "width");
    const heightWidget = getWidget(node, "height");
    if (!widthWidget || !heightWidget) return;

    if (mode === "manual") {
        if (runtime.manualWidth > 0 && runtime.manualHeight > 0) {
            setResolutionMirrorValues(node, runtime, runtime.manualWidth, runtime.manualHeight);
        }
        runtime.resolutionMirrorActive = false;
        return;
    }

    const flGuide = runtime.state?.generation_mode === "fl2va" ? dimensionsFromFl2vaKeyframe(runtime) : null;
    const guideRef = runtime.state?.generation_mode === "fl2va" ? null : currentGuideRefNumber(runtime);
    if (!flGuide && guideRef == null) {
        // Auto without an applicable keyframe/reference is the editable Manual fallback.
        if (runtime.manualWidth > 0 && runtime.manualHeight > 0) {
            setResolutionMirrorValues(node, runtime, runtime.manualWidth, runtime.manualHeight);
        }
        runtime.resolutionMirrorActive = false;
        return;
    }

    let source = flGuide || dimensionsFromInternalRef(runtime, guideRef);
    const executedGuide = /^ref_(\d+)$/.exec(String(runtime.resolutionGuide || ""));
    if (!source && executedGuide && Number(executedGuide[1]) === Number(guideRef)) {
        if (runtime.guideSourceWidth > 0 && runtime.guideSourceHeight > 0) {
            source = { width: runtime.guideSourceWidth, height: runtime.guideSourceHeight };
        }
    }

    if (!source) {
        // Internal metadata normally carries the exact source dimensions. Keep
        // the last backend result as a defensive fallback for older saved state.
        if (
            executedGuide &&
            Number(executedGuide[1]) === Number(guideRef) &&
            runtime.resolvedWidth > 0 && runtime.resolvedHeight > 0
        ) {
            setResolutionMirrorValues(node, runtime, runtime.resolvedWidth, runtime.resolvedHeight);
            runtime.resolutionMirrorActive = true;
        }
        return;
    }

    const resolved = autoResolutionFromDimensions(
        source.width,
        source.height,
        Number(getWidget(node, "megapixels")?.value ?? DEFAULT_MEGAPIXELS),
    );
    if (!resolved) return;
    runtime.guideSourceWidth = Number(source.width);
    runtime.guideSourceHeight = Number(source.height);
    setResolutionMirrorValues(node, runtime, resolved.width, resolved.height);
    runtime.resolutionMirrorActive = true;
    node.graph?.setDirtyCanvas(true, true);
}

function wrapResolutionWidgetCallbacks(node, runtime) {
    if (!node || !runtime || runtime.resolutionCallbacksInstalled) return;
    runtime.resolutionCallbacksInstalled = true;

    const widthWidget = getWidget(node, "width");
    const heightWidget = getWidget(node, "height");
    const modeWidget = getWidget(node, "resolution_mode");
    const mpWidget = getWidget(node, "megapixels");

    const wrap = (widget, handler) => {
        if (!widget) return;
        const old = widget.callback;
        widget.callback = function (value) {
            const result = old ? old.apply(this, arguments) : undefined;
            handler(value);
            return result;
        };
    };

    wrap(widthWidget, (value) => {
        if (runtime.applyingResolutionMirror) return;
        const mode = String(modeWidget?.value || "auto_from_ref");
        if (mode === "manual" || !hasAutoResolutionGuide(runtime)) {
            // Manual edits update the independent fallback geometry.
            runtime.projectResolutionLoaded = false;
            rememberManualResolution(
                node,
                runtime,
                Number(value || widthWidget?.value || runtime.manualWidth || 896),
                runtime.manualHeight,
            );
            invalidateForResolutionChange(node, runtime);
        } else {
            requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
        }
    });
    wrap(heightWidget, (value) => {
        if (runtime.applyingResolutionMirror) return;
        const mode = String(modeWidget?.value || "auto_from_ref");
        if (mode === "manual" || !hasAutoResolutionGuide(runtime)) {
            runtime.projectResolutionLoaded = false;
            rememberManualResolution(
                node,
                runtime,
                runtime.manualWidth,
                Number(value || heightWidget?.value || runtime.manualHeight || 576),
            );
            invalidateForResolutionChange(node, runtime);
        } else {
            requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
        }
    });
    wrap(modeWidget, (value) => {
        runtime.projectResolutionLoaded = false;
        const mode = String(value || modeWidget?.value || "auto_from_ref");
        if (mode === "auto_from_ref" && !runtime.resolutionMirrorActive) {
            rememberManualResolution(
                node,
                runtime,
                Number(widthWidget?.value || runtime.manualWidth || 896),
                Number(heightWidget?.value || runtime.manualHeight || 576),
            );
        }
        requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
    });
    wrap(mpWidget, () => {
        // Compatibility guard for older runtimes that may still carry the
        // one-shot projectResolutionLoaded flag. Megapixels is an Auto-only
        // control, so an explicit edit releases that legacy lock.
        if (runtime.projectResolutionLoaded) {
            runtime.projectResolutionLoaded = false;
            setWidgetValue(node, "resolution_mode", "auto_from_ref");
        }
        requestAnimationFrame(() => syncResolutionAndInvalidate(node, runtime));
    });
}

// Nodes 2.0 (Vue) can render the native multiline STRING row before our
// onNodeCreated code gets a chance to touch the widget object. Hide that row
// pre-emptively with CSS, using the same proven strategy as ComfyUI_Stem_Mixer.
// MiniMaxH3Extender keeps clips_json + refs_json as native serialized textareas.
(function injectStateJsonHideRule() {
    if (document.getElementById("h3-extender-hide-state-json")) return;
    const style = document.createElement("style");
    style.id = "h3-extender-hide-state-json";
    style.textContent = `
        .lg-node-widget:has(> [node-type="${TARGET}"] > textarea) {
            display: none !important;
        }
    `;
    document.head.appendChild(style);
})();

function hideNativeWidget(node, widget) {
    if (!widget) return;

    // Modern Nodes 2.0 renders native widgets from its own widget store. Merely
    // giving a widget a zero layout size is not enough: the control can remain
    // visible while the following DOM widget is laid out in the same row, which
    // causes the overlap seen with the hidden generation_mode combo. Mark the
    // widget hidden as well, while keeping it alive for normal serialization.
    widget.hidden = true;

    // LiteGraph / Nodes 1.0: also remove the logical footprint but keep the
    // widget itself intact so workflow serialization continues to work.
    widget.computeSize = () => [0, -4];
    widget.computeLayoutSize = () => ({
        minWidth: 0,
        minHeight: 0,
        maxWidth: 0,
        maxHeight: 0,
    });

    // LiteGraph may recreate the textarea when the node leaves/re-enters the
    // viewport, so re-hide the actual legacy DOM element on every foreground
    // draw, exactly as Stem Mixer does for its state widget.
    const oldDrawForeground = node?.onDrawForeground;
    if (node) {
        node.onDrawForeground = function (ctx) {
            if (oldDrawForeground) oldDrawForeground.apply(this, arguments);
            const inputEl = widget.inputEl;
            if (inputEl) {
                if (inputEl.style.display !== "none") inputEl.style.display = "none";
                const parent = inputEl.parentElement;
                if (parent && parent.style.display !== "none") {
                    parent.style.display = "none";
                }
            }
        };
    }
}


function setNativeWidgetVisibility(node, widget, visible) {
    if (!widget) return;
    if (!Object.prototype.hasOwnProperty.call(widget, "__h3OriginalComputeSize")) {
        widget.__h3OriginalComputeSize = widget.computeSize;
        widget.__h3OriginalComputeLayoutSize = widget.computeLayoutSize;
    }

    widget.hidden = !visible;
    if (visible) {
        if (widget.__h3OriginalComputeSize !== undefined) widget.computeSize = widget.__h3OriginalComputeSize;
        else delete widget.computeSize;
        if (widget.__h3OriginalComputeLayoutSize !== undefined) widget.computeLayoutSize = widget.__h3OriginalComputeLayoutSize;
        else delete widget.computeLayoutSize;
        const inputEl = widget.inputEl;
        if (inputEl) {
            inputEl.style.display = "";
            if (inputEl.parentElement) inputEl.parentElement.style.display = "";
        }
    } else {
        widget.computeSize = () => [0, -4];
        widget.computeLayoutSize = () => ({
            minWidth: 0,
            minHeight: 0,
            maxWidth: 0,
            maxHeight: 0,
        });
        const inputEl = widget.inputEl;
        if (inputEl) {
            inputEl.style.display = "none";
            if (inputEl.parentElement) inputEl.parentElement.style.display = "none";
        }
    }
    node?.graph?.setDirtyCanvas(true, true);
}

function syncModeSpecificNativeWidgets(node, runtime, fl2vaMode) {
    // Motion Context controls have no meaning in FL2VA. Hide them only in that
    // mode while keeping their values intact for the independent Ref2VA state.
    setNativeWidgetVisibility(node, runtime?.contextLengthWidget, !fl2vaMode);
    setNativeWidgetVisibility(node, runtime?.audioContextLengthWidget, !fl2vaMode);
}

function domWidgetRenderMode(element) {
    // ComfyUI exposes the renderer state on LiteGraph.vueNodesMode. Use that
    // as the authority, but wait while the DOM widget is being re-parented so
    // we never apply Legacy sizing with a stale Vue last_y (or vice versa).
    const LG = globalThis.LiteGraph;
    const hasModeFlag = typeof LG?.vueNodesMode === "boolean";
    if (!element?.isConnected) return "pending";

    const insideVueRow = Boolean(element.closest?.(".lg-node-widget"));
    if (hasModeFlag) {
        if (LG.vueNodesMode && !insideVueRow) return "pending";
        if (!LG.vueNodesMode && insideVueRow) return "pending";
        return LG.vueNodesMode ? "nodes2" : "legacy";
    }

    // Older frontends may not expose vueNodesMode; fall back to the wrapper.
    return insideVueRow ? "nodes2" : "legacy";
}

function obviouslyPoisonedHeight(height, minimumHeight) {
    const h = Number(height);
    if (!Number.isFinite(h) || h <= 0) return false;
    return h > Math.max(1800, Number(minimumHeight || 0) * 3);
}

function invalidateFrom(state, index) {
    for (let i = Math.max(0, index); i < state.clips.length; i++) {
        state.clips[i].validated = false;
    }
}

function currentResolutionFromWidgets(node) {
    const width = Number(getWidget(node, "width")?.value || 0);
    const height = Number(getWidget(node, "height")?.value || 0);
    if (!(width > 0) || !(height > 0)) return null;
    return effectiveManualResolution(width, height);
}

function invalidateForResolutionChange(node, runtime) {
    if (!node || !runtime?.state) return false;
    const expected = runtime.expectedResolution;
    const current = currentResolutionFromWidgets(node);
    if (!expected || !current) return false;

    const expectedW = Number(expected.width || 0);
    const expectedH = Number(expected.height || 0);
    if (!(expectedW > 0) || !(expectedH > 0)) return false;
    if (current.width === expectedW && current.height === expectedH) return false;

    const hadValidated = runtime.state.clips.some((clip) => Boolean(clip?.validated));
    const hadCached = Number(runtime.cachedCount || 0) > 0;

    // Once the requested geometry differs from the cache/project geometry,
    // every latent in that chain is incompatible. Reflect that immediately in
    // the cards instead of waiting for the backend to discover it at Queue.
    for (const clip of runtime.state.clips) clip.validated = false;
    runtime.validatedCount = 0;
    runtime.cachedCount = 0;
    runtime.resolutionInvalidated = true;
    runtime.statusText =
        `Resolution changed: ${expectedW}x${expectedH} → ${current.width}x${current.height} | ` +
        `clips invalidated; cache resets on next run`;

    if (hadValidated || hadCached) updateHidden(node, runtime);
    render(node, runtime);
    node.graph?.setDirtyCanvas(true, true);
    return hadValidated || hadCached;
}

function syncResolutionAndInvalidate(node, runtime) {
    syncResolutionMirror(node, runtime);
    invalidateForResolutionChange(node, runtime);
}


function advanceSeedAfterGenerate(clip) {
    const mode = String(clip?.seed_mode || "randomize");
    const max = Number.MAX_SAFE_INTEGER;
    const current = Math.max(0, Math.min(max, Math.trunc(Number(clip?.seed || 0))));

    if (mode === "randomize") {
        let next = randomSeed();
        // Extremely unlikely, but never leave the node cache-identical.
        if (next === current) next = (current + 1) % (max + 1);
        clip.seed = next;
    } else if (mode === "increment") {
        clip.seed = current >= max ? 0 : current + 1;
    } else if (mode === "decrement") {
        clip.seed = current <= 0 ? max : current - 1;
    }
    // fixed deliberately does nothing.
}

function cardStatus(runtime, clip, index) {
    if (
        Number(runtime.activeClipIndex) === index &&
        ["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""))
    ) {
        return "rendering";
    }

    const fl2va = runtime.state?.generation_mode === "fl2va";
    const cached = fl2va
        ? runtime.cachedClipIds?.has(String(clip.id))
        : index < Number(runtime.cachedCount || 0);
    if (clip.validated && cached) return "validated";
    const firstOpen = runtime.state.clips.findIndex((c) => !c.validated);
    if (index === firstOpen) return cached ? "candidate" : "current";
    if (cached) return "cached";
    return "future";
}

function snapshotModeValidation(runtime, mode = null) {
    if (!runtime?.state) return;
    if (!runtime.modeValidationState) runtime.modeValidationState = {};
    const key = String(mode || runtime.state.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    runtime.modeValidationState[key] = new Map(
        (runtime.state.clips || []).map((clip) => [String(clip.id), Boolean(clip.validated)])
    );
}

function restoreModeValidation(runtime, mode) {
    const key = String(mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    const saved = runtime?.modeValidationState?.[key];
    if (!(saved instanceof Map)) return false;
    for (const clip of runtime.state?.clips || []) {
        clip.validated = Boolean(saved.get(String(clip.id)));
    }
    return true;
}

function explicitGenerationModeFromStateJson(raw) {
    if (typeof raw !== "string" || !raw.trim()) return "";
    try {
        const parsed = JSON.parse(raw);
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") return "";
        if (!Object.prototype.hasOwnProperty.call(parsed, "generation_mode")) return "";
        const mode = String(parsed.generation_mode || "").toLowerCase();
        return mode === "fl2va" || mode === "ref2va" ? mode : "";
    } catch (_) {
        return "";
    }
}

function persistentGenerationMode(node, raw = "") {
    // Both sources below are native ComfyUI widgets and are therefore restored
    // by LGraphNode.configure(). clips_json is preferred for compatibility with
    // workflows saved before the trailing generation_mode combo existed.
    const explicit = explicitGenerationModeFromStateJson(raw);
    if (explicit) return explicit;
    const widgetMode = String(getWidget(node, "generation_mode")?.value || "").toLowerCase();
    return widgetMode === "fl2va" ? "fl2va" : "ref2va";
}

function captureNativeWorkflowState(node, runtime = null) {
    // Custom DOM buttons mutate hidden native widgets on `click`, but Nodes 2.0
    // captures normal UI edits on the preceding `mouseup`. Without an explicit
    // post-mutation capture, ChangeTracker.activeState can therefore lag one
    // interaction behind the canvas and a quick refresh can restore the old mode.
    if (runtime?.hydrating || isH3GraphConfiguring()) return false;
    try {
        const workflow = app?.extensionManager?.workflow?.activeWorkflow;
        const tracker = workflow?.changeTracker;
        if (!tracker) return false;
        if (typeof tracker.captureCanvasState === "function") {
            tracker.captureCanvasState();
            return true;
        }
        // Compatibility fallback for older frontends.
        if (typeof tracker.checkState === "function") {
            tracker.checkState();
            return true;
        }
    } catch (_) {}
    return false;
}

function notifyWorkflowChanged(node, runtime = null) {
    const graph = node?.graph || app.graph;
    // Never mark the graph changed while ComfyUI is hydrating/configuring this
    // node from an existing workflow. Doing so can make the frontend serialize
    // the temporary schema defaults before the saved widgets have finished
    // restoring; a second browser refresh would then load that poisoned snapshot.
    if (runtime?.hydrating || isH3GraphConfiguring()) {
        graph?.setDirtyCanvas?.(true, true);
        return;
    }
    // setDirtyCanvas() only repaints. graph.change() is the actual LiteGraph /
    // Nodes 2.0 mutation notification used for genuine custom-DOM edits.
    try { graph?.change?.(); } catch (_) {}
    graph?.setDirtyCanvas?.(true, true);
}

function updateHidden(node, runtime) {
    snapshotModeValidation(runtime);
    const raw = serializeState(runtime.state);
    runtime.jsonWidget.value = raw;
    if (runtime.generationModeWidget) {
        runtime.generationModeWidget.value = runtime.state?.generation_mode === "fl2va" ? "fl2va" : "ref2va";
    }
    notifyWorkflowChanged(node, runtime);
}

function updateRefsHidden(node, runtime) {
    if (!runtime?.refsWidget) return;
    runtime.refsState.refs = normalizeRefsArray(runtime.refsState?.refs || []);
    runtime.refsWidget.value = serializeRefsState(runtime.refsState);
    notifyWorkflowChanged(node, runtime);
}

function handleReferenceChange(node, runtime, message = "Image references changed") {
    if (!node || !runtime?.state) return;

    // Reference edits are deliberately user-controlled. Do not infer any
    // Ref-to-Clip relationship and do not change validation automatically.
    updateRefsHidden(node, runtime);

    // Auto resolution still follows the active guide ref. If the ref edit changes
    // the effective geometry, the existing resolution safety rule necessarily
    // invalidates the whole latent chain; that is independent of ref semantics.
    syncResolutionAndInvalidate(node, runtime);

    if (!runtime.resolutionInvalidated) {
        runtime.statusText = `${message} | validations unchanged`;
        render(node, runtime);
    }
}

function openReferenceEditor(node, runtime, slotIndex, ref, target = null) {
    if (!ref?.id || !node || !runtime) return;
    const isFrame = Boolean(target && ["first", "last", "guide"].includes(String(target.kind)));
    const frameClipIndex = isFrame ? Number(target.clipIndex) : -1;
    const frameKind = isFrame ? String(target.kind) : "";
    const frameGuideIndex = frameKind === "guide" ? Number(target?.guideIndex) : -1;
    const frameKindLabel = frameKind === "first"
        ? "First frame"
        : frameKind === "last"
            ? "Last frame"
            : `Guide ${Number.isInteger(frameGuideIndex) && frameGuideIndex >= 0 ? frameGuideIndex + 1 : 1}`;
    const frameLabel = isFrame
        ? `Clip ${frameClipIndex + 1} ${frameKindLabel}`
        : `Ref ${slotIndex + 1}`;
    const defaultName = isFrame
        ? (frameKind === "guide"
            ? `clip_${frameClipIndex + 1}_guide_${Math.max(0, frameGuideIndex) + 1}.png`
            : `clip_${frameClipIndex + 1}_${frameKind}.png`)
        : `ref_${slotIndex + 1}.png`;
    if (projectBusy(runtime) || runtime.refBusy || runtime.projectOperationBusy) {
        alert("Wait for the current clip generation to finish before editing a reference image.");
        return;
    }

    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "100000";
    overlay.style.background = "rgba(0,0,0,.86)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.padding = "24px";
    overlay.style.boxSizing = "border-box";

    const panel = document.createElement("div");
    panel.style.width = "min(1180px, 94vw)";
    panel.style.height = "min(820px, 92vh)";
    panel.style.minWidth = "0";
    panel.style.minHeight = "0";
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.background = "#191919";
    panel.style.border = "1px solid rgba(255,255,255,.18)";
    panel.style.borderRadius = "10px";
    panel.style.boxShadow = "0 18px 60px rgba(0,0,0,.65)";
    panel.style.overflow = "hidden";
    overlay.appendChild(panel);

    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "12px";
    header.style.padding = "10px 12px";
    header.style.borderBottom = "1px solid rgba(255,255,255,.12)";

    const title = document.createElement("div");
    title.textContent = `Reference Editor — ${frameLabel}`;
    title.style.fontWeight = "650";
    title.style.fontSize = "13px";
    title.style.overflow = "hidden";
    title.style.textOverflow = "ellipsis";
    title.style.whiteSpace = "nowrap";
    title.title = ref.original_name || frameLabel;

    const closeButton = document.createElement("button");
    closeButton.textContent = "×";
    closeButton.title = "Close";
    closeButton.style.width = "28px";
    closeButton.style.minWidth = "28px";
    closeButton.style.height = "26px";
    closeButton.style.padding = "0";
    closeButton.style.fontSize = "18px";
    header.append(title, closeButton);
    panel.appendChild(header);

    const body = document.createElement("div");
    body.style.flex = "1 1 auto";
    body.style.minHeight = "0";
    body.style.minWidth = "0";
    body.style.display = "flex";
    body.style.gap = "0";
    panel.appendChild(body);

    const previewWrap = document.createElement("div");
    previewWrap.style.flex = "1 1 auto";
    previewWrap.style.minWidth = "0";
    previewWrap.style.minHeight = "0";
    previewWrap.style.display = "flex";
    previewWrap.style.alignItems = "center";
    previewWrap.style.justifyContent = "center";
    previewWrap.style.padding = "14px";
    previewWrap.style.boxSizing = "border-box";
    previewWrap.style.background = "#0f0f0f";

    const image = document.createElement("img");
    const sourceRef = { ...ref, id: ref.source_id || ref.id };
    image.src = refImageUrl(sourceRef);
    image.alt = ref.original_name || "Reference image";
    image.style.maxWidth = "100%";
    image.style.maxHeight = "100%";
    image.style.objectFit = "contain";
    image.style.borderRadius = "6px";
    image.style.boxShadow = "0 8px 30px rgba(0,0,0,.45)";
    image.draggable = false;
    previewWrap.appendChild(image);
    body.appendChild(previewWrap);

    const controls = document.createElement("div");
    controls.style.flex = "0 0 235px";
    controls.style.width = "235px";
    controls.style.boxSizing = "border-box";
    controls.style.padding = "14px";
    controls.style.borderLeft = "1px solid rgba(255,255,255,.12)";
    controls.style.display = "flex";
    controls.style.flexDirection = "column";
    controls.style.gap = "12px";
    controls.style.overflowY = "auto";
    body.appendChild(controls);

    const makeControl = (labelText) => {
        const wrap = document.createElement("div");
        wrap.style.display = "block";
        wrap.style.fontSize = "11px";
        wrap.style.fontWeight = "600";

        const headerRow = document.createElement("div");
        headerRow.style.display = "flex";
        headerRow.style.alignItems = "center";
        headerRow.style.justifyContent = "space-between";
        headerRow.style.gap = "8px";
        headerRow.style.marginBottom = "4px";

        const label = document.createElement("div");
        label.textContent = labelText;

        const number = document.createElement("input");
        number.type = "number";
        number.min = "0";
        number.max = "200";
        number.step = "1";
        number.value = "100";
        number.style.width = "58px";
        number.style.boxSizing = "border-box";
        number.style.padding = "3px 5px";
        number.style.borderRadius = "5px";
        number.style.border = "1px solid rgba(255,255,255,.18)";
        number.style.background = "rgba(0,0,0,.28)";
        number.style.color = "inherit";
        number.style.textAlign = "right";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "200";
        slider.step = "1";
        slider.value = "100";
        slider.style.width = "100%";
        slider.style.margin = "0";
        slider.style.padding = "0";
        slider.style.boxSizing = "border-box";
        slider.title = `${labelText}: 100`;

        headerRow.append(label, number);
        wrap.append(headerRow, slider);
        controls.appendChild(wrap);
        return { slider, number, labelText };
    };

    const saturation = makeControl("Saturation (%)");
    const contrast = makeControl("Contrast (%)");
    const brightness = makeControl("Brightness (%)");

    const help = document.createElement("div");
    help.textContent = "100 = original image. Edits are always calculated from the initially loaded reference, so Reset truly restores the original pixels.";
    help.style.fontSize = "10px";
    help.style.lineHeight = "1.35";
    help.style.opacity = ".66";
    controls.appendChild(help);

    const spacer = document.createElement("div");
    spacer.style.flex = "1 1 auto";
    controls.appendChild(spacer);

    const buttons = document.createElement("div");
    buttons.style.display = "grid";
    buttons.style.gridTemplateColumns = "1fr 1fr";
    buttons.style.gap = "7px";

    const reset = document.createElement("button");
    reset.textContent = "Reset";
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.style.gridColumn = "1 / -1";
    apply.style.fontWeight = "650";
    buttons.append(reset, cancel, apply);
    controls.appendChild(buttons);

    const numericValue = (control) => {
        const value = Number(control.slider.value);
        if (!Number.isFinite(value)) return 100;
        return Math.min(200, Math.max(0, value));
    };

    const setControlValue = (control, value) => {
        const parsed = Number(value);
        const clamped = Number.isFinite(parsed) ? Math.min(200, Math.max(0, parsed)) : 100;
        const text = String(Math.round(clamped));
        control.slider.value = text;
        control.number.value = text;
        control.slider.title = `${control.labelText}: ${text}`;
    };

    setControlValue(saturation, ref.saturation ?? 100);
    setControlValue(contrast, ref.contrast ?? 100);
    setControlValue(brightness, ref.brightness ?? 100);

    const updatePreview = () => {
        const b = numericValue(brightness);
        const c = numericValue(contrast);
        const sat = numericValue(saturation);
        image.style.filter = `brightness(${b}%) contrast(${c}%) saturate(${sat}%)`;
    };
    for (const control of [saturation, contrast, brightness]) {
        control.slider.addEventListener("input", () => {
            control.number.value = control.slider.value;
            control.slider.title = `${control.labelText}: ${control.slider.value}`;
            updatePreview();
        });
        control.number.addEventListener("input", () => {
            const value = Number(control.number.value);
            if (Number.isFinite(value)) {
                setControlValue(control, value);
                updatePreview();
            }
        });
        control.number.addEventListener("change", () => {
            setControlValue(control, control.number.value);
            updatePreview();
        });
    }
    updatePreview();

    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        window.removeEventListener("keydown", onKey);
        overlay.remove();
    };
    const onKey = (event) => {
        if (event.key === "Escape") close();
    };
    closeButton.addEventListener("click", close);
    cancel.addEventListener("click", close);
    overlay.addEventListener("click", close);
    panel.addEventListener("click", (event) => event.stopPropagation());
    window.addEventListener("keydown", onKey);

    reset.addEventListener("click", () => {
        setControlValue(saturation, 100);
        setControlValue(contrast, 100);
        setControlValue(brightness, 100);
        updatePreview();
    });

    apply.addEventListener("click", async () => {
        if (projectBusy(runtime) || runtime.refBusy || runtime.projectOperationBusy) {
            alert("Wait for the current clip generation to finish before editing a reference image.");
            return;
        }
        apply.disabled = true;
        reset.disabled = true;
        cancel.disabled = true;
        runtime.refBusy = true;
        runtime.statusText = `Applying ${frameLabel} adjustments…`;
        render(node, runtime);
        try {
            const response = await fetch(api.apiURL("/h3_extender/ref/edit"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ref_id: ref.id,
                    source_id: ref.source_id || ref.id,
                    original_name: ref.original_name || defaultName,
                    saturation: numericValue(saturation),
                    contrast: numericValue(contrast),
                    brightness: numericValue(brightness),
                    external_signature: ref.external_signature || "",
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload?.ok || !payload?.ref) {
                throw new Error(payload?.error || `Reference edit failed (${response.status}).`);
            }
            const newRef = normalizeRefDescriptor(payload.ref);
            if (!newRef) throw new Error("The backend returned invalid reference metadata.");

            const current = isFrame
                ? (frameKind === "guide"
                    ? runtime.state?.clips?.[frameClipIndex]?.guides?.[frameGuideIndex]?.frame
                    : runtime.state?.clips?.[frameClipIndex]?.[`${frameKind}_frame`])
                : runtime.refsState.refs[slotIndex];
            if (!current || String(current.id) !== String(ref.id)) {
                throw new Error(`${frameLabel} changed while the editor was open.`);
            }

            if (isFrame) {
                if (frameKind === "guide") {
                    const guide = runtime.state?.clips?.[frameClipIndex]?.guides?.[frameGuideIndex];
                    if (!guide) throw new Error(`${frameLabel} changed while the editor was open.`);
                    guide.frame = newRef;
                } else {
                    runtime.state.clips[frameClipIndex][`${frameKind}_frame`] = newRef;
                }
                updateHidden(node, runtime);
                captureNativeWorkflowState(node, runtime);
                runtime.statusText = sameRefContent(ref, newRef)
                    ? `${frameLabel} unchanged`
                    : `${frameLabel} adjusted | validations unchanged`;
                render(node, runtime);
            } else {
                runtime.refsState.refs[slotIndex] = newRef;
                if (sameRefContent(ref, newRef)) {
                    updateRefsHidden(node, runtime);
                    runtime.statusText = `Ref ${slotIndex + 1} unchanged`;
                    render(node, runtime);
                } else {
                    handleReferenceChange(node, runtime, `Ref ${slotIndex + 1} adjusted`);
                }
            }
            close();
        } catch (error) {
            runtime.statusText = "Reference edit failed";
            render(node, runtime);
            alert(String(error?.message || error));
            apply.disabled = false;
            reset.disabled = false;
            cancel.disabled = false;
        } finally {
            runtime.refBusy = false;
            render(node, runtime);
        }
    });

    document.body.appendChild(overlay);
}

async function uploadReference(node, runtime, slotIndex, file) {
    if (!node || !runtime || !file) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before changing a reference image.");
        return;
    }

    runtime.refBusy = true;
    runtime.statusText = `Loading Ref ${slotIndex + 1}: ${file.name}…`;
    render(node, runtime);
    try {
        const form = new FormData();
        form.append("ref_file", file, file.name);
        const response = await fetch(api.apiURL("/h3_extender/ref/upload"), {
            method: "POST",
            body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok || !payload?.ref) {
            throw new Error(payload?.error || `Reference upload failed (${response.status}).`);
        }

        const newRef = normalizeRefDescriptor(payload.ref);
        if (!newRef) throw new Error("The backend returned invalid reference metadata.");
        const previous = runtime.refsState.refs[slotIndex];
        if (sameRefContent(previous, newRef)) {
            runtime.refsState.refs[slotIndex] = newRef;
            updateRefsHidden(node, runtime);
            runtime.statusText = `Ref ${slotIndex + 1} unchanged`;
            render(node, runtime);
            return;
        }

        runtime.refsState.refs[slotIndex] = newRef;
        handleReferenceChange(node, runtime, `Ref ${slotIndex + 1} loaded`);
    } catch (error) {
        runtime.statusText = "Reference load failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        runtime.refBusy = false;
        render(node, runtime);
    }
}

async function uploadClipFrame(node, runtime, clipIndex, kind, file, guideIndex = -1) {
    if (!node || !runtime || !file) return;
    const clip = runtime.state?.clips?.[clipIndex];
    if (!clip || !["first", "last", "guide"].includes(kind)) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before changing an FL2VA keyframe.");
        return;
    }
    if (kind === "guide" && (!Number.isInteger(guideIndex) || guideIndex < 0 || guideIndex > MAX_FL2VA_GUIDES)) return;
    runtime.refBusy = true;
    const label = kind === "guide" ? `guide ${guideIndex + 1}` : `${kind} frame`;
    runtime.statusText = `Loading Clip ${clipIndex + 1} ${label}: ${file.name}…`;
    render(node, runtime);
    try {
        const form = new FormData();
        form.append("ref_file", file, file.name);
        const response = await fetch(api.apiURL("/h3_extender/ref/upload"), { method: "POST", body: form });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok || !payload?.ref) {
            throw new Error(payload?.error || `FL2VA frame upload failed (${response.status}).`);
        }
        const ref = normalizeRefDescriptor(payload.ref);
        if (!ref) throw new Error("The backend returned invalid FL2VA frame metadata.");

        if (kind === "guide") {
            clip.guides = normalizeGuideList(clip);
            if (guideIndex < clip.guides.length) {
                clip.guides[guideIndex].frame = ref;
            } else if (guideIndex === clip.guides.length && clip.guides.length < MAX_FL2VA_GUIDES) {
                clip.guides.push({ frame: ref, frame_idx: 0 });
            } else {
                throw new Error(`A maximum of ${MAX_FL2VA_GUIDES} image guides is supported per FL2VA clip.`);
            }
        } else {
            clip[`${kind}_frame`] = ref;
            if (kind === "first" && String(clip.first_source || "manual") === "previous_clip") {
                clip.first_source = "manual";
                invalidateFl2vaPlanAndFollowers(runtime, clipIndex, true);
            }
        }
        updateHidden(node, runtime);
        captureNativeWorkflowState(node, runtime);
        runtime.statusText = kind === "guide"
            ? `Clip ${clipIndex + 1} Guide ${guideIndex + 1} loaded`
            : `Clip ${clipIndex + 1} ${kind} frame loaded`;
    } catch (error) {
        runtime.statusText = "FL2VA frame load failed";
        alert(String(error?.message || error));
    } finally {
        runtime.refBusy = false;
        render(node, runtime);
    }
}

function removeClipFrame(node, runtime, clipIndex, kind, guideIndex = -1) {
    const clip = runtime?.state?.clips?.[clipIndex];
    if (!clip || !["first", "last", "guide"].includes(kind)) return;
    if (kind === "guide") {
        clip.guides = normalizeGuideList(clip);
        if (!Number.isInteger(guideIndex) || guideIndex < 0 || guideIndex >= clip.guides.length) return;
        clip.guides.splice(guideIndex, 1);
    } else {
        clip[`${kind}_frame`] = null;
    }
    updateHidden(node, runtime);
    captureNativeWorkflowState(node, runtime);
    render(node, runtime);
}

function removeReference(node, runtime, slotIndex) {
    if (!runtime?.refsState?.refs?.[slotIndex]) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before changing a reference image.");
        return;
    }
    const oldName = runtime.refsState.refs[slotIndex]?.original_name || `Ref ${slotIndex + 1}`;
    runtime.refsState.refs[slotIndex] = null;
    handleReferenceChange(node, runtime, `${oldName} removed`);
}

function nodeIs(node, className) {
    return node?.comfyClass === className || node?.type === className;
}

function connectedFinalDecode(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return null;
    const output = (node.outputs || []).find((o) => o?.name === "cache") || node.outputs?.[0];
    for (const linkId of output?.links || []) {
        const link = graph.links?.[linkId];
        if (!link) continue;
        const target = graph.getNodeById?.(link.target_id)
            || (graph._nodes || []).find((n) => String(n?.id) === String(link.target_id));
        if (target && nodeIs(target, FINAL_TARGET)) return target;
    }
    return null;
}

function colorMediaUrl(info) {
    const params = new URLSearchParams();
    params.set("filename", info?.filename || "");
    params.set("type", info?.type || "temp");
    params.set("subfolder", info?.subfolder || "");
    return api.apiURL("/view?" + params.toString());
}

function colorAtTimelineTime(timeline, time, targetIndex, liveAdjustment) {
    const t = Number(time || 0);
    for (const item of timeline || []) {
        const start = Number(item?.start || 0);
        const end = Number(item?.end || start);
        if (t >= start && t < end) {
            if (Number(item?.index) === Number(targetIndex)) return liveAdjustment;
            return normalizeColorAdjustment(item?.adjustment);
        }
    }
    return normalizeColorAdjustment();
}

function closeColorEditor(overlay) {
    try {
        const video = overlay?.querySelector?.("video");
        if (video) {
            video.pause();
            video.removeAttribute("src");
            video.load();
        }
    } catch (_) {}
    overlay?.remove?.();
}

async function openClipColorEditor(node, runtime, clipIndex) {
    const finalNode = connectedFinalDecode(node);
    if (!finalNode) {
        alert("Connect the Extender cache output to Final Decode / Preview first.");
        return;
    }

    const params = new URLSearchParams();
    params.set("owner_id", String(node.id));
    params.set("final_id", String(finalNode.id));
    params.set("clip_index", String(clipIndex));
    params.set("mode", String(runtime.state?.generation_mode || "ref2va"));

    let payload;
    try {
        const response = await fetch(
            api.apiURL("/h3_extender/color_editor_info?" + params.toString())
        );
        payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Color editor failed (${response.status}).`);
        }
    } catch (error) {
        alert(`Color editor unavailable:\n${error?.message || error}`);
        return;
    }

    const timeline = Array.isArray(payload.timeline) ? payload.timeline : [];
    const target = timeline.find((item) => Number(item?.index) === Number(clipIndex));
    if (!target || !payload?.video?.filename) {
        alert("The decoded clip preview is not available yet.");
        return;
    }

    const clip = runtime.state.clips[clipIndex];
    let adjustment = normalizeColorAdjustment(
        clip?.color_adjustment || target?.adjustment
    );

    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "100000";
    overlay.style.background = "rgba(0,0,0,.78)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.padding = "24px";
    overlay.style.boxSizing = "border-box";

    const dialog = document.createElement("div");
    dialog.style.width = "min(1040px, 94vw)";
    dialog.style.maxHeight = "92vh";
    dialog.style.overflow = "auto";
    dialog.style.background = "#171717";
    dialog.style.color = "#f0f0f0";
    dialog.style.border = "1px solid rgba(255,255,255,.18)";
    dialog.style.borderRadius = "10px";
    dialog.style.boxShadow = "0 18px 60px rgba(0,0,0,.65)";
    dialog.style.padding = "14px";
    dialog.style.boxSizing = "border-box";

    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "12px";
    header.style.marginBottom = "10px";

    const title = document.createElement("strong");
    const clipName = String(clip?.name || "").trim();
    title.textContent = `Color Edit — Clip ${clipIndex + 1}${clipName ? ` — ${clipName}` : ""}`;
    title.style.fontSize = "15px";

    const close = document.createElement("button");
    close.textContent = "✕";
    close.title = "Close";
    close.style.width = "30px";
    close.style.height = "26px";
    close.style.cursor = "pointer";
    close.addEventListener("click", () => closeColorEditor(overlay));
    header.append(title, close);

    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "auto";
    video.style.display = "block";
    video.style.width = "100%";
    video.style.maxHeight = "58vh";
    video.style.objectFit = "contain";
    video.style.background = "#000";
    video.style.borderRadius = "6px";

    const totalEnd = timeline.length ? Number(timeline[timeline.length - 1]?.end || 0) : Number(target.end || 0);
    const loopStart = Math.max(0, Number(target.start || 0) - 2.0);
    const loopEnd = Math.min(totalEnd, Number(target.end || 0) + 2.0);

    const loopInfo = document.createElement("div");
    loopInfo.textContent = `Loop: ${loopStart.toFixed(2)}s → ${loopEnd.toFixed(2)}s  •  target ${Number(target.start).toFixed(2)}s → ${Number(target.end).toFixed(2)}s`;
    loopInfo.style.fontSize = "11px";
    loopInfo.style.opacity = ".72";
    loopInfo.style.margin = "7px 0 10px";

    const controls = document.createElement("div");
    controls.style.display = "grid";
    controls.style.gridTemplateColumns = "1fr";
    controls.style.gap = "8px";

    const valueInputs = {};
    const sliderRows = [];
    const makeSlider = (key, label, min, max) => {
        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "100px 1fr 64px";
        row.style.gap = "10px";
        row.style.alignItems = "center";

        const text = document.createElement("span");
        text.textContent = label;
        text.style.fontSize = "12px";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(min);
        slider.max = String(max);
        slider.step = "1";
        slider.value = String(Math.round(adjustment[key]));
        slider.style.width = "100%";

        const number = document.createElement("input");
        number.type = "number";
        number.min = String(min);
        number.max = String(max);
        number.step = "1";
        number.value = String(Math.round(adjustment[key]));
        number.style.width = "64px";
        number.style.boxSizing = "border-box";
        number.style.background = "rgba(0,0,0,.35)";
        number.style.color = "inherit";
        number.style.border = "1px solid rgba(255,255,255,.18)";
        number.style.borderRadius = "4px";
        number.style.padding = "3px 5px";

        const update = (raw) => {
            const n = Math.max(min, Math.min(max, Number(raw)));
            adjustment = { ...adjustment, [key]: Number.isFinite(n) ? n : 100 };
            slider.value = String(Math.round(adjustment[key]));
            number.value = String(Math.round(adjustment[key]));
            updateLiveFilter();
        };
        slider.addEventListener("input", () => update(slider.value));
        number.addEventListener("input", () => update(number.value));
        valueInputs[key] = { slider, number, update };
        row.append(text, slider, number);
        sliderRows.push(row);
        controls.appendChild(row);
    };

    const updateLiveFilter = () => {
        const c = colorAtTimelineTime(timeline, video.currentTime, clipIndex, adjustment);
        video.style.filter = cssColorFilter(c);
    };

    makeSlider("saturation", "Saturation", 0, 200);
    makeSlider("contrast", "Contrast", 50, 150);
    makeSlider("brightness", "Brightness", 50, 150);

    const buttons = document.createElement("div");
    buttons.style.display = "flex";
    buttons.style.justifyContent = "flex-end";
    buttons.style.gap = "8px";
    buttons.style.marginTop = "12px";

    const reset = document.createElement("button");
    reset.textContent = "Reset";
    reset.title = "Return this clip to neutral 100 / 100 / 100";
    reset.addEventListener("click", () => {
        adjustment = normalizeColorAdjustment();
        for (const [key, pair] of Object.entries(valueInputs)) {
            pair.slider.value = String(Math.round(adjustment[key]));
            pair.number.value = String(Math.round(adjustment[key]));
        }
        updateLiveFilter();
    });

    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => closeColorEditor(overlay));

    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.style.fontWeight = "700";
    apply.style.minWidth = "84px";
    apply.addEventListener("click", async () => {
        apply.disabled = true;
        apply.textContent = "Applying...";
        try {
            const response = await fetch(api.apiURL("/h3_extender/color_adjust"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    owner_id: String(node.id),
                    clip_index: Number(clipIndex),
                    generation_mode: String(runtime.state?.generation_mode || "ref2va"),
                    adjustment: normalizeColorAdjustment(adjustment),
                }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok || !result?.ok) {
                throw new Error(result?.error || `Color adjustment failed (${response.status}).`);
            }
            clip.color_adjustment = normalizeColorAdjustment(result.adjustment);
            updateHidden(node, runtime);
            runtime.statusText = result.modified
                ? `Clip ${clipIndex + 1} color correction saved`
                : `Clip ${clipIndex + 1} color correction reset`;
            render(node, runtime);
            window.dispatchEvent(new CustomEvent("h3-extender-color-updated", {
                detail: {
                    owner_id: String(node.id),
                    color_timeline: Array.isArray(result.timeline) ? result.timeline : [],
                },
            }));
            node.graph?.setDirtyCanvas(true, true);
            closeColorEditor(overlay);
        } catch (error) {
            alert(`Color adjustment failed:\n${error?.message || error}`);
            apply.disabled = false;
            apply.textContent = "Apply";
        }
    });

    buttons.append(reset, cancel, apply);
    dialog.append(header, video, loopInfo, controls, buttons);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener("mousedown", (event) => {
        if (event.target === overlay) closeColorEditor(overlay);
    });
    const keyHandler = (event) => {
        if (event.key === "Escape" && overlay.isConnected) {
            closeColorEditor(overlay);
            document.removeEventListener("keydown", keyHandler);
        }
    };
    document.addEventListener("keydown", keyHandler);

    video.addEventListener("loadedmetadata", () => {
        video.currentTime = loopStart;
        updateLiveFilter();
        video.play().catch(() => {});
    });
    video.addEventListener("timeupdate", () => {
        if (video.currentTime >= loopEnd - 0.015 || video.currentTime < loopStart - 0.05) {
            video.currentTime = loopStart;
        }
        updateLiveFilter();
    });
    video.addEventListener("seeked", updateLiveFilter);
    if (typeof video.requestVideoFrameCallback === "function") {
        const colorFrameTick = () => {
            if (!overlay.isConnected) return;
            if (video.currentTime >= loopEnd - 0.015 || video.currentTime < loopStart - 0.05) {
                video.currentTime = loopStart;
            }
            updateLiveFilter();
            video.requestVideoFrameCallback(colorFrameTick);
        };
        video.requestVideoFrameCallback(colorFrameTick);
    }
    video.src = colorMediaUrl(payload.video) + "&t=" + Date.now();
    video.load();
}

function collectWidgetValues(node, names) {
    const out = {};
    for (const name of names) {
        const widget = getWidget(node, name);
        if (widget) out[name] = widget.value;
    }
    return out;
}

function collectConnectionSummary(node) {
    const out = {};
    for (const input of node?.inputs || []) {
        out[String(input?.name || "")] = input?.link != null;
    }
    return out;
}

function collectProjectPayload(node, runtime) {
    updateHidden(node, runtime);
    updateRefsHidden(node, runtime);
    const finalNode = connectedFinalDecode(node);
    const settings = collectWidgetValues(node, PROJECT_WIDGETS);
    // A portable .ext archive represents the currently selected generation
    // mode only. Workflow serialization keeps both independent card timelines,
    // but the project archive must not retain dangling inactive FL2VA frame ids.
    settings.clips_json = serializeProjectState(runtime.state);
    // In Auto mode the visible width/height widgets are mirrors of the active
    // derived resolution. Preserve the user's Manual fallback separately so a
    // later Auto -> Manual switch restores what they actually entered.
    settings.width = Number(runtime.manualWidth || settings.width || 896);
    settings.height = Number(runtime.manualHeight || settings.height || 576);
    return {
        schema_version: 2,
        extender: {
            class_name: TARGET,
            generation_mode: String(runtime.state?.generation_mode || getWidget(node, "generation_mode")?.value || "ref2va"),
            node_title: String(node?.title || "MiniMax H3 Extender"),
            settings,
            resolution: {
                mode: String(getWidget(node, "resolution_mode")?.value || "manual"),
                megapixels: Number(getWidget(node, "megapixels")?.value ?? 0.40),
                manual_width: Number(runtime.manualWidth || settings.width || 0),
                manual_height: Number(runtime.manualHeight || settings.height || 0),
                resolved_width: Number(runtime.resolvedWidth || runtime.expectedResolution?.width || 0),
                resolved_height: Number(runtime.resolvedHeight || runtime.expectedResolution?.height || 0),
                guide_ref: String(runtime.resolutionGuide || ""),
                fallback: Boolean(runtime.resolutionFallback),
            },
            clips_json: serializeProjectState(runtime.state),
            clips: runtime.state.clips.map((clip) => ({ ...clip })),
            refs_json: serializeRefsState(runtime.refsState),
            references: runtime.refsState.refs.map((ref) => ref ? { ...ref } : null),
            connections: collectConnectionSummary(node),
        },
        final_decode: finalNode ? {
            class_name: FINAL_TARGET,
            node_id: String(finalNode.id),
            settings: collectWidgetValues(finalNode, FINAL_PROJECT_WIDGETS),
            preview: (() => {
                const previewState = finalNode.__h3LivePreview;
                const previewMeta = previewState?.currentPreviewMeta || {};
                const info = previewState?.currentVideoInfo;
                return {
                    available: Boolean(info?.filename),
                    clip_count: Number(previewMeta.clip_count || 0),
                    frame_count: Number(previewMeta.frame_count || 0),
                };
            })(),
        } : null,
    };
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget || value === undefined) return false;
    widget.value = value;
    return true;
}

function applyProjectPayload(node, runtime, projectPayload) {
    const extender = projectPayload?.extender || {};
    const settings = extender?.settings || {};
    if (typeof extender?.node_title === "string" && extender.node_title.trim()) {
        node.title = extender.node_title;
    }
    for (const name of PROJECT_WIDGETS) {
        if (name === "clips_json" || name === "refs_json") continue;
        if (Object.prototype.hasOwnProperty.call(settings, name)) {
            setWidgetValue(node, name, settings[name]);
        }
    }

    const projectMode = String(extender?.generation_mode || settings?.generation_mode || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
    setWidgetValue(node, "generation_mode", projectMode);

    const savedResolution = extender?.resolution;
    const hasSavedMode =
        Object.prototype.hasOwnProperty.call(settings, "resolution_mode")
        || (savedResolution && Object.prototype.hasOwnProperty.call(savedResolution, "mode"));
    // v14.24 and older .ext projects did not know about automatic resolution.
    // Preserve their exact historical Manual behavior. Newer projects restore
    // the mode that was actually saved instead of forcing every imported cache
    // into Manual.
    const savedMode = hasSavedMode && String(savedResolution?.mode || settings?.resolution_mode || "") === "auto_from_ref"
        ? "auto_from_ref"
        : "manual";
    setWidgetValue(node, "resolution_mode", savedMode);

    const savedManualW = Number(savedResolution?.manual_width || settings?.width || 0);
    const savedManualH = Number(savedResolution?.manual_height || settings?.height || 0);
    rememberManualResolution(node, runtime, savedManualW, savedManualH);
    runtime.resolutionGuide = String(savedResolution?.guide_ref || "");
    runtime.resolutionFallback = Boolean(savedResolution?.fallback);

    const savedW = Number(savedResolution?.resolved_width || 0);
    const savedH = Number(savedResolution?.resolved_height || 0);
    if (savedW > 0 && savedH > 0) {
        runtime.expectedResolution = { width: savedW, height: savedH };
        runtime.resolvedWidth = savedW;
        runtime.resolvedHeight = savedH;
        // The imported cache geometry is authoritative, but its UI mode is not
        // changed. Auto projects therefore reopen as Auto while keeping the
        // exact archived resolved size until the next genuine resolution
        // change. The separate Manual fallback above is left untouched.
        setResolutionMirrorValues(node, runtime, savedW, savedH);
        runtime.resolutionMirrorActive = savedMode === "auto_from_ref" && !runtime.resolutionFallback;
        runtime.projectResolutionLoaded = false;
    }

    const rawRefs =
        extender?.refs_json
        || settings?.refs_json
        || JSON.stringify({ version: 2, refs: extender?.references || [] });
    runtime.refsState = parseRefsState(rawRefs);
    updateRefsHidden(node, runtime);

    const rawClips = String(
        extender?.clips_json
        || settings?.clips_json
        || JSON.stringify({ version: 1, clips: extender?.clips || [] })
    );
    runtime.state = parseState(rawClips);
    activateModeState(runtime.state, projectMode);
    // Loading a project mutates the disk cache outside ComfyUI's executor. A
    // one-shot token forces the Extender input hash to change even if every
    // visible setting happens to match the workflow that was previously run.
    runtime.state.load_token = `${Date.now().toString(36)}_${randomSeed().toString(36)}`;
    updateHidden(node, runtime);
    captureNativeWorkflowState(node, runtime);

    const finalSettings = projectPayload?.final_decode?.settings;
    const finalNode = connectedFinalDecode(node);
    if (finalNode && finalSettings && typeof finalSettings === "object") {
        for (const name of FINAL_PROJECT_WIDGETS) {
            if (Object.prototype.hasOwnProperty.call(finalSettings, name)) {
                setWidgetValue(finalNode, name, finalSettings[name]);
            }
        }
        finalNode.graph?.setDirtyCanvas(true, true);
    }

    node.graph?.setDirtyCanvas(true, true);
}

function projectBusy(runtime) {
    return ["preparing", "sampling", "complete"].includes(String(runtime?.activePhase || ""));
}

function setProjectButtonsBusy(runtime, busy) {
    if (!runtime) return;
    runtime.projectOperationBusy = Boolean(busy);
    if (runtime.saveProjectButton) runtime.saveProjectButton.disabled = Boolean(busy);
    if (runtime.loadProjectButton) runtime.loadProjectButton.disabled = Boolean(busy);
}

async function saveProject(node, runtime) {
    if (!node || !runtime) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before saving the project.");
        return;
    }
    if (runtime.resolutionInvalidated) {
        alert(
            "The resolution has changed and the previous cache is no longer compatible. " +
            "Queue the Extender once to start the new-resolution cache before saving the project."
        );
        return;
    }

    const suggested = runtime.projectName || "MiniMax_H3_Project";
    const requested = prompt("Project name (.ext)", suggested);
    if (requested == null) return;
    const projectName = String(requested || suggested).trim() || suggested;

    setProjectButtonsBusy(runtime, true);
    runtime.statusText = "Saving project…";
    render(node, runtime);
    try {
        const response = await fetch(api.apiURL("/h3_extender/project/prepare_save"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                owner_id: String(node.id),
                project_name: projectName,
                project: collectProjectPayload(node, runtime),
            }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok || !payload?.token) {
            throw new Error(payload?.error || `Save Project failed (${response.status}).`);
        }

        runtime.projectName = String(payload.filename || projectName).replace(/\.ext$/i, "");
        node.properties = node.properties || {};
        node.properties.h3_project_name = runtime.projectName;
        runtime.statusText = `Project ready: ${payload.filename || projectName} | refs ${Number(payload?.references?.count ?? refCount(runtime))} embedded`;
        render(node, runtime);

        // Do not fetch the archive into a JS Blob: .ext files may be many GB.
        // A normal browser download streams it directly from the backend.
        const a = document.createElement("a");
        a.href = api.apiURL(
            "/h3_extender/project/download?token=" + encodeURIComponent(String(payload.token))
        );
        a.download = String(payload.filename || `${runtime.projectName}.ext`);
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        setTimeout(() => a.remove(), 0);
    } catch (error) {
        runtime.statusText = "Save Project failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        setProjectButtonsBusy(runtime, false);
        render(node, runtime);
    }
}

async function loadProjectFile(node, runtime, file) {
    if (!node || !runtime || !file) return;
    if (projectBusy(runtime)) {
        alert("Wait for the current clip generation to finish before loading a project.");
        return;
    }
    if (!confirm(
        "Load this .ext project?\n\nThe current Extender cache, image references and project settings will be replaced."
    )) return;

    setProjectButtonsBusy(runtime, true);
    runtime.statusText = `Loading ${file.name}…`;
    render(node, runtime);
    try {
        const form = new FormData();
        form.append("owner_id", String(node.id));
        form.append("project_file", file, file.name);
        const response = await fetch(api.apiURL("/h3_extender/project/load"), {
            method: "POST",
            body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Load Project failed (${response.status}).`);
        }

        applyProjectPayload(node, runtime, payload.project || {});
        runtime.cachedCount = Number(payload?.cache?.cached_count || 0);
        runtime.validatedCount = Number(payload?.cache?.validated_count || 0);
        runtime.cachedClipIds = new Set(Array.isArray(payload?.cache?.cached_clip_ids) ? payload.cache.cached_clip_ids.map(String) : []);
        runtime.validatedClipIds = new Set(Array.isArray(payload?.cache?.validated_clip_ids) ? payload.cache.validated_clip_ids.map(String) : []);
        runtime.continuitySignatures = new Map(
            Object.entries(payload?.cache?.continuity_signatures || {}).map(([key, value]) => [String(key), String(value || "")]).filter(([, value]) => Boolean(value))
        );
        const loadedW = Number(payload?.cache?.resolved_width || runtime.expectedResolution?.width || 0);
        const loadedH = Number(payload?.cache?.resolved_height || runtime.expectedResolution?.height || 0);
        if (loadedW > 0 && loadedH > 0) {
            runtime.expectedResolution = { width: loadedW, height: loadedH };
            runtime.resolvedWidth = loadedW;
            runtime.resolvedHeight = loadedH;
            // applyProjectPayload already restored the saved Auto/Manual mode
            // and the independent Manual fallback. Only mirror the exact cache
            // geometry returned by the backend here; never force Auto projects
            // back to Manual.
            setResolutionMirrorValues(node, runtime, loadedW, loadedH);
            const loadedMode = String(getWidget(node, "resolution_mode")?.value || "manual");
            runtime.resolutionMirrorActive = loadedMode === "auto_from_ref" && !runtime.resolutionFallback;
            runtime.projectResolutionLoaded = false;
            runtime.resolutionInvalidated = false;
        }
        runtime.cacheStateRestored = true;
        runtime.projectName = String(payload.project_name || file.name).replace(/\.ext$/i, "");
        node.properties = node.properties || {};
        node.properties.h3_project_name = runtime.projectName;
        const resolutionText = loadedW > 0 && loadedH > 0 ? ` | ${loadedW}x${loadedH}` : "";
        runtime.statusText =
            `Loaded ${runtime.projectName}${resolutionText} | refs ${refCount(runtime)} | cached ${runtime.cachedCount}/${runtime.state.clips.length} | ` +
            `validated ${runtime.validatedCount}`;
        render(node, runtime);
        syncDomHeight(node, runtime, false);

        // Final Decode / Preview can rebuild the full preview from decoded blobs
        // already inside the imported cache, with no sampler or VAE execution.
        // Pass the imported mode explicitly. During Nodes 2.0 restore the hidden
        // native combo can transiently expose its Ref2VA schema default; Final
        // Decode must restore the cache that belongs to the project just loaded.
        window.dispatchEvent(new CustomEvent("h3-extender-project-loaded", {
            detail: {
                owner_id: String(node.id),
                generation_mode: runtime.state?.generation_mode === "fl2va" ? "fl2va" : "ref2va",
            },
        }));
    } catch (error) {
        runtime.statusText = "Load Project failed";
        render(node, runtime);
        alert(String(error?.message || error));
    } finally {
        setProjectButtonsBusy(runtime, false);
        render(node, runtime);
    }
}

function makeFieldLabel(text) {
    const label = document.createElement("div");
    label.textContent = text;
    label.style.fontSize = "11px";
    label.style.opacity = "0.72";
    label.style.margin = "5px 0 3px";
    return label;
}

function makeNumberInput(value, min, max, step) {
    const input = document.createElement("input");
    input.type = "number";
    input.value = String(value);
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.background = "rgba(0,0,0,.25)";
    input.style.border = "1px solid rgba(255,255,255,.15)";
    input.style.color = "inherit";
    input.style.borderRadius = "5px";
    input.style.padding = "5px 7px";
    return input;
}

function renderReferences(node, runtime) {
    const row = runtime?.refsRow;
    if (!row) return;
    row.replaceChildren();

    const refs = runtime.refsState?.refs || [];

    for (let index = 0; index < MAX_IMAGE_REFS; index++) {
        const ref = refs[index] || null;
        const slot = document.createElement("div");
        // Fill the whole available node width with nine equal reference slots.
        // REF_SLOT_WIDTH is a hard minimum for each slot, not for the node.
        // The node itself may shrink well below the combined strip width; once
        // that happens this row owns the horizontal overflow and exposes its scrollbar.
        slot.style.flex = "1 1 0px";
        slot.style.minWidth = `${REF_SLOT_WIDTH}px`;
        slot.style.boxSizing = "border-box";
        slot.style.position = "relative";

        const load = document.createElement("button");
        load.textContent = ref ? `Replace Ref ${index + 1}` : `Load Ref ${index + 1}`;
        load.title = ref
            ? `Replace Ref ${index + 1}: ${ref.original_name || "reference"}`
            : `Load image reference ${index + 1}`;
        load.style.width = "100%";
        load.style.height = "23px";
        load.style.padding = "2px 4px";
        load.style.fontSize = "10px";
        load.disabled = Boolean(
            runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime)
        );
        load.addEventListener("click", (event) => {
            event.preventDefault();
            if (load.disabled) return;
            runtime.pendingRefSlot = index;
            runtime.refFileInput?.click();
        });
        slot.appendChild(load);

        const thumb = document.createElement("div");
        thumb.style.marginTop = "2px";
        thumb.style.width = "100%";
        thumb.style.height = `${REF_THUMB_HEIGHT}px`;
        thumb.style.boxSizing = "border-box";
        thumb.style.border = "1px solid rgba(255,255,255,.15)";
        thumb.style.borderRadius = "6px";
        thumb.style.background = "rgba(0,0,0,.24)";
        thumb.style.display = "flex";
        thumb.style.alignItems = "center";
        thumb.style.justifyContent = "center";
        thumb.style.position = "relative";
        thumb.style.overflow = "hidden";

        if (ref) {
            const img = document.createElement("img");
            img.src = refImageUrl(ref);
            img.alt = ref.original_name || `Ref ${index + 1}`;
            img.title = `${ref.original_name || `Ref ${index + 1}`} — double-click to edit`;
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
            img.style.cursor = "pointer";
            img.draggable = false;
            img.addEventListener("dblclick", (event) => {
                event.preventDefault();
                event.stopPropagation();
                openReferenceEditor(node, runtime, index, ref);
            });
            thumb.appendChild(img);

            const remove = document.createElement("button");
            remove.textContent = "×";
            remove.title = `Remove Ref ${index + 1}`;
            remove.style.position = "absolute";
            remove.style.top = "3px";
            remove.style.right = "3px";
            remove.style.width = "20px";
            remove.style.height = "20px";
            remove.style.minWidth = "20px";
            remove.style.padding = "0";
            remove.style.lineHeight = "16px";
            remove.style.borderRadius = "10px";
            remove.style.background = "rgba(0,0,0,.68)";
            remove.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
            remove.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                removeReference(node, runtime, index);
            });
            thumb.appendChild(remove);
        } else {
            const empty = document.createElement("span");
            empty.textContent = "+";
            empty.style.fontSize = "24px";
            empty.style.opacity = ".55";
            thumb.appendChild(empty);
        }
        slot.appendChild(thumb);

        const meta = document.createElement("div");
        meta.style.marginTop = "1px";
        meta.style.fontSize = "9px";
        meta.style.lineHeight = "9px";
        meta.style.opacity = ".6";
        meta.style.textAlign = "center";
        meta.style.whiteSpace = "nowrap";
        meta.style.overflow = "hidden";
        meta.style.textOverflow = "ellipsis";
        meta.textContent = ref && ref.width > 0 && ref.height > 0
            ? `${Math.trunc(ref.width)}×${Math.trunc(ref.height)}`
            : "empty";
        meta.title = ref?.original_name || "";
        slot.appendChild(meta);

        row.appendChild(slot);
    }
}



function fl2vaPreviousFrameUrl(node, runtime, previousClipId) {
    const clipId = String(previousClipId || "");
    const params = new URLSearchParams();
    params.set("owner_id", String(node?.id ?? ""));
    params.set("clip_id", clipId);
    // A known content signature gives the PNG a stable immutable URL. While the
    // Final Decode has not produced metadata yet, use one stable revalidated URL
    // instead of Date.now() cache-busting every UI render.
    const signature = String(runtime?.continuitySignatures?.get(clipId) || "current");
    params.set("v", signature);
    return api.apiURL("/h3_extender/fl2va/last_frame?" + params.toString());
}

async function refreshFl2vaContinuitySignature(node, runtime, clipId) {
    clipId = String(clipId || "");
    if (!clipId || !runtime || runtime.continuitySignatures?.has(clipId)) return;
    if (runtime.continuitySignatureRequests?.has(clipId)) return;
    runtime.continuitySignatureRequests?.add(clipId);
    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(node?.id ?? ""));
        params.set("clip_id", clipId);
        const response = await fetch(
            api.apiURL("/h3_extender/fl2va/continuity_meta?" + params.toString()),
            { cache: "no-store" },
        );
        if (!response.ok) return;
        const payload = await response.json().catch(() => ({}));
        const signature = String(payload?.signature || "");
        if (payload?.found && signature) {
            runtime.continuitySignatures.set(clipId, signature);
            render(node, runtime);
        }
    } catch (_) {
        // The sidecar may legitimately not exist until Final Decode has run.
    } finally {
        runtime.continuitySignatureRequests?.delete(clipId);
    }
}

function invalidateFl2vaPlanAndFollowers(runtime, startIndex, includeStart = true) {
    const clips = runtime?.state?.clips || [];
    let i = Math.max(0, Number(startIndex) || 0);
    if (!includeStart) i += 1;
    let touched = 0;
    for (; i < clips.length; i++) {
        if (i > startIndex && String(clips[i]?.first_source || "manual") !== "previous_clip") break;
        const clip = clips[i];
        clip.validated = false;
        runtime.validatedClipIds?.delete(String(clip.id));
        runtime.cachedClipIds?.delete(String(clip.id));
        runtime.continuitySignatures?.delete(String(clip.id));
        touched++;
    }
    if (touched) {
        runtime.validatedCount = runtime.validatedClipIds?.size || 0;
        runtime.cachedCount = runtime.cachedClipIds?.size || 0;
    }
    return touched;
}

function healFirstPlanPreviousSource(runtime) {
    const first = runtime?.state?.clips?.[0];
    if (!first || String(first.first_source || "manual") !== "previous_clip") return false;
    first.first_source = "manual";
    first.validated = false;
    runtime.validatedClipIds?.delete(String(first.id));
    runtime.cachedClipIds?.delete(String(first.id));
    return true;
}

function renderFl2vaFrames(node, runtime) {
    const row = runtime?.refsRow;
    if (!row) return;
    row.replaceChildren();

    // Keep the FL2VA keyframes horizontally aligned with the clip cards below.
    // One fixed-width group represents one card, and the First/Last thumbnails
    // are centered inside that group. The cards use the same CARD_WIDTH and 9px
    // inter-card gap, so scrolling the media strip reads naturally as a plan row.
    for (let clipIndex = 0; clipIndex < (runtime.state?.clips || []).length; clipIndex++) {
        const clip = runtime.state.clips[clipIndex];
        const group = document.createElement("div");
        group.style.flex = `0 0 ${CARD_WIDTH}px`;
        group.style.width = `${CARD_WIDTH}px`;
        group.style.minWidth = `${CARD_WIDTH}px`;
        group.style.boxSizing = "border-box";
        group.style.display = "flex";
        group.style.justifyContent = "center";
        group.style.alignItems = "flex-start";
        group.style.gap = "7px";

        for (const kind of ["first", "last"]) {
            const key = `${kind}_frame`;
            const label = kind === "first" ? "First" : "Last";
            const ref = normalizeRefDescriptor(clip?.[key]);
            clip[key] = ref;

            const slot = document.createElement("div");
            slot.style.flex = `0 0 ${FL2VA_FRAME_SLOT_WIDTH}px`;
            slot.style.width = `${FL2VA_FRAME_SLOT_WIDTH}px`;
            slot.style.minWidth = `${FL2VA_FRAME_SLOT_WIDTH}px`;
            slot.style.boxSizing = "border-box";
            slot.style.position = "relative";

            const usingPrevious = kind === "first" && clipIndex > 0 && String(clip.first_source || "manual") === "previous_clip";
            const controls = document.createElement("div");
            controls.style.display = "flex";
            controls.style.gap = "3px";
            controls.style.width = "100%";

            const load = document.createElement("button");
            load.textContent = usingPrevious
                ? "Manual"
                : (ref ? `Replace ${label}` : `Load ${label}`);
            load.title = usingPrevious
                ? `Load a manual Clip ${clipIndex + 1} First frame and leave Previous mode`
                : (ref
                    ? `Replace Clip ${clipIndex + 1} ${label} frame: ${ref.original_name || "keyframe"}`
                    : `Load Clip ${clipIndex + 1} ${label} frame`);
            load.style.flex = "1 1 0";
            load.style.minWidth = "0";
            load.style.height = "23px";
            load.style.padding = "2px 4px";
            load.style.fontSize = "10px";
            load.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
            load.addEventListener("click", (event) => {
                event.preventDefault();
                if (load.disabled) return;
                runtime.pendingFrameClip = clipIndex;
                runtime.pendingFrameKind = kind;
                runtime.frameFileInput?.click();
            });
            controls.appendChild(load);

            if (kind === "first") {
                const previous = document.createElement("button");
                previous.type = "button";
                previous.textContent = "⛓ Prev";
                previous.title = clipIndex === 0
                    ? "The first FL2VA plan has no previous generated frame"
                    : (usingPrevious
                        ? `Use the manual First frame instead of Clip ${clipIndex} selected pre-end continuity frame`
                        : `Use Clip ${clipIndex} selected pre-end continuity frame as this plan's First frame`);
                previous.style.flex = "0 0 49px";
                previous.style.width = "49px";
                previous.style.height = "23px";
                previous.style.padding = "2px";
                previous.style.fontSize = "9px";
                previous.style.fontWeight = usingPrevious ? "700" : "400";
                previous.style.background = usingPrevious ? "rgba(70,150,230,.42)" : "";
                previous.disabled = clipIndex === 0 || Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
                previous.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (previous.disabled) return;
                    clip.first_source = usingPrevious ? "manual" : "previous_clip";
                    invalidateFl2vaPlanAndFollowers(runtime, clipIndex, true);
                    runtime.statusText = clip.first_source === "previous_clip"
                        ? `Clip ${clipIndex + 1} First → continuity frame of Clip ${clipIndex}`
                        : `Clip ${clipIndex + 1} First → manual`;
                    updateHidden(node, runtime);
                    render(node, runtime);
                });
                controls.appendChild(previous);
            }
            slot.appendChild(controls);

            const thumb = document.createElement("div");
            thumb.style.marginTop = "2px";
            thumb.style.width = "100%";
            thumb.style.height = `${REF_THUMB_HEIGHT}px`;
            thumb.style.boxSizing = "border-box";
            thumb.style.border = "1px solid rgba(255,255,255,.15)";
            thumb.style.borderRadius = "6px";
            thumb.style.background = "rgba(0,0,0,.24)";
            thumb.style.display = "flex";
            thumb.style.alignItems = "center";
            thumb.style.justifyContent = "center";
            thumb.style.position = "relative";
            thumb.style.overflow = "hidden";

            if (usingPrevious) {
                const fallback = document.createElement("div");
                fallback.textContent = `⛓ C${clipIndex} end`;
                fallback.style.fontSize = "11px";
                fallback.style.fontWeight = "700";
                fallback.style.opacity = ".72";
                fallback.style.textAlign = "center";
                fallback.style.padding = "4px";
                thumb.appendChild(fallback);

                const img = document.createElement("img");
                const previousClipId = String(runtime.state.clips[clipIndex - 1]?.id || "");
                void refreshFl2vaContinuitySignature(node, runtime, previousClipId);
                img.src = fl2vaPreviousFrameUrl(node, runtime, previousClipId);
                img.alt = `Selected pre-end continuity frame of Clip ${clipIndex}`;
                img.title = `Selected pre-end continuity frame of Clip ${clipIndex} — used automatically as this First frame`;
                img.style.position = "absolute";
                img.style.inset = "0";
                img.style.width = "100%";
                img.style.height = "100%";
                img.style.objectFit = "contain";
                img.draggable = false;
                img.addEventListener("load", () => { fallback.style.display = "none"; });
                img.addEventListener("error", () => { img.remove(); fallback.style.display = "block"; });
                thumb.appendChild(img);
            } else if (ref) {
                const img = document.createElement("img");
                img.src = refImageUrl(ref);
                img.alt = ref.original_name || `Clip ${clipIndex + 1} ${label} frame`;
                img.title = `${ref.original_name || `Clip ${clipIndex + 1} ${label} frame`} — double-click to edit`;
                img.style.width = "100%";
                img.style.height = "100%";
                img.style.objectFit = "contain";
                img.style.cursor = "pointer";
                img.draggable = false;
                img.addEventListener("dblclick", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    openReferenceEditor(node, runtime, -1, ref, { clipIndex, kind });
                });
                thumb.appendChild(img);

                const remove = document.createElement("button");
                remove.textContent = "×";
                remove.title = `Remove Clip ${clipIndex + 1} ${label} frame`;
                remove.style.position = "absolute";
                remove.style.top = "3px";
                remove.style.right = "3px";
                remove.style.width = "20px";
                remove.style.height = "20px";
                remove.style.minWidth = "20px";
                remove.style.padding = "0";
                remove.style.lineHeight = "16px";
                remove.style.borderRadius = "10px";
                remove.style.background = "rgba(0,0,0,.68)";
                remove.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
                remove.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (remove.disabled) return;
                    removeClipFrame(node, runtime, clipIndex, kind);
                });
                thumb.appendChild(remove);
            } else {
                const empty = document.createElement("div");
                empty.textContent = "+";
                empty.style.fontSize = "24px";
                empty.style.opacity = ".55";
                thumb.appendChild(empty);
            }
            slot.appendChild(thumb);

            const meta = document.createElement("div");
            meta.style.marginTop = "1px";
            meta.style.fontSize = "9px";
            meta.style.lineHeight = "9px";
            meta.style.opacity = ".6";
            meta.style.textAlign = "center";
            meta.style.whiteSpace = "nowrap";
            meta.style.overflow = "hidden";
            meta.style.textOverflow = "ellipsis";
            meta.textContent = usingPrevious
                ? `← C${clipIndex} continuity`
                : (ref && ref.width > 0 && ref.height > 0
                    ? `${Math.trunc(ref.width)}×${Math.trunc(ref.height)}`
                    : "empty");
            meta.title = usingPrevious
                ? `Selected pre-end continuity frame of Clip ${clipIndex}`
                : (ref?.original_name || "");
            slot.appendChild(meta);

            group.appendChild(slot);
        }
        row.appendChild(group);
    }
}


function renderMediaStrip(node, runtime, fl2vaMode) {
    if (runtime?.refsHeader) {
        runtime.refsHeader.textContent = fl2vaMode
            ? "FL2VA FIRST / LAST FRAMES — per-clip IMAGE GUIDES are inside each card"
            : "REFERENCE IMAGES — double-click a thumbnail to edit";
    }
    if (runtime?.refsRow) runtime.refsRow.style.gap = fl2vaMode ? "9px" : "7px";
    if (fl2vaMode) renderFl2vaFrames(node, runtime);
    else renderReferences(node, runtime);
}

function render(node, runtime) {
    const { state, cards, counter, status } = runtime;
    cards.replaceChildren();

    const fl2vaMode = state.generation_mode === "fl2va";
    syncModeSpecificNativeWidgets(node, runtime, fl2vaMode);
    renderMediaStrip(node, runtime, fl2vaMode);
    if (runtime.modeButton) {
        runtime.modeButton.textContent = fl2vaMode ? "MODE: FL2VA" : "MODE: REF2VA";
    }
    if (runtime.generationModeWidget) runtime.generationModeWidget.value = fl2vaMode ? "fl2va" : "ref2va";
    if (runtime.refsSection) runtime.refsSection.style.display = "block";
    counter.textContent = fl2vaMode
        ? `${state.clips.length} plan${state.clips.length > 1 ? "s" : ""} • FL2VA`
        : `${state.clips.length} clip${state.clips.length > 1 ? "s" : ""} • ${refCount(runtime)} ref${refCount(runtime) === 1 ? "" : "s"}`;
    status.textContent = runtime.statusText || "Ready";

    state.clips.forEach((clip, index) => {
        const card = document.createElement("div");
        card.className = "h3-extender-card";
        card.dataset.clipIndex = String(index);
        card.style.flex = `0 0 ${CARD_WIDTH}px`;
        card.style.width = `${CARD_WIDTH}px`;
        card.style.boxSizing = "border-box";
        card.style.padding = "9px";
        card.style.borderRadius = "8px";
        card.style.background = "rgba(20,20,20,.72)";
        card.style.border = "1px solid rgba(255,255,255,.13)";
        card.style.display = "flex";
        card.style.flexDirection = "column";
        card.style.minHeight = `${cardMinHeightForState(state)}px`;

        const st = cardStatus(runtime, clip, index);
        if (st === "rendering") {
            card.style.border = "3px solid rgba(70,210,255,1)";
            card.style.boxShadow = "0 0 0 1px rgba(70,210,255,.25), 0 0 16px rgba(70,210,255,.38)";
            card.style.background = "rgba(24,40,46,.88)";
        } else if (st === "validated") {
            card.style.borderColor = "rgba(80,210,120,.8)";
        } else if (st === "candidate" || st === "current") {
            card.style.borderColor = "rgba(255,180,60,.9)";
        } else if (st === "cached") {
            card.style.borderColor = "rgba(90,155,230,.65)";
        }

        const head = document.createElement("div");
        head.style.display = "flex";
        head.style.alignItems = "center";
        head.style.gap = "7px";
        head.style.marginBottom = "5px";

        const title = document.createElement("strong");
        title.textContent = `CLIP ${index + 1}`;
        title.style.flex = "0 0 auto";
        title.style.whiteSpace = "nowrap";

        const name = document.createElement("input");
        name.type = "text";
        name.value = clip.name || "";
        name.placeholder = "name";
        name.title = "Optional clip/card name";
        name.style.flex = "1 1 0";
        name.style.minWidth = "0";
        name.style.height = "22px";
        name.style.boxSizing = "border-box";
        name.style.background = "rgba(0,0,0,.22)";
        name.style.border = "1px solid rgba(255,255,255,.12)";
        name.style.color = "inherit";
        name.style.borderRadius = "4px";
        name.style.padding = "2px 5px";
        name.style.fontSize = "11px";
        name.addEventListener("input", () => {
            if (name.value === clip.name) return;
            clip.name = name.value;
            updateHidden(node, runtime);
            // Keep focus while typing; no DOM rebuild here.
        });

        const colorWrap = document.createElement("div");
        colorWrap.style.display = "flex";
        colorWrap.style.alignItems = "center";
        colorWrap.style.gap = "2px";
        colorWrap.style.flex = "0 0 auto";

        const colorButton = document.createElement("button");
        colorButton.type = "button";
        colorButton.textContent = "🎨";
        const colorBusy = ["preparing", "sampling", "complete"].includes(String(runtime.activePhase || ""));
        const colorCached = fl2vaMode
            ? runtime.cachedClipIds?.has(String(clip.id))
            : index < Number(runtime.cachedCount || 0);
        colorButton.title = colorBusy
            ? "Color editing is disabled while the Extender is rendering"
            : colorCached
                ? "Edit color for this decoded clip"
                : "Color editor becomes available after this clip has been decoded";
        colorButton.disabled = colorBusy || !colorCached;
        colorButton.style.width = "27px";
        colorButton.style.height = "22px";
        colorButton.style.padding = "0";
        colorButton.style.borderRadius = "4px";
        colorButton.style.cursor = colorButton.disabled ? "default" : "pointer";
        colorButton.style.opacity = colorButton.disabled ? ".35" : ".9";
        colorButton.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (!colorButton.disabled) openClipColorEditor(node, runtime, index);
        });

        const colorCheck = document.createElement("span");
        colorCheck.textContent = colorAdjustmentIsNeutral(clip.color_adjustment) ? "" : "✓";
        colorCheck.title = colorCheck.textContent ? "Color correction applied" : "";
        colorCheck.style.width = "10px";
        colorCheck.style.fontSize = "11px";
        colorCheck.style.fontWeight = "700";
        colorCheck.style.color = "rgba(115,225,145,.95)";
        colorCheck.style.textAlign = "center";

        colorWrap.append(colorButton, colorCheck);

        const badge = document.createElement("span");
        badge.style.fontSize = "10px";
        badge.style.opacity = ".8";
        badge.textContent =
            st === "rendering"
                ? (
                    runtime.activePhase === "preparing"
                        ? "◆ PREPARING"
                        : runtime.activePhase === "complete"
                            ? "✓ COMPLETE"
                            : "▶ RENDERING"
                ) :
            st === "validated" ? "VALIDATED" :
            st === "candidate" ? "● CANDIDATE" :
            st === "current" ? "● NEXT" :
            st === "cached" ? "CACHE" : "○";

        head.append(title, name, colorWrap, badge);
        if (fl2vaMode) {
            const insertButton = document.createElement("button");
            insertButton.type = "button";
            insertButton.textContent = "+";
            insertButton.title = "Insert a new independent FL2VA plan after this one";
            insertButton.style.width = "24px";
            insertButton.style.height = "22px";
            insertButton.style.padding = "0";
            insertButton.addEventListener("click", (e) => {
                e.preventDefault();
                state.clips.splice(index + 1, 0, newClip(index + 1));
                // A former follower moved one position to the right. If it was
                // linked to "Previous", its predecessor changed and its chain
                // must be regenerated.
                if (String(state.clips[index + 2]?.first_source || "manual") === "previous_clip") {
                    invalidateFl2vaPlanAndFollowers(runtime, index + 2, true);
                }
                updateHidden(node, runtime);
                render(node, runtime);
            });
            const deleteButton = document.createElement("button");
            deleteButton.type = "button";
            deleteButton.textContent = "×";
            deleteButton.title = "Remove this FL2VA plan";
            deleteButton.style.width = "24px";
            deleteButton.style.height = "22px";
            deleteButton.style.padding = "0";
            deleteButton.disabled = state.clips.length <= 1;
            deleteButton.addEventListener("click", (e) => {
                e.preventDefault();
                if (state.clips.length <= 1) return;
                state.clips.splice(index, 1);
                healFirstPlanPreviousSource(runtime);
                if (String(state.clips[index]?.first_source || "manual") === "previous_clip") {
                    invalidateFl2vaPlanAndFollowers(runtime, index, true);
                }
                updateHidden(node, runtime);
                render(node, runtime);
            });
            head.append(insertButton, deleteButton);
        }
        card.appendChild(head);

        // FL2VA First/Last frames live in the shared media strip above the
        // cards. AddGuide anchors are dynamic and stay compact in a horizontal
        // per-card strip because every guide owns its own exact frame index.
        if (fl2vaMode) {
            clip.guides = normalizeGuideList(clip);
            const guideFrameCount = h3FrameCountForDuration(clip.duration);
            for (const guide of clip.guides) {
                guide.frame_idx = Math.max(
                    -guideFrameCount,
                    Math.min(guideFrameCount - 1, normalizeGuideFrameIdx(guide.frame_idx ?? 0)),
                );
            }

            const guideWrap = document.createElement("div");
            guideWrap.style.display = "flex";
            guideWrap.style.flexDirection = "column";
            guideWrap.style.gap = "4px";
            guideWrap.style.marginBottom = "7px";
            guideWrap.style.padding = "5px";
            guideWrap.style.border = "1px solid rgba(255,255,255,.11)";
            guideWrap.style.borderRadius = "6px";
            guideWrap.style.background = "rgba(0,0,0,.16)";

            const guideHeader = document.createElement("div");
            guideHeader.style.display = "flex";
            guideHeader.style.alignItems = "center";
            guideHeader.style.justifyContent = "space-between";
            guideHeader.style.gap = "6px";

            const guideTitle = document.createElement("div");
            guideTitle.textContent = `IMAGE GUIDES${clip.guides.length ? ` • ${clip.guides.length}` : ""}`;
            guideTitle.style.fontSize = "10px";
            guideTitle.style.fontWeight = "700";
            guideTitle.style.opacity = ".78";

            const guideAdd = document.createElement("button");
            guideAdd.type = "button";
            guideAdd.textContent = "+ Add Guide";
            guideAdd.title = "Add another MiniMax H3 image guide";
            guideAdd.style.height = "21px";
            guideAdd.style.fontSize = "9px";
            guideAdd.style.padding = "1px 7px";
            guideAdd.disabled = Boolean(
                runtime.refBusy
                || runtime.projectOperationBusy
                || projectBusy(runtime)
                || clip.guides.length >= MAX_FL2VA_GUIDES
            );
            guideAdd.addEventListener("click", (event) => {
                event.preventDefault();
                if (guideAdd.disabled) return;
                runtime.pendingFrameClip = index;
                runtime.pendingFrameKind = "guide";
                runtime.pendingFrameGuideIndex = clip.guides.length;
                runtime.frameFileInput?.click();
            });
            guideHeader.append(guideTitle, guideAdd);
            guideWrap.appendChild(guideHeader);

            const guideRow = document.createElement("div");
            guideRow.style.display = "flex";
            guideRow.style.gap = "6px";
            guideRow.style.alignItems = "flex-start";
            guideRow.style.overflowX = "auto";
            guideRow.style.overflowY = "hidden";
            guideRow.style.paddingBottom = clip.guides.length > 3 ? "3px" : "0";
            guideRow.style.minHeight = "82px";

            if (!clip.guides.length) {
                const emptyGuide = document.createElement("button");
                emptyGuide.type = "button";
                emptyGuide.textContent = "+";
                emptyGuide.title = "Load the first image guide";
                emptyGuide.style.flex = "0 0 72px";
                emptyGuide.style.width = "72px";
                emptyGuide.style.height = "76px";
                emptyGuide.style.fontSize = "24px";
                emptyGuide.style.opacity = ".55";
                emptyGuide.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
                emptyGuide.addEventListener("click", (event) => {
                    event.preventDefault();
                    if (emptyGuide.disabled) return;
                    runtime.pendingFrameClip = index;
                    runtime.pendingFrameKind = "guide";
                    runtime.pendingFrameGuideIndex = 0;
                    runtime.frameFileInput?.click();
                });
                guideRow.appendChild(emptyGuide);
            }

            clip.guides.forEach((guide, guideIndex) => {
                const guideRef = normalizeRefDescriptor(guide.frame);
                if (!guideRef) return;
                guide.frame = guideRef;

                const slot = document.createElement("div");
                slot.style.flex = "0 0 82px";
                slot.style.width = "82px";
                slot.style.minWidth = "82px";

                const guideThumb = document.createElement("div");
                guideThumb.style.width = "82px";
                guideThumb.style.height = "54px";
                guideThumb.style.position = "relative";
                guideThumb.style.display = "flex";
                guideThumb.style.alignItems = "center";
                guideThumb.style.justifyContent = "center";
                guideThumb.style.overflow = "hidden";
                guideThumb.style.border = "1px solid rgba(255,255,255,.15)";
                guideThumb.style.borderRadius = "5px";
                guideThumb.style.background = "rgba(0,0,0,.25)";
                guideThumb.title = `Guide ${guideIndex + 1} — double-click to edit`;

                const img = document.createElement("img");
                img.src = refImageUrl(guideRef);
                img.alt = `Clip ${index + 1} Guide ${guideIndex + 1}`;
                img.style.width = "100%";
                img.style.height = "100%";
                img.style.objectFit = "contain";
                img.draggable = false;
                img.addEventListener("dblclick", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    openReferenceEditor(node, runtime, -1, guideRef, {
                        clipIndex: index,
                        kind: "guide",
                        guideIndex,
                    });
                });
                guideThumb.appendChild(img);

                const badge = document.createElement("div");
                badge.textContent = `G${guideIndex + 1}`;
                badge.style.position = "absolute";
                badge.style.left = "3px";
                badge.style.top = "3px";
                badge.style.padding = "1px 4px";
                badge.style.fontSize = "8px";
                badge.style.lineHeight = "12px";
                badge.style.borderRadius = "4px";
                badge.style.background = "rgba(0,0,0,.68)";
                badge.style.pointerEvents = "none";
                guideThumb.appendChild(badge);

                const clear = document.createElement("button");
                clear.type = "button";
                clear.textContent = "×";
                clear.title = `Remove Guide ${guideIndex + 1}`;
                clear.style.position = "absolute";
                clear.style.top = "3px";
                clear.style.right = "3px";
                clear.style.width = "19px";
                clear.style.height = "19px";
                clear.style.minWidth = "19px";
                clear.style.padding = "0";
                clear.style.borderRadius = "10px";
                clear.style.background = "rgba(0,0,0,.68)";
                clear.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
                clear.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (!clear.disabled) removeClipFrame(node, runtime, index, "guide", guideIndex);
                });
                guideThumb.appendChild(clear);
                slot.appendChild(guideThumb);

                const guideControls = document.createElement("div");
                guideControls.style.display = "grid";
                guideControls.style.gridTemplateColumns = "53px 25px";
                guideControls.style.gap = "4px";
                guideControls.style.marginTop = "3px";

                const guideIdx = document.createElement("input");
                guideIdx.type = "number";
                guideIdx.min = String(-guideFrameCount);
                guideIdx.max = String(guideFrameCount - 1);
                guideIdx.step = "1";
                guideIdx.value = String(guide.frame_idx);
                guideIdx.title = `Guide ${guideIndex + 1} frame index. Valid range: ${-guideFrameCount} to ${guideFrameCount - 1}. Negative values count from the end.`;
                guideIdx.style.width = "53px";
                guideIdx.style.height = "22px";
                guideIdx.style.boxSizing = "border-box";
                guideIdx.style.fontSize = "9px";
                guideIdx.style.padding = "1px 3px";
                guideIdx.addEventListener("change", () => {
                    const next = Math.max(
                        -guideFrameCount,
                        Math.min(guideFrameCount - 1, normalizeGuideFrameIdx(guideIdx.value)),
                    );
                    guideIdx.value = String(next);
                    if (next !== guide.frame_idx) {
                        guide.frame_idx = next;
                        updateHidden(node, runtime);
                        captureNativeWorkflowState(node, runtime);
                        runtime.statusText = `Clip ${index + 1} Guide ${guideIndex + 1} → frame ${next}`;
                        render(node, runtime);
                    }
                });

                const replace = document.createElement("button");
                replace.type = "button";
                replace.textContent = "↻";
                replace.title = `Replace Guide ${guideIndex + 1}`;
                replace.style.width = "25px";
                replace.style.height = "22px";
                replace.style.padding = "0";
                replace.style.fontSize = "12px";
                replace.disabled = Boolean(runtime.refBusy || runtime.projectOperationBusy || projectBusy(runtime));
                replace.addEventListener("click", (event) => {
                    event.preventDefault();
                    if (replace.disabled) return;
                    runtime.pendingFrameClip = index;
                    runtime.pendingFrameKind = "guide";
                    runtime.pendingFrameGuideIndex = guideIndex;
                    runtime.frameFileInput?.click();
                });

                guideControls.append(guideIdx, replace);
                slot.appendChild(guideControls);
                guideRow.appendChild(slot);
            });

            guideWrap.appendChild(guideRow);
            card.appendChild(guideWrap);
        }

        card.appendChild(makeFieldLabel("Prompt"));
        const prompt = document.createElement("textarea");
        prompt.value = clip.prompt;
        prompt.spellcheck = false;
        prompt.style.width = "100%";
        // The prompt is the flexible section of the card. Keep a real minimum
        // but allow it to absorb extra height without pushing controls on top
        // of one another.
        const promptMinHeight = 170;
        prompt.style.height = `${promptMinHeight}px`;
        prompt.style.minHeight = `${promptMinHeight}px`;
        prompt.style.flex = `1 1 ${promptMinHeight}px`;
        prompt.style.resize = "vertical";
        prompt.style.boxSizing = "border-box";
        prompt.style.background = "rgba(0,0,0,.27)";
        prompt.style.border = "1px solid rgba(255,255,255,.15)";
        prompt.style.color = "inherit";
        prompt.style.borderRadius = "5px";
        prompt.style.padding = "6px";
        prompt.addEventListener("input", () => {
            if (prompt.value === clip.prompt) return;
            clip.prompt = prompt.value;
            updateHidden(node, runtime);
            // Do not rebuild the DOM while typing: that would steal focus.
        });
        prompt.addEventListener("blur", () => render(node, runtime));
        card.appendChild(prompt);

        clip.loras = normalizeClipLoras(clip.loras, clip.lora);
        // Remove the old single-LoRA property after migration so serialized
        // state has one authoritative representation from v14.79 onward.
        if (Object.prototype.hasOwnProperty.call(clip, "lora")) delete clip.lora;

        const loraGroup = document.createElement("div");
        loraGroup.style.display = "flex";
        loraGroup.style.flexDirection = "column";
        loraGroup.style.gap = "5px";
        loraGroup.style.marginTop = "6px";

        // Autogrow pattern: every selected LoRA gets its own row, followed by
        // exactly one empty selector. Choosing that empty selector appends a new
        // LoRA row; choosing (no LoRA) on an existing row removes it and compacts
        // the stack automatically.
        const loraRows = [...clip.loras, normalizeClipLora()];
        for (let loraIndex = 0; loraIndex < loraRows.length; loraIndex++) {
            const cfg = loraRows[loraIndex];
            const isAddRow = loraIndex >= clip.loras.length;
            const selectedLora = String(cfg.name || "");

            const loraRow = document.createElement("div");
            loraRow.style.display = "grid";
            loraRow.style.gridTemplateColumns = "minmax(0, 1fr) 70px";
            loraRow.style.gap = "6px";
            loraRow.style.alignItems = "end";

            const loraBox = document.createElement("div");
            loraBox.appendChild(makeFieldLabel(isAddRow ? "Add LoRA" : `LoRA ${loraIndex + 1}`));
            const loraSelect = document.createElement("select");
            loraSelect.style.width = "100%";
            loraSelect.style.minWidth = "0";
            loraSelect.style.boxSizing = "border-box";
            loraSelect.style.background = "rgba(0,0,0,.25)";
            loraSelect.style.border = "1px solid rgba(255,255,255,.15)";
            loraSelect.style.color = "inherit";
            loraSelect.style.borderRadius = "5px";
            loraSelect.style.padding = "4px 5px";
            loraSelect.title = runtime.loraListError
                ? `Could not refresh ComfyUI LoRAs: ${runtime.loraListError}`
                : "Optional model-only LoRA stack applied only to this clip. Selecting a LoRA automatically creates another empty selector.";

            const noneOption = document.createElement("option");
            noneOption.value = "";
            noneOption.textContent = runtime.loraListLoading
                ? "Loading LoRAs…"
                : (isAddRow ? "(no additional LoRA)" : "(remove LoRA)");
            loraSelect.appendChild(noneOption);

            const usedByOtherRows = new Set(
                clip.loras
                    .filter((_, index) => index !== loraIndex)
                    .map((entry) => String(entry?.name || ""))
                    .filter(Boolean)
            );
            const loraNames = Array.isArray(runtime.loraNames) ? [...runtime.loraNames] : [];
            if (selectedLora && !loraNames.includes(selectedLora)) {
                loraNames.unshift(selectedLora);
            }
            for (const loraName of loraNames) {
                if (usedByOtherRows.has(loraName) && loraName !== selectedLora) continue;
                const option = document.createElement("option");
                option.value = loraName;
                option.textContent = loraName === selectedLora && !runtime.loraNames?.includes?.(loraName)
                    ? `${loraName} (missing)`
                    : loraName;
                loraSelect.appendChild(option);
            }
            loraSelect.value = selectedLora;
            loraSelect.addEventListener("change", () => {
                const name = String(loraSelect.value || "").trim();
                if (isAddRow) {
                    if (name) clip.loras.push(normalizeClipLora({ name, strength: 1.0 }));
                } else if (!name) {
                    clip.loras.splice(loraIndex, 1);
                } else {
                    clip.loras[loraIndex] = normalizeClipLora({
                        ...clip.loras[loraIndex],
                        name,
                    });
                }
                updateHidden(node, runtime);
                render(node, runtime);
            });
            loraBox.appendChild(loraSelect);

            const loraStrengthBox = document.createElement("div");
            loraStrengthBox.appendChild(makeFieldLabel("Strength"));
            const loraStrength = makeNumberInput(cfg.strength, -100, 100, 0.01);
            loraStrength.title = "Per-clip LoRA strength applied to the H3 diffusion model.";
            loraStrength.disabled = isAddRow || !selectedLora;
            loraStrength.style.opacity = selectedLora && !isAddRow ? "1" : ".45";
            loraStrength.addEventListener("change", () => {
                if (isAddRow || !clip.loras[loraIndex]) return;
                clip.loras[loraIndex].strength = Math.max(-100, Math.min(100, Number(loraStrength.value || 0)));
                updateHidden(node, runtime);
            });
            loraStrengthBox.appendChild(loraStrength);

            loraRow.append(loraBox, loraStrengthBox);
            loraGroup.appendChild(loraRow);
        }
        card.appendChild(loraGroup);

        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "1fr 92px";
        row.style.gap = "7px";
        row.style.alignItems = "end";

        const seedBox = document.createElement("div");
        seedBox.appendChild(makeFieldLabel("Seed"));
        const seedRow = document.createElement("div");
        seedRow.style.display = "flex";
        seedRow.style.gap = "5px";
        const seed = makeNumberInput(clip.seed, 0, Number.MAX_SAFE_INTEGER, 1);
        seed.style.minWidth = "0";
        seed.addEventListener("change", () => {
            const v = Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(Number(seed.value || 0))));
            if (v !== clip.seed) {
                clip.seed = v;
                updateHidden(node, runtime);
                render(node, runtime);
            }
        });
        const dice = document.createElement("button");
        dice.textContent = "🎲";
        dice.title = "Randomize seed";
        dice.style.width = "32px";
        dice.addEventListener("click", (e) => {
            e.preventDefault();
            clip.seed = randomSeed();
            updateHidden(node, runtime);
            render(node, runtime);
        });
        seedRow.append(seed, dice);
        seedBox.appendChild(seedRow);

        const seedMode = document.createElement("select");
        seedMode.title = "Seed behavior after a generated candidate";
        seedMode.style.width = "100%";
        seedMode.style.marginTop = "4px";
        seedMode.style.boxSizing = "border-box";
        seedMode.style.background = "rgba(0,0,0,.25)";
        seedMode.style.border = "1px solid rgba(255,255,255,.15)";
        seedMode.style.color = "inherit";
        seedMode.style.borderRadius = "5px";
        seedMode.style.padding = "4px 5px";
        for (const [value, label] of [
            ["randomize", "after: randomize"],
            ["fixed", "after: fixed"],
            ["increment", "after: increment"],
            ["decrement", "after: decrement"],
        ]) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            seedMode.appendChild(option);
        }
        seedMode.value = clip.seed_mode || "randomize";
        seedMode.addEventListener("change", () => {
            clip.seed_mode = seedMode.value;
            updateHidden(node, runtime);
        });
        seedBox.appendChild(seedMode);

        const durBox = document.createElement("div");
        durBox.appendChild(makeFieldLabel("Duration s"));
        const duration = makeNumberInput(clip.duration, 0.25, 150, 0.1);
        duration.addEventListener("change", () => {
            const v = Math.max(0.25, Math.min(150, Number(duration.value || 10)));
            if (Math.abs(v - clip.duration) > 1e-9) {
                clip.duration = v;
                const frameCount = h3FrameCountForDuration(v);
                clip.guides = normalizeGuideList(clip);
                for (const guide of clip.guides) {
                    const guideIdx = normalizeGuideFrameIdx(guide.frame_idx ?? 0);
                    guide.frame_idx = Math.max(-frameCount, Math.min(frameCount - 1, guideIdx));
                }
                updateHidden(node, runtime);
                render(node, runtime);
            }
        });
        durBox.appendChild(duration);
        row.append(seedBox, durBox);
        card.appendChild(row);

        const foot = document.createElement("div");
        foot.style.display = "flex";
        foot.style.alignItems = "center";
        foot.style.justifyContent = "space-between";
        foot.style.marginTop = "9px";

        const validateLabel = document.createElement("label");
        validateLabel.style.display = "flex";
        validateLabel.style.alignItems = "center";
        validateLabel.style.gap = "6px";
        validateLabel.style.cursor = "pointer";
        const validated = document.createElement("input");
        validated.type = "checkbox";
        validated.checked = clip.validated;
        validated.addEventListener("change", () => {
            if (fl2vaMode) {
                // FL2VA plans are independent: validation is per card and never
                // forces later plans open.
                clip.validated = Boolean(validated.checked);
            } else {
                if (validated.checked) {
                    clip.validated = true;
                } else {
                    invalidateFrom(state, index);
                }
                let open = false;
                for (const c of state.clips) {
                    if (open) c.validated = false;
                    else if (!c.validated) open = true;
                }
            }
            updateHidden(node, runtime);
            render(node, runtime);
        });
        validateLabel.append(validated, document.createTextNode("Validated"));

        const info = document.createElement("span");
        const aligned = h3FrameCountForDuration(clip.duration);
        info.textContent = `${aligned}f / ${(aligned / 24).toFixed(3)}s`;
        info.style.fontSize = "10px";
        info.style.opacity = ".65";

        foot.append(validateLabel, info);
        card.appendChild(foot);
        cards.appendChild(card);
    });
}

function syncDomHeight(node, runtime, forceMin = false, retry = 0) {
    if (!node || !runtime?.domWidget || runtime.syncingDomHeight) return;

    const mode = domWidgetRenderMode(runtime.root);
    if (mode === "pending") {
        if (retry < 12) {
            requestAnimationFrame(() => syncDomHeight(node, runtime, forceMin, retry + 1));
        }
        return;
    }

    // Nodes 2.0 owns the DOM-widget row height. Never derive a new getHeight
    // value from node.size here: node.size -> DOM getHeight -> node.size is the
    // feedback loop that created the infinite-height nodes.
    if (mode === "nodes2") {
        const currentH = Number(node.size?.[1] || 0);
        const y = Number(runtime.domWidget.last_y);
        const nodes2MinH = nodes2MinHeightForState(runtime.state);
        const fallbackH = Number.isFinite(y) && y > 0
            ? y + nodes2MinH + BOTTOM_PAD
            : nodes2MinH + 180;

        // One-time recovery for workflows that were already saved with a
        // runaway height by an older build. This is not DOM-driven resizing;
        // it only removes a clearly corrupted value.
        if (
            runtime.lastRenderMode !== "nodes2" &&
            obviouslyPoisonedHeight(currentH, fallbackH)
        ) {
            runtime.syncingDomHeight = true;
            try {
                const rememberedLegacyH = Number(runtime.legacyNodeHeight);
                const targetH = (
                    Number.isFinite(rememberedLegacyH) &&
                    !obviouslyPoisonedHeight(rememberedLegacyH, fallbackH)
                )
                    ? Math.max(fallbackH, rememberedLegacyH)
                    : fallbackH;
                const targetW = Math.max(
                    NODE_MIN_WIDTH,
                    Number(node.size?.[0] || NODE_MIN_WIDTH)
                );
                node.setSize([targetW, targetH]);
            } finally {
                runtime.syncingDomHeight = false;
            }
        }

        runtime.lastRenderMode = "nodes2";

        // Nodes 2.0 mounts this element inside WidgetDOM.vue's flex wrapper
        // (`flex flex-col *:flex-1`) and NodeWidgets.vue owns the grid row.
        // Do NOT use percentage heights here. A `height: 100%` has no stable
        // intrinsic size while CSS Grid is resolving an `auto` row; after a
        // manual resize that row can collapse to 0 and WidgetDOM will not
        // remount the element until a page refresh. Keep a real intrinsic
        // minimum instead and let Vue stretch the row/child naturally.
        runtime.root.style.height = "auto";
        runtime.root.style.minHeight = `${nodes2MinH}px`;
        runtime.root.style.setProperty("--comfy-widget-min-height", `${nodes2MinH}px`);
        runtime.root.style.maxHeight = "none";
        runtime.root.style.flex = "1 1 auto";
        runtime.root.style.paddingTop = `${5 + NODES2_TOP_GAP}px`;
        // Avoid a second vertical clipping boundary at fractional canvas zooms.
        // Horizontal clipping/scrolling is still owned by `cards`.
        runtime.root.style.overflow = "visible";

        runtime.cards.style.height = "auto";
        runtime.cards.style.flex = "1 1 auto";
        // The horizontal scrollbar has reserved space below the cards. Give the
        // row enough intrinsic height for both the card and that gutter so the
        // top/bottom cannot be shaved off by grid rounding at certain zooms.
        runtime.cards.style.minHeight = `${cardMinHeightForState(runtime.state) + CARD_SCROLLBAR_SPACE}px`;
        return;
    }

    const y = Number(runtime.domWidget.last_y);
    if (!Number.isFinite(y) || y <= 0) {
        if (retry < 12) {
            requestAnimationFrame(() => syncDomHeight(node, runtime, forceMin, retry + 1));
        }
        return;
    }

    // Remove Nodes 2.0-only intrinsic sizing when returning to Legacy.
    runtime.root.style.paddingTop = "5px";
    runtime.root.style.minHeight = "0";
    runtime.root.style.setProperty("--comfy-widget-min-height", `${UI_MIN_HEIGHT}px`);
    runtime.root.style.maxHeight = "none";
    runtime.root.style.flex = "0 0 auto";
    runtime.root.style.overflow = "hidden";

    runtime.syncingDomHeight = true;
    try {
        let w = Math.max(NODE_MIN_WIDTH, Number(node.size?.[0] || NODE_MIN_WIDTH));
        let h = Number(node.size?.[1] || 0);
        const uiMinH = uiMinHeightForState(runtime.state);
        const minNodeH = y + uiMinH + BOTTOM_PAD;
        const returningFromNodes2 = runtime.lastRenderMode === "nodes2";

        if (returningFromNodes2) {
            // Restore the last real Legacy height. If this node was first opened
            // in Nodes 2.0 (so there is no stored Legacy size), start from the
            // calculated Legacy minimum instead of inheriting a Vue runaway.
            const rememberedLegacyH = Number(runtime.legacyNodeHeight);
            h = (
                Number.isFinite(rememberedLegacyH) &&
                !obviouslyPoisonedHeight(rememberedLegacyH, minNodeH)
            )
                ? Math.max(minNodeH, rememberedLegacyH)
                : minNodeH;
        } else if (
            runtime.lastRenderMode == null &&
            obviouslyPoisonedHeight(h, minNodeH)
        ) {
            // Also heal workflows that are opened directly in Legacy after an
            // older version serialized an absurd height.
            h = minNodeH;
        } else if (forceMin && h < minNodeH) {
            h = minNodeH;
        }

        if (w !== Number(node.size?.[0]) || h !== Number(node.size?.[1])) {
            node.setSize([w, h]);
        }

        const actualH = Number(node.size?.[1] || h);
        const available = Math.max(uiMinH, actualH - y - BOTTOM_PAD);
        runtime.root.style.height = `${available}px`;
        runtime.cards.style.height = `${Math.max(340, available - 55 - REF_SECTION_HEIGHT)}px`;
        runtime.cards.style.flex = "0 0 auto";
        runtime.cards.style.minHeight = "";
        runtime.domHeight = available;
        if (!obviouslyPoisonedHeight(actualH, minNodeH)) {
            runtime.legacyNodeHeight = actualH;
        }
        runtime.lastRenderMode = "legacy";
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        runtime.syncingDomHeight = false;
    }
}

function installInvalidationHooks(node, runtime) {
    // Image references are no longer graph sockets. Other input/parameter
    // changes deliberately preserve explicit clip validation as before.
}


function hydrateRuntimeFromNativeWidgets(node, runtime, restoreCache = false) {
    if (!node || !runtime) return;

    // Native workflow widgets are the only persistence source. No onSerialize
    // interception, no node-property mirror, no widgets_values rewriting.
    const rawState = String(runtime.jsonWidget?.value || "");
    const state = parseState(rawState);
    const mode = persistentGenerationMode(node, rawState);
    activateModeState(state, mode);
    runtime.state = state;
    if (runtime.generationModeWidget) runtime.generationModeWidget.value = mode;

    runtime.refsState = parseRefsState(runtime.refsWidget?.value);
    snapshotModeValidation(runtime, mode);
    const restoredValidatedPrefix = validatedPrefixFromState(runtime.state);
    runtime.cachedCount = restoredValidatedPrefix;
    runtime.validatedCount = restoredValidatedPrefix;

    const removedLegacyRefs = removeLegacyImageRefInputs(node);
    syncDynamicAVReferenceInputs(node);
    if (removedLegacyRefs && refCount(runtime) === 0) {
        runtime.statusText = "Legacy image-ref sockets removed — load references in the Extender";
    }

    if (String(getWidget(node, "resolution_mode")?.value || "manual") === "manual") {
        rememberManualResolution(
            node,
            runtime,
            Number(getWidget(node, "width")?.value || runtime.manualWidth || 896),
            Number(getWidget(node, "height")?.value || runtime.manualHeight || 576),
        );
    }

    render(node, runtime);
    syncResolutionMirror(node, runtime);
    syncDomHeight(node, runtime, true);
    if (restoreCache) restoreCacheState(node, runtime);
}

function finalizeRuntimeAfterGraphLoad(node, runtime) {
    if (!node || !runtime) return;
    runtime.hydrating = false;
    runtime.ready = true;
    hydrateRuntimeFromNativeWidgets(node, runtime, true);
}

function buildUi(node) {
    if (node.__h3Extender) return node.__h3Extender;

    const jsonWidget = getWidget(node, "clips_json");
    const refsWidget = getWidget(node, "refs_json");
    const generationModeWidget = getWidget(node, "generation_mode");
    const contextLengthWidget = getWidget(node, "context_length");
    const audioContextLengthWidget = getWidget(node, "audio_context_length");
    if (!jsonWidget || !refsWidget || !generationModeWidget) return null;
    hideNativeWidget(node, jsonWidget);
    hideNativeWidget(node, refsWidget);
    hideNativeWidget(node, generationModeWidget);

    const state = parseState(jsonWidget.value);
    // Initial node construction can happen before a saved workflow has been
    // configured. This state is display-only until loadedGraphNode hydrates it
    // from the native serialized widgets.
    const persistedMode = persistentGenerationMode(node, jsonWidget.value);
    generationModeWidget.value = persistedMode;
    activateModeState(state, persistedMode);
    const refsState = parseRefsState(refsWidget.value);

    const root = document.createElement("div");
    root.style.width = "100%";
    root.style.minWidth = "0";
    const initialUiMinHeight = uiMinHeightForState(state);
    root.style.height = `${initialUiMinHeight}px`;
    root.style.minHeight = `${initialUiMinHeight}px`;
    // Official DOMWidgetImpl.computeLayoutSize() reads this CSS variable as a
    // fallback to getMinHeight. Keeping both makes the intrinsic contract clear
    // to current and slightly older Nodes 2.0 frontends.
    root.style.setProperty("--comfy-widget-min-height", `${nodes2MinHeightForState(state)}px`);
    root.style.boxSizing = "border-box";
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.padding = "5px 0 0";
    root.style.overflow = "hidden";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.minWidth = "0";
    toolbar.style.gap = "7px";
    toolbar.style.alignItems = "center";
    toolbar.style.marginBottom = "7px";

    const modeButton = document.createElement("button");
    modeButton.title = "Switch between Ref2VA + Motion Context and independent FL2VA plans";
    modeButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (projectBusy(runtime)) return;
        const current = runtime.state.generation_mode === "fl2va" ? "fl2va" : "ref2va";
        const next = current === "fl2va" ? "ref2va" : "fl2va";
        // REF2VA and FL2VA own completely independent card timelines. Store the
        // active array before switching and restore the other mode's array;
        // edits, insertions and deletions in one mode never mutate the other.
        activateModeState(runtime.state, next);
        generationModeWidget.value = next;
        runtime.cachedClipIds = new Set();
        runtime.validatedClipIds = new Set();
        runtime.cachedCount = 0;
        runtime.validatedCount = 0;
        runtime.cacheStateRestored = false;
        updateHidden(node, runtime);
        captureNativeWorkflowState(node, runtime);
        render(node, runtime);
        restoreCacheState(node, runtime);
        requestAnimationFrame(() => syncDomHeight(node, runtime, true));
    });

    const add = document.createElement("button");
    add.textContent = "+ Add Clip";
    add.addEventListener("click", (e) => {
        e.preventDefault();
        runtime.state.clips.push(newClip(runtime.state.clips.length));
        updateHidden(node, runtime);
        render(node, runtime);
        requestAnimationFrame(() => {
            cards.scrollLeft = cards.scrollWidth;
        });
    });

    const remove = document.createElement("button");
    remove.textContent = "− Remove Last";
    remove.addEventListener("click", (e) => {
        e.preventDefault();
        if (runtime.state.clips.length <= 1) return;
        runtime.state.clips.pop();
        updateHidden(node, runtime);
        render(node, runtime);
    });

    const saveProjectButton = document.createElement("button");
    saveProjectButton.textContent = "Save Project";
    saveProjectButton.title = "Save settings + disk cache as a portable .ext project";
    saveProjectButton.addEventListener("click", (e) => {
        e.preventDefault();
        saveProject(node, runtime);
    });

    const loadProjectButton = document.createElement("button");
    loadProjectButton.textContent = "Load Project";
    loadProjectButton.title = "Load a .ext project into this Extender node";

    const projectFileInput = document.createElement("input");
    projectFileInput.type = "file";
    projectFileInput.accept = ".ext,application/zip,application/octet-stream";
    projectFileInput.style.display = "none";
    projectFileInput.addEventListener("change", async () => {
        const file = projectFileInput.files?.[0];
        projectFileInput.value = "";
        if (file) await loadProjectFile(node, runtime, file);
    });
    loadProjectButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (projectBusy(runtime)) {
            alert("Wait for the current clip generation to finish before loading a project.");
            return;
        }
        projectFileInput.click();
    });

    const counter = document.createElement("span");
    counter.style.fontSize = "11px";
    counter.style.opacity = ".8";

    const status = document.createElement("span");
    status.style.fontSize = "11px";
    status.style.opacity = ".72";
    status.style.marginLeft = "auto";
    status.style.whiteSpace = "nowrap";
    status.style.overflow = "hidden";
    status.style.textOverflow = "ellipsis";
    status.style.maxWidth = "55%";

    toolbar.append(modeButton, add, remove, saveProjectButton, loadProjectButton, counter, status, projectFileInput);

    const refFileInput = document.createElement("input");
    refFileInput.type = "file";
    refFileInput.accept = "image/*,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff";
    refFileInput.style.display = "none";

    const frameFileInput = document.createElement("input");
    frameFileInput.type = "file";
    frameFileInput.accept = "image/*,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff";
    frameFileInput.style.display = "none";

    const refsSection = document.createElement("div");
    refsSection.style.height = `${REF_SECTION_HEIGHT}px`;
    refsSection.style.minWidth = "0";
    refsSection.style.flex = `0 0 ${REF_SECTION_HEIGHT}px`;
    refsSection.style.boxSizing = "border-box";
    refsSection.style.marginBottom = "7px";

    const refsHeader = document.createElement("div");
    refsHeader.textContent = "REFERENCE IMAGES — double-click a thumbnail to edit";
    refsHeader.style.fontSize = "10px";
    refsHeader.style.fontWeight = "600";
    refsHeader.style.opacity = ".75";
    refsHeader.style.height = "13px";
    refsHeader.style.lineHeight = "13px";
    refsHeader.style.marginBottom = "1px";

    const refsRow = document.createElement("div");
    refsRow.style.display = "flex";
    refsRow.style.flexDirection = "row";
    refsRow.style.width = "100%";
    refsRow.style.maxWidth = "100%";
    refsRow.style.minWidth = "0";
    refsRow.style.gap = "7px";
    refsRow.style.overflowX = "auto";
    refsRow.style.overflowY = "hidden";
    refsRow.style.paddingBottom = `${REF_SCROLLBAR_SPACE}px`;
    refsRow.style.boxSizing = "border-box";
    refsRow.style.height = `${REF_SECTION_HEIGHT - 14}px`;
    refsRow.style.scrollbarGutter = "stable";
    refsSection.append(refsHeader, refsRow);

    const cards = document.createElement("div");
    cards.style.display = "flex";
    cards.style.minWidth = "0";
    cards.style.flexDirection = "row";
    cards.style.gap = "9px";
    cards.style.overflowX = "auto";
    cards.style.overflowY = "hidden";
    cards.style.padding = `0 0 ${CARD_SCROLLBAR_SPACE}px 0`;
    cards.style.scrollbarGutter = "stable";
    cards.style.boxSizing = "border-box";
    cards.style.scrollBehavior = "smooth";
    cards.style.height = `${Math.max(340, initialUiMinHeight - 55 - REF_SECTION_HEIGHT)}px`;
    cards.style.minHeight = `${cardMinHeightForState(state) + CARD_SCROLLBAR_SPACE}px`;

    root.append(toolbar, refsSection, cards, refFileInput, frameFileInput);

    const restoredValidatedPrefix = validatedPrefixFromState(state);
    const runtime = {
        state,
        jsonWidget,
        refsState,
        refsWidget,
        root,
        toolbar,
        refsSection,
        refsHeader,
        refsRow,
        cards,
        counter,
        status,
        saveProjectButton,
        loadProjectButton,
        projectFileInput,
        refFileInput,
        frameFileInput,
        generationModeWidget,
        contextLengthWidget,
        audioContextLengthWidget,
        modeButton,
        pendingRefSlot: -1,
        pendingFrameClip: -1,
        pendingFrameKind: "",
        pendingFrameGuideIndex: -1,
        refBusy: false,
        projectOperationBusy: false,
        projectName: String(node?.properties?.h3_project_name || ""),
        domWidget: null,
        domHeight: UI_MIN_HEIGHT,
        syncingDomHeight: false,
        lastRenderMode: null,
        legacyNodeHeight: null,
        // clips_json already preserves the validated flags. Seed the visual state
        // immediately, then replace it with the authoritative disk manifest below.
        cachedCount: restoredValidatedPrefix,
        validatedCount: restoredValidatedPrefix,
        statusText: restoredValidatedPrefix
            ? `Restoring cache | validated ${restoredValidatedPrefix}`
            : "Ready",
        activeClipIndex: -1,
        activePhase: "idle",
        cacheStateRequestRunning: false,
        cacheStateRestored: false,
        expectedResolution: null,
        resolvedWidth: 0,
        resolvedHeight: 0,
        resolutionGuide: "",
        guideSourceWidth: 0,
        guideSourceHeight: 0,
        resolutionFallback: false,
        resolutionMismatch: false,
        manualWidth: Number(node?.properties?.h3_manual_width || getWidget(node, "width")?.value || 896),
        manualHeight: Number(node?.properties?.h3_manual_height || getWidget(node, "height")?.value || 576),
        applyingResolutionMirror: false,
        resolutionMirrorActive: false,
        resolutionCallbacksInstalled: false,
        // True only after an explicit .ext Load has imposed its archived
        // geometry. Any user resolution edit clears it; editing megapixels
        // also switches straight back to Auto because MP has no Manual meaning.
        projectResolutionLoaded: false,
        // True after a live resolution change has made the on-disk cache stale.
        // The backend clears/rebuilds that cache on the next Queue.
        resolutionInvalidated: false,
        loraNames: [],
        loraListLoading: false,
        loraListLoaded: false,
        loraListError: "",
        cachedClipIds: new Set(),
        validatedClipIds: new Set(),
        continuitySignatures: new Map(),
        continuitySignatureRequests: new Set(),
        modeValidationState: {
            [state.generation_mode === "fl2va" ? "fl2va" : "ref2va"]: new Map(
                (state.clips || []).map((clip) => [String(clip.id), Boolean(clip.validated)])
            ),
        },
        // True while ComfyUI is reconstructing a serialized graph. The official
        // lifecycle hooks clear this only after native widget restoration has
        // completed; custom controls never serialize a parallel state.
        hydrating: isH3GraphConfiguring(),
        ready: false,
    };

    refFileInput.addEventListener("change", async () => {
        const file = refFileInput.files?.[0];
        const slot = Number(runtime.pendingRefSlot);
        refFileInput.value = "";
        runtime.pendingRefSlot = -1;
        if (file && Number.isInteger(slot) && slot >= 0 && slot < MAX_IMAGE_REFS) {
            await uploadReference(node, runtime, slot, file);
        }
    });

    frameFileInput.addEventListener("change", async () => {
        const file = frameFileInput.files?.[0];
        const clipIndex = Number(runtime.pendingFrameClip);
        const kind = String(runtime.pendingFrameKind || "");
        const guideIndex = Number(runtime.pendingFrameGuideIndex);
        frameFileInput.value = "";
        runtime.pendingFrameClip = -1;
        runtime.pendingFrameKind = "";
        runtime.pendingFrameGuideIndex = -1;
        if (file && Number.isInteger(clipIndex) && clipIndex >= 0 && ["first", "last", "guide"].includes(kind)) {
            await uploadClipFrame(node, runtime, clipIndex, kind, file, guideIndex);
        }
    });

    const domWidget = node.addDOMWidget("h3_extender_timeline", "timeline", root, {
        serialize: false,
        hideOnZoom: false,
        // DOMWidgetImpl.computeLayoutSize() is the official size contract.
        // Give Nodes 2.0 a little more intrinsic room, while keeping the old
        // Legacy minimum unchanged.
        getMinHeight: () =>
            globalThis.LiteGraph?.vueNodesMode
                ? nodes2MinHeightForState(runtime.state)
                : uiMinHeightForState(runtime.state),
        getHeight: () => runtime.domHeight,
        afterResize: (resizedNode) => {
            const mode = domWidgetRenderMode(root);
            if (mode === "nodes2") {
                // Re-assert only intrinsic CSS. Never derive anything from
                // node.size while Vue is resolving its grid.
                const nodes2MinH = nodes2MinHeightForState(runtime.state);
                root.style.height = "auto";
                root.style.minHeight = `${nodes2MinH}px`;
                root.style.setProperty("--comfy-widget-min-height", `${nodes2MinH}px`);
                root.style.maxHeight = "none";
                root.style.flex = "1 1 auto";
                root.style.paddingTop = `${5 + NODES2_TOP_GAP}px`;
                root.style.overflow = "visible";
                cards.style.height = "auto";
                cards.style.flex = "1 1 auto";
                cards.style.minHeight = `${cardMinHeightForState(runtime.state) + CARD_SCROLLBAR_SPACE}px`;
                runtime.lastRenderMode = "nodes2";
            } else if (mode === "legacy") {
                requestAnimationFrame(() => syncDomHeight(resizedNode, runtime, false));
            } else {
                requestAnimationFrame(() => syncDomHeight(resizedNode, runtime, false));
            }
        },
    });
    runtime.domWidget = domWidget;
    node.__h3Extender = runtime;

    installInvalidationHooks(node, runtime);
    wrapResolutionWidgetCallbacks(node, runtime);
    render(node, runtime);
    refreshLoraNames(node, runtime);

    const oldConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;

        // Keep only the legacy resolution migration here. Native ComfyUI
        // configure() already restored clips_json/refs_json/generation_mode by
        // the time loadedGraphNode is emitted; do not shadow that mechanism.
        const savedWidgetValues = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        const hasSavedResolutionMode = Boolean(
            savedWidgetValues?.some((value) => value === "auto_from_ref" || value === "manual")
        );
        if (savedWidgetValues && !hasSavedResolutionMode) {
            setWidgetValue(this, "resolution_mode", "manual");
        }

        // Direct configure() calls outside a full graph load (copy/paste and a
        // few legacy paths) still hydrate from the native widget values, but no
        // custom serialization is involved.
        if (!isH3GraphConfiguring()) {
            hydrateRuntimeFromNativeWidgets(this, runtime, false);
            runtime.hydrating = false;
            finalizeRuntimeAfterGraphLoad(this, runtime);
        }
        return result;
    };

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            if (runtime.hydrating || isH3GraphConfiguring()) return;
            finalizeRuntimeAfterGraphLoad(node, runtime);
        });
    });

    return runtime;
}


function findExtenderNodeByExecutionId(nodeId) {
    const graph = app.graph;
    if (!graph) return null;

    const wanted = String(nodeId);
    for (const node of graph._nodes || []) {
        if (
            String(node?.id) === wanted &&
            (node?.comfyClass === TARGET || node?.type === TARGET)
        ) {
            return node;
        }
    }
    return null;
}

function scrollActiveCard(runtime, index) {
    if (!runtime?.cards || index < 0) return;
    const card = runtime.cards.querySelector(
        `[data-clip-index="${index}"]`
    );
    if (!card) return;

    const left = Math.max(
        0,
        card.offsetLeft -
            Math.max(0, (runtime.cards.clientWidth - card.offsetWidth) / 2)
    );
    runtime.cards.scrollTo({
        left,
        behavior: "smooth",
    });
}

// A cancelled/failed ComfyUI execution does not call this node's onExecuted
// callback. Without an explicit terminal-event reset, the last custom progress
// event (usually "sampling") leaves the active card permanently blue until a
// page refresh. ComfyUI exposes official execution_interrupted/error/success
// websocket events, so clear only the transient rendering state when a prompt
// terminates. Cache/validation/card data are deliberately left untouched.
function clearTransientRenderingState(statusText = null) {
    const graph = app.graph;
    if (!graph) return;

    for (const node of graph._nodes || []) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;

        const runtime = node.__h3Extender;
        if (!runtime) continue;

        const wasActive =
            Number(runtime.activeClipIndex) >= 0 ||
            ["preparing", "sampling", "complete"].includes(
                String(runtime.activePhase || "")
            );
        if (!wasActive) continue;

        runtime.activeClipIndex = -1;
        runtime.activePhase = "idle";
        if (statusText) runtime.statusText = statusText;

        render(node, runtime);
        node.graph?.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "MiniMaxH3.Extender",

    beforeConfigureGraph() {
        h3GraphConfiguring = true;
        for (const node of app.graph?._nodes || []) {
            if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
            if (node.__h3Extender) node.__h3Extender.hydrating = true;
        }
    },

    loadedGraphNode(node) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) return;
        const runtime = buildUi(node);
        if (!runtime) return;
        runtime.hydrating = true;
        // This hook runs after LGraphNode.configure() restored the native widget
        // values. Hydrate runtime/UI from them, but wait for afterConfigureGraph
        // before touching disk cache/preview state.
        hydrateRuntimeFromNativeWidgets(node, runtime, false);
    },

    afterConfigureGraph() {
        h3GraphConfiguring = false;
        for (const node of app.graph?._nodes || []) {
            if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
            const runtime = buildUi(node);
            if (!runtime) continue;
            finalizeRuntimeAfterGraphLoad(node, runtime);
        }
    },

    setup() {
        // Nodes 2.0 persists workflow drafts on graphChanged with a debounce and
        // current frontends flush that debounce on pagehide. Run our capture in
        // the capture phase so ComfyUI's own pagehide flush serializes the latest
        // hidden widget values even when the user hits F5 immediately after a
        // custom-DOM action.
        if (!window.__h3ExtenderPagehideCaptureInstalled) {
            window.__h3ExtenderPagehideCaptureInstalled = true;
            window.addEventListener("pagehide", () => {
                for (const node of app.graph?._nodes || []) {
                    if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
                    const runtime = node.__h3Extender || null;
                    captureNativeWorkflowState(node, runtime);
                }
            }, true);
        }

        // Official ComfyUI terminal execution events. In particular, pressing
        // Kill/Interrupt raises execution_interrupted and bypasses onExecuted.
        api.addEventListener("execution_interrupted", () => {
            clearTransientRenderingState("Rendering interrupted");
        });
        api.addEventListener("execution_error", () => {
            clearTransientRenderingState("Execution stopped by error");
        });
        // Defensive cleanup: a successful prompt should never leave a stale
        // rendering highlight even if another frontend/backend change prevents
        // the expected node UI callback from arriving.
        api.addEventListener("execution_success", () => {
            clearTransientRenderingState();
        });

        api.addEventListener(PROMPT_PACK_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime || !detail?.clips_json) return;

            runtime.state = mergeActiveStateJson(
                runtime,
                detail.clips_json,
                runtime.state?.generation_mode || "ref2va",
            );
            runtime.jsonWidget.value = serializeState(runtime.state);
            const count = Number(detail?.prompt_count || runtime.state.clips.length || 0);
            const source = String(detail?.source || "External prompt pack");
            runtime.statusText = `${source}: imported ${count} prompt${count === 1 ? "" : "s"} → ${count} clip${count === 1 ? "" : "s"}`;
            updateHidden(node, runtime);
            render(node, runtime);
            syncDomHeight(node, runtime, false);
            node.graph?.setDirtyCanvas(true, true);
        });

        api.addEventListener(REF_PACK_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime || !detail?.refs_json) return;

            runtime.refsWidget.value = String(detail.refs_json);
            runtime.refsState = parseRefsState(detail.refs_json);
            updateRefsHidden(node, runtime);
            const slots = Array.isArray(detail?.imported_slots)
                ? detail.imported_slots.map((value) => Number(value)).filter((value) => Number.isInteger(value) && value >= 1 && value <= MAX_IMAGE_REFS)
                : [];
            const source = String(detail?.source || "External reference pack");
            runtime.statusText = slots.length
                ? `${source}: imported Ref ${slots.join(", ")} into internal slots`
                : `${source}: synchronized`;
            render(node, runtime);
            node.graph?.setDirtyCanvas(true, true);
        });

        api.addEventListener(PROGRESS_EVENT, ({ detail }) => {
            const node = findExtenderNodeByExecutionId(detail?.node);
            if (!node) return;

            const runtime = buildUi(node);
            if (!runtime) return;

            const index = Number(detail?.clip_index ?? -1);
            runtime.activeClipIndex = Number.isFinite(index) ? index : -1;
            runtime.activePhase = String(detail?.phase || "idle");
            runtime.statusText = String(detail?.message || runtime.statusText || "Ready");

            render(node, runtime);

            if (runtime.activeClipIndex >= 0) {
                requestAnimationFrame(() => {
                    scrollActiveCard(runtime, runtime.activeClipIndex);
                });
            }

            node.graph?.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = oldCreated ? oldCreated.apply(this, arguments) : undefined;

            // New nodes must start in Auto resolution mode. Older workflows are
            // still migrated to Manual later in onConfigure when they do not
            // contain the v14.25+ resolution widgets.
            setWidgetValue(this, "resolution_mode", "auto_from_ref");

            const runtime = buildUi(this);
            removeLegacyImageRefInputs(this);
            deferDynamicAVReferenceSync(this);
            if (runtime) {
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => syncDomHeight(this, runtime, true));
                });
            }
            return r;
        };

        const oldConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = oldConnectionsChange
                ? oldConnectionsChange.apply(this, arguments)
                : undefined;
            // LiteGraph mutates link target slots during the callback; defer the
            // socket grow/shrink pass until that mutation has completed.
            deferDynamicAVReferenceSync(this);
            return result;
        };

        const oldExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (oldExecuted) oldExecuted.apply(this, arguments);
            const runtime = buildUi(this);
            if (!runtime) return;

            const info = message?.h3_extender_state?.[0];
            if (!info) return;

            if (info.clips_json) {
                runtime.state = mergeActiveStateJson(
                    runtime,
                    info.clips_json,
                    info.generation_mode || runtime.state?.generation_mode || "ref2va",
                );
                runtime.jsonWidget.value = serializeState(runtime.state);
            }
            if (info.generation_mode) {
                activateModeState(
                    runtime.state,
                    String(info.generation_mode) === "fl2va" ? "fl2va" : "ref2va",
                );
                if (runtime.generationModeWidget) runtime.generationModeWidget.value = runtime.state.generation_mode;
            }
            if (info.refs_json) {
                runtime.refsWidget.value = info.refs_json;
                runtime.refsState = parseRefsState(info.refs_json);
            }
    
            const generated = Array.isArray(info.generated) ? info.generated : [];
            for (const humanIndex of generated) {
                const i = Number(humanIndex) - 1;
                const clip = runtime.state.clips[i];
                // Only prepare a next seed for a candidate. A validated cached
                // clip is never touched by this automatic seed behavior.
                if (clip && !clip.validated) {
                    advanceSeedAfterGenerate(clip);
                }
            }

            // Critical: persist the next seed into clips_json. This changes the
            // node input hash, so pressing Queue again really re-executes it.
            if (generated.length) {
                updateHidden(this, runtime);
            }

            runtime.cachedCount = Number(info.cached_count || 0);
            runtime.validatedCount = Number(info.validated_count || 0);
            runtime.cachedClipIds = new Set(Array.isArray(info.cached_clip_ids) ? info.cached_clip_ids.map(String) : []);
            runtime.validatedClipIds = new Set(Array.isArray(info.validated_clip_ids) ? info.validated_clip_ids.map(String) : []);
            runtime.continuitySignatures = new Map(
                Object.entries(info?.continuity_signatures || {}).map(([key, value]) => [String(key), String(value || "")]).filter(([, value]) => Boolean(value))
            );
            snapshotModeValidation(runtime, runtime.state?.generation_mode);
            runtime.resolvedWidth = Number(info.resolved_width || 0);
            runtime.resolvedHeight = Number(info.resolved_height || 0);
            runtime.resolutionGuide = String(info.resolution_guide || "");
            runtime.guideSourceWidth = Number(info.resolution_guide_width || 0);
            runtime.guideSourceHeight = Number(info.resolution_guide_height || 0);
            runtime.resolutionFallback = Boolean(info.resolution_fallback);
            runtime.resolutionMismatch = Boolean(info.resolution_mismatch);
            if (runtime.resolvedWidth > 0 && runtime.resolvedHeight > 0) {
                // Backend execution is authoritative. After a resolution-change
                // run, this becomes the new baseline for future invalidation.
                runtime.expectedResolution = {
                    width: runtime.resolvedWidth,
                    height: runtime.resolvedHeight,
                };
                runtime.resolutionInvalidated = false;
            }
            runtime.activeClipIndex = -1;
            runtime.activePhase = "idle";
            runtime.statusText = String(info.status || "Ready");
            if (runtime.resolutionMismatch && Number(info.cache_width || 0) > 0) {
                runtime.statusText +=
                    ` | WARNING cache ${Number(info.cache_width)}x${Number(info.cache_height)} differs`;
            }
            syncResolutionMirror(this, runtime);
            render(this, runtime);
            syncDomHeight(this, runtime, false);
        };
    },
});
