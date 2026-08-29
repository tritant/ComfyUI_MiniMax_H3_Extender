import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3MotionContextDiskFinalDecode";
const DISK_JOIN_TARGET = "MiniMaxH3MotionContextDiskJoin";
const EXTENDER_TARGET = "MiniMaxH3Extender";

function ensureSavePreviewButtonStyle() {
    if (document.getElementById("h3-save-preview-button-style")) return;
    const style = document.createElement("style");
    style.id = "h3-save-preview-button-style";
    style.textContent = `
        .h3-save-preview-button {
            height: 22px;
            min-width: 118px;
            padding: 0 14px;
            border: 1px solid rgba(120, 220, 160, 0.72);
            border-radius: 5px;
            background: linear-gradient(180deg, rgba(51, 145, 92, 0.96), rgba(35, 108, 70, 0.96));
            color: #ffffff;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.35px;
            line-height: 20px;
            cursor: pointer;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
            transition: filter 100ms ease, transform 100ms ease, opacity 100ms ease;
        }
        .h3-save-preview-button:hover:not(:disabled) {
            filter: brightness(1.18);
        }
        .h3-save-preview-button:active:not(:disabled) {
            transform: translateY(1px);
        }
        .h3-save-preview-button:disabled {
            border-color: rgba(255, 255, 255, 0.20);
            background: rgba(70, 70, 70, 0.88);
            color: rgba(255, 255, 255, 0.55);
            cursor: default;
            box-shadow: none;
            opacity: 0.78;
        }
    `;
    document.head.appendChild(style);
}

function stripFinalDecodeOutputs(node) {
    if (!node?.outputs?.length) return;

    // Old workflows serialize the nine legacy outputs in the graph JSON.
    // RETURN_TYPES=() prevents them on newly-defined nodes, but LiteGraph can
    // restore the serialized sockets during configure(). Remove them here too.
    while (node.outputs?.length) {
        const index = node.outputs.length - 1;
        if (typeof node.removeOutput === "function") {
            node.removeOutput(index);
        } else {
            // Fallback for older LiteGraph builds. These outputs were never
            // meant to be consumed; remove the visual slots at minimum.
            node.outputs.splice(index, 1);
        }
    }

    node.graph?.setDirtyCanvas(true, true);
}

function isDiskJoin(node) {
    return (
        node?.comfyClass === DISK_JOIN_TARGET ||
        node?.type === DISK_JOIN_TARGET
    );
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function hideCompatibilityWidget(node, name) {
    const widget = getWidget(node, name);
    if (!widget || widget.__h3CompatibilityHidden) return;

    // Keep the widget in node.widgets so legacy widgets_values arrays retain
    // their original positional layout, but remove it from both legacy canvas
    // layout and Nodes 2.0 DOM rendering. Its backend value is ignored.
    widget.__h3CompatibilityHidden = true;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    try { widget.type = "converted-widget"; } catch (_) {}
    for (const element of [widget.element, widget.inputEl, widget.domWidget?.element]) {
        if (element?.style) element.style.display = "none";
    }
    node?.graph?.setDirtyCanvas(true, true);
}

function isFalseValue(value) {
    return value === false || value === 0 || value === "false";
}

function setValidatedFalse(node) {
    const widget = getWidget(node, "validated");
    if (!widget) return false;

    if (!isFalseValue(widget.value)) {
        // Do NOT invoke its callback here: the recursive graph traversal below
        // already propagates invalidation and avoids callback recursion.
        widget.value = false;
        node.graph?.setDirtyCanvas(true, true);
        return true;
    }
    return false;
}

function downstreamDiskJoins(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return [];

    const output = node?.outputs?.find((o) => o?.name === "cache");
    const links = output?.links || [];
    const result = [];

    for (const linkId of links) {
        const link = graph.links?.[linkId];
        if (!link) continue;

        const target = graph.getNodeById?.(link.target_id);
        if (target && isDiskJoin(target)) {
            result.push(target);
        }
    }
    return result;
}

function invalidateDownstream(node) {
    const visited = new Set();

    function walk(current) {
        for (const next of downstreamDiskJoins(current)) {
            const key = next.id ?? next;
            if (visited.has(key)) continue;
            visited.add(key);

            setValidatedFalse(next);
            walk(next);
        }
    }

    walk(node);
    node?.graph?.setDirtyCanvas(true, true);
}

function installValidationCascade(node) {
    if (!node || node.__h3ValidationCascadeInstalled) return;

    const widget = getWidget(node, "validated");
    if (!widget) {
        // Widgets can finish materializing just after onNodeCreated.
        requestAnimationFrame(() => installValidationCascade(node));
        return;
    }

    const originalCallback = widget.callback;

    widget.callback = function (value) {
        const result = originalCallback
            ? originalCallback.apply(this, arguments)
            : undefined;

        // If clip N is invalidated, clips N+1... are no longer valid because
        // their Motion Context depended on the old version of clip N.
        if (isFalseValue(value)) {
            invalidateDownstream(node);
        }

        return result;
    };

    node.__h3ValidationCascadeInstalled = true;
}


const PLAYER_MIN_WIDTH = 380;
const PLAYER_MIN_HEIGHT = 227;
const LABEL_HEIGHT = 22;
const PREVIEW_HEADER_GAP = 7;
const BOTTOM_PAD = 14;

function previewDomRenderMode(element) {
    const LG = globalThis.LiteGraph;
    const hasModeFlag = typeof LG?.vueNodesMode === "boolean";
    if (!element?.isConnected) return "pending";

    const insideVueRow = Boolean(element.closest?.(".lg-node-widget"));
    if (hasModeFlag) {
        if (LG.vueNodesMode && !insideVueRow) return "pending";
        if (!LG.vueNodesMode && insideVueRow) return "pending";
        return LG.vueNodesMode ? "nodes2" : "legacy";
    }
    return insideVueRow ? "nodes2" : "legacy";
}

function previewHeightIsPoisoned(height, minimumHeight) {
    const h = Number(height);
    if (!Number.isFinite(h) || h <= 0) return false;
    return h > Math.max(1400, Number(minimumHeight || 0) * 4);
}

function mediaUrl(info) {
    const params = new URLSearchParams();
    params.set("filename", info.filename || "");
    params.set("type", info.type || "temp");
    params.set("subfolder", info.subfolder || "");
    return api.apiURL("/view?" + params.toString());
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

function cssColorFilter(value) {
    const c = normalizeColorAdjustment(value);
    return `saturate(${c.saturation}%) contrast(${c.contrast}%) brightness(${c.brightness}%)`;
}

function colorAdjustmentAtTime(timeline, time) {
    const t = Number(time || 0);
    for (const item of timeline || []) {
        const start = Number(item?.start || 0);
        const end = Number(item?.end || start);
        if (t >= start && t < end) return item?.adjustment || null;
    }
    return null;
}

function syncPreviewColorFilter(state) {
    if (!state?.video) return;
    const adjustment = colorAdjustmentAtTime(state.colorTimeline, state.video.currentTime);
    state.video.style.filter = adjustment ? cssColorFilter(adjustment) : "none";
}


function findUpstreamExtenderNode(node) {
    const graph = node?.graph || app.graph;
    if (!graph) return null;
    const cacheInput = (node.inputs || []).find((input) => input?.name === "cache");
    const linkId = cacheInput?.link;
    if (linkId == null) return null;
    const link = graph.links?.[linkId];
    if (!link) return null;
    const origin = graph.getNodeById?.(link.origin_id)
        || (graph._nodes || []).find((n) => String(n?.id) === String(link.origin_id));
    if (!origin) return null;
    if (origin?.comfyClass === EXTENDER_TARGET || origin?.type === EXTENDER_TARGET) return origin;
    return null;
}

function findUpstreamExtenderId(node) {
    return findUpstreamExtenderNode(node)?.id ?? null;
}

function upstreamGenerationMode(node) {
    const origin = findUpstreamExtenderNode(node);

    // The Extender custom runtime owns the authoritative mode once its UI has
    // been restored. Prefer it over the hidden native combo, whose default can
    // transiently be Ref2VA during Nodes 2.0 workflow startup/refresh.
    const runtimeMode = String(origin?.__h3Extender?.state?.generation_mode || "").toLowerCase();
    if (runtimeMode === "fl2va" || runtimeMode === "ref2va") return runtimeMode;

    // clips_json also persists the active mode. It gives us a second stable
    // source before falling back to the native generation_mode widget.
    const clipsWidget = (origin?.widgets || []).find((w) => w?.name === "clips_json");
    if (typeof clipsWidget?.value === "string") {
        try {
            const parsed = JSON.parse(clipsWidget.value);
            const persisted = String(parsed?.generation_mode || "").toLowerCase();
            if (persisted === "fl2va" || persisted === "ref2va") return persisted;
        } catch (_) {}
    }

    const widget = (origin?.widgets || []).find((w) => w?.name === "generation_mode");
    return String(widget?.value || "ref2va") === "fl2va" ? "fl2va" : "ref2va";
}

function loadPreviewSource(node, state, url) {
    if (!state?.video) return;

    const video = state.video;
    const autoplayWidget = getWidget(node, "autoplay");
    const rawAutoplay = autoplayWidget?.value;
    const wantsAutoplay = rawAutoplay === true || rawAutoplay === 1 || String(rawAutoplay).toLowerCase() === "true";

    // Always fetch enough media to paint frame 0 after workflow restore.
    video.preload = "auto";
    video.autoplay = wantsAutoplay;

    // Each source load gets its own token so delayed media events from an older
    // preview cannot pause/play the newly restored preview.
    state.previewLoadToken = Number(state.previewLoadToken || 0) + 1;
    const loadToken = state.previewLoadToken;

    const paintPausedFrame = () => {
        if (state.previewLoadToken !== loadToken) return;
        try {
            video.pause();
            const target = Number.isFinite(video.duration) && video.duration > 0
                ? Math.min(0.001, Math.max(0, video.duration / 100000))
                : 0.001;
            if (Math.abs(Number(video.currentTime || 0) - target) > 1e-6) {
                video.currentTime = target;
            }
        } catch (_) {}
    };

    let autoplayStarted = false;
    const tryAutoplay = () => {
        if (!wantsAutoplay || autoplayStarted || state.previewLoadToken !== loadToken) return;
        if (video.readyState < 2) return;
        autoplayStarted = true;
        // play() is intentionally attempted only once decoded data exists.
        // Calling it immediately after assigning src can reject during a page
        // restore even when autoplay is enabled for the site.
        const playPromise = video.play();
        if (playPromise?.catch) {
            playPromise.catch(() => {
                autoplayStarted = false;
                // Keep a visible first frame if the browser itself blocks
                // audible autoplay; a later canplay event gets one final try.
                if (video.readyState >= 2) {
                    const retry = () => {
                        if (state.previewLoadToken !== loadToken || !wantsAutoplay || !video.paused) return;
                        video.play().catch(() => {});
                    };
                    video.addEventListener("canplay", retry, { once: true });
                }
            });
        }
    };

    video.addEventListener("loadeddata", () => {
        if (state.previewLoadToken !== loadToken) return;
        if (wantsAutoplay) tryAutoplay();
        else paintPausedFrame();
    }, { once: true });

    video.addEventListener("canplay", () => {
        if (state.previewLoadToken !== loadToken) return;
        if (wantsAutoplay && video.paused) tryAutoplay();
    }, { once: true });

    video.src = url;
    video.load();
}

async function restorePreviewOnLoad(node, state, attempt = 0) {
    if (!node || !state || state.liveLoaded || state.restoreLoaded) return;

    const ownerId = findUpstreamExtenderId(node);
    if (ownerId == null) {
        // Workflow links may be restored a little after node.configure().
        if (attempt < 12) {
            setTimeout(() => restorePreviewOnLoad(node, state, attempt + 1), 80);
        }
        return;
    }

    if (state.restoreRequestRunning) return;
    state.restoreRequestRunning = true;

    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(ownerId));
        params.set("final_id", String(node.id));
        params.set("mode", upstreamGenerationMode(node));

        const response = await fetch(
            api.apiURL("/h3_extender/restored_preview?" + params.toString())
        );
        if (!response.ok) {
            if (attempt < 12) setTimeout(() => restorePreviewOnLoad(node, state, attempt + 1), 120);
            return;
        }

        const payload = await response.json();
        if (!payload?.found || !payload?.video?.filename) {
            // During workflow restore the upstream Extender can finish restoring
            // its FL/REF state a few frames after Final Decode. Retry briefly so
            // we do not permanently miss the preview because the first request
            // looked at the transient/default mode.
            if (attempt < 12) setTimeout(() => restorePreviewOnLoad(node, state, attempt + 1), 120);
            return;
        }
        if (state.liveLoaded) return;

        const clips = Number(payload.clip_count || 0);
        const frames = Number(payload.frame_count || 0);
        state.label.textContent =
            `RESTORED PREVIEW — ${clips} clip${clips === 1 ? "" : "s"} (${frames} frames)`;

        state.currentVideoInfo = { ...payload.video };
        state.currentPreviewMeta = {
            clip_count: clips,
            frame_count: frames,
            mode: "restored",
        };
        state.currentFps = Number(payload.video?.frame_rate || state.currentFps || 24);
        state.colorTimeline = Array.isArray(payload.color_timeline) ? payload.color_timeline : [];
        state.saveButton.disabled = false;
        loadPreviewSource(node, state, mediaUrl(payload.video) + "&t=" + Date.now());
        state.restoreLoaded = true;

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                syncPlayerToNode(node, state, true);
            });
        });
    } catch (_) {
        // Startup preview is convenience only; never break workflow loading.
    } finally {
        state.restoreRequestRunning = false;
    }
}

function syncPlayerToNode(node, state, growNodeIfNeeded = false, retry = 0) {
    if (!node || !state?.widget || state.syncingPlayer) return;

    const mode = previewDomRenderMode(state.box);
    if (mode === "pending") {
        if (retry < 12) {
            requestAnimationFrame(() =>
                syncPlayerToNode(node, state, growNodeIfNeeded, retry + 1)
            );
        }
        return;
    }

    if (mode === "nodes2") {
        const currentH = Number(node.size?.[1] || 0);
        const widgetY = Number(state.widget.last_y);
        const fallbackH = Number.isFinite(widgetY) && widgetY > 0
            ? widgetY + PLAYER_MIN_HEIGHT + BOTTOM_PAD
            : PLAYER_MIN_HEIGHT + 180;

        // Recover a node that was already poisoned by the old resize feedback
        // loop, but otherwise never write Vue's allocated height back to size.
        if (
            state.lastRenderMode !== "nodes2" &&
            previewHeightIsPoisoned(currentH, fallbackH)
        ) {
            state.syncingPlayer = true;
            try {
                const rememberedLegacyH = Number(state.legacyNodeHeight);
                const targetH = (
                    Number.isFinite(rememberedLegacyH) &&
                    !previewHeightIsPoisoned(rememberedLegacyH, fallbackH)
                )
                    ? Math.max(fallbackH, rememberedLegacyH)
                    : fallbackH;
                const targetW = Math.max(
                    PLAYER_MIN_WIDTH,
                    Number(node.size?.[0] || PLAYER_MIN_WIDTH)
                );
                node.setSize([targetW, targetH]);
            } finally {
                state.syncingPlayer = false;
            }
        }

        state.lastRenderMode = "nodes2";

        // WidgetDOM.vue places the element in a flex child of an auto grid row.
        // Percentage heights are unstable during Nodes 2.0 resize and can make
        // the row collapse until WidgetDOM remounts on page refresh. Preserve an
        // intrinsic minimum and let Vue stretch the player naturally.
        state.box.style.height = "auto";
        state.box.style.minHeight = `${PLAYER_MIN_HEIGHT}px`;
        state.box.style.maxHeight = "none";
        state.box.style.flex = "1 1 auto";
        state.box.style.overflow = "visible";
        state.video.style.height = "auto";
        state.video.style.minHeight = `${Math.max(80, PLAYER_MIN_HEIGHT - LABEL_HEIGHT - PREVIEW_HEADER_GAP - 4)}px`;
        state.video.style.flex = "1 1 auto";
        return;
    }

    const widgetY = Number(state.widget.last_y);

    // LiteGraph only knows the real widget Y after layout/draw.
    if (!Number.isFinite(widgetY) || widgetY <= 0) {
        if (retry < 12) {
            requestAnimationFrame(() =>
                syncPlayerToNode(node, state, growNodeIfNeeded, retry + 1)
            );
        }
        return;
    }

    // Restore the explicit Legacy sizing contract.
    state.box.style.minHeight = "0";
    state.box.style.maxHeight = "none";
    state.box.style.flex = "0 0 auto";
    state.box.style.overflow = "hidden";
    state.video.style.minHeight = "0";
    state.video.style.flex = "0 0 auto";

    state.syncingPlayer = true;
    try {
        let nodeW = Math.max(
            PLAYER_MIN_WIDTH,
            Number(node.size?.[0] || PLAYER_MIN_WIDTH)
        );
        let nodeH = Number(node.size?.[1] || 0);
        const minimumNodeH = widgetY + PLAYER_MIN_HEIGHT + BOTTOM_PAD;
        const returningFromNodes2 = state.lastRenderMode === "nodes2";

        if (returningFromNodes2) {
            const rememberedLegacyH = Number(state.legacyNodeHeight);
            nodeH = (
                Number.isFinite(rememberedLegacyH) &&
                !previewHeightIsPoisoned(rememberedLegacyH, minimumNodeH)
            )
                ? Math.max(minimumNodeH, rememberedLegacyH)
                : minimumNodeH;
        } else if (
            state.lastRenderMode == null &&
            previewHeightIsPoisoned(nodeH, minimumNodeH)
        ) {
            nodeH = minimumNodeH;
        } else if (growNodeIfNeeded && nodeH < minimumNodeH) {
            nodeH = minimumNodeH;
        }

        if (
            nodeW !== Number(node.size?.[0]) ||
            nodeH !== Number(node.size?.[1])
        ) {
            node.setSize([nodeW, nodeH]);
        }

        const actualH = Number(node.size?.[1] || nodeH);
        const availableH = Math.max(
            PLAYER_MIN_HEIGHT,
            actualH - widgetY - BOTTOM_PAD
        );

        state.currentHeight = availableH;
        if (!previewHeightIsPoisoned(actualH, minimumNodeH)) {
            state.legacyNodeHeight = actualH;
        }
        state.lastRenderMode = "legacy";
        state.box.style.height = `${availableH}px`;
        state.video.style.height =
            `${Math.max(80, availableH - LABEL_HEIGHT - PREVIEW_HEADER_GAP - 4)}px`;
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        state.syncingPlayer = false;
    }
}

async function saveCurrentPreview(node, state) {
    const info = state?.currentVideoInfo;
    if (!info?.filename || state.saveInProgress) return;

    state.saveInProgress = true;
    const button = state.saveButton;
    const oldText = button?.textContent || "SAVE PREVIEW";
    if (button) {
        button.disabled = true;
        button.textContent = "Saving...";
    }

    try {
        let promptData = null;
        try {
            promptData = await app.graphToPrompt();
        } catch (_) {
            // Workflow metadata is still useful even if API-prompt serialization
            // fails for an unrelated custom widget.
        }

        const workflow = promptData?.workflow
            ?? app.graph?.serialize?.()
            ?? null;
        const prompt = promptData?.output ?? null;

        const response = await fetch(api.apiURL("/h3_extender/save_preview"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                owner_id: findUpstreamExtenderId(node),
                generation_mode: upstreamGenerationMode(node),
                filename: info.filename,
                subfolder: info.subfolder || "",
                type: info.type || "temp",
                fps: Number(info.frame_rate || state.currentFps || 24),
                workflow,
                prompt,
            }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Save Preview failed (${response.status}).`);
        }

        if (button) {
            button.textContent = "Saved ✓";
            setTimeout(() => {
                if (!state.saveInProgress && button.textContent === "Saved ✓") {
                    button.textContent = oldText;
                }
            }, 1800);
        }
    } catch (error) {
        console.error("MiniMax H3 Save Preview failed", error);
        if (button) button.textContent = "Save failed";
        alert(`Save Preview failed:\n${error?.message || error}`);
        setTimeout(() => {
            if (!state.saveInProgress && button) button.textContent = oldText;
        }, 1800);
    } finally {
        state.saveInProgress = false;
        if (button) button.disabled = !state.currentVideoInfo?.filename;
    }
}

function makePlayer(node) {
    if (node.__h3LivePreview) return node.__h3LivePreview;

    const box = document.createElement("div");
    box.style.width = "100%";
    box.style.height = `${PLAYER_MIN_HEIGHT}px`;
    box.style.minHeight = `${PLAYER_MIN_HEIGHT}px`;
    box.style.setProperty("--comfy-widget-min-height", `${PLAYER_MIN_HEIGHT}px`);
    box.style.boxSizing = "border-box";
    box.style.display = "flex";
    box.style.flexDirection = "column";
    box.style.padding = "4px 0 0 0";
    box.style.background = "transparent";
    box.style.overflow = "hidden";

    const header = document.createElement("div");
    header.style.height = `${LABEL_HEIGHT}px`;
    header.style.minHeight = `${LABEL_HEIGHT}px`;
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.gap = "8px";
    header.style.marginBottom = `${PREVIEW_HEADER_GAP}px`;
    header.style.overflow = "hidden";

    const label = document.createElement("div");
    label.textContent = "FULL LIVE PREVIEW";
    label.style.fontSize = "11px";
    label.style.opacity = "0.78";
    label.style.height = `${LABEL_HEIGHT}px`;
    label.style.lineHeight = `${LABEL_HEIGHT}px`;
    label.style.margin = "0";
    label.style.whiteSpace = "nowrap";
    label.style.overflow = "hidden";
    label.style.textOverflow = "ellipsis";
    label.style.flex = "1 1 auto";
    label.style.minWidth = "0";

    ensureSavePreviewButtonStyle();
    const saveButton = document.createElement("button");
    saveButton.type = "button";
    saveButton.textContent = "SAVE PREVIEW";
    saveButton.title = "Save the currently assembled preview to ComfyUI output with workflow metadata";
    saveButton.className = "h3-save-preview-button";
    saveButton.disabled = true;
    saveButton.style.flex = "0 0 auto";

    header.appendChild(label);
    header.appendChild(saveButton);

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "auto";
    video.style.display = "block";
    video.style.width = "100%";
    video.style.height = `${PLAYER_MIN_HEIGHT - LABEL_HEIGHT - PREVIEW_HEADER_GAP - 4}px`;
    video.style.flex = "1 1 auto";
    video.style.minHeight = "0";
    video.style.objectFit = "contain";
    video.style.background = "#000";
    video.style.borderRadius = "4px";

    box.appendChild(header);
    box.appendChild(video);

    const state = {
        box,
        header,
        label,
        saveButton,
        video,
        widget: null,
        currentVideoInfo: null,
        currentPreviewMeta: null,
        currentFps: 24,
        colorTimeline: [],
        saveInProgress: false,
        currentHeight: PLAYER_MIN_HEIGHT,
        syncingPlayer: false,
        lastRenderMode: null,
        legacyNodeHeight: null,
        liveLoaded: false,
        restoreLoaded: false,
        restoreRequestRunning: false,
    };

    video.addEventListener("timeupdate", () => syncPreviewColorFilter(state));
    video.addEventListener("seeked", () => syncPreviewColorFilter(state));
    video.addEventListener("loadedmetadata", () => syncPreviewColorFilter(state));
    if (typeof video.requestVideoFrameCallback === "function") {
        const colorFrameTick = () => {
            if (!box.isConnected) return;
            syncPreviewColorFilter(state);
            video.requestVideoFrameCallback(colorFrameTick);
        };
        video.requestVideoFrameCallback(colorFrameTick);
    }

    const widget = node.addDOMWidget("h3_live_preview", "preview", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => PLAYER_MIN_HEIGHT,
        getHeight: () => state.currentHeight,
        afterResize: (resizedNode) => {
            const mode = previewDomRenderMode(box);
            if (mode === "nodes2") {
                // Current WidgetDOM.vue already stretches its child. Keep only
                // an intrinsic minimum; never write a percentage height back.
                box.style.height = "auto";
                box.style.minHeight = `${PLAYER_MIN_HEIGHT}px`;
                box.style.maxHeight = "none";
                box.style.flex = "1 1 auto";
                box.style.overflow = "visible";
                video.style.height = "auto";
                video.style.minHeight = `${Math.max(80, PLAYER_MIN_HEIGHT - LABEL_HEIGHT - PREVIEW_HEADER_GAP - 4)}px`;
                video.style.flex = "1 1 auto";
                state.lastRenderMode = "nodes2";
            } else {
                requestAnimationFrame(() =>
                    syncPlayerToNode(resizedNode, state, false)
                );
            }
        },
    });
    state.widget = widget;
    saveButton.addEventListener("click", () => saveCurrentPreview(node, state));

    node.__h3LivePreview = state;

    const oldRemove = node.onRemoved;
    node.onRemoved = function () {
        try {
            video.pause();
            video.removeAttribute("src");
            video.load();
        } catch (_) {}

        if (oldRemove) oldRemove.apply(this, arguments);
    };

    // First layout: guarantee minimum useful player size.
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            syncPlayerToNode(node, state, true);
        });
    });

    return state;
}

function refreshImportedProjectPreview(ownerId) {
    const graph = app.graph;
    if (!graph) return;
    const wanted = String(ownerId);
    for (const node of graph._nodes || []) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
        if (String(findUpstreamExtenderId(node)) !== wanted) continue;

        const state = makePlayer(node);
        state.liveLoaded = false;
        state.restoreLoaded = false;
        state.restoreRequestRunning = false;
        state.currentVideoInfo = null;
        state.currentPreviewMeta = null;
        state.colorTimeline = [];
        state.video.style.filter = "none";
        state.saveButton.disabled = true;
        state.label.textContent = "PROJECT LOADED — preview will restore when cached render data is available";
        try {
            state.video.pause();
            state.video.removeAttribute("src");
            state.video.load();
        } catch (_) {}
        restorePreviewOnLoad(node, state);
    }
}

app.registerExtension({
    name: "MiniMaxH3.MotionContext.LivePreview",

    setup() {
        window.addEventListener("h3-extender-project-loaded", (event) => {
            const ownerId = event?.detail?.owner_id;
            if (ownerId == null) return;
            refreshImportedProjectPreview(ownerId);
        });
        window.addEventListener("h3-extender-color-updated", (event) => {
            const ownerId = event?.detail?.owner_id;
            const timeline = event?.detail?.color_timeline;
            if (ownerId == null || !Array.isArray(timeline)) return;
            const graph = app.graph;
            for (const node of graph?._nodes || []) {
                if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
                if (String(findUpstreamExtenderId(node)) !== String(ownerId)) continue;
                const state = makePlayer(node);
                state.colorTimeline = timeline;
                syncPreviewColorFilter(state);
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === DISK_JOIN_TARGET) {
            const oldJoinCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = oldJoinCreated
                    ? oldJoinCreated.apply(this, arguments)
                    : undefined;

                installValidationCascade(this);

                // One extra frame covers workflows restored from JSON where
                // widgets can be populated just after node creation.
                requestAnimationFrame(() => {
                    installValidationCascade(this);
                });

                return r;
            };

            return;
        }

        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = oldCreated
                ? oldCreated.apply(this, arguments)
                : undefined;

            stripFinalDecodeOutputs(this);
            hideCompatibilityWidget(this, "fps");
            const state = makePlayer(this);

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    stripFinalDecodeOutputs(this);
                    hideCompatibilityWidget(this, "fps");
                    syncPlayerToNode(this, state, true);
                    restorePreviewOnLoad(this, state);
                });
            });

            return r;
        };

        // Existing workflow JSON can restore legacy output slots after
        // onNodeCreated. Strip them again immediately after configuration.
        const oldConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = oldConfigure
                ? oldConfigure.apply(this, arguments)
                : undefined;

            stripFinalDecodeOutputs(this);
            hideCompatibilityWidget(this, "fps");
            const state = makePlayer(this);
            requestAnimationFrame(() => {
                stripFinalDecodeOutputs(this);
                hideCompatibilityWidget(this, "fps");
                restorePreviewOnLoad(this, state);
            });
            return r;
        };

        const oldExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (oldExecuted) oldExecuted.apply(this, arguments);

            const info = message?.h3_video?.[0];
            if (!info?.filename) return;

            const state = makePlayer(this);
            state.liveLoaded = true;
            const meta = message?.h3_preview_info?.[0];

            if (meta?.mode === "clip_by_clip") {
                const s = Number(meta.seam_shift || 0);
                state.label.textContent =
                    `FULL LIVE PREVIEW — ${meta.total_clips} clip${meta.total_clips > 1 ? "s" : ""} — shift ${s >= 0 ? "+" : ""}${s}`;
            } else if (meta?.mode === "full_batch") {
                state.label.textContent =
                    `FINAL PREVIEW — ${meta.total_clips} clips (${meta.preview_frames} frames)`;
            } else {
                state.label.textContent = "FULL LIVE PREVIEW";
            }

            state.currentVideoInfo = { ...info };
            state.currentPreviewMeta = {
                clip_count: Number(meta?.total_clips || 0),
                frame_count: Number(meta?.preview_frames || 0),
                mode: String(meta?.mode || ""),
            };
            state.currentFps = Number(info.frame_rate || state.currentFps || 24);
            state.colorTimeline = Array.isArray(meta?.color_timeline) ? meta.color_timeline : [];
            state.saveButton.disabled = false;
            loadPreviewSource(this, state, mediaUrl(info) + "&t=" + Date.now());

            // Do not reset a node the user already enlarged.
            // Only enforce the minimum if necessary, then fit the player
            // to the CURRENT node height.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    syncPlayerToNode(this, state, true);
                });
            });
        };
    },
});
