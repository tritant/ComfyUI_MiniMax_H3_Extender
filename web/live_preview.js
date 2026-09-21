import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MiniMaxH3MotionContextDiskFinalDecode";
const DISK_JOIN_TARGET = "MiniMaxH3MotionContextDiskJoin";
const EXTENDER_TARGET = "MiniMaxH3Extender";

let h3PreviewGraphConfiguring = false;

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
        .h3-layer-toggle {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            height: 22px;
            padding: 0 10px;
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(55, 55, 55, 0.92);
            color: rgba(255, 255, 255, 0.72);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.2px;
            line-height: 20px;
            cursor: pointer;
            flex: 0 0 auto;
            transition: background 100ms ease, border-color 100ms ease, color 100ms ease;
        }
        .h3-layer-toggle.is-active {
            border-color: rgba(120, 180, 255, 0.85);
            background: linear-gradient(180deg, rgba(55, 110, 185, 0.96), rgba(38, 82, 145, 0.96));
            color: #ffffff;
        }
        .h3-layer-toggle:disabled {
            cursor: default;
            opacity: 0.72;
        }
        .h3-layer-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #7a7a7a;
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.35);
            flex: 0 0 auto;
        }
        .h3-layer-dot.is-ready {
            background: #3dcf6a;
            box-shadow: 0 0 0 1px rgba(40, 120, 70, 0.55);
        }
    `;
    document.head.appendChild(style);
}

function stripFinalDecodeOutputs(node) {
    if (!node) return;

    // Old workflows may still serialize the nine historical Final Decode
    // outputs. Since v2.1 the node intentionally exposes one native VIDEO
    // output, so only remove legacy sockets and preserve that one.
    let keptVideo = false;
    for (let index = (node.outputs?.length || 0) - 1; index >= 0; index--) {
        const output = node.outputs[index];
        const isVideo =
            String(output?.type || "").toUpperCase() === "VIDEO" ||
            String(output?.name || "").toLowerCase() === "video";

        if (isVideo && !keptVideo) {
            keptVideo = true;
            // Normalize workflows that may have restored an older label.
            output.name = "video";
            output.type = "VIDEO";
            continue;
        }

        if (typeof node.removeOutput === "function") {
            node.removeOutput(index);
        } else {
            node.outputs.splice(index, 1);
        }
    }

    // Defensive recovery for workflow/configure paths where LiteGraph restored
    // legacy sockets after the Python node definition was applied.
    if (!keptVideo && typeof node.addOutput === "function") {
        node.addOutput("video", "VIDEO");
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

function ensureLatentUpscaleWidgetDefaults(node) {
    // Newly appended Final Decode combos are '' in older workflow JSON until
    // the user touches them. Coerce to safe defaults so prompt validation passes.
    // Also repair shifted widgets_values junk (e.g. ai_skip_first=1 → latent_layer,
    // fp16 → latent_upscale_model).
    const layer = getWidget(node, "latent_layer");
    if (layer) {
        const text = String(layer.value ?? "").trim().toLowerCase();
        if (!["auto", "draft", "refine"].includes(text)) {
            layer.value = "auto";
        }
    }
    const model = getWidget(node, "latent_upscale_model");
    if (model) {
        const raw = String(model.value ?? "").trim();
        const low = raw.toLowerCase();
        const junk = (
            raw === ""
            || low === "null"
            || ["auto", "draft", "refine", "fp16", "bf16", "fp32"].includes(low)
            || (!Number.isNaN(Number(raw)) && raw !== "")
        );
        if (junk) {
            model.value = "None";
        } else {
            const options = Array.isArray(model.options) ? model.options.map(String) : null;
            if (options && options.length && !options.includes(raw)) {
                model.value = "None";
            }
        }
    }
    const precision = getWidget(node, "latent_upscale_precision");
    if (precision) {
        const text = String(precision.value ?? "").trim().toLowerCase();
        if (!["fp16", "bf16", "fp32"].includes(text)) {
            precision.value = "bf16";
        }
    }
    const mp = getWidget(node, "latent_upscale_megapixels");
    if (mp) {
        const n = Number(mp.value);
        if (mp.value === "" || mp.value == null || Number.isNaN(n) || n < 0.1 || n > 8) {
            mp.value = 1.2;
        }
    }
    ensureStitchWidgetDefaults(node);
}

const STITCH_DEFAULTS = {
    ref_frames_offset: 20,
    rife_multiplier: 2,
    rife_ckpt: "rife49.pth",
    rife_fast_mode: false,
    rife_ensemble: true,
    ai_skip_first: 1,
};

const STITCH_WIDGET_NAMES = [
    "ref_frames_offset",
    "rife_multiplier",
    "rife_ckpt",
    "rife_fast_mode",
    "rife_ensemble",
    "ai_skip_first",
];

function isMissingWidgetValue(value) {
    if (value === "" || value === null || value === undefined) return true;
    // Booleans are valid for checkbox widgets. String "true"/"false" is bad when
    // a shifted widgets_values entry lands in a combo/number field.
    if (typeof value === "boolean") return false;
    const s = String(value).trim().toLowerCase();
    return s === "true" || s === "false" || s === "none" || s === "null";
}

function hideFinalDecodeGenerationWidgets(node) {
    // Generation / layer controls live on the Extender (Latent refine section).
    // Final Decode keeps the values for serialization + decode, but the player UI
    // should stay play-only.
    // Soft-hide only — never converted-widget (that shifts widgets_values and
    // used to land ai_skip_first=1 into latent_layer).
    for (const name of [
        "latent_layer",
        "latent_upscale_model",
        "latent_upscale_megapixels",
        "latent_upscale_precision",
        ...STITCH_WIDGET_NAMES,
        "stitch_json",
    ]) {
        softHideSerializedWidget(node, name);
    }
}

function softHideSerializedWidget(node, name) {
    const widget = getWidget(node, name);
    if (!widget || widget.__h3SoftHidden) return;
    widget.__h3SoftHidden = true;
    widget.hidden = true;
    const prevCompute = widget.computeSize?.bind(widget);
    widget.computeSize = () => [0, -4];
    widget.__h3PrevComputeSize = prevCompute;
    for (const element of [widget.element, widget.inputEl, widget.domWidget?.element]) {
        if (element?.style) element.style.display = "none";
    }
}

function defaultStitchState() {
    return { ...STITCH_DEFAULTS };
}

function stitchJsonLooksValid(parsed) {
    if (!parsed || typeof parsed !== "object") return false;
    // At least one known numeric key must be present — empty {} is not authoritative.
    return STITCH_WIDGET_NAMES.some((name) => Object.prototype.hasOwnProperty.call(parsed, name));
}

function readStitchState(node) {
    const out = defaultStitchState();
    const jsonWidget = getWidget(node, "stitch_json");
    if (jsonWidget && typeof jsonWidget.value === "string" && jsonWidget.value.trim()) {
        try {
            const parsed = JSON.parse(jsonWidget.value);
            if (stitchJsonLooksValid(parsed)) {
                Object.assign(out, parsed);
                return out;
            }
        } catch (_) {}
    }
    // First load / old graphs: migrate from native widgets when they look valid.
    for (const name of STITCH_WIDGET_NAMES) {
        const widget = getWidget(node, name);
        if (!widget) continue;
        if (name === "rife_ckpt") {
            const cur = String(widget.value ?? "").trim();
            if (cur.toLowerCase().endsWith(".pth") && typeof widget.value !== "boolean") {
                out.rife_ckpt = cur;
            }
            continue;
        }
        if (name === "rife_fast_mode" || name === "rife_ensemble") {
            if (typeof widget.value === "boolean") out[name] = widget.value;
            else if (!isMissingWidgetValue(widget.value)) out[name] = coerceWidgetBool(widget.value);
            continue;
        }
        const n = Number(widget.value);
        if (Number.isFinite(n) && typeof widget.value !== "boolean") out[name] = n;
    }
    return out;
}

function writeStitchState(node, state) {
    const next = { ...defaultStitchState(), ...(state || {}) };
    next.ref_frames_offset = Math.max(0, Math.min(256, Number(next.ref_frames_offset) || 0));
    next.ai_skip_first = Math.max(0, Math.min(16, Number(next.ai_skip_first) || 0));
    let mult = Number(next.rife_multiplier) || 2;
    if (mult % 2 !== 0) mult += 1;
    next.rife_multiplier = Math.max(2, Math.min(8, mult));
    next.rife_fast_mode = Boolean(next.rife_fast_mode);
    next.rife_ensemble = Boolean(next.rife_ensemble);
    next.rife_ckpt = String(next.rife_ckpt || STITCH_DEFAULTS.rife_ckpt);

    const jsonWidget = getWidget(node, "stitch_json");
    if (jsonWidget) {
        jsonWidget.value = JSON.stringify(next);
        if (typeof jsonWidget.callback === "function") {
            try { jsonWidget.callback(jsonWidget.value); } catch (_) {}
        }
    }
    for (const name of STITCH_WIDGET_NAMES) {
        const widget = getWidget(node, name);
        if (!widget) continue;
        setWidgetValueFromUi(widget, next[name]);
    }
    return next;
}

function ensureStitchWidgetDefaults(node) {
    // Migrate / repair once from widgets → stitch_json, then trust stitch_json.
    const state = readStitchState(node);
    if (typeof state.rife_ckpt === "boolean" || !String(state.rife_ckpt).toLowerCase().endsWith(".pth")) {
        state.rife_ckpt = STITCH_DEFAULTS.rife_ckpt;
    }
    writeStitchState(node, state);
}

function ensureStitchSectionStyles() {
    if (document.getElementById("h3-final-stitch-style")) return;
    const style = document.createElement("style");
    style.id = "h3-final-stitch-style";
    style.textContent = `
        .h3-final-stitch {
            margin: 0 0 8px;
            border: 1px solid rgba(140, 210, 155, 0.42);
            border-radius: 7px;
            background: rgba(70, 150, 95, 0.10);
            overflow: hidden;
            flex: 0 0 auto;
            min-width: 0;
        }
        .h3-final-stitch.is-off {
            border-color: rgba(255, 255, 255, 0.16);
            background: rgba(0, 0, 0, 0.18);
        }
        .h3-final-stitch-head {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            padding: 7px 9px;
            border: 0;
            background: rgba(255, 255, 255, 0.04);
            color: inherit;
            cursor: pointer;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.2px;
            text-align: left;
        }
        .h3-final-stitch-head:hover { background: rgba(255, 255, 255, 0.08); }
        .h3-final-stitch-chevron { opacity: 0.7; width: 12px; flex: 0 0 auto; }
        .h3-final-stitch-badge {
            margin-left: auto;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.3px;
            padding: 1px 7px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            opacity: 0.9;
            flex: 0 0 auto;
        }
        .h3-final-stitch-badge.on {
            border-color: rgba(140, 210, 155, 0.55);
            background: rgba(70, 150, 95, 0.25);
            color: rgba(190, 245, 205, 0.98);
        }
        .h3-final-stitch-badge.off {
            background: rgba(255, 255, 255, 0.06);
            color: rgba(255, 255, 255, 0.62);
        }
        .h3-final-stitch-body {
            display: none;
            padding: 7px 9px 9px;
            gap: 5px;
            flex-direction: column;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .h3-final-stitch.open .h3-final-stitch-body { display: flex; }
        .h3-final-stitch-hint {
            font-size: 10px;
            opacity: 0.72;
            line-height: 1.35;
            margin: 0 0 3px;
        }
        .h3-final-stitch-row {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }
        .h3-final-stitch-row label {
            flex: 0 0 128px;
            font-size: 10px;
            opacity: 0.78;
        }
        .h3-final-stitch-row select,
        .h3-final-stitch-row input[type="number"] {
            flex: 1 1 auto;
            min-width: 0;
            font-size: 11px;
        }
        .h3-final-stitch-row input[type="checkbox"] {
            width: 14px;
            height: 14px;
        }
        .h3-final-stitch.is-off .h3-final-stitch-body {
            opacity: 0.55;
            pointer-events: none;
        }
        .h3-final-toolbar {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            width: 100%;
            height: 22px;
            min-height: 22px;
            margin-top: 8px;
            flex: 0 0 auto;
            overflow: hidden;
        }
    `;
    document.head.appendChild(style);
}

function coerceWidgetBool(value) {
    if (value === true || value === 1) return true;
    if (value === false || value === 0 || value == null) return false;
    if (typeof value === "string") {
        const s = value.trim().toLowerCase();
        if (s === "true" || s === "1" || s === "yes" || s === "on") return true;
        if (s === "false" || s === "0" || s === "no" || s === "off" || s === "") return false;
    }
    return Boolean(value);
}

function setWidgetValueFromUi(widget, value) {
    if (!widget) return;
    widget.value = value;
    if (typeof widget.callback === "function") {
        try { widget.callback(value); } catch (_) {}
    }
}

function widgetComboValues(widget) {
    const values = widget?.options?.values;
    if (Array.isArray(values)) return values.map((v) => String(v));
    if (values && typeof values === "object") return Object.keys(values).map(String);
    return [];
}

function createStitchSelectRow(node, labelText, key, widget) {
    const row = document.createElement("div");
    row.className = "h3-final-stitch-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    const select = document.createElement("select");
    const opts = widgetComboValues(widget);
    for (const value of opts) {
        const opt = document.createElement("option");
        opt.value = String(value);
        opt.textContent = String(value);
        select.appendChild(opt);
    }
    if (!opts.includes(String(widget?.value ?? "")) && widget?.value != null) {
        const opt = document.createElement("option");
        opt.value = String(widget.value);
        opt.textContent = String(widget.value);
        select.appendChild(opt);
    }
    select.value = String(widget?.value ?? opts[0] ?? "");
    select.addEventListener("change", () => {
        const next = readStitchState(node);
        next[key] = select.value;
        writeStitchState(node, next);
        select.value = String(readStitchState(node)[key] ?? "");
    });
    row.append(label, select);
    row.__h3Select = select;
    row.__h3Key = key;
    row.__h3Widget = widget;
    return row;
}

function createStitchNumberRow(node, labelText, key, widget, { min = null, max = null, step = null } = {}) {
    const row = document.createElement("div");
    row.className = "h3-final-stitch-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    const input = document.createElement("input");
    input.type = "number";
    if (min != null) input.min = String(min);
    if (max != null) input.max = String(max);
    if (step != null) input.step = String(step);
    input.value = String(widget?.value ?? "");
    input.addEventListener("change", () => {
        const n = Number(input.value);
        const next = readStitchState(node);
        if (Number.isFinite(n)) next[key] = n;
        writeStitchState(node, next);
        input.value = String(readStitchState(node)[key] ?? "");
    });
    row.append(label, input);
    row.__h3Input = input;
    row.__h3Key = key;
    row.__h3Widget = widget;
    return row;
}

function createStitchCheckboxRow(node, labelText, key, widget) {
    const row = document.createElement("div");
    row.className = "h3-final-stitch-row";
    const label = document.createElement("label");
    label.textContent = labelText;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = coerceWidgetBool(widget?.value);
    input.addEventListener("change", () => {
        const next = readStitchState(node);
        next[key] = Boolean(input.checked);
        writeStitchState(node, next);
        input.checked = Boolean(readStitchState(node)[key]);
    });
    row.append(label, input);
    row.__h3Input = input;
    row.__h3Key = key;
    row.__h3Widget = widget;
    return row;
}

function isOriginalImagesConnected(node) {
    const input = (node?.inputs || []).find((i) => String(i?.name || "") === "original_images");
    return input?.link != null;
}

function syncStitchSection(node, state) {
    if (!state?.stitchSection) return;
    ensureStitchWidgetDefaults(node);
    ensureStitchRows(node, state);
    const active = isOriginalImagesConnected(node);
    state.stitchSection.classList.toggle("is-off", !active);
    if (state.stitchBadge) {
        state.stitchBadge.textContent = active ? "ON" : "OFF";
        state.stitchBadge.className = "h3-final-stitch-badge " + (active ? "on" : "off");
    }
    if (state.stitchTitle) {
        state.stitchTitle.textContent = active
            ? "Seamless Stitch (RIFE)"
            : "Seamless Stitch (RIFE) — connect original_images";
    }
    if (state.stitchHint) {
        state.stitchHint.textContent = active
            ? "Exports original[:cut] + RIFE bridge + AI. Soft-calls Comfyui-MinimaxUtils; no separate stitcher node needed."
            : "Connect Load Video → original_images (left socket) to enable. Without it, Final Decode exports AI-only as usual.";
    }

    // Mirror stitch_json (source of truth) into DOM — not the fragile native widgets alone.
    const stitch = readStitchState(node);
    for (const row of state.stitchRows || []) {
        const key = row.__h3Key;
        if (!key) continue;
        const value = stitch[key];
        const widget = row.__h3Widget;
        if (row.__h3Select) {
            const opts = widgetComboValues(widget);
            const shown = String(value ?? "");
            if (opts.length && !opts.includes(shown)) {
                let found = false;
                for (const child of row.__h3Select.options) {
                    if (child.value === shown) { found = true; break; }
                }
                if (!found && shown && !isMissingWidgetValue(shown)) {
                    const opt = document.createElement("option");
                    opt.value = shown;
                    opt.textContent = shown;
                    row.__h3Select.appendChild(opt);
                }
            }
            row.__h3Select.value = shown;
        }
        if (row.__h3Input) {
            if (row.__h3Input.type === "checkbox") {
                row.__h3Input.checked = Boolean(value);
            } else {
                row.__h3Input.value = String(value ?? "");
            }
        }
    }
}

function ensureStitchRows(node, state) {
    if (!state?.stitchSection) return;
    if ((state.stitchRows || []).length) return;
    const body = state.stitchSection.querySelector(".h3-final-stitch-body");
    if (!body) return;

    const rows = [
        createStitchNumberRow(node, "Ref frames offset", "ref_frames_offset", getWidget(node, "ref_frames_offset"), { min: 0, max: 256, step: 1 }),
        createStitchNumberRow(node, "AI skip first", "ai_skip_first", getWidget(node, "ai_skip_first"), { min: 0, max: 16, step: 1 }),
        createStitchNumberRow(node, "RIFE multiplier", "rife_multiplier", getWidget(node, "rife_multiplier"), { min: 2, max: 8, step: 2 }),
        createStitchSelectRow(node, "RIFE checkpoint", "rife_ckpt", getWidget(node, "rife_ckpt")),
        createStitchCheckboxRow(node, "RIFE fast mode", "rife_fast_mode", getWidget(node, "rife_fast_mode")),
        createStitchCheckboxRow(node, "RIFE ensemble", "rife_ensemble", getWidget(node, "rife_ensemble")),
    ].filter((row) => row.__h3Widget);

    if (!rows.length) return;
    for (const row of rows) body.appendChild(row);
    state.stitchRows = rows;
}

function buildStitchSection(node, state) {
    if (state.stitchSection) {
        syncStitchSection(node, state);
        return state.stitchSection;
    }
    ensureStitchSectionStyles();

    const section = document.createElement("div");
    section.className = "h3-final-stitch open";

    const head = document.createElement("button");
    head.type = "button";
    head.className = "h3-final-stitch-head";
    const chevron = document.createElement("span");
    chevron.className = "h3-final-stitch-chevron";
    chevron.textContent = "▾";
    const title = document.createElement("span");
    title.textContent = "Seamless Stitch (RIFE)";
    const badge = document.createElement("span");
    badge.className = "h3-final-stitch-badge off";
    badge.textContent = "OFF";
    head.append(chevron, title, badge);
    head.addEventListener("click", () => {
        const next = !section.classList.contains("open");
        section.classList.toggle("open", next);
        chevron.textContent = next ? "▾" : "▸";
        requestAnimationFrame(() => syncPlayerToNode(node, state, true));
    });

    const body = document.createElement("div");
    body.className = "h3-final-stitch-body";
    const hint = document.createElement("p");
    hint.className = "h3-final-stitch-hint";

    const refWidget = getWidget(node, "ref_frames_offset");
    const multWidget = getWidget(node, "rife_multiplier");
    const ckptWidget = getWidget(node, "rife_ckpt");
    const fastWidget = getWidget(node, "rife_fast_mode");
    const ensembleWidget = getWidget(node, "rife_ensemble");
    const skipWidget = getWidget(node, "ai_skip_first");

    const rows = [
        createStitchNumberRow(node, "Ref frames offset", "ref_frames_offset", refWidget, { min: 0, max: 256, step: 1 }),
        createStitchNumberRow(node, "AI skip first", "ai_skip_first", skipWidget, { min: 0, max: 16, step: 1 }),
        createStitchNumberRow(node, "RIFE multiplier", "rife_multiplier", multWidget, { min: 2, max: 8, step: 2 }),
        createStitchSelectRow(node, "RIFE checkpoint", "rife_ckpt", ckptWidget),
        createStitchCheckboxRow(node, "RIFE fast mode", "rife_fast_mode", fastWidget),
        createStitchCheckboxRow(node, "RIFE ensemble", "rife_ensemble", ensembleWidget),
    ].filter((row) => row.__h3Widget);

    body.append(hint, ...rows);
    section.append(head, body);

    state.stitchSection = section;
    state.stitchWrap = section;
    state.stitchBadge = badge;
    state.stitchTitle = title;
    state.stitchHint = hint;
    state.stitchRows = rows;
    state.stitchChevron = chevron;

    syncStitchSection(node, state);
    return section;
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
const CLIP_STRIP_HEIGHT = 26;
const CLIP_STRIP_GAP = 6;
const STITCH_LABEL_HEIGHT = 14;
const STITCH_LABEL_GAP = 3;
const TOOLBAR_HEIGHT = 22;
const TOOLBAR_GAP = 8;
const BOTTOM_PAD = 14;

// Accent colors for clip borders (fills stay dark gray).
const CLIP_STRIP_PALETTE = [
    "#3d8bfd",
    "#5cb85c",
    "#f0ad4e",
    "#d9534f",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#3498db",
    "#e74c3c",
    "#2ecc71",
];

function ensureClipStripStyles() {
    if (document.getElementById("h3-clip-strip-style")) return;
    const style = document.createElement("style");
    style.id = "h3-clip-strip-style";
    style.textContent = `
        .h3-clip-strip-host {
            display: none;
            width: 100%;
            margin-top: ${CLIP_STRIP_GAP}px;
            flex: 0 0 auto;
            min-width: 0;
        }
        .h3-clip-strip {
            position: relative;
            display: flex;
            width: 100%;
            height: ${CLIP_STRIP_HEIGHT}px;
            min-height: ${CLIP_STRIP_HEIGHT}px;
            box-sizing: border-box;
            border-radius: 4px;
            overflow: hidden;
            background: rgba(18, 18, 18, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .h3-clip-seg {
            margin: 0;
            padding: 0 6px;
            min-width: 0;
            height: 100%;
            border: 1px solid transparent;
            border-right: 1px solid rgba(0, 0, 0, 0.45);
            border-radius: 0;
            background: #2a2a2a;
            color: rgba(255, 255, 255, 0.82);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.2px;
            line-height: ${CLIP_STRIP_HEIGHT - 2}px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            cursor: pointer;
            box-sizing: border-box;
            position: relative;
            z-index: 1;
        }
        .h3-clip-seg.is-original {
            background: #222;
            color: rgba(255, 255, 255, 0.55);
            cursor: pointer;
            font-weight: 600;
        }
        .h3-clip-seg.is-active {
            background: #3a3a3a;
            color: #fff;
            z-index: 2;
        }
        .h3-stitch-overlay {
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 3;
        }
        .h3-stitch-fill {
            position: absolute;
            top: 0;
            bottom: 0;
            background: rgba(70, 190, 110, 0.28);
        }
        .h3-stitch-mark {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            margin-left: -1px;
            background: rgba(90, 220, 130, 0.95);
            box-shadow: 0 0 0 1px rgba(20, 60, 30, 0.35);
        }
        .h3-stitch-label {
            display: none;
            margin-top: ${STITCH_LABEL_GAP}px;
            height: ${STITCH_LABEL_HEIGHT}px;
            line-height: ${STITCH_LABEL_HEIGHT}px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.15px;
            color: rgba(140, 220, 155, 0.88);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    `;
    document.head.appendChild(style);
}

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

function setLegacyPreviewWidgetFullWidth(state, enabled) {
    const widget = state?.widget;
    if (!widget) return;

    if (enabled) {
        if (state.legacyWidthPinInstalled) return;
        try {
            state.legacyWidthOwnDescriptor = Object.getOwnPropertyDescriptor(widget, "width") || null;
            Object.defineProperty(widget, "width", {
                configurable: true,
                enumerable: state.legacyWidthOwnDescriptor?.enumerable ?? true,
                get: () => undefined,
                set: () => {},
            });
            state.legacyWidthPinInstalled = true;
        } catch (_) {
            // Best-effort workaround for the upstream Legacy DOM-widget width bug.
        }
        return;
    }

    if (!state.legacyWidthPinInstalled) return;
    try {
        const previous = state.legacyWidthOwnDescriptor;
        if (previous) Object.defineProperty(widget, "width", previous);
        else delete widget.width;
    } catch (_) {}
    state.legacyWidthPinInstalled = false;
    state.legacyWidthOwnDescriptor = null;
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

function applyLayerToggleUi(state) {
    if (!state) return;
    const active = String(state.activeLayer || "draft") === "refine" ? "refine" : "draft";
    const layers = state.layerStatus || {};
    const draftReady = Boolean(layers?.draft?.ready);
    const refineReady = Boolean(layers?.refine?.ready);

    if (state.lowresDot) state.lowresDot.classList.toggle("is-ready", draftReady);
    if (state.refinedDot) state.refinedDot.classList.toggle("is-ready", refineReady);
    if (state.lowresButton) {
        state.lowresButton.classList.toggle("is-active", active === "draft");
        state.lowresButton.title = draftReady
            ? "Show lowres (draft) preview"
            : "Lowres preview not ready — queue Final Decode with draft/auto";
    }
    if (state.refinedButton) {
        state.refinedButton.classList.toggle("is-active", active === "refine");
        const count = Number(layers?.refine?.clip_count || 0);
        state.refinedButton.title = refineReady
            ? `Show refined preview (${count} clip${count === 1 ? "" : "s"})`
            : "Refined preview not ready — run refine + Final Decode with layer=refine";
    }
}

async function refreshLayerStatus(node, state) {
    if (!node || !state) return;
    const ownerId = findUpstreamExtenderId(node);
    if (ownerId == null) return;
    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(ownerId));
        params.set("final_id", String(node.id));
        params.set("layer", String(state.activeLayer || "draft"));
        params.set("mode", upstreamGenerationMode(node));
        params.set("status_only", "1");
        const response = await fetch(
            api.apiURL("/h3_extender/layer_preview?" + params.toString())
        );
        if (!response.ok) return;
        const payload = await response.json();
        if (payload?.layers) {
            const prev = state.layerStatus || {};
            const next = payload.layers;
            const active = String(state.activeLayer || "draft");
            const liveReady = Boolean(state.liveLoaded && state.currentVideoInfo?.filename);
            state.layerStatus = {
                draft: {
                    ...(prev.draft || {}),
                    ...(next.draft || {}),
                    ready: Boolean(next?.draft?.ready) || (liveReady && active === "draft"),
                },
                refine: {
                    ...(prev.refine || {}),
                    ...(next.refine || {}),
                    ready: Boolean(next?.refine?.ready) || (liveReady && active === "refine"),
                },
            };
        }
        applyLayerToggleUi(state);
    } catch (_) {
        // Layer status is convenience only.
    }
}

async function switchPreviewLayer(node, state, layer) {
    if (!node || !state) return;
    const next = String(layer || "draft") === "refine" ? "refine" : "draft";
    if (next === String(state.activeLayer || "draft") && state.currentVideoInfo?.filename) {
        applyLayerToggleUi(state);
        return;
    }

    const ready = Boolean(state.layerStatus?.[next]?.ready);
    if (!ready) {
        // Refresh once in case Final Decode just finished.
        await refreshLayerStatus(node, state);
        if (!state.layerStatus?.[next]?.ready) {
            applyLayerToggleUi(state);
            return;
        }
    }

    const ownerId = findUpstreamExtenderId(node);
    if (ownerId == null) return;
    if (state.layerSwitchRunning) return;
    state.layerSwitchRunning = true;
    try {
        const params = new URLSearchParams();
        params.set("owner_id", String(ownerId));
        params.set("final_id", String(node.id));
        params.set("layer", next);
        params.set("mode", upstreamGenerationMode(node));
        const response = await fetch(
            api.apiURL("/h3_extender/layer_preview?" + params.toString())
        );
        if (!response.ok) return;
        const payload = await response.json();
        if (payload?.layers) state.layerStatus = payload.layers;
        if (!payload?.ok || !payload?.video?.filename) {
            applyLayerToggleUi(state);
            return;
        }

        const clips = Number(payload.clip_count || 0);
        const frames = Number(payload.frame_count || 0);
        const labelBit = next === "refine" ? "REFINED" : "LOWRES";
        const baseLabel =
            `${labelBit} PREVIEW — ${clips} clip${clips === 1 ? "" : "s"} (${frames} frames)`;

        state.activeLayer = next;
        state.currentVideoInfo = { ...payload.video };
        state.currentPreviewMeta = {
            clip_count: clips,
            frame_count: frames,
            mode: String(payload.cache_mode || next),
            active_layer: next,
        };
        state.currentFps = Number(payload.video?.frame_rate || state.currentFps || 24);
        // Layer cache previews are AI-only; clear any previous stitch overlay.
        state.stitchMeta = null;
        setPreviewTimeline(node, state, payload.color_timeline, baseLabel);
        state.saveButton.disabled = false;
        loadPreviewSource(node, state, mediaUrl(payload.video) + "&t=" + Date.now());
        applyLayerToggleUi(state);
        requestAnimationFrame(() => {
            requestAnimationFrame(() => syncPlayerToNode(node, state, true));
        });
    } catch (_) {
        // Toggle failure must never break the player.
    } finally {
        state.layerSwitchRunning = false;
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

function clipStripColor(index) {
    return CLIP_STRIP_PALETTE[Math.abs(Number(index) || 0) % CLIP_STRIP_PALETTE.length];
}

function timelineItemAtTime(timeline, time) {
    const items = timeline || [];
    if (!items.length) return null;
    const t = Number(time || 0);
    for (const item of items) {
        const start = Number(item?.start || 0);
        const end = Number(item?.end || start);
        if (t >= start && t < end) return item;
    }
    const last = items[items.length - 1];
    if (t >= Number(last?.start || 0)) return last;
    return items[0];
}

function stripChromeHeight(state) {
    const hasStrip = (state?.colorTimeline || []).length > 0
        || Boolean(state?.stitchMeta?.total_frames);
    if (!hasStrip) return 0;
    let h = CLIP_STRIP_HEIGHT + CLIP_STRIP_GAP;
    if (state?.stitchMeta?.total_frames) h += STITCH_LABEL_HEIGHT + STITCH_LABEL_GAP;
    return h;
}

function stitchChromeHeight(state) {
    if (!state?.stitchWrap) return 0;
    const measured = Number(state.stitchWrap.offsetHeight || 0);
    if (measured > 0) return measured + 2;
    // Fallback before first layout: closed header ~34px, open panel ~210px.
    return state.stitchSection?.classList.contains("open") ? 210 : 34;
}

function toolbarChromeHeight() {
    return TOOLBAR_HEIGHT + TOOLBAR_GAP;
}

function effectivePlayerMinHeight(state) {
    return (
        PLAYER_MIN_HEIGHT
        + stripChromeHeight(state)
        + stitchChromeHeight(state)
        + toolbarChromeHeight()
    );
}

function videoChromePadding(state) {
    return (
        stitchChromeHeight(state)
        + LABEL_HEIGHT
        + PREVIEW_HEADER_GAP
        + stripChromeHeight(state)
        + toolbarChromeHeight()
        + 4
    );
}

function upstreamClipNames(node) {
    const origin = findUpstreamExtenderNode(node);
    const runtimeClips = origin?.__h3Extender?.state?.clips;
    if (Array.isArray(runtimeClips) && runtimeClips.length) {
        return runtimeClips.map((clip) => String(clip?.name || "").trim());
    }

    const clipsWidget = (origin?.widgets || []).find((w) => w?.name === "clips_json");
    if (typeof clipsWidget?.value === "string" && clipsWidget.value) {
        try {
            const parsed = JSON.parse(clipsWidget.value);
            const clips = Array.isArray(parsed?.clips) ? parsed.clips : parsed;
            if (Array.isArray(clips)) {
                return clips.map((clip) => String(clip?.name || "").trim());
            }
        } catch (_) {}
    }
    return [];
}

function clipStripLabel(index, names) {
    const name = String(names?.[index] || "").trim();
    return name ? `Clip ${index + 1}: ${name}` : `Clip ${index + 1}`;
}

function formatPreviewLabel(state, activeItem) {
    const base = String(state?.baseLabel || "FULL LIVE PREVIEW");
    if (!activeItem) return base;
    const index = Number(activeItem.index);
    if (!Number.isFinite(index)) return base;
    return `${base}  ·  ${clipStripLabel(index, state.clipNames)}`;
}

function syncClipStripActive(state) {
    if (!state?.video) return;

    const item = timelineItemAtTime(state.colorTimeline, state.video.currentTime);
    const t = Number(state.video.currentTime || 0);
    const stitch = state.stitchMeta;
    let activeIndex = item == null ? -1 : Number(item.index);
    // While the playhead is still in the prepended original, no AI clip is active.
    if (stitch && Number(stitch.original_kept) > 0) {
        const fps = Number(stitch.fps || state.currentFps || 24) || 24;
        const originalEnd = Number(stitch.original_kept) / fps;
        if (t < originalEnd) activeIndex = -2; // original region
    }
    if (state.stripActiveIndex !== activeIndex) {
        state.stripActiveIndex = activeIndex;
        for (const seg of state.stripSegments || []) {
            const isOriginal = seg.dataset.clipIndex === "original";
            const isActive = isOriginal
                ? activeIndex === -2
                : Number(seg.dataset.clipIndex) === activeIndex;
            seg.classList.toggle("is-active", isActive);
            if (!isOriginal) {
                const accent = clipStripColor(Number(seg.dataset.clipIndex) || 0);
                seg.style.borderLeft = `3px solid ${accent}`;
                seg.style.boxShadow = isActive
                    ? `inset 0 0 0 1px ${accent}`
                    : "none";
            } else {
                seg.style.borderLeft = "3px solid rgba(255,255,255,0.28)";
                seg.style.boxShadow = isActive
                    ? "inset 0 0 0 1px rgba(255,255,255,0.35)"
                    : "none";
            }
        }
    }

    if (state.label) {
        let labelItem = item;
        if (activeIndex === -2) {
            labelItem = { index: -1, label: "Original" };
        }
        const next = activeIndex === -2
            ? `${String(state?.baseLabel || "FULL LIVE PREVIEW")}  ·  Original`
            : formatPreviewLabel(state, labelItem);
        if (state.label.textContent !== next) state.label.textContent = next;
    }
}

function stitchRegionSeconds(stitch, fpsFallback) {
    if (!stitch) return null;
    const fps = Number(stitch.fps || fpsFallback || 24) || 24;
    const total = Math.max(1, Number(stitch.total_frames) || 0);
    const startFrame = Math.max(0, Number(stitch.bridge_start_frame ?? stitch.original_kept) || 0);
    const endFrame = Math.max(
        startFrame,
        Number(stitch.bridge_end_frame ?? (startFrame + Number(stitch.bridge || 0))) || startFrame
    );
    return {
        fps,
        total,
        startFrame,
        endFrame,
        startSec: startFrame / fps,
        endSec: endFrame / fps,
        bridgeFrames: Math.max(0, endFrame - startFrame),
    };
}

function formatStitchLabel(region) {
    if (!region) return "";
    const a = region.startSec.toFixed(2);
    const b = region.endSec.toFixed(2);
    const n = region.bridgeFrames;
    if (n <= 0) return `seam @ ${a}s`;
    return `stitch @ ${a}s–${b}s (${n}f)`;
}

function rebuildClipStrip(node, state) {
    if (!state?.strip) return;
    ensureClipStripStyles();

    const timeline = Array.isArray(state.colorTimeline) ? state.colorTimeline : [];
    const stitch = state.stitchMeta && Number(state.stitchMeta.total_frames) > 0
        ? state.stitchMeta
        : null;
    state.clipNames = upstreamClipNames(node);
    state.strip.innerHTML = "";
    state.stripSegments = [];
    state.stripActiveIndex = -1;
    if (state.stitchOverlay) {
        state.stitchOverlay.innerHTML = "";
        state.stitchOverlay.style.display = "none";
    }
    if (state.stitchLabel) {
        state.stitchLabel.textContent = "";
        state.stitchLabel.style.display = "none";
    }

    if (!timeline.length && !stitch) {
        if (state.stripHost) state.stripHost.style.display = "none";
        else state.strip.style.display = "none";
        syncClipStripActive(state);
        return;
    }

    if (state.stripHost) state.stripHost.style.display = "block";
    state.strip.style.display = "flex";

    const region = stitchRegionSeconds(stitch, state.currentFps);
    const fps = region?.fps || Number(state.currentFps || 24) || 24;

    // Prefer stitched total duration so the original prefix + clips share one scale.
    let total = 0;
    if (region) {
        total = region.total / fps;
    } else {
        total = timeline.reduce((sum, item) => {
            const start = Number(item?.start || 0);
            const end = Number(item?.end || start);
            return sum + Math.max(0, end - start);
        }, 0);
    }
    total = Math.max(0.001, total);

    if (region && Number(stitch.original_kept) > 0) {
        const origDur = Number(stitch.original_kept) / fps;
        const seg = document.createElement("button");
        seg.type = "button";
        seg.className = "h3-clip-seg is-original";
        seg.dataset.clipIndex = "original";
        seg.title = `Original  (0.0s–${origDur.toFixed(1)}s)`;
        seg.textContent = "Original";
        seg.style.flex = `${Math.max(origDur / total, 0.04)} 1 0`;
        seg.style.borderLeft = "3px solid rgba(255,255,255,0.28)";
        seg.addEventListener("click", (event) => {
            if (!state.video) return;
            const rect = seg.getBoundingClientRect();
            const width = Math.max(1, rect.width);
            const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / width));
            const target = ratio * origDur;
            const durationLimit = Number(state.video.duration);
            const clamped = Number.isFinite(durationLimit) && durationLimit > 0
                ? Math.min(Math.max(0, target), Math.max(0, durationLimit - 0.001))
                : Math.max(0, target);
            try { state.video.currentTime = clamped; } catch (_) {}
            syncClipStripActive(state);
            syncPreviewColorFilter(state);
        });
        state.strip.appendChild(seg);
        state.stripSegments.push(seg);
    }

    for (const item of timeline) {
        const index = Number(item?.index || 0);
        const start = Number(item?.start || 0);
        const end = Number(item?.end || start);
        const duration = Math.max(0, end - start);
        const accent = clipStripColor(index);
        const seg = document.createElement("button");
        seg.type = "button";
        seg.className = "h3-clip-seg";
        seg.dataset.clipIndex = String(index);
        seg.title = `${clipStripLabel(index, state.clipNames)}  (${start.toFixed(1)}s–${end.toFixed(1)}s)`;
        seg.textContent = clipStripLabel(index, state.clipNames);
        seg.style.flex = `${Math.max(duration / total, 0.04)} 1 0`;
        seg.style.borderLeft = `3px solid ${accent}`;

        seg.addEventListener("click", (event) => {
            if (!state.video) return;
            const rect = seg.getBoundingClientRect();
            const width = Math.max(1, rect.width);
            const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / width));
            const target = start + ratio * Math.max(0, end - start);
            const durationLimit = Number(state.video.duration);
            const clamped = Number.isFinite(durationLimit) && durationLimit > 0
                ? Math.min(Math.max(0, target), Math.max(0, durationLimit - 0.001))
                : Math.max(0, target);
            try {
                state.video.currentTime = clamped;
            } catch (_) {}
            syncClipStripActive(state);
            syncPreviewColorFilter(state);
        });

        state.strip.appendChild(seg);
        state.stripSegments.push(seg);
    }

    if (region && state.stitchOverlay) {
        const leftPct = (region.startFrame / region.total) * 100;
        const widthPct = Math.max(0, (region.endFrame - region.startFrame) / region.total) * 100;
        state.stitchOverlay.style.display = "block";

        if (widthPct > 0) {
            const fill = document.createElement("div");
            fill.className = "h3-stitch-fill";
            fill.style.left = `${leftPct}%`;
            fill.style.width = `${Math.max(widthPct, 0.35)}%`;
            state.stitchOverlay.appendChild(fill);
        }

        const markA = document.createElement("div");
        markA.className = "h3-stitch-mark";
        markA.style.left = `${leftPct}%`;
        markA.title = `stitch start @ ${region.startSec.toFixed(2)}s`;
        state.stitchOverlay.appendChild(markA);

        const markB = document.createElement("div");
        markB.className = "h3-stitch-mark";
        markB.style.left = `${(region.endFrame / region.total) * 100}%`;
        markB.title = `stitch end @ ${region.endSec.toFixed(2)}s`;
        state.stitchOverlay.appendChild(markB);

        state.strip.appendChild(state.stitchOverlay);
    }

    if (region && state.stitchLabel) {
        state.stitchLabel.textContent = formatStitchLabel(region);
        state.stitchLabel.style.display = "block";
        state.stitchLabel.title = formatStitchLabel(region);
    }

    syncClipStripActive(state);
}

function setPreviewTimeline(node, state, timeline, baseLabel = null) {
    if (!state) return;
    if (baseLabel != null) state.baseLabel = String(baseLabel);
    state.colorTimeline = Array.isArray(timeline) ? timeline : [];
    rebuildClipStrip(node, state);
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

function boolValue(value, defaultValue = true) {
    if (value === undefined || value === null || value === "") return Boolean(defaultValue);
    if (value === false || value === 0) return false;
    const text = String(value).trim().toLowerCase();
    if (["false", "0", "off", "no"].includes(text)) return false;
    if (["true", "1", "on", "yes"].includes(text)) return true;
    return Boolean(value);
}

function upstreamGenerationMode(node) {
    const origin = findUpstreamExtenderNode(node);

    // Native serialized widgets are authoritative in Nodes 2.0. Prefer them
    // over the custom runtime so preview restore can never follow a transient
    // UI state left over from node construction.
    const clipsWidget = (origin?.widgets || []).find((w) => w?.name === "clips_json");
    if (typeof clipsWidget?.value === "string") {
        try {
            const parsed = JSON.parse(clipsWidget.value);
            const persisted = String(parsed?.generation_mode || "").toLowerCase();
            if (persisted === "fl2va" || persisted === "ref2va") return persisted;
        } catch (_) {}
    }

    const widget = (origin?.widgets || []).find((w) => w?.name === "generation_mode");
    const widgetMode = String(widget?.value || "").toLowerCase();
    if (widgetMode === "fl2va" || widgetMode === "ref2va") return widgetMode;

    const runtimeMode = String(origin?.__h3Extender?.state?.generation_mode || "").toLowerCase();
    return runtimeMode === "fl2va" ? "fl2va" : "ref2va";
}

function upstreamMotionContext(node) {
    const origin = findUpstreamExtenderNode(node);
    const clipsWidget = (origin?.widgets || []).find((w) => w?.name === "clips_json");
    if (typeof clipsWidget?.value === "string") {
        try {
            const parsed = JSON.parse(clipsWidget.value);
            if (parsed && !Array.isArray(parsed) && Object.prototype.hasOwnProperty.call(parsed, "motion_context")) {
                return boolValue(parsed.motion_context, true);
            }
        } catch (_) {}
    }
    const widget = (origin?.widgets || []).find((w) => w?.name === "motion_context");
    if (widget) return boolValue(widget.value, true);
    return origin?.__h3Extender?.state?.motion_context !== false;
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
        const restoreMode = String(state.restoreModeOverride || upstreamGenerationMode(node)) === "fl2va"
            ? "fl2va"
            : "ref2va";
        params.set("mode", restoreMode);
        const restoreMotion = state.restoreMotionOverride == null
            ? upstreamMotionContext(node)
            : Boolean(state.restoreMotionOverride);
        params.set("motion_context", restoreMotion ? "true" : "false");

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
        const totalClips = Number(payload.project_total_clips || clips);
        const interrupted = Boolean(payload.interrupted);
        const baseLabel = interrupted
            ? `INTERRUPTED PREVIEW — ${clips}/${totalClips} clips (${frames} frames)`
            : `RESTORED PREVIEW — ${clips} clip${clips === 1 ? "" : "s"} (${frames} frames)`;

        state.currentVideoInfo = { ...payload.video };
        state.currentPreviewMeta = {
            clip_count: clips,
            frame_count: frames,
            mode: interrupted ? "interrupted_restored" : "restored",
            interrupted,
            total_clips: totalClips,
            active_layer: String(payload.active_layer || "draft"),
        };
        state.activeLayer = String(payload.active_layer || "draft") === "refine" ? "refine" : "draft";
        if (payload.layers) state.layerStatus = payload.layers;
        applyLayerToggleUi(state);
        state.currentFps = Number(payload.video?.frame_rate || state.currentFps || 24);
        state.stitchMeta = null;
        setPreviewTimeline(node, state, payload.color_timeline, baseLabel);
        state.saveButton.disabled = false;
        loadPreviewSource(node, state, mediaUrl(payload.video) + "&t=" + Date.now());
        state.restoreLoaded = true;
        state.restoreModeOverride = null;
        state.restoreMotionOverride = null;

        // Extender clip names can finish restoring a tick after Final Decode.
        setTimeout(() => {
            if (state.liveLoaded) return;
            rebuildClipStrip(node, state);
            syncPlayerToNode(node, state, true);
        }, 250);

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
        // Never carry the Legacy-only width workaround into Nodes 2.0.
        setLegacyPreviewWidgetFullWidth(state, false);

        const currentH = Number(node.size?.[1] || 0);
        const widgetY = Number(state.widget.last_y);
        const minH = effectivePlayerMinHeight(state);
        const fallbackH = Number.isFinite(widgetY) && widgetY > 0
            ? widgetY + minH + BOTTOM_PAD
            : minH + 180;

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
        state.box.style.minHeight = `${minH}px`;
        state.box.style.maxHeight = "none";
        state.box.style.flex = "1 1 auto";
        state.box.style.overflow = "visible";
        state.video.style.height = "auto";
        state.video.style.minHeight = `${Math.max(80, minH - videoChromePadding(state))}px`;
        state.video.style.flex = "1 1 auto";
        return;
    }

    // ComfyUI frontend currently writes the right-panel host width into
    // widget.width in Legacy mode. LiteGraph then stops falling back to the
    // live node width and the DOM preview is clipped to roughly half the node.
    // Keep width undefined only in Legacy so layout always follows node.size[0].
    setLegacyPreviewWidgetFullWidth(state, true);

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
        const minH = effectivePlayerMinHeight(state);
        const minimumNodeH = widgetY + minH + BOTTOM_PAD;
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
            minH,
            actualH - widgetY - BOTTOM_PAD
        );

        state.currentHeight = availableH;
        if (!previewHeightIsPoisoned(actualH, minimumNodeH)) {
            state.legacyNodeHeight = actualH;
        }
        state.lastRenderMode = "legacy";
        state.box.style.height = `${availableH}px`;
        state.video.style.height =
            `${Math.max(80, availableH - videoChromePadding(state))}px`;
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
                motion_context: upstreamMotionContext(node),
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
    const lowresButton = document.createElement("button");
    lowresButton.type = "button";
    lowresButton.className = "h3-layer-toggle is-active";
    lowresButton.title = "Show lowres (draft) preview";
    const lowresDot = document.createElement("span");
    lowresDot.className = "h3-layer-dot";
    lowresButton.append(lowresDot, document.createTextNode("lowres"));

    const refinedButton = document.createElement("button");
    refinedButton.type = "button";
    refinedButton.className = "h3-layer-toggle";
    refinedButton.title = "Show refined preview";
    const refinedDot = document.createElement("span");
    refinedDot.className = "h3-layer-dot";
    refinedButton.append(refinedDot, document.createTextNode("refined"));

    const saveButton = document.createElement("button");
    saveButton.type = "button";
    saveButton.textContent = "SAVE PREVIEW";
    saveButton.title = "Save the currently assembled preview to ComfyUI output with workflow metadata";
    saveButton.className = "h3-save-preview-button";
    saveButton.disabled = true;
    saveButton.style.flex = "0 0 auto";

    header.appendChild(label);

    const toolbar = document.createElement("div");
    toolbar.className = "h3-final-toolbar";
    toolbar.appendChild(lowresButton);
    toolbar.appendChild(refinedButton);
    toolbar.appendChild(saveButton);

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

    const stripHost = document.createElement("div");
    stripHost.className = "h3-clip-strip-host";

    const strip = document.createElement("div");
    strip.className = "h3-clip-strip";
    strip.title = "Clip timeline — click a segment to seek";

    const stitchOverlay = document.createElement("div");
    stitchOverlay.className = "h3-stitch-overlay";
    stitchOverlay.style.display = "none";

    const stitchLabel = document.createElement("div");
    stitchLabel.className = "h3-stitch-label";

    ensureClipStripStyles();
    stripHost.append(strip, stitchLabel);

    const state = {
        box,
        header,
        toolbar,
        label,
        lowresButton,
        refinedButton,
        lowresDot,
        refinedDot,
        saveButton,
        video,
        stripHost,
        strip,
        stitchOverlay,
        stitchLabel,
        stripSegments: [],
        stripActiveIndex: -1,
        stitchMeta: null,
        widget: null,
        currentVideoInfo: null,
        currentPreviewMeta: null,
        currentFps: 24,
        colorTimeline: [],
        clipNames: [],
        baseLabel: "FULL LIVE PREVIEW",
        saveInProgress: false,
        currentHeight: PLAYER_MIN_HEIGHT,
        syncingPlayer: false,
        lastRenderMode: null,
        legacyNodeHeight: null,
        legacyWidthPinInstalled: false,
        legacyWidthOwnDescriptor: null,
        liveLoaded: false,
        restoreLoaded: false,
        restoreRequestRunning: false,
        activeLayer: "draft",
        layerStatus: {
            draft: { ready: false },
            refine: { ready: false },
        },
        layerSwitchRunning: false,
        stitchSection: null,
        stitchWrap: null,
        stitchRows: [],
    };

    // Layout: stitch controls on top, preview in the middle, layer/save toolbar under video.
    ensureStitchWidgetDefaults(node);
    const stitchSection = buildStitchSection(node, state);
    box.appendChild(stitchSection);
    box.appendChild(header);
    box.appendChild(video);
    box.appendChild(stripHost);
    box.appendChild(toolbar);

    lowresButton.addEventListener("click", () => {
        switchPreviewLayer(node, state, "draft");
    });
    refinedButton.addEventListener("click", () => {
        switchPreviewLayer(node, state, "refine");
    });
    applyLayerToggleUi(state);

    const onPlaybackTick = () => {
        syncPreviewColorFilter(state);
        syncClipStripActive(state);
    };
    video.addEventListener("timeupdate", onPlaybackTick);
    video.addEventListener("seeked", onPlaybackTick);
    video.addEventListener("loadedmetadata", onPlaybackTick);
    if (typeof video.requestVideoFrameCallback === "function") {
        const colorFrameTick = () => {
            if (!box.isConnected) return;
            onPlaybackTick();
            video.requestVideoFrameCallback(colorFrameTick);
        };
        video.requestVideoFrameCallback(colorFrameTick);
    }

    const widget = node.addDOMWidget("h3_live_preview", "preview", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => effectivePlayerMinHeight(state),
        getHeight: () => state.currentHeight,
        afterResize: (resizedNode) => {
            const mode = previewDomRenderMode(box);
            if (mode === "nodes2") {
                // Current WidgetDOM.vue already stretches its child. Keep only
                // an intrinsic minimum; never write a percentage height back.
                const minH = effectivePlayerMinHeight(state);
                box.style.height = "auto";
                box.style.minHeight = `${minH}px`;
                box.style.maxHeight = "none";
                box.style.flex = "1 1 auto";
                box.style.overflow = "visible";
                video.style.height = "auto";
                video.style.minHeight = `${Math.max(80, minH - videoChromePadding(state))}px`;
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
        setLegacyPreviewWidgetFullWidth(state, false);
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

function clearPreviewForNewProject(ownerId) {
    const graph = app.graph;
    if (!graph) return;
    const wanted = String(ownerId);
    for (const node of graph._nodes || []) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
        if (String(findUpstreamExtenderId(node)) !== wanted) continue;

        const state = makePlayer(node);
        state.liveLoaded = false;
        // The active Extender cache is known to be empty after New Project; do
        // not spend the normal workflow-restore retry window looking for it.
        state.restoreLoaded = true;
        state.restoreRequestRunning = false;
        state.restoreModeOverride = null;
        state.restoreMotionOverride = null;
        state.currentVideoInfo = null;
        state.currentPreviewMeta = null;
        state.colorTimeline = [];
        state.video.style.filter = "none";
        state.saveButton.disabled = true;
        state.label.textContent = "NEW PROJECT — preview cleared";
        try {
            state.video.pause();
            state.video.removeAttribute("src");
            state.video.load();
        } catch (_) {}
    }
}

function refreshImportedProjectPreview(ownerId, generationMode = null, motionContext = null) {
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
        state.restoreModeOverride = String(generationMode || "") === "fl2va"
            ? "fl2va"
            : (String(generationMode || "") === "ref2va" ? "ref2va" : null);
        state.restoreMotionOverride = motionContext == null ? null : boolValue(motionContext, true);
        state.currentVideoInfo = null;
        state.currentPreviewMeta = null;
        state.stitchMeta = null;
        setPreviewTimeline(
            node,
            state,
            [],
            "PROJECT LOADED — preview will restore when cached render data is available"
        );
        state.video.style.filter = "none";
        state.saveButton.disabled = true;
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

    beforeConfigureGraph() {
        h3PreviewGraphConfiguring = true;
    },

    loadedGraphNode(node) {
        if (!(node?.comfyClass === TARGET || node?.type === TARGET)) return;
        stripFinalDecodeOutputs(node);
        hideCompatibilityWidget(node, "fps");
        makePlayer(node);
    },

    afterConfigureGraph() {
        h3PreviewGraphConfiguring = false;
        requestAnimationFrame(() => {
            for (const node of app.graph?._nodes || []) {
                if (!(node?.comfyClass === TARGET || node?.type === TARGET)) continue;
                stripFinalDecodeOutputs(node);
                hideCompatibilityWidget(node, "fps");
                const state = makePlayer(node);
                syncPlayerToNode(node, state, true);
                restorePreviewOnLoad(node, state);
            }
        });
    },

    setup() {
        window.addEventListener("h3-extender-new-project", (event) => {
            const ownerId = event?.detail?.owner_id;
            if (ownerId == null) return;
            clearPreviewForNewProject(ownerId);
        });
        window.addEventListener("h3-extender-project-loaded", (event) => {
            const ownerId = event?.detail?.owner_id;
            if (ownerId == null) return;
            refreshImportedProjectPreview(
                ownerId,
                event?.detail?.generation_mode,
                event?.detail?.motion_context,
            );
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
                setPreviewTimeline(node, state, timeline);
                syncPreviewColorFilter(state);
                syncPlayerToNode(node, state, true);
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
            ensureLatentUpscaleWidgetDefaults(this);
            hideFinalDecodeGenerationWidgets(this);
            const state = makePlayer(this);
            syncStitchSection(this, state);

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    stripFinalDecodeOutputs(this);
                    hideCompatibilityWidget(this, "fps");
                    ensureLatentUpscaleWidgetDefaults(this);
                    hideFinalDecodeGenerationWidgets(this);
                    syncStitchSection(this, state);
                    syncPlayerToNode(this, state, true);
                    if (!h3PreviewGraphConfiguring) restorePreviewOnLoad(this, state);
                });
            });

            return r;
        };

        const oldConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = oldConnectionsChange
                ? oldConnectionsChange.apply(this, arguments)
                : undefined;
            const state = this.__h3LivePreview;
            if (state) {
                syncStitchSection(this, state);
                requestAnimationFrame(() => syncPlayerToNode(this, state, true));
            }
            return result;
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
            ensureLatentUpscaleWidgetDefaults(this);
            hideFinalDecodeGenerationWidgets(this);
            const state = makePlayer(this);
            syncStitchSection(this, state);
            requestAnimationFrame(() => {
                stripFinalDecodeOutputs(this);
                hideCompatibilityWidget(this, "fps");
                ensureLatentUpscaleWidgetDefaults(this);
                hideFinalDecodeGenerationWidgets(this);
                syncStitchSection(this, state);
                syncPlayerToNode(this, state, true);
                if (!h3PreviewGraphConfiguring) restorePreviewOnLoad(this, state);
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

            let baseLabel = "FULL LIVE PREVIEW";
            if (meta?.interrupted) {
                const shown = Number(meta?.preview_clips || meta?.clip || 0);
                const total = Number(meta?.total_clips || shown || 0);
                baseLabel =
                    `INTERRUPTED PREVIEW — ${shown}/${total} clips (${meta.preview_frames} frames)`;
            } else if (meta?.mode === "clip_by_clip") {
                const s = Number(meta.seam_shift || 0);
                baseLabel =
                    `FULL LIVE PREVIEW — ${meta.total_clips} clip${meta.total_clips > 1 ? "s" : ""} — shift ${s >= 0 ? "+" : ""}${s}`;
            } else if (meta?.mode === "full_batch" || meta?.mode === "full_batch_stitch" || meta?.mode === "full_batch_incremental") {
                baseLabel = meta?.seamless_stitch
                    ? `STITCHED PREVIEW — ${meta.total_clips} clips (${meta.preview_frames} frames)`
                    : `FINAL PREVIEW — ${meta.total_clips} clips (${meta.preview_frames} frames)`;
            } else if (meta?.mode === "refine_cache") {
                baseLabel =
                    `REFINED PREVIEW — ${meta.total_clips} clips (${meta.preview_frames} frames)`;
            }

            const activeLayer = String(meta?.active_layer || "").toLowerCase() === "refine"
                || String(meta?.cache_mode || "").includes("refine")
                || String(meta?.mode || "").includes("refine")
                ? "refine"
                : "draft";
            state.activeLayer = activeLayer;
            if (activeLayer === "refine") {
                state.layerStatus = {
                    ...(state.layerStatus || {}),
                    refine: {
                        ...(state.layerStatus?.refine || {}),
                        ready: true,
                        clip_count: Number(meta?.total_clips || 0),
                        frame_count: Number(meta?.preview_frames || 0),
                    },
                };
            } else {
                state.layerStatus = {
                    ...(state.layerStatus || {}),
                    draft: {
                        ...(state.layerStatus?.draft || {}),
                        ready: true,
                        clip_count: Number(meta?.total_clips || 0),
                        frame_count: Number(meta?.preview_frames || 0),
                    },
                };
            }
            applyLayerToggleUi(state);
            refreshLayerStatus(this, state);
            if (meta?.project_autosave_error) {
                state.label.textContent += " — PROJECT SAVE FAILED";
                state.label.title = String(meta.project_autosave_error);
            } else if (meta?.project_autosave_path) {
                state.label.textContent += " — PROJECT SAVED";
                state.label.title = String(meta.project_autosave_path);
            } else {
                state.label.title = "";
            }

            state.currentVideoInfo = { ...info };
            state.currentPreviewMeta = {
                clip_count: Number(meta?.preview_clips || meta?.total_clips || 0),
                frame_count: Number(meta?.preview_frames || 0),
                mode: String(meta?.mode || ""),
                active_layer: activeLayer,
            };
            state.currentFps = Number(info.frame_rate || state.currentFps || 24);
            state.stitchMeta = meta?.stitch || null;
            setPreviewTimeline(this, state, meta?.color_timeline, baseLabel);
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
