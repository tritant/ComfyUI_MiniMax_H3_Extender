import { app } from "../../scripts/app.js";

const TARGET = "MiniMaxH3PromptPackMerge";
const MAX_PACKS = 128;
const PACK_RE = /^pack_(\d+)$/;

function packIndex(input) {
    const match = String(input?.name || "").match(PACK_RE);
    if (!match) return 0;
    const index = Number(match[1]);
    return Number.isInteger(index) && index >= 1 && index <= MAX_PACKS ? index : 0;
}

function isConnected(input) {
    return input?.link !== null && input?.link !== undefined;
}

function currentPackInputs(node) {
    return (node?.inputs || [])
        .map((input, slot) => ({ input, slot, index: packIndex(input) }))
        .filter((entry) => entry.index > 0)
        .sort((a, b) => a.slot - b.slot);
}

function addPackInput(node, index) {
    if (!node || index < 1 || index > MAX_PACKS) return;
    if ((node.inputs || []).some((input) => packIndex(input) === index)) return;
    node.addInput(`pack_${index}`, "H3_PROMPT_PACK");
}

function removePackInput(node, slot) {
    if (!node || slot < 0 || slot >= (node.inputs || []).length) return;
    try {
        node.removeInput(slot);
    } catch (_) {
        /* retry on next sync */
    }
}

function renamePackInput(input, index) {
    if (!input || index < 1 || index > MAX_PACKS) return false;
    const nextName = `pack_${index}`;
    const oldName = String(input.name || "");
    if (oldName === nextName) return false;

    input.name = nextName;
    if (typeof input.label === "string" && PACK_RE.test(input.label)) {
        input.label = nextName;
    }
    return true;
}

function fitNodeHeight(node) {
    try {
        const computed = node?.computeSize?.();
        const height = Number(computed?.[1]);
        const width = Number(node?.size?.[0]);
        if (Number.isFinite(height) && height > 0 && Number.isFinite(width) && width > 0) {
            node.setSize?.([width, height]);
        }
    } catch (_) {}
}

function syncPackInputs(node) {
    if (!node || node.__h3PromptMergeSyncing) return;
    node.__h3PromptMergeSyncing = true;

    let changed = false;
    try {
        let entries = currentPackInputs(node);

        const emptyEntries = entries
            .filter(({ input }) => !isConnected(input))
            .sort((a, b) => b.slot - a.slot);

        for (const { slot } of emptyEntries) {
            const before = (node.inputs || []).length;
            removePackInput(node, slot);
            changed = changed || (node.inputs || []).length !== before;
        }

        entries = currentPackInputs(node).filter(({ input }) => isConnected(input));
        for (let i = 0; i < entries.length; i++) {
            changed = renamePackInput(entries[i].input, i + 1) || changed;
        }

        const nextIndex = Math.min(MAX_PACKS, entries.length + 1);
        if (entries.length < MAX_PACKS) {
            const before = (node.inputs || []).length;
            addPackInput(node, nextIndex);
            changed = changed || (node.inputs || []).length !== before;
        }

        if (changed) fitNodeHeight(node);
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__h3PromptMergeSyncing = false;
    }
}

function deferSync(node) {
    if (!node || node.__h3PromptMergeSyncQueued) return;
    node.__h3PromptMergeSyncQueued = true;
    requestAnimationFrame(() => {
        node.__h3PromptMergeSyncQueued = false;
        syncPackInputs(node);
    });
}

app.registerExtension({
    name: "MiniMaxH3.PromptPackMerge.DynamicInputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = oldCreated ? oldCreated.apply(this, arguments) : undefined;
            deferSync(this);
            return result;
        };

        const oldConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
            deferSync(this);
            return result;
        };

        const oldConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = oldConnectionsChange
                ? oldConnectionsChange.apply(this, arguments)
                : undefined;
            deferSync(this);
            return result;
        };
    },
});
