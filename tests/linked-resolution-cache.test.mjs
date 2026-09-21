import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
    process.env.H3_EXTENDER_SOURCE || new URL("../web/extender.js", import.meta.url),
    "utf8",
).replace(/^import .*;\r?\n/gm, "");

function contextFor(node) {
    const context = vm.createContext({
        app: { registerExtension() {}, graph: null },
        api: {},
        document: {
            getElementById() { return {}; },
            createElement() { return { dataset: {}, style: { setProperty() {} }, append() {}, addEventListener() {} }; },
        },
        requestAnimationFrame() {},
        console,
        Set,
    });
    vm.runInContext(source, context);
    context.node = node;
    return context;
}

test("linked resolution inputs do not invalidate restored cache from stale widget fallbacks", () => {
    const node = {
        inputs: [
            { name: "width", link: 72 },
            { name: "height", link: 73 },
        ],
        widgets: [
            { name: "width", value: 896 },
            { name: "height", value: 576 },
        ],
    };
    const context = contextFor(node);
    context.runtime = {
        state: { clips: [{ validated: true }, { validated: true }] },
        expectedResolution: { width: 736, height: 992 },
        cachedCount: 2,
        validatedCount: 2,
    };

    assert.equal(vm.runInContext("currentResolutionFromWidgets(node)", context), null);
    assert.equal(vm.runInContext("invalidateForResolutionChange(node, runtime)", context), false);
    assert.deepEqual(context.runtime.state.clips.map((clip) => clip.validated), [true, true]);
    assert.equal(context.runtime.cachedCount, 2);
    assert.equal(context.runtime.validatedCount, 2);
});

test("unlinked resolution inputs still invalidate an incompatible cache", () => {
    const node = {
        inputs: [
            { name: "width", link: null },
            { name: "height", link: null },
        ],
        widgets: [
            { name: "width", value: 896 },
            { name: "height", value: 576 },
        ],
        graph: { setDirtyCanvas() {} },
    };
    const context = contextFor(node);
    vm.runInContext("updateHidden = () => {}; render = () => {};", context);
    context.runtime = {
        state: { clips: [{ validated: true }, { validated: true }] },
        expectedResolution: { width: 736, height: 992 },
        cachedCount: 2,
        validatedCount: 2,
        computedIndices: new Set([0, 1]),
        computedClipIds: new Set(["a", "b"]),
    };

    const resolution = vm.runInContext("currentResolutionFromWidgets(node)", context);
    assert.equal(resolution.width, 896);
    assert.equal(resolution.height, 576);
    assert.equal(vm.runInContext("invalidateForResolutionChange(node, runtime)", context), true);
    assert.deepEqual(context.runtime.state.clips.map((clip) => clip.validated), [false, false]);
    assert.equal(context.runtime.cachedCount, 0);
    assert.equal(context.runtime.validatedCount, 0);
});
