import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

// Exercise the real widget and execution hooks without ComfyUI or a GPU.
const source = readFileSync(
    process.env.H3_EXTENDER_SOURCE || new URL("../web/extender.js", import.meta.url),
    "utf8",
).replace(/^import .*;\r?\n/gm, "");

function clip(id, seed_mode = "increment", seed = 10, validated = false) {
    return { id, seed_mode, seed, validated, prompt: id };
}

async function fixture(clips, mode = "ref2va", inactive = [clip("inactive")]) {
    let extension;
    const context = vm.createContext({
        app: { registerExtension(value) { extension = value; } },
        api: {},
        document: {
            getElementById() { return {}; },
            createElement() {
                return { style: { setProperty() {} }, append() {}, addEventListener() {} };
            },
        },
        requestAnimationFrame() {},
        console,
    });
    vm.runInContext(source, context);
    // Layout, network requests and resolution controls are unrelated to seeds.
    vm.runInContext(`
        installInvalidationHooks = () => {};
        wrapResolutionWidgetCallbacks = () => {};
        refreshLoraNames = () => {};
        render = (node, runtime) => { node.displayedSeeds = runtime.state.clips.map(c => c.seed); };
        syncResolutionMirror = () => {};
        syncDomHeight = () => {};
    `, context);
    class Node {
        constructor() {
            this.properties = {};
            this.widgets = [
                { name: "clips_json", value: JSON.stringify({
                    version: 2, generation_mode: mode, clips,
                    mode_clips: { [mode]: clips, [mode === "ref2va" ? "fl2va" : "ref2va"]: inactive },
                }) },
                { name: "refs_json", value: "{}" },
                { name: "generation_mode", value: mode },
            ];
            this.graph = { change() {}, setDirtyCanvas() {} };
        }
        addDOMWidget() { return {}; }
    }
    await extension.beforeRegisterNodeDef(Node, { name: "MiniMaxH3Extender" });
    const node = new Node();
    context.node = node;
    const widget = node.widgets[0];
    let previousCalls = 0;
    widget.afterQueued = function (arg) {
        assert.equal(this, widget);
        assert.equal(arg.isPartialExecution, false);
        previousCalls++;
    };
    vm.runInContext("buildUi(node); updateHidden(node, node.__h3Extender)", context);
    const state = () => JSON.parse(widget.value);
    return {
        node, context, state,
        previousCalls: () => previousCalls,
        queue() {
            const submitted = state();
            widget.afterQueued?.({ isPartialExecution: false });
            return submitted;
        },
        complete(submitted, extra = {}) {
            node.onExecuted({ h3_extender_state: [{
                generation_mode: submitted.generation_mode,
                clips_json: JSON.stringify(submitted),
                generated: submitted.clips.map((_, i) => i + 1),
                ...extra,
            }] });
        },
    };
}

for (const mode of ["ref2va", "fl2va"]) {
    test(`${mode}: eight submissions advance candidates once and preserve other clips`, async () => {
        // Deliberately reuse an ID in the other mode: timelines are independent.
        const f = await fixture([
            clip("up"), clip("down", "decrement", 20),
            clip("fixed", "fixed", 30), clip("validated", "increment", 40, true),
        ], mode, [clip("up", "increment", 90)]);
        const inactiveMode = mode === "ref2va" ? "fl2va" : "ref2va";
        const inactive = f.state().mode_clips[inactiveMode];
        const submitted = Array.from({ length: 8 }, () => f.queue());
        assert.deepEqual(submitted.map(s => s.clips.map(c => c.seed)),
            Array.from({ length: 8 }, (_, i) => [10 + i, 20 - i, 30, 40]));
        for (const payload of submitted) {
            assert.deepEqual(payload.clips, payload.mode_clips[mode]);
            assert.deepEqual(payload.mode_clips[inactiveMode], inactive);
            f.complete(payload);
            assert.deepEqual(f.state().clips.map(c => c.seed), [18, 12, 30, 40]);
        }
        assert.deepEqual(Array.from(f.node.displayedSeeds), [18, 12, 30, 40]);
        assert.equal(f.previousCalls(), 8);
        assert.equal(f.queue().clips[0].seed, 18);
    });
}

test("completion between submissions cannot rewind or double-advance seeds", async () => {
    const f = await fixture([clip("a"), clip("b")]);
    const first = f.queue();
    f.queue();
    // Also retain the next seed for a clip that execution did not reach.
    f.complete(first, { generated: [1], checkpoint_interrupted: true });
    assert.deepEqual(f.queue().clips.map(c => c.seed), [12, 12]);
    assert.ok(f.state().resume_nonce);
    const last = f.queue();
    f.complete(last);
    assert.deepEqual(f.queue().clips.map(c => c.seed), [14, 14]);
    assert.equal(f.state().resume_nonce, undefined);
});

test("randomize changes every payload, including a random seed collision", async () => {
    const f = await fixture([clip("random", "randomize", 10)]);
    vm.runInContext("let testSeed = 100; randomSeed = () => testSeed++", f.context);
    const batch = Array.from({ length: 8 }, () => f.queue());
    assert.equal(new Set(batch.map(s => s.clips[0].seed)).size, 8);
    f.complete(batch[0]);
    assert.equal(f.state().clips[0].seed, 107);
    vm.runInContext("node.__h3Extender.state.clips[0].seed = 10; updateHidden(node, node.__h3Extender)", f.context);
    vm.runInContext("randomSeed = () => 10", f.context);
    const first = f.queue();
    const second = f.queue();
    assert.equal(first.clips[0].seed, 10);
    assert.equal(second.clips[0].seed, 11);
    assert.equal(f.state().clips[0].seed, 10);
    f.complete(first);
    assert.equal(f.state().clips[0].seed, 10);
});

test("increment and decrement wrap at safe integer boundaries", async () => {
    const f = await fixture([clip("up", "increment", Number.MAX_SAFE_INTEGER), clip("down", "decrement", 0)]);
    f.queue();
    assert.deepEqual(f.state().clips.map(c => c.seed), [0, Number.MAX_SAFE_INTEGER]);
});

test("validated backend results stay authoritative and new clips get a next seed", async () => {
    const f = await fixture([clip("a")]);
    const submitted = f.queue();
    submitted.clips[0].validated = true;
    submitted.clips.push(clip("imported", "increment", 50));
    f.complete(submitted);
    assert.deepEqual(f.state().clips.map(c => [c.seed, c.validated]), [[10, true], [51, false]]);
});

test("rebuilding UI does not install the queue hook twice", async () => {
    const f = await fixture([clip("a")]);
    vm.runInContext("buildUi(node); buildUi(node)", f.context);
    f.queue();
    assert.equal(f.state().clips[0].seed, 11);
    assert.equal(f.previousCalls(), 1);
});

test("fixed and validated clips alone keep the queued input cache-identical", async () => {
    const f = await fixture([clip("fixed", "fixed"), clip("validated", "randomize", 42, true)]);
    const before = f.node.widgets[0].value;
    f.queue();
    assert.equal(f.node.widgets[0].value, before);
});

test("queue hook uses replacement runtime state after workflow hydration", async () => {
    const f = await fixture([clip("a")]);
    f.node.widgets[0].value = JSON.stringify({ clips: [clip("loaded", "increment", 80)] });
    vm.runInContext("node.__h3Extender.state = parseState(node.widgets[0].value)", f.context);
    f.queue();
    assert.equal(f.state().clips[0].id, "loaded");
    assert.equal(f.state().clips[0].seed, 81);
});

test("completion in another mode preserves each timeline's next seed", async () => {
    const f = await fixture([clip("shared")], "ref2va", [clip("shared", "increment", 70)]);
    const submitted = f.queue();
    vm.runInContext(`
        activateModeState(node.__h3Extender.state, "fl2va");
        updateHidden(node, node.__h3Extender);
    `, f.context);
    f.queue();
    f.complete(submitted);
    assert.equal(f.state().clips[0].seed, 11);
    assert.equal(f.state().mode_clips.fl2va[0].seed, 71);
});
