import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
    process.env.H3_EXTENDER_SOURCE || new URL("../web/extender.js", import.meta.url),
    "utf8",
).replace(/^import .*;\r?\n/gm, "");

const ref = (character) => ({ id: character.repeat(64), original_name: `${character}.png` });

function context() {
    const scope = vm.createContext({
        app: { registerExtension() {} },
        api: { apiURL(value) { return value; } },
        document: { getElementById() { return {}; } },
        console,
        Set,
        alert() {},
    });
    vm.runInContext(source, scope);
    return scope;
}

test("library persists more than nine uploads while clip selection stays limited", () => {
    const scope = context();
    scope.input = {
        refs: [ref("a"), ref("b")],
        library: "cdef0123456789".split("").map(ref),
    };
    const state = vm.runInContext("parseRefsState(input)", scope);
    assert.equal(state.library.length, 14);
    assert.equal(state.refs.filter(Boolean).length, 2);
    assert.equal(JSON.parse(vm.runInContext("serializeRefsState(parseRefsState(input))", scope)).library.length, 14);
});

test("managed Picture numbering places selected locals after globals", () => {
    const scope = context();
    scope.runtime = {
        refsState: { refs: [ref("a"), null, ref("b"), ...Array(6).fill(null)] },
    };
    scope.clip = { local_refs: { selected_images: [ref("c"), ref("d")] } };
    const keys = Array.from(vm.runInContext("pictureBindingKeys(runtime, clip)", scope));
    assert.deepEqual(keys, ["global:0", "global:2", `local:${"c".repeat(64)}`, `local:${"d".repeat(64)}`]);
    scope.before = ["global:0", `local:${"c".repeat(64)}`];
    scope.after = keys;
    assert.equal(
        vm.runInContext('remapPicturePrompt("<Picture 1> and <Picture 2>", before, after)', scope),
        "<Picture 1> and <Picture 3>",
    );
});

test("legacy workflow hydration promotes local images without clearing validation", () => {
    const scope = context();
    scope.runtime = {
        state: {
            generation_mode: "ref2va",
            mode_clips: { ref2va: [{ validated: true, local_refs: { images: [{ slot: 4, ref: ref("a") }] } }] },
        },
        refsState: { refs: Array(9).fill(null), library: [] },
    };
    vm.runInContext("syncLibraryFromClips(runtime)", scope);
    assert.equal(scope.runtime.refsState.library[0].id, "a".repeat(64));
    assert.equal(scope.runtime.state.mode_clips.ref2va[0].validated, true);
    assert.equal(scope.runtime.state.mode_clips.ref2va[0].local_refs.images[0].slot, 4);
});

test("managed selections and validated flags survive workflow serialization", () => {
    const scope = context();
    scope.saved = JSON.stringify({
        generation_mode: "ref2va",
        motion_context: true,
        clips: [{ id: "clip1", validated: true, local_refs: { selected_images: [ref("a")] } }],
    });
    const state = vm.runInContext("parseState(saved)", scope);
    scope.state = state;
    const restored = vm.runInContext("parseState(serializeState(state))", scope);
    assert.equal(restored.clips[0].validated, true);
    assert.equal(restored.clips[0].local_refs.selected_images[0].id, "a".repeat(64));
});

test("uploading to the library does not invalidate a validated clip", async () => {
    const scope = context();
    scope.node = { id: 1 };
    scope.runtime = {
        state: { clips: [{ id: "clip1", validated: true }] },
        refsState: { refs: Array(9).fill(null), library: [] },
        refsWidget: { value: "" },
    };
    scope.FormData = class { append() {} };
    scope.fetch = async () => ({ ok: true, json: async () => ({ ok: true, ref: ref("a") }) });
    vm.runInContext(`
        invalidations = 0;
        prepareLocalRefMutation = async () => { invalidations++; return true; };
        updateRefsHidden = () => {};
        captureNativeWorkflowState = () => {};
        render = () => {};
        projectBusy = () => false;
    `, scope);
    const uploaded = await vm.runInContext(
        'uploadLibraryPictures(node, runtime, [{ name: "a.png", type: "image/png" }])', scope,
    );
    assert.equal(uploaded.length, 1);
    assert.equal(scope.runtime.refsState.library.length, 1);
    assert.equal(scope.runtime.state.clips[0].validated, true);
    assert.equal(vm.runInContext("invalidations", scope), 0);
});

test("selecting from the library invalidates only through the local-ref mutation path", async () => {
    const scope = context();
    scope.node = { id: 1 };
    scope.runtime = {
        state: { clips: [{ id: "clip1", prompt: "<Picture 2>", local_refs: { images: [{ slot: 2, ref: ref("c") }] } }] },
        refsState: { refs: [ref("a"), null, ...Array(7).fill(null)], library: [ref("c"), ref("d")] },
    };
    vm.runInContext(`
        invalidations = [];
        prepareLocalRefMutation = async (_node, _runtime, index) => { invalidations.push(index); return true; };
        updateHidden = () => {};
        captureNativeWorkflowState = () => {};
        render = () => {};
        projectBusy = () => false;
    `, scope);
    assert.equal(await vm.runInContext(`toggleLibraryPicture(node, runtime, 0, "${"d".repeat(64)}")`, scope), true);
    const local = scope.runtime.state.clips[0].local_refs;
    assert.equal(local.selected_images.length, 2);
    assert.equal(local.images.length, 0);
    assert.equal(scope.runtime.state.clips[0].prompt, "<Picture 2>");
    assert.deepEqual(Array.from(vm.runInContext("invalidations", scope)), [0]);
});
