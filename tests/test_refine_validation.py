"""CPU-only regression tests for #83 Draft/Refine validation persistence."""
import ast
import asyncio
import copy
import logging
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _validated_prefix_count(segments):
    count = 0
    for seg in segments:
        if not seg.get("validated", False):
            break
        count += 1
    return count


class _Web:
    @staticmethod
    def json_response(data, status=200):
        return {"data": data, "status": status}


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class RefineValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "motion_context_disk.py").read_text(encoding="utf-8"))
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "h3_extender_local_ref_invalidate"
        )
        fn = copy.deepcopy(fn)
        fn.decorator_list = []
        cls.fn_ast = fn

    def _fixture(self, draft_manifest=None, refine_manifest=None):
        writes = {}
        draft = copy.deepcopy(draft_manifest)
        refine = copy.deepcopy(refine_manifest)

        def load_manifest(_data, _manifest):
            return copy.deepcopy(draft)

        def load_refine(_data, _draft, require_complete=False):
            if refine is None:
                return None
            return "refine.data", "refine.json", copy.deepcopy(refine)

        def write_json(path, payload):
            writes[str(path)] = copy.deepcopy(payload)

        ns = {
            "web": _Web,
            "time": time,
            "_LOG": logging.getLogger("refine-validation-test"),
            "_request_bool": lambda value, default=False: bool(value) if value is not None else bool(default),
            "_extender_runtime_mode": lambda generation_mode, motion: (
                "ref2va_motion" if generation_mode == "ref2va" and motion else
                "ref2va_independent" if generation_mode == "ref2va" else "fl2va"
            ),
            "_extender_cache_owner_id": lambda owner, mode, motion: owner,
            "_chain_paths": lambda owner: ("draft.data", "draft.json"),
            "_load_manifest_from_paths": load_manifest,
            "_load_refine_sidecar": load_refine,
            "_write_json_atomic": write_json,
            "_validated_prefix_count": _validated_prefix_count,
        }
        exec(compile(ast.Module(body=[self.fn_ast], type_ignores=[]), "validation_route", "exec"), ns)
        return ns["h3_extender_local_ref_invalidate"], writes

    @staticmethod
    def _body(**extra):
        body = {
            "owner_id": "1",
            "generation_mode": "ref2va",
            "motion_context": True,
            "clip_index": 0,
            "clip_id": "clip_1",
            "validated": False,
            "layer": "draft",
        }
        body.update(extra)
        return body

    def test_missing_cache_can_be_unvalidated_but_not_validated(self):
        fn, _ = self._fixture(None, None)
        result = asyncio.run(fn(_Request(self._body(validated=False))))
        self.assertEqual(result["status"], 200)
        self.assertFalse(result["data"]["found"])

        result = asyncio.run(fn(_Request(self._body(validated=True))))
        self.assertEqual(result["status"], 400)
        self.assertIn("physical cache", result["data"]["error"])

    def test_refine_requires_validated_draft_and_physical_refine_cache(self):
        draft = {"segments": [{"clip_id": "clip_1", "validated": False}]}
        refine = {"segments": [{"clip_id": "clip_1", "validated": False}]}
        fn, _ = self._fixture(draft, refine)
        result = asyncio.run(fn(_Request(self._body(layer="refine", validated=True))))
        self.assertEqual(result["status"], 400)
        self.assertIn("Draft", result["data"]["error"])

        draft["segments"][0]["validated"] = True
        fn, writes = self._fixture(draft, None)
        result = asyncio.run(fn(_Request(self._body(layer="refine", validated=True))))
        self.assertEqual(result["status"], 400)
        self.assertIn("Refine cache", result["data"]["error"])
        self.assertEqual(writes, {})

    def test_refine_validation_persists_to_refine_manifest(self):
        draft = {"segments": [{"clip_id": "clip_1", "validated": True}]}
        refine = {"segments": [{"clip_id": "clip_1", "validated": False}]}
        fn, writes = self._fixture(draft, refine)
        result = asyncio.run(fn(_Request(self._body(layer="refine", validated=True))))
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["data"]["refine_validated_count"], 1)
        self.assertTrue(writes["refine.json"]["segments"][0]["validated"])

    def test_draft_unvalidation_clears_dependent_refine_suffix(self):
        draft = {"segments": [
            {"clip_id": "clip_1", "validated": True},
            {"clip_id": "clip_2", "validated": True},
        ]}
        refine = {"segments": [
            {"clip_id": "clip_1", "validated": True},
            {"clip_id": "clip_2", "validated": True},
        ]}
        fn, writes = self._fixture(draft, refine)
        result = asyncio.run(fn(_Request(self._body(clip_index=1, clip_id="clip_2", validated=False))))
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["data"]["validated_count"], 1)
        self.assertEqual(result["data"]["refine_validated_count"], 1)
        self.assertTrue(writes["refine.json"]["segments"][0]["validated"])
        self.assertFalse(writes["refine.json"]["segments"][1]["validated"])


if __name__ == "__main__":
    unittest.main()
