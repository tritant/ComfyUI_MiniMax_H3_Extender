import ast
import copy
import json
import logging
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "extender.py").read_text())
NAMES = {
    "_normalize_local_refs", "_clip_picture_refs", "_max_selected_picture_count", "_ref_library_from_json",
    "_local_ref_assets_from_clips", "_write_refs_to_project_payload",
    "_sync_refs_from_ref_pack", "_reference_count",
}
SCOPE = {
    "json": json,
    "copy": copy,
    "LOCAL_REFS_VERSION": 1,
    "MAX_IMAGE_REFS": 9,
    "MAX_VIDEO_REFS": 3,
    "MAX_STANDALONE_AUDIO_REFS": 3,
    "_normalize_ref_descriptor": lambda value: value if isinstance(value, dict) and value.get("id") else None,
    "_normalize_media_descriptor": lambda value, kind: None,
    "_ref_id_is_safe": lambda value: isinstance(value, str) and len(value) == 64,
    "_normalize_ref_descriptors": lambda refs: refs,
    "REFS_JSON_VERSION": 3,
    "_LOG": logging.getLogger(__name__),
    "_store_external_reference": lambda image, index, previous: ({"id": f"{index:064x}"}, True),
}
exec(compile(ast.Module(
    body=[node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in NAMES],
    type_ignores=[],
), "extender.py", "exec"), SCOPE)


class LocalPictureLibraryTests(unittest.TestCase):
    def test_legacy_slots_and_managed_order_remain_distinct(self):
        legacy = SCOPE["_normalize_local_refs"]({"images": [{"slot": 4, "ref": {"id": "a" * 64}}]})
        self.assertIsNone(legacy["selected_images"])
        self.assertEqual(legacy["images"][0]["slot"], 4)

        managed = SCOPE["_normalize_local_refs"]({
            "images": [{"slot": 4, "ref": {"id": "a" * 64}}],
            "selected_images": [{"id": str(index).zfill(64)} for index in range(12)],
        })
        self.assertEqual(len(managed["selected_images"]), 9)
        self.assertEqual(managed["selected_images"][0]["id"], str(0).zfill(64))

    def test_unselected_library_images_are_portable(self):
        library_ref = {"id": "a" * 64, "source_id": "b" * 64}
        selected_ref = {"id": "c" * 64}
        clips = [{"local_refs": {"selected_images": [selected_ref]}}]
        images, media = SCOPE["_local_ref_assets_from_clips"](clips, [library_ref])
        self.assertEqual(set(images), {"a" * 64, "b" * 64, "c" * 64})
        self.assertEqual(media, {})

    def test_selected_images_follow_globals_and_respect_nine_picture_limit(self):
        globals_ = [{"id": "a" * 64}, None, {"id": "b" * 64}] + [None] * 6
        selected = SCOPE["_normalize_local_refs"]({"selected_images": [{"id": "c" * 64}, {"id": "d" * 64}]})
        packed, local_visual = SCOPE["_clip_picture_refs"](globals_, selected, 0)
        self.assertTrue(local_visual)
        self.assertEqual([ref["id"] for ref in packed if ref], ["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        self.assertEqual(len(packed), 9)
        too_many = {"selected_images": [{"id": f"{index:064x}"} for index in range(8)]}
        with self.assertRaisesRegex(ValueError, "maximum is 9"):
            SCOPE["_clip_picture_refs"](globals_, too_many, 0)
        self.assertEqual(SCOPE["_max_selected_picture_count"]([
            {"local_refs": {"images": [{"slot": 1, "ref": {"id": "e" * 64}}]}},
            {"local_refs": {"selected_images": [{"id": "c" * 64}, {"id": "d" * 64}]}},
        ]), 2)

    def test_library_round_trip_exceeds_active_picture_limit(self):
        refs = [{"id": f"{index:064x}"} for index in range(14)]
        library = SCOPE["_ref_library_from_json"](json.dumps({"library": refs}))
        self.assertEqual(len(library), 14)
        payload = {"extender": {"refs_json": json.dumps({"refs": [], "library": refs}), "settings": {}}}
        SCOPE["_write_refs_to_project_payload"](payload, [])
        self.assertEqual(len(json.loads(payload["extender"]["refs_json"])["library"]), 14)

    def test_external_pack_keeps_room_for_selected_local_pictures(self):
        refs = [{"id": "a" * 64}] + [None] * 8
        pack = {"slots": [None] + [object()] * 8}
        imported, slots, skipped = SCOPE["_sync_refs_from_ref_pack"](refs, pack, set(), 2)
        self.assertEqual(SCOPE["_reference_count"](imported), 2)
        self.assertEqual(slots, [2])
        self.assertEqual(skipped, list(range(3, 10)))


if __name__ == "__main__":
    unittest.main()
