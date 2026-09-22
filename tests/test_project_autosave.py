"""CPU-only autosave orchestration tests; disk archives use a small test writer."""
import ast
import copy
import functools
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]

def load_functions(filename, names, ns):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), ns)
    return tree

class Autosave(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ns = dict(math=math, copy=copy, functools=functools, inspect=inspect, json=json, os=os, Path=Path, uuid=uuid,
                       DEFAULT_SEED_MAX=2**53-1)
        tree = load_functions('extender.py', {'_capture_full_batch_project', '_auto_save_full_batch_project', '_restore_project_generation_seeds', '_generation_mode_from_project_payload'}, self.ns)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROJECT_SETTING_NAMES' for t in n.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'settings', 'exec'), self.ns)
        load_functions('fl2va_engine.py', {'normalize_mode'}, self.ns)
        self.ns['_normalize_generation_mode'] = self.ns['normalize_mode']
        self.ns['_clips_from_project_payload'] = lambda p: json.loads(p['extender']['clips_json'])['clips']
        self.calls = []
        def writer(owner, name, payload, output):
            self.calls.append((owner, name, copy.deepcopy(payload)))
            self.ns['_restore_project_generation_seeds'](payload, {'segments': [{'clip_id': 'a', 'generation_seed': 42}]})
            with ZipFile(output, 'w') as z:
                z.writestr('project.json', json.dumps(payload))
                z.writestr('cache/chain.h3cache', b'cached latent')
            return {'cache': {'present': True, 'clip_count': 1}}
        self.ns['_build_project_archive'] = writer
        module = types.ModuleType('autosave_test.extender')
        module._auto_save_full_batch_project = self.ns['_auto_save_full_batch_project']
        package = types.ModuleType('autosave_test')
        package.__path__ = []
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'autosave_test': package, 'autosave_test.extender': module}).start()
        self.gate_ns = {'__package__': 'autosave_test', '_LOG': logging.getLogger('autosave-test')}
        load_functions('motion_context_disk.py', {'_maybe_auto_save_project'}, self.gate_ns)
        self.gate = self.gate_ns['_maybe_auto_save_project']

    def capture(self, mode='ref2va', motion=True, run='full_batch', seed=42, manual=None):
        raw = json.dumps({'clips': [{'id': 'a', 'seed': seed, 'seed_mode': 'increment', 'validated': False}]})
        if manual is not None:
            raw = json.dumps({**json.loads(raw), 'project_manual_resolution': manual})
        @self.ns['_capture_full_batch_project']
        def execute(run_mode='full_batch', generation_mode='ref2va', motion_context=True, unique_id='7', clips_json=raw,
                    refs_json='{"refs":[]}', resolution_mode='manual', megapixels=0.4, width=896, height=576, steps=4):
            return {'ui': {'h3_extender_state': [{'clips_json': clips_json, 'refs_json': '{"refs":[],"imported":true}',
                                                'generation_mode': generation_mode, 'resolved_width': 896, 'resolved_height': 576}]},
                    'result': ({'run_mode': run_mode}, 1, 0, 'done')}
        return execute(run_mode=run, generation_mode=mode, motion_context=motion)['result'][0]

    def test_three_modes_completed_archive_matches_output_and_real_seed(self):
        for index, (mode, motion) in enumerate([('ref2va', True), ('ref2va', False), ('fl2va', False)]):
            cache = self.capture(mode, motion, seed=43)
            before = copy.deepcopy(cache)
            video = self.root / f'video_{index:05d}.mp4'
            video.write_bytes(b'video')
            result = self.gate(cache, video, '8', {'auto_save_project': True}, 1, 241)
            self.assertEqual(Path(result['project_autosave_path']), video.with_suffix('.ext'))
            with ZipFile(video.with_suffix('.ext')) as z:
                payload = json.loads(z.read('project.json'))
                self.assertEqual(json.loads(payload['extender']['clips_json'])['clips'][0]['seed'], 42)
                self.assertEqual(payload['extender']['generation_mode'], mode)
                self.assertEqual(payload['extender']['motion_context'], motion)
                self.assertTrue(json.loads(payload['extender']['refs_json'])['imported'])
                self.assertEqual(payload['final_decode']['preview']['frame_count'], 241)
                self.assertEqual(z.read('cache/chain.h3cache'), b'cached latent')
            self.assertEqual(cache, before)
            self.assertEqual(video.read_bytes(), b'video')
        self.assertEqual(len(self.calls), 3)

    def test_disabled_clip_by_clip_and_interrupt_never_write(self):
        full = self.capture()
        for cache, settings in [(full, None), (full, {'auto_save_project': False}),
                                (self.capture(run='clip_by_clip'), {'auto_save_project': True}),
                                ({**full, 'interrupted': True}, {'auto_save_project': True})]:
            self.assertEqual(self.gate(cache, self.root/'x.mp4', '8', settings, 1, 241), {})
        self.assertEqual(self.calls, [])
        self.assertNotIn('project_snapshot', self.capture(run='clip_by_clip'))

    def test_failed_archive_preserves_video_and_removes_temporary_file(self):
        def fail(owner, name, payload, path):
            Path(path).write_bytes(b'incomplete')
            raise OSError('disk full')
        self.ns['_build_project_archive'] = fail
        video = self.root/'x.mp4'
        video.write_bytes(b'good video')
        with self.assertLogs('autosave-test', level='ERROR'):
            result = self.gate(self.capture(), video, '8', {'auto_save_project': True}, 1, 241)
        self.assertIn('disk full', result['project_autosave_error'])
        self.assertEqual(list(self.root.iterdir()), [video])
        self.assertEqual(video.read_bytes(), b'good video')

    def test_existing_archive_missing_context_and_incomplete_batch_are_reported(self):
        video = self.root/'x.mp4'
        destination = video.with_suffix('.ext')
        destination.write_bytes(b'previous project')
        with self.assertLogs('autosave-test', level='ERROR'):
            result = self.gate(self.capture(), video, '8', {'auto_save_project': True}, 1, 241)
        self.assertIn('already exists', result['project_autosave_error'])
        self.assertEqual(destination.read_bytes(), b'previous project')
        for cache, count in [({'run_mode':'full_batch'}, 1), (self.capture(), 0)]:
            with self.assertLogs('autosave-test', level='ERROR'):
                self.assertIn('project_autosave_error', self.gate(cache, self.root/'y.mp4', '8', {'auto_save_project': True}, count, 241))
        self.assertEqual(self.calls, [])

    def test_manual_fallback_is_separate_from_resolved_canvas(self):
        for mode, motion in [('ref2va', True), ('ref2va', False), ('fl2va', False)]:
            cache = self.capture(mode, motion, manual={'width': 1024, 'height': 640})
            payload = cache['project_snapshot']['project']['extender']
            self.assertEqual((payload['settings']['width'], payload['settings']['height']), (1024, 640))
            self.assertEqual((payload['resolution']['manual_width'], payload['resolution']['manual_height']), (1024, 640))
            self.assertEqual((payload['resolution']['resolved_width'], payload['resolution']['resolved_height']), (896, 576))
        for invalid in [{'width': -1, 'height': '640'}, {'width': 33, 'height': None}, []]:
            payload = self.capture(manual=invalid)['project_snapshot']['project']['extender']
            self.assertEqual((payload['settings']['width'], payload['settings']['height']), (896, 576))

    def test_queued_snapshots_do_not_share_mutable_settings(self):
        first, second = self.capture(seed=10), self.capture(seed=11)
        second['project_snapshot']['project']['extender']['settings']['width'] = 32
        self.assertEqual(first['project_snapshot']['project']['extender']['settings']['width'], 896)
        self.assertEqual(json.loads(first['project_snapshot']['project']['extender']['clips_json'])['clips'][0]['seed'], 10)

    def test_option_default_false_appended_and_export_hooks_after_video(self):
        source = (ROOT/'motion_context_disk.py').read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3MotionContextDiskFinalDecode')
        input_types = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'INPUT_TYPES')
        method = copy.deepcopy(input_types)
        method.decorator_list = []
        ns = {'CACHE_TYPE': 'CACHE'}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'schema', 'exec'), ns)
        required = ns['INPUT_TYPES'](None)['required']
        self.assertIn('auto_save_project', required)
        self.assertIs(required['auto_save_project'][1]['default'], False)
        # Existing positional widgets stay in their historical order; #83-only
        # Final Decode widgets are appended afterwards for old-workflow safety.
        keys = list(required)
        self.assertLess(keys.index('auto_save_project'), keys.index('save_individual_clips'))
        self.assertLess(keys.index('save_individual_clips'), keys.index('latent_layer'))
        for filename, function in [('motion_context_disk.py', '_export_after_layer_select'), ('fl2va_engine.py', 'export_fl2va_final')]:
            module = ast.parse((ROOT/filename).read_text())
            node = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == function)
            arg_names = [arg.arg for arg in node.args.args]
            self.assertIn("auto_save_project" if function == "_export_after_layer_select" else "project_autosave_settings", arg_names)
            text = ast.unparse(node)
            self.assertLess(text.index('_embed_final_metadata_in_place(output_path'), text.index('_maybe_auto_save_project('))
            self.assertLess(text.index('_maybe_auto_save_project('), text.index('**project_autosave_info'))

if __name__ == '__main__':
    unittest.main()
