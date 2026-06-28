"""step13: 复用 step07 的模型"""
import importlib.util, os, sys

_src = os.path.join(os.path.dirname(__file__), '..', 'step07_kvcache_single', 'model.py')
_spec = importlib.util.spec_from_file_location('step03a_model', os.path.abspath(_src))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache  # noqa: F401
