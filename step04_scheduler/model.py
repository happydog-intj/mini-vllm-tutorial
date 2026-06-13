"""step04: 复用 step03a 的模型"""
import sys, os, importlib.util

_src = os.path.join(os.path.dirname(__file__), '..', 'step03a_kvcache_single', 'model.py')
_spec = importlib.util.spec_from_file_location("step03a_model", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache
