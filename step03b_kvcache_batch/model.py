"""step03b: 复用 step03a 的 TinyTransformerWithKVCache"""
import sys, os, importlib
_step03a_path = os.path.join(os.path.dirname(__file__), '..', 'step03a_kvcache_single')
sys.path.insert(0, os.path.abspath(_step03a_path))
_mod = importlib.import_module('model')
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache
