"""adv01: 复用 step07 的 TinyTransformerWithKVCache"""
import sys, os, importlib

_path = os.path.join(os.path.dirname(__file__), '..', '..', 'step07_kvcache_for_single_request')
sys.path.insert(0, os.path.abspath(_path))
_mod = importlib.import_module('model')
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache
