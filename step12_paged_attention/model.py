"""step12: 复用 step07 的模型"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step07_kvcache_single'))
from model import TinyTransformerWithKVCache  # noqa: F401
