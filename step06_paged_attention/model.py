"""step06: 复用 step03a 的模型"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step03a_kvcache_single'))
from model import TinyTransformerWithKVCache  # noqa: F401
