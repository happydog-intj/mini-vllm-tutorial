"""step02: 复用 step01 的 TinyTransformer"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step01_naive'))
from model import TinyTransformer  # noqa: F401
