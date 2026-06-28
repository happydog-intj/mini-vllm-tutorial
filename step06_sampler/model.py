"""step06: 复用 step05 的 TinyTransformer"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step05_naive'))
from model import TinyTransformer  # noqa: F401
