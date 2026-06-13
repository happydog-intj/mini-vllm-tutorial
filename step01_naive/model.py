"""
step01: TinyTransformer（复用 step00d，独立引用保证每步自包含）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step00d_transformer'))
from transformer import TinyTransformer  # noqa: F401
