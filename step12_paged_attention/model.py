"""step12: 复用 step07 的模型（importlib 加载，避免与本地 model.py 同名循环导入）"""
import importlib.util, os

_src = os.path.join(os.path.dirname(__file__), '..', 'step07_kvcache_single', 'model.py')
_spec = importlib.util.spec_from_file_location('step07_model', os.path.abspath(_src))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache  # noqa: F401
