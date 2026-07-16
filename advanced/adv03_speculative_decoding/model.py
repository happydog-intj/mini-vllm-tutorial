"""adv03: 复用 step07 的 TinyTransformerWithKVCache"""
import sys, os, importlib.util

_step07_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'step07_kvcache_for_single_request')
)
_model_path = os.path.join(_step07_dir, 'model.py')

# 用 spec_from_file_location 加载,避免与本文件同名产生循环引入
_spec = importlib.util.spec_from_file_location('step07_model', _model_path)
_mod = importlib.util.module_from_spec(_spec)

# step07/model.py 内部 sys.path.insert 依赖其自身目录在路径中
sys.path.insert(0, _step07_dir)
_spec.loader.exec_module(_mod)

TinyTransformerWithKVCache = _mod.TinyTransformerWithKVCache
