"""step12: 复用 step09 的调度器（importlib 加载，避免与本地 scheduler.py 同名循环导入）"""
import importlib.util, os

_src = os.path.join(os.path.dirname(__file__), '..', 'step09_kvcache_continuous_batching_for_multi_requests', 'scheduler.py')
_spec = importlib.util.spec_from_file_location('step09_kvcache_continuous_batching_for_multi_requests', os.path.abspath(_src))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Sequence = _mod.Sequence
Scheduler = _mod.Scheduler
SequenceStatus = _mod.SequenceStatus  # noqa: F401
