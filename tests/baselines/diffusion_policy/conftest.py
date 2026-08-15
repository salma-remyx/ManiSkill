"""Optional-dependency shims for the diffusion_policy baseline tests.

``torch.utils.tensorboard`` is imported at module scope by the training
script, so importing that script fails on machines without TensorBoard even
though nothing under test touches logging. Pre-registering a no-op SummaryWriter
keeps the import working; the real package is used whenever it is installed.
"""

import sys
import types

try:  # pragma: no cover - exercised implicitly by the imports below
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    _tb = types.ModuleType("torch.utils.tensorboard")

    class _SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            return None

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

    _tb.SummaryWriter = _SummaryWriter
    _tb.FileWriter = object
    sys.modules.setdefault("torch.utils.tensorboard", _tb)
