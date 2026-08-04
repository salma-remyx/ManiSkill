---
name: python-init-reexports
description: Use explicit same-name imports for Python package re-exports. Apply when adding, changing, or reviewing convenience imports in an __init__.py file.
---

# Python Init Re-exports

Re-export each symbol explicitly with `X as X`:

```python
from .actor import ActorBuilder as ActorBuilder
from .records import PoseRecord as PoseRecord
```

Use the same convention inside grouped imports. Avoid wildcard imports and do not add `__all__` solely to declare these re-exports.
