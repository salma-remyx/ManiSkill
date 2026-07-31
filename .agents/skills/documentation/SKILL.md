---
name: documentation
description: Establish and maintain project documentation at file and method levels. Use when creating, editing, reviewing, or auditing human-authored files, or when adding new methods, so files have concise purpose comments and new methods have properly formatted docstrings.
---

<!-- Defines the file-level and method-level documentation standards for the project. -->

# Documentation

Apply the relevant sections to every new or modified artifact in the requested scope.
Follow an established project or language convention when one exists.

## File Documentation

1. Inspect the beginning of each file before changing it.
2. Keep an existing purpose comment if it is short, accurate, and easy to identify.
3. Otherwise, add or update one sentence that states what responsibility the file
   has. Describe why the file exists, not its implementation details.
4. Use the file format's native comment syntax.
5. Place the comment at the earliest syntax-valid location. Keep shebangs, encoding
   declarations, required frontmatter, XML declarations, legal notices, and similar
   mandatory headers first.
6. Avoid duplicate purpose comments and update comments that become inaccurate when
   the file's responsibility changes.

Do not insert comments into strict formats that forbid them, binaries, generated
files, lockfiles, snapshots, vendored dependencies, or third-party artifacts. For an
exceptional file, document its purpose in the nearest owning documentation or
manifest without corrupting the file.

## Method Documentation

1. Add a docstring to every new method as part of its implementation.
2. Put the docstring directly under the method declaration, properly indented, before
   any executable statement.
3. Use the language's native docstring syntax and match the repository's established
   format. For Python, use triple double quotes and follow the existing Google,
   NumPy, or Sphinx convention in that package.
4. Start with a concise sentence describing the method's behavior or responsibility.
5. Document parameters, return values, yielded values, raised exceptions, and
   important side effects when applicable. Keep type information in annotations
   unless the local convention repeats it in docstrings.
6. Keep inherited documentation when it is accurate. Do not add empty, placeholder,
   duplicated, or implementation-narrating docstrings.

Example:

```python
def resolve_asset(self, name: str) -> Path:
    """Resolve an asset name to an existing local path.

    Args:
        name: Asset identifier relative to the configured root.

    Returns:
        The resolved local path.

    Raises:
        FileNotFoundError: If the asset does not exist.
    """
```

## Verify

Review every file and new method in scope before finishing. Confirm that each eligible
file has one accurate purpose comment and every new method has a correctly placed,
properly formatted, accurate docstring.
