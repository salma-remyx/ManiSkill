# Checks that BaseEnv and make expose the same typed environment options.
# NOTE (stao): file is entirely vibe-coded

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ENV_PATH = REPO_ROOT / "mani_skill/envs/base_env.py"
MAKE_PATH = REPO_ROOT / "mani_skill/envs/make.py"
DOC_ARG_PATTERN = re.compile(r"^\s+(\w+):\s*(.*)$")

Parameter = tuple[str, str, str]


class ApiSyncError(Exception):
    """Raised when the public environment construction APIs differ."""


def _parse_file(path: Path) -> ast.Module:
    """Parse a Python source file without importing its dependencies."""
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ApiSyncError(
            f"Could not parse {path.relative_to(REPO_ROOT)}: {error}"
        ) from error


def _find_base_env_init(module: ast.Module) -> ast.FunctionDef:
    """Find the BaseEnv constructor in a parsed module."""
    base_env = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "BaseEnv"
        ),
        None,
    )
    if base_env is None:
        raise ApiSyncError(
            f"{BASE_ENV_PATH.relative_to(REPO_ROOT)} has no BaseEnv class"
        )

    init = next(
        (
            node
            for node in base_env.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise ApiSyncError("BaseEnv has no __init__ method")
    return init


def _find_make(module: ast.Module) -> ast.FunctionDef:
    """Find the top-level make function in a parsed module."""
    make = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "make"
        ),
        None,
    )
    if make is None:
        raise ApiSyncError(f"{MAKE_PATH.relative_to(REPO_ROOT)} has no make function")
    return make


def _shared_parameters(function: ast.FunctionDef, leading_name: str) -> list[Parameter]:
    """Extract the typed keyword-only parameters after a required leading parameter."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if [argument.arg for argument in positional] != [leading_name]:
        raise ApiSyncError(
            f"{function.name} must have only {leading_name!r} before its keyword-only options"
        )
    if arguments.defaults:
        raise ApiSyncError(
            f"{function.name}'s {leading_name!r} parameter must be required"
        )
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise ApiSyncError(f"{function.name} must not accept *args or **kwargs")

    parameters: list[Parameter] = []
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if argument.annotation is None:
            raise ApiSyncError(
                f"{function.name}'s {argument.arg!r} parameter has no type annotation"
            )
        annotation_text = ast.unparse(argument.annotation)
        default_text = "<required>" if default is None else ast.unparse(default)
        parameters.append((argument.arg, annotation_text, default_text))
    return parameters


def _documented_arguments(function: ast.FunctionDef) -> dict[str, str]:
    """Extract argument descriptions from a Google-style Args section."""
    docstring = ast.get_docstring(function, clean=True)
    if docstring is None:
        raise ApiSyncError(f"{function.name} has no docstring")

    descriptions: dict[str, list[str]] = {}
    current_name: str | None = None
    in_args = False
    for line in docstring.splitlines():
        if line == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if line and not line[0].isspace():
            break

        match = DOC_ARG_PATTERN.match(line)
        if match:
            argument_name = match.group(1)
            descriptions[argument_name] = [match.group(2).strip()]
            current_name = argument_name
        elif current_name is not None and line.strip():
            descriptions[current_name].append(line.strip())

    if not in_args:
        raise ApiSyncError(f"{function.name}'s docstring has no Args section")
    return {
        name: " ".join(part for part in parts if part)
        for name, parts in descriptions.items()
    }


def _check_documentation(
    base_init: ast.FunctionDef,
    make: ast.FunctionDef,
    parameter_names: list[str],
) -> None:
    """Check that both APIs document the same shared parameters identically."""
    base_docs = _documented_arguments(base_init)
    make_docs = _documented_arguments(make)
    expected_base_names = set(parameter_names)
    expected_make_names = {"env_id", *parameter_names}

    if set(base_docs) != expected_base_names:
        raise ApiSyncError(
            "BaseEnv.__init__ documentation does not match its parameters: "
            f"expected {sorted(expected_base_names)}, found {sorted(base_docs)}"
        )
    if set(make_docs) != expected_make_names:
        raise ApiSyncError(
            "make documentation does not match its parameters: "
            f"expected {sorted(expected_make_names)}, found {sorted(make_docs)}"
        )

    mismatches = [
        name for name in parameter_names if base_docs[name] != make_docs[name]
    ]
    if mismatches:
        details = "\n".join(
            f"  {name}: BaseEnv={base_docs[name]!r}, make={make_docs[name]!r}"
            for name in mismatches
        )
        raise ApiSyncError(f"Argument documentation differs:\n{details}")


def main() -> int:
    """Check the BaseEnv and make signatures and report actionable failures."""
    try:
        base_init = _find_base_env_init(_parse_file(BASE_ENV_PATH))
        make = _find_make(_parse_file(MAKE_PATH))
        base_parameters = _shared_parameters(base_init, "self")
        make_parameters = _shared_parameters(make, "env_id")

        if base_parameters != make_parameters:
            raise ApiSyncError(
                "BaseEnv.__init__ and make keyword parameters differ:\n"
                f"  BaseEnv: {base_parameters}\n"
                f"  make:    {make_parameters}"
            )

        _check_documentation(
            base_init,
            make,
            [name for name, _, _ in base_parameters],
        )
    except ApiSyncError as error:
        print(f"Environment API sync check failed: {error}", file=sys.stderr)
        return 1

    print("BaseEnv.__init__ and make arguments and documentation are synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
