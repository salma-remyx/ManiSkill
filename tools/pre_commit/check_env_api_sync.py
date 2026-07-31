# Synchronizes the public make API with BaseEnv.__init__ before changes are committed.

from __future__ import annotations

import ast
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ENV_PATH = REPO_ROOT / "mani_skill/envs/base_env.py"
MAKE_PATH = REPO_ROOT / "mani_skill/envs/make.py"
DOC_ARG_PATTERN = re.compile(r"^\s+(\w+):\s*(.*)$")


class ApiSyncError(Exception):
    """Raised when an environment API cannot be read or synchronized safely."""


@dataclass(frozen=True)
class Parameter:
    """Store the source-level parts of a keyword-only parameter."""

    name: str
    annotation: str
    default: str | None

    def render(self) -> str:
        """Render the parameter for a generated function signature."""
        default = "" if self.default is None else f" = {self.default}"
        return f"{self.name}: {self.annotation}{default}"


def _parse_source(source: str, path: Path) -> ast.Module:
    """Parse Python source without importing its dependencies."""
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise ApiSyncError(
            f"Could not parse {path.relative_to(REPO_ROOT)}: {error}"
        ) from error


def _read_source(path: Path) -> tuple[str, ast.Module]:
    """Read and parse a Python source file."""
    try:
        source = path.read_text()
    except OSError as error:
        raise ApiSyncError(
            f"Could not read {path.relative_to(REPO_ROOT)}: {error}"
        ) from error
    return source, _parse_source(source, path)


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


def _validate_function_shape(function: ast.FunctionDef, leading_name: str) -> ast.arg:
    """Validate the fixed leading argument and return it."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if [argument.arg for argument in positional] != [leading_name]:
        raise ApiSyncError(
            f"{function.name} must have only {leading_name!r} before its "
            "keyword-only options"
        )
    if arguments.defaults:
        raise ApiSyncError(
            f"{function.name}'s {leading_name!r} parameter must be required"
        )
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise ApiSyncError(f"{function.name} must not accept *args or **kwargs")
    return positional[0]


def _shared_parameters(
    function: ast.FunctionDef, leading_name: str, source: str | None = None
) -> list[Parameter]:
    """Extract typed keyword-only parameters after a required leading parameter."""
    _validate_function_shape(function, leading_name)
    parameters: list[Parameter] = []
    for argument, default in zip(
        function.args.kwonlyargs, function.args.kw_defaults, strict=True
    ):
        if argument.annotation is None:
            raise ApiSyncError(
                f"{function.name}'s {argument.arg!r} parameter has no type annotation"
            )
        parameters.append(
            Parameter(
                name=argument.arg,
                annotation=(
                    ast.get_source_segment(source, argument.annotation)
                    if source is not None
                    else ast.unparse(argument.annotation)
                )
                or ast.unparse(argument.annotation),
                default=(
                    None
                    if default is None
                    else (
                        ast.get_source_segment(source, default)
                        if source is not None
                        else ast.unparse(default)
                    )
                    or ast.unparse(default)
                ),
            )
        )
    return parameters


def _documented_arguments(function: ast.FunctionDef) -> dict[str, str]:
    """Extract normalized argument descriptions from a Google-style Args section."""
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
            if argument_name in descriptions:
                raise ApiSyncError(
                    f"{function.name} documents {argument_name!r} more than once"
                )
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


def _validate_base_documentation(
    base_init: ast.FunctionDef, parameters: list[Parameter]
) -> dict[str, str]:
    """Return complete BaseEnv constructor documentation for its parameters."""
    base_docs = _documented_arguments(base_init)
    expected_names = {parameter.name for parameter in parameters}
    if set(base_docs) != expected_names:
        raise ApiSyncError(
            "BaseEnv.__init__ documentation does not match its parameters: "
            f"expected {sorted(expected_names)}, found {sorted(base_docs)}"
        )
    return base_docs


def _format_doc_argument(name: str, description: str) -> list[str]:
    """Format one argument description within a cleaned docstring."""
    return textwrap.wrap(
        description,
        width=84,
        initial_indent=f"    {name}: ",
        subsequent_indent="        ",
        break_long_words=False,
        break_on_hyphens=False,
    ) or [f"    {name}:"]


def _synchronized_docstring(
    make: ast.FunctionDef,
    parameters: list[Parameter],
    base_docs: dict[str, str],
) -> str:
    """Build make's docstring with BaseEnv's ordered argument documentation."""
    docstring = ast.get_docstring(make, clean=True)
    if docstring is None:
        raise ApiSyncError("make has no docstring")
    make_docs = _documented_arguments(make)
    if "env_id" not in make_docs:
        raise ApiSyncError("make's docstring does not document 'env_id'")

    lines = docstring.splitlines()
    args_index = lines.index("Args:")
    section_end = len(lines)
    for index in range(args_index + 1, len(lines)):
        if lines[index] and not lines[index][0].isspace():
            section_end = index
            break

    argument_lines: list[str] = []
    descriptions = [("env_id", make_docs["env_id"])]
    descriptions.extend(
        (parameter.name, base_docs[parameter.name]) for parameter in parameters
    )
    for name, description in descriptions:
        argument_lines.extend(_format_doc_argument(name, description))

    suffix = lines[section_end:]
    while suffix and not suffix[0]:
        suffix.pop(0)
    replacement = ["Args:", *argument_lines]
    if suffix:
        replacement.append("")
    return "\n".join([*lines[:args_index], *replacement, *suffix])


def _render_docstring(docstring: str) -> str:
    """Render a cleaned docstring at make's function-body indentation."""
    if '"""' in docstring:
        raise ApiSyncError("make's docstring contains an unsupported triple quote")
    lines = docstring.splitlines()
    if not lines:
        return '""""""'
    rendered = [f'"""{lines[0]}']
    rendered.extend(f"    {line}" if line else "" for line in lines[1:])
    rendered.append('    """')
    return "\n".join(rendered)


def _render_signature(
    make: ast.FunctionDef, leading_argument: ast.arg, parameters: list[Parameter]
) -> str:
    """Render make's signature using BaseEnv's shared parameters."""
    if leading_argument.annotation is None:
        raise ApiSyncError("make's 'env_id' parameter has no type annotation")
    leading = f"{leading_argument.arg}: {ast.unparse(leading_argument.annotation)}"
    returns = "" if make.returns is None else f" -> {ast.unparse(make.returns)}"
    lines = ["def make(", f"    {leading},", "    *,"]
    lines.extend(f"    {parameter.render()}," for parameter in parameters)
    lines.append(f"){returns}:")
    return "\n".join(lines) + "\n"


def _find_constructor_call(make: ast.FunctionDef) -> ast.Call:
    """Find the direct constructor call returned by make."""
    returns = [node for node in make.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise ApiSyncError("make must directly return exactly one constructor call")
    call = returns[0].value
    if call.args:
        raise ApiSyncError("make's constructor call must use keyword arguments")
    return call


def _render_constructor_call(
    source: str, call: ast.Call, parameters: list[Parameter]
) -> str:
    """Render make's constructor call with every shared parameter forwarded."""
    function = ast.get_source_segment(source, call.func)
    if function is None:
        raise ApiSyncError("Could not read make's constructor expression")
    lines = [f"{function}("]
    lines.extend(
        f"        {parameter.name}={parameter.name}," for parameter in parameters
    )
    lines.append("    )")
    return "\n".join(lines)


def _line_offsets(source: str) -> list[int]:
    """Return absolute offsets for the start of each source line."""
    offsets = [0]
    for match in re.finditer("\n", source):
        offsets.append(match.end())
    return offsets


def _node_span(node: ast.AST, offsets: list[int]) -> tuple[int, int]:
    """Return an AST node's absolute character span."""
    if not all(
        hasattr(node, attribute)
        for attribute in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        raise ApiSyncError("Could not determine a source node's location")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping source replacements from bottom to top."""
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _synchronize_make(
    base_source: str,
    base_init: ast.FunctionDef,
    make_source: str,
    make: ast.FunctionDef,
) -> str:
    """Rewrite make's signature, documentation, and forwarded arguments."""
    parameters = _shared_parameters(base_init, "self", base_source)
    base_docs = _validate_base_documentation(base_init, parameters)
    leading_argument = _validate_function_shape(make, "env_id")
    if not make.body or not (
        isinstance(make.body[0], ast.Expr)
        and isinstance(make.body[0].value, ast.Constant)
        and isinstance(make.body[0].value.value, str)
    ):
        raise ApiSyncError("make's docstring must be its first statement")
    docstring_node = make.body[0].value
    constructor_call = _find_constructor_call(make)
    offsets = _line_offsets(make_source)

    signature_start = offsets[make.lineno - 1]
    signature_end = offsets[docstring_node.lineno - 1]
    replacements = [
        (
            signature_start,
            signature_end,
            _render_signature(make, leading_argument, parameters),
        ),
        (
            *_node_span(docstring_node, offsets),
            _render_docstring(_synchronized_docstring(make, parameters, base_docs)),
        ),
        (
            *_node_span(constructor_call, offsets),
            _render_constructor_call(make_source, constructor_call, parameters),
        ),
    ]
    synchronized = _apply_replacements(make_source, replacements)

    synchronized_make = _find_make(_parse_source(synchronized, MAKE_PATH))
    if _shared_parameters(synchronized_make, "env_id", synchronized) != parameters:
        raise ApiSyncError("Generated make signature did not match BaseEnv.__init__")
    synchronized_docs = _documented_arguments(synchronized_make)
    expected_docs = {"env_id": _documented_arguments(make)["env_id"], **base_docs}
    if synchronized_docs != expected_docs:
        raise ApiSyncError(
            "Generated make documentation did not match BaseEnv.__init__"
        )
    return synchronized


def main() -> int:
    """Synchronize make with BaseEnv.__init__, writing changes when needed."""
    try:
        base_source, base_module = _read_source(BASE_ENV_PATH)
        make_source, make_module = _read_source(MAKE_PATH)
        synchronized = _synchronize_make(
            base_source,
            _find_base_env_init(base_module),
            make_source,
            _find_make(make_module),
        )
        if synchronized != make_source:
            MAKE_PATH.write_text(synchronized)
            print(
                "Updated mani_skill/envs/make.py from BaseEnv.__init__. "
                "Stage the changes and run pre-commit again."
            )
        else:
            print("BaseEnv.__init__ and make are synchronized.")
    except (ApiSyncError, OSError) as error:
        print(f"Environment API synchronization failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
