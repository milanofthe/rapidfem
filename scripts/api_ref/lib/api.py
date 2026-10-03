# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""
API extraction utilities using griffe.
"""

import re
import textwrap
from pathlib import Path
from typing import Any

try:
    import griffe
    from griffe import GriffeLoader
except ImportError:
    raise ImportError("griffe not installed. Run: pip install griffe")

try:
    from docutils.core import publish_parts
    HAS_DOCUTILS = True
except ImportError:
    HAS_DOCUTILS = False

from .config import SKIP_PATTERNS


def extract_api(package_id: str, source_path: Path, root_modules: list[str]) -> dict[str, Any]:
    """Extract API documentation for a package."""
    if not source_path.exists():
        print(f"    Warning: Source path not found: {source_path}")
        return {"package": package_id, "modules": {}}

    result = {
        "package": package_id,
        "modules": {},
    }

    for root_module in root_modules:
        discovered = _discover_modules(source_path, root_module)
        for module_path in discovered:
            if _should_skip_module(module_path):
                continue

            try:
                module_data = _extract_module(source_path, module_path)
                if module_data and (module_data["classes"] or module_data["functions"]):
                    result["modules"][module_path] = module_data
            except Exception as e:
                print(f"    Warning: Failed to extract {module_path}: {e}")

    return result


def _discover_modules(source_path: Path, root_module: str) -> list[str]:
    """Recursively discover all modules and subpackages under a root module path."""
    modules = []
    parts = root_module.split(".")
    module_dir = source_path / Path(*parts)

    if not module_dir.exists():
        return modules

    if module_dir.is_dir():
        init_file = module_dir / "__init__.py"
        if init_file.exists():
            modules.append(root_module)

        for py_file in sorted(module_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            submodule = f"{root_module}.{py_file.stem}"
            if not _should_skip_module(submodule):
                modules.append(submodule)

        # recurse into subpackages
        for subdir in sorted(module_dir.iterdir()):
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                subpackage = f"{root_module}.{subdir.name}"
                if not _should_skip_module(subpackage):
                    modules.extend(_discover_modules(source_path, subpackage))

    return modules


def _should_skip_module(name: str) -> bool:
    """Check if a module name matches skip patterns."""
    for pattern in SKIP_PATTERNS:
        if pattern in name:
            return True
    return False


def _extract_module(source_path: Path, module_path: str) -> dict | None:
    """Extract a single module."""
    try:
        loader = GriffeLoader(
            search_paths=[str(source_path)],
            docstring_parser="numpy",
            allow_inspection=False,
        )
        module = loader.load(module_path)
        return _extract_module_obj(module, module_path)
    except Exception:
        return None


def _extract_module_obj(obj: griffe.Object, module_path: str) -> dict:
    """Extract module data from griffe object."""
    module_data = {
        "name": module_path,
        "description": "",
        "docstring_html": "",
        "classes": [],
        "functions": [],
    }

    if not hasattr(obj, "members"):
        return module_data

    for name, member in obj.members.items():
        if _should_skip_member(name):
            continue

        if not _is_defined_here(member, module_path):
            continue

        try:
            if member.is_class:
                class_data = _extract_class(member)
                if class_data:
                    module_data["classes"].append(class_data)
            elif member.is_function:
                func_data = _extract_function(member)
                if func_data:
                    module_data["functions"].append(func_data)
        except Exception:
            continue

    return module_data


def _should_skip_member(name: str) -> bool:
    """Skip private members, only the public API is documented.

    `__init__` is kept so constructor parameters can be extracted; every
    other underscore-prefixed name (dunders and single-underscore privates)
    is omitted.
    """
    if name == "__init__":
        return False
    return name.startswith("_")


def _is_defined_here(member: griffe.Object, module_path: str) -> bool:
    """Check if a member is defined in this module (not imported)."""
    try:
        if hasattr(member, 'canonical_path'):
            canonical = str(member.canonical_path)
            source_module = canonical.rsplit('.', 1)[0] if '.' in canonical else canonical
            return source_module == module_path
        return True
    except Exception:
        return True


def _extract_class(cls: griffe.Class) -> dict | None:
    """Extract class documentation."""
    try:
        docstring = cls.docstring.value if cls.docstring else ""

        class_data = {
            "name": cls.name,
            "description": _extract_first_line(docstring),
            "docstring_html": _rst_to_html(docstring),
            "source": _extract_source(cls),
            "bases": [],
            "methods": [],
            "attributes": [],
            "parameters": [],
        }

        # Extract bases
        try:
            if cls.bases:
                for base in cls.bases:
                    try:
                        if hasattr(base, 'canonical_path') and base.canonical_path:
                            class_data["bases"].append(str(base.canonical_path))
                        else:
                            class_data["bases"].append(str(base))
                    except Exception:
                        class_data["bases"].append(str(base))
        except Exception:
            pass

        # Extract __init__ parameters
        try:
            if hasattr(cls, "members") and "__init__" in cls.members:
                init = cls.members["__init__"]
                init_doc = init.docstring.value if init.docstring else docstring
                class_data["parameters"] = _extract_parameters(init, init_doc)
        except Exception:
            pass

        # Extract methods and attributes
        if hasattr(cls, "members"):
            for name, member in cls.members.items():
                if _should_skip_member(name):
                    continue

                try:
                    if member.is_function:
                        method_data = _extract_method(member)
                        if method_data:
                            class_data["methods"].append(method_data)
                    elif member.is_attribute:
                        attr_data = _extract_attribute(member)
                        if attr_data:
                            class_data["attributes"].append(attr_data)
                except Exception:
                    continue

        return class_data
    except Exception:
        return None


def _extract_function(func: griffe.Function) -> dict | None:
    """Extract function documentation."""
    try:
        docstring = func.docstring.value if func.docstring else ""

        return {
            "name": func.name,
            "description": _extract_first_line(docstring),
            "docstring_html": _rst_to_html(docstring),
            "source": _extract_source(func),
            "signature": _get_signature(func),
            "parameters": _extract_parameters(func, docstring),
            "returns": str(func.returns) if hasattr(func, "returns") and func.returns else None,
        }
    except Exception:
        return None


def _extract_method(method: griffe.Function) -> dict | None:
    """Extract method documentation."""
    try:
        docstring = method.docstring.value if method.docstring else ""

        method_type = "method"
        try:
            for decorator in method.decorators:
                dec_str = str(decorator.value)
                if "classmethod" in dec_str:
                    method_type = "classmethod"
                elif "staticmethod" in dec_str:
                    method_type = "staticmethod"
                elif "property" in dec_str:
                    method_type = "property"
        except Exception:
            pass

        return {
            "name": method.name,
            "description": _extract_first_line(docstring),
            "docstring_html": _rst_to_html(docstring),
            "source": _extract_source(method),
            "signature": _get_signature(method),
            "parameters": _extract_parameters(method, docstring),
            "returns": str(method.returns) if hasattr(method, "returns") and method.returns else None,
            "method_type": method_type,
        }
    except Exception:
        return None


def _extract_attribute(attr: griffe.Attribute) -> dict | None:
    """Extract attribute documentation."""
    try:
        docstring = attr.docstring.value if attr.docstring else ""

        return {
            "name": attr.name,
            "description": _extract_first_line(docstring),
            "type": str(attr.annotation) if hasattr(attr, "annotation") and attr.annotation else None,
            "value": str(attr.value) if hasattr(attr, "value") and attr.value else None,
        }
    except Exception:
        return None


def _extract_source(obj: griffe.Object) -> str | None:
    """Extract source code for a griffe object."""
    try:
        filepath = obj.filepath
        lineno = obj.lineno
        endlineno = obj.endlineno

        if not filepath or not lineno or not endlineno:
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        source_lines = lines[lineno - 1:endlineno]
        source = ''.join(source_lines)

        if source:
            source = textwrap.dedent(source)

        return source.rstrip() if source else None
    except Exception:
        return None


# Sphinx cross-reference role, e.g. :meth:`~pkg.Class.method` or :class:`X <Y>`.
_ROLE_RE = re.compile(r':\w+:`(?:[^`<]*<)?~?([^`<>]+)>?`')


def _strip_roles(text: str) -> str:
    """Reduce Sphinx roles to their plain target text."""
    return _ROLE_RE.sub(lambda m: m.group(1).strip(), text)


def _extract_first_line(docstring: str | None) -> str:
    """Extract first line/sentence as brief description."""
    if not docstring:
        return ""

    lines = docstring.strip().split("\n")
    first_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if ". " in first_line and len(first_line) > 100:
        first_line = first_line.split(". ")[0] + "."

    return _strip_roles(first_line)


def _get_signature(obj: griffe.Object) -> str | None:
    """Get function/method signature as string."""
    try:
        if not hasattr(obj, "parameters"):
            return None

        params = []
        for param in obj.parameters:
            if param.name == "self":
                continue
            param_str = param.name
            if param.annotation:
                param_str += f": {param.annotation}"
            if param.default is not None and str(param.default) != "":
                default_str = str(param.default)
                if len(default_str) > 50:
                    default_str = default_str[:47] + "..."
                param_str += f" = {default_str}"
            params.append(param_str)

        return f"({', '.join(params)})"
    except Exception:
        return None


def _extract_parameters(obj: griffe.Object, docstring: str = "") -> list[dict]:
    """Extract function/method parameters."""
    try:
        if not hasattr(obj, "parameters"):
            return []

        params = []
        for param in obj.parameters:
            if param.name == "self":
                continue

            param_info = {
                "name": param.name,
                "type": str(param.annotation) if param.annotation else None,
                "default": str(param.default) if param.default is not None and str(param.default) != "" else None,
                "description": _extract_param_description(docstring, param.name),
            }
            params.append(param_info)

        return params
    except Exception:
        return []


def _extract_param_description(docstring: str, param_name: str) -> str:
    """Extract parameter description from docstring."""
    if not docstring:
        return ""

    # NumPy style
    pattern = rf'{param_name}\s*:\s*[^\n]*\n\s+(.+?)(?=\n\s*\w+\s*:|\n\n|$)'
    match = re.search(pattern, docstring, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        desc = re.sub(r'\s+', ' ', desc)
        return desc

    # Google style
    pattern = rf'{param_name}:\s*(.+?)(?=\n\s*\w+:|\n\n|$)'
    match = re.search(pattern, docstring, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        desc = re.sub(r'\s+', ' ', desc)
        return desc

    return ""


def _rst_to_html(rst_text: str) -> str:
    """Convert RST docstring to HTML."""
    if not rst_text:
        return ""

    if not HAS_DOCUTILS:
        return _markdown_fallback(rst_text)

    try:
        processed = _preprocess_numpy_docstring(rst_text)
        processed = _preprocess_rst_roles(processed)

        parts = publish_parts(
            processed,
            writer_name="html",
            settings_overrides={
                "report_level": 5,
                "halt_level": 5,
                "initial_header_level": 4,
                "math_output": "LaTeX",
                "syntax_highlight": "short",
            }
        )
        return parts["body"]
    except Exception:
        return _markdown_fallback(rst_text)


def _preprocess_rst_roles(text: str) -> str:
    """Convert Sphinx-specific RST roles to standard RST."""
    # :doi:`...` -> hyperlink to https://doi.org/...
    text = re.sub(r':doi:`([^`]+)`', r'`doi:\1 <https://doi.org/\1>`_', text)
    # Cross-reference roles (:meth:, :class:, :func:, ...) -> inline literal.
    text = _ROLE_RE.sub(lambda m: f'``{m.group(1).strip()}``', text)
    return text


def _preprocess_numpy_docstring(docstring: str) -> str:
    """Convert NumPy-style docstring sections to RST."""
    sections = [
        "Parameters", "Returns", "Yields", "Raises", "Warns",
        "Attributes", "Methods", "See Also", "Notes", "References",
        "Examples", "Other Parameters"
    ]

    result = docstring
    for section in sections:
        pattern = rf'^({section})\s*\n-+\s*\n'
        replacement = rf'**{section}**\n\n'
        result = re.sub(pattern, replacement, result, flags=re.MULTILINE)

    return result


def _markdown_fallback(text: str) -> str:
    """Basic markdown-style conversion as fallback."""
    if not text:
        return ""

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'```(\w*)\n(.*?)\n```', r'<pre><code class="language-\1">\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'``(.*?)``', r'<code>\1</code>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)

    paragraphs = text.split('\n\n')
    text = '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())

    return text
