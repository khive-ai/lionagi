# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from typing import Any

from lionagi.utils import to_dict

# Try to import rich for enhanced console output
try:
    from rich.align import Align
    from rich.box import ROUNDED
    from rich.console import Console
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Custom themes for dark and light modes
DARK_THEME = Theme(
    {
        "info": "bright_cyan",
        "warning": "bright_yellow",
        "error": "bold bright_red",
        "success": "bold bright_green",
        "panel.border": "bright_blue",
        "panel.title": "bold bright_cyan",
        "markdown.h1": "bold bright_magenta",
        "markdown.h2": "bold bright_blue",
        "markdown.h3": "bold bright_cyan",
        "markdown.h4": "bold bright_green",
        "markdown.code": "bright_yellow on grey23",
        "markdown.code_block": "bright_white on grey15",
        "markdown.paragraph": "bright_white",
        "markdown.text": "bright_white",
        "markdown.emph": "italic bright_yellow",
        "markdown.strong": "bold bright_white",
        "markdown.item": "bright_cyan",
        "markdown.item.bullet": "bright_blue",
        "json.key": "bright_cyan",
        "json.string": "bright_green",
        "json.number": "bright_yellow",
        "json.boolean": "bright_magenta",
        "json.null": "bright_red",
        "yaml.key": "bright_cyan",
        "yaml.string": "bright_green",
        "yaml.number": "bright_yellow",
        "yaml.boolean": "bright_magenta",
    }
)

LIGHT_THEME = Theme(
    {
        "info": "blue",
        "warning": "dark_orange",
        "error": "bold red",
        "success": "bold green4",
        "panel.border": "blue",
        "panel.title": "bold blue",
        "markdown.h1": "bold dark_magenta",
        "markdown.h2": "bold dark_blue",
        "markdown.h3": "bold dark_cyan",
        "markdown.h4": "bold dark_green",
        "markdown.code": "dark_orange on grey93",
        "markdown.code_block": "black on grey82",
        "markdown.paragraph": "black",
        "markdown.text": "black",
        "markdown.emph": "italic dark_orange",
        "markdown.strong": "bold black",
        "markdown.item": "dark_blue",
        "markdown.item.bullet": "blue",
        "json.key": "dark_blue",
        "json.string": "dark_green",
        "json.number": "dark_orange",
        "json.boolean": "dark_magenta",
        "json.null": "dark_red",
        "yaml.key": "dark_blue",
        "yaml.string": "dark_green",
        "yaml.number": "dark_orange",
        "yaml.boolean": "dark_magenta",
    }
)


def in_notebook() -> bool:
    """
    Checks if we're running inside a Jupyter notebook.
    Returns True if yes, False otherwise.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return "ZMQInteractiveShell" in shell
    except Exception:
        return False


def in_console() -> bool:
    """
    Checks if we're running in a console/terminal environment.
    Returns True if stdout is a TTY and not in a notebook.
    """
    return (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and not in_notebook()
    )


def format_dict(data: Any, indent: int = 0) -> str:
    """
    Recursively format Python data (dicts, lists, strings, etc.) into a
    YAML-like readable string.

    - Multi-line strings are displayed using a '|' block style, each line indented.
    - Lists are shown with a '- ' prefix per item at the appropriate indentation.
    - Dict keys are shown as "key:" lines, with values on subsequent lines if complex.
    """
    lines = []
    prefix = "  " * indent  # 2 spaces per indent level

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dict
                lines.append(f"{prefix}{key}:")
                lines.append(format_dict(value, indent + 1))
            elif isinstance(value, list):
                # List under a key
                lines.append(f"{prefix}{key}:")
                for item in value:
                    item_str = format_dict(item, indent + 2).lstrip()
                    lines.append(f"{prefix}  - {item_str}")
            elif isinstance(value, str) and "\n" in value:
                # Multi-line string
                lines.append(f"{prefix}{key}: |")
                subprefix = "  " * (indent + 1)
                for line in value.splitlines():
                    lines.append(f"{subprefix}{line}")
            else:
                # Simple single-line scalar
                item_str = format_dict(value, indent + 1).lstrip()
                lines.append(f"{prefix}{key}: {item_str}")
        return "\n".join(lines)

    elif isinstance(data, list):
        # For top-level or nested lists
        for item in data:
            item_str = format_dict(item, indent + 1).lstrip()
            lines.append(f"{prefix}- {item_str}")
        return "\n".join(lines)

    # Base case: single-line scalar
    return prefix + str(data)


def as_readable(
    input_: Any,
    /,
    *,
    md: bool = False,
    format_curly: bool = False,
    display_str: bool = False,
    max_chars: int | None = None,
    use_rich: bool = True,
    theme: str = "dark",
    max_panel_width: int = 140,
) -> str:
    """
    Convert `input_` into a human-readable string. If `format_curly=True`, uses
    a YAML-like style (`format_dict`). Otherwise, pretty-printed JSON.

    - For Pydantic models or nested data, uses `to_dict` to get a dictionary.
    - If the result is a list of items, each is processed and concatenated.
    - When in console and rich is available, provides syntax highlighting.

    Args:
        input_: The data to convert (could be a single item or list).
        md: If True, wraps the final output in code fences for Markdown display.
        format_curly: If True, use `format_dict`. Otherwise, produce JSON text.
        display_str: If True, prints the output instead of returning it.
        max_chars: If set, truncates output to this many characters.
        use_rich: If True and rich is available, uses rich for console output.
        theme: Color theme - "dark" (default) or "light". Dark uses GitHub Dark Dimmed,
               light uses Solarized Light inspired colors.
        max_panel_width: Maximum width for panels and code blocks in characters.

    Returns:
        A formatted string representation of `input_` (unless display_str=True).
    """

    # 1) Convert the input to a Python dict/list structure
    #    (handles recursion, Pydantic models, etc.)
    def to_dict_safe(obj: Any) -> Any:
        # Attempt to call to_dict with typical recursion flags
        to_dict_kwargs = {
            "use_model_dump": True,
            "fuzzy_parse": True,
            "recursive": True,
            "recursive_python_only": False,
            "max_recursive_depth": 5,
        }
        return to_dict(obj, **to_dict_kwargs)

    def _inner(i_: Any) -> Any:
        items = []
        try:
            if isinstance(i_, list):
                # Already a list. Convert each item
                items = [to_dict_safe(x) for x in i_]
            else:
                # Single item
                maybe_list = to_dict_safe(i_)
                # If it's a list, store as items; else just single
                items = (
                    maybe_list
                    if isinstance(maybe_list, list)
                    else [maybe_list]
                )
        except Exception:
            # If conversion fails, fallback to str
            return str(i_)

        # 2) For each item in `items`, either format with YAML-like or JSON
        rendered = []
        for item in items:
            if format_curly:
                # YAML-like
                rendered.append(format_dict(item))
            else:
                # JSON approach
                try:
                    # Provide indentation, ensure ASCII not forced
                    rendered.append(
                        json.dumps(item, indent=2, ensure_ascii=False)
                    )
                except Exception:
                    # fallback
                    rendered.append(str(item))

        # 3) Combine
        final_str = "\n\n".join(rendered).strip()

        # 4) If Markdown requested, wrap with code fences
        #    - If we used format_curly, we might do "```yaml" instead. But user specifically asked for JSON code blocks previously
        if md:
            if format_curly:
                return f"```yaml\n{final_str}\n```"
            else:
                return f"```json\n{final_str}\n```"

        return final_str

    str_ = _inner(input_).strip()
    if max_chars is not None and len(str_) > max_chars:
        str1 = str_[:max_chars] + "...\n\n[Truncated output]\n\n"
        if str_.endswith("\n```"):
            str1 += "```"
        str_ = str1
    if display_str:
        if md and in_notebook():
            # If in IPython environment, display Markdown
            from IPython.display import Markdown, display

            display(Markdown(str_))
        elif RICH_AVAILABLE and in_console() and use_rich:
            # Use rich for enhanced console output
            # Select theme and syntax highlighting based on user preference
            console_theme = DARK_THEME if theme == "dark" else LIGHT_THEME
            syntax_theme = (
                "github-dark" if theme == "dark" else "solarized-light"
            )
            panel_style = "bright_blue" if theme == "dark" else "blue"

            console = Console(theme=console_theme)

            # Check if content looks like markdown prose (not code)
            is_markdown_prose = isinstance(str_, str) and (
                str_.startswith("#")
                or str_.startswith("**")
                or str_.startswith("- ")
                or str_.startswith("1.")
                or "<multi_reasoning>" in str_
                or "\n### " in str_
                or "\n## " in str_
                or "\n# " in str_
                or "│" in str_  # Rich table content
            )

            if md and is_markdown_prose:
                # Display as formatted markdown
                # Create markdown with max width
                from rich.markdown import Markdown as RichMarkdown

                md_content = RichMarkdown(str_, code_theme=syntax_theme)

                # Calculate appropriate width
                console_width = console.width
                panel_width = min(console_width - 4, max_panel_width)

                # Add left margin padding for better alignment
                panel = Panel(
                    Padding(md_content, (0, 2)),
                    border_style=panel_style,
                    box=ROUNDED,
                    width=panel_width,
                    expand=False,
                )

                # Left align with margin
                aligned_panel = Align.left(panel, pad=True)
                console.print(Padding(aligned_panel, (0, 0, 0, 4)))

            elif md:
                # Extract content from markdown code blocks if present
                content = str_
                if content.startswith("```") and content.endswith("```"):
                    # Remove code fences
                    lines = content.split("\n")
                    if len(lines) > 2:
                        lang = lines[0][3:].strip() or "json"
                        content = "\n".join(lines[1:-1])
                    else:
                        lang = "json"
                else:
                    lang = "yaml" if format_curly else "json"

                # Calculate appropriate width
                console_width = console.width
                panel_width = min(console_width - 4, max_panel_width)

                # Create syntax highlighted output
                syntax = Syntax(
                    content,
                    lang,
                    theme=syntax_theme,
                    line_numbers=True,
                    background_color="default",
                    word_wrap=True,
                )

                # Add left margin padding for better alignment
                panel = Panel(
                    syntax,
                    border_style=panel_style,
                    box=ROUNDED,
                    width=panel_width,
                    expand=False,
                )

                # Left align with margin
                aligned_panel = Align.left(panel, pad=True)
                console.print(Padding(aligned_panel, (0, 0, 0, 4)))

            else:
                # Plain text output with rich formatting
                if format_curly:
                    syntax = Syntax(
                        str_,
                        "yaml",
                        theme=syntax_theme,
                        background_color="default",
                        word_wrap=True,
                    )
                else:
                    syntax = Syntax(
                        str_,
                        "json",
                        theme=syntax_theme,
                        background_color="default",
                        word_wrap=True,
                    )

                # For plain syntax, add left margin
                # Create a constrained width container if console is too wide
                if console.width > max_panel_width:
                    content = Align.left(
                        syntax, width=max_panel_width, pad=False
                    )
                    # Add left margin
                    console.print(Padding(content, (0, 0, 0, 4)))
                else:
                    # Just add left margin
                    console.print(Padding(syntax, (0, 0, 0, 4)))
        else:
            # Fallback to regular print
            print(str_)
    else:
        return str_
