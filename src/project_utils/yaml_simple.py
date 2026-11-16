from __future__ import annotations


def load_simple_yaml(path: str) -> dict:
    """
    Load a small YAML config without external dependencies.

    Supported:
    - nested dicts via indentation (spaces)
    - scalars: int/float/bool/quoted-or-unquoted strings
    - comments starting with '#'
    """

    def parse_scalar(s: str):
        s = s.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        try:
            if any(ch in s for ch in (".", "e", "E")):
                return float(s)
            return int(s)
        except ValueError:
            return s

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines: list[tuple[int, str]] = []
    for line in raw_lines:
        if "#" in line:
            line = line.split("#", 1)[0]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for indent, content in lines:
        if ":" not in content:
            raise ValueError(f"Invalid line (missing ':'): {content!r}")
        key, rest = content.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Bad indentation near: {content!r}")
        parent = stack[-1][1]

        if rest == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_scalar(rest)

    if not isinstance(root, dict):
        raise TypeError(f"Expected dict at root of {path}")
    return root
