#!/usr/bin/env python3
"""Check tracked source without importing it or contacting external services.

Run from any directory with Python 3.10+. This is a structural check, not
validation of experiments, hardware behaviour, dependencies or remote links.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        ).decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        print("Run this check in a Git checkout (git clone), with Git installed.")
        return 2
    files = sorted({root / name for name in tracked if name})
    failures = []
    counts = {"python": 0, "json": 0, "notebook": 0, "local_links": 0}
    for path in files:
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        try:
            if path.suffix == ".py":
                ast.parse(path.read_bytes(), filename=str(rel))
                counts["python"] += 1
            elif path.suffix in {".json", ".ipynb"}:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                key = "notebook" if path.suffix == ".ipynb" else "json"
                if key == "notebook" and not (isinstance(data, dict) and data.get("nbformat") == 4 and isinstance(data.get("cells"), list)):
                    raise ValueError("Expected a version 4 Jupyter notebook with cells")
                counts[key] += 1
        except (SyntaxError, ValueError, UnicodeError) as exc:
            failures.append(f"{rel}: {exc}")

    # Deliberately check navigation documents, not illustrative links inside
    # code examples or private/generated research notes.
    for rel in ("README.md", "docs/REUSE.md", "CONTRIBUTING.md"):
        path = root / rel
        if not path.is_file():
            failures.append(f"Missing navigation document: {rel}")
            continue
        text = re.sub(r"```.*?```|~~~.*?~~~", "", path.read_text(encoding="utf-8"), flags=re.S)
        for match in re.finditer(r'!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)', text):
            target = match.group(1) or match.group(2)
            parsed = urlsplit(target)
            if parsed.scheme == "file":
                failures.append(f"{rel}: computer-local link {target}")
                continue
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (root if target.startswith("/") else path.parent) / unquote(parsed.path).lstrip("/")
            counts["local_links"] += 1
            if not destination.exists():
                failures.append(f"{rel}: missing local target {target}")
    for failure in failures:
        print(f"ERROR: {failure}")
    print("Structural checks: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("FAIL" if failures else "PASS (no training, hardware or external services executed)")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
