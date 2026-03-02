import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement


def exists_on_pypi(name):
    try:
        urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json")
        return True
    except urllib.error.HTTPError:
        return False


def parse_req(line):
    try:
        return Requirement(line).name
    except Exception:
        return None


def extract_deps(root):
    names = set()

    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            name = parse_req(line)
            if name:
                names.add(name)

    toml = root / "pyproject.toml"
    if toml.exists():
        # Match all dependencies arrays in pyproject.toml
        content = toml.read_text()
        for match in re.finditer(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL):
            block = match.group(1)
            for m in re.finditer(r'"([^"]+)"', block):
                name = parse_req(m.group(1))
                if name:
                    names.add(name)

    return names


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    deps = extract_deps(root)
    missing = [n for n in sorted(deps) if not exists_on_pypi(n)]

    for n in missing:
        print(f"MISSING: {n}")

    if missing:
        sys.exit(1)

    print(f"All {len(deps)} packages found on PyPI.")
