"""Tệp mẫu và tài liệu cấu hình phải nói đúng những gì mã thực sự đọc.

Ba thứ được ghim ở đây, vì cả ba đều đã từng sai trong repo này:

* Một khoá trong ``.env.example`` mà mã không đọc là một biến bịa — người dùng
  đặt nó, không có gì xảy ra, và không có gì báo. Tên biến được **đọc ra từ
  ``src/``** chứ không chép tay, nên thêm biến mới vào mã là test này nhắc cập
  nhật tệp mẫu.
* Bản mẫu cũ gán giá trị cho gần như mọi biến, nên mặc định trong mã không bao
  giờ tới được lượt chạy. Giờ chỉ bốn nhóm endpoint là **đang bật**; phần còn
  lại là comment kèm mặc định, và mặc định đó phải khớp mã.
* Một bảng trong tài liệu chỉ có giá trị khi nó phủ hết biến — thiếu một dòng
  nghĩa là người đọc không biết biến đó tồn tại.

Định dạng bảng của ``docs/configuration.md`` là một phần giao kèo của test:
``| `TÊN` | `MẶC ĐỊNH` | giải thích |`` — ô thứ hai và thứ ba được đọc ra ở đây.
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "polyglot_lc"
CONFIG = SRC / "config.py"
EXAMPLE = ROOT / ".env.example"
GUIDE = ROOT / "docs" / "configuration.md"

#: Bốn nhóm endpoint: thứ duy nhất một ``.env`` mới nhận sẵn.
REQUIRED_ACTIVE_KEYS = frozenset(
    {
        "PLANNER_BASE_URL",
        "PLANNER_MODEL",
        "PLANNER_API_KEY",
        "SOURCE_GROUP_BASE_URL",
        "SOURCE_GROUP_MODEL",
        "SOURCE_GROUP_API_KEY",
        "VERIFIER_BASE_URL",
        "VERIFIER_MODEL",
        "VERIFIER_API_KEY",
        "EMBED_BASE_URL",
        "EMBED_MODEL",
        "EMBED_API_KEY",
    }
)


# --------------------------------------------------------------- what the code reads
def _config_text() -> str:
    return CONFIG.read_text(encoding="utf-8")


def _agent_names() -> list[str]:
    """The five agents that own a ``<NAME>_*`` set (``AGENT_GROUPS``' keys)."""
    block = re.search(r"AGENT_GROUPS[^=]*=\s*\{(.*?)\}", _config_text(), re.DOTALL)
    assert block, "AGENT_GROUPS moved or changed shape; this test needs updating"
    return re.findall(r'"([A-Z_]+)"\s*:', block.group(1))


def _group_prefix() -> str:
    found = re.search(r'SOURCE_GROUP_PREFIX\s*=\s*"([A-Z_]+)"', _config_text())
    assert found, "SOURCE_GROUP_PREFIX moved; this test needs updating"
    return found.group(1)


def _prefixed_env_names() -> set[str]:
    """``<NAME>_SUFFIX`` names, expanded from the code's own f-string reads.

    ``config.py`` builds these from two locals — ``prefix = f"{name}_"`` for an
    agent and ``p = SOURCE_GROUP_PREFIX`` for the shared group — so the suffix
    list and the prefixes both come from the source rather than from a copy
    pasted into this test.
    """
    suffixes: dict[str, set[str]] = {}
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for var, suffix in re.findall(
            r'(?:os\.getenv|os\.environ\.get|_get_[a-z]+_env)\(f"\{([a-z_]+)\}([A-Z_]+)"',
            text,
        ):
            suffixes.setdefault(var, set()).add(suffix)
    assert suffixes, "no f-string env read found in the code; this test needs updating"

    names: set[str] = set()
    for var, found in suffixes.items():
        if var == "p":
            prefixes = [_group_prefix()]
        else:
            prefixes = [f"{agent}_" for agent in _agent_names()]
        for prefix in prefixes:
            names.update(f"{prefix}{suffix}" for suffix in found)
    return names


def _python_sources() -> list[Path]:
    paths = [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]
    paths += [p for p in (ROOT / "webui").rglob("*.py") if "__pycache__" not in str(p)]
    return paths


def _code_env_names() -> set[str]:
    """Every environment variable name the code reads, by any of its spellings."""
    names: set[str] = set()
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        names.update(re.findall(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"', text))
        names.update(re.findall(r'os\.environ(?:\.get\(|\[)\s*"([A-Z][A-Z0-9_]*)"', text))
        names.update(re.findall(r'_get_[a-z]+_env\(\s*"([A-Z][A-Z0-9_]*)"', text))
    names.update(_prefixed_env_names())
    return names


def _numeric_and_bool_defaults() -> dict[str, str]:
    """Code defaults for the settings the guide states as a value.

    Numbers are returned as written (``"7.0"``, ``"45"``), booleans as the
    ``1``/``0`` a ``.env`` file speaks. Module constants used as a fallback
    (``DEFAULT_TIMEOUT``) are resolved, so ``LLM_TIMEOUT`` is pinned too.
    """
    text = _config_text()
    constants = {
        name: str(value)
        for name, value in re.findall(
            r"^([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*$", text, re.MULTILINE
        )
    }
    out: dict[str, str] = {}
    for kind, name, fallback in re.findall(
        r'_get_(int|float|bool|str)_env\(\s*"([A-Z][A-Z0-9_]*)"\s*,\s*([^,)]+)', text
    ):
        value = fallback.strip()
        if kind in {"int", "float"}:
            if value in constants:
                value = constants[value]
            if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                out[name] = value
        elif kind == "bool" and value in {"True", "False"}:
            out[name] = "1" if value == "True" else "0"
    return out


# --------------------------------------------------------------- what the files say
def _assignments(text: str) -> list[tuple[str, str, bool]]:
    """``(name, value, active)`` for every assignment-shaped line, comments included."""
    rows: list[tuple[str, str, bool]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(#?)\s*([A-Z][A-Z0-9_]*)\s*=(.*)$", line)
        if match:
            rows.append((match.group(2), match.group(3).strip(), match.group(1) != "#"))
    return rows


def _guide_defaults() -> dict[str, str]:
    """``{name: stated default}`` from the guide's tables."""
    stated: dict[str, str] = {}
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 4:
            continue
        name = cells[1].strip("`")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            continue
        stated[name] = cells[2].strip("`")
    return stated


# --------------------------------------------------------------------------- tests
def test_every_key_in_the_template_is_a_variable_the_code_reads():
    known = _code_env_names()
    unknown = sorted(
        name
        for name, _value, _active in _assignments(EXAMPLE.read_text(encoding="utf-8"))
        if name not in known
    )
    assert not unknown, f".env.example names variables no code reads: {unknown}"


def test_the_template_activates_only_the_required_endpoints():
    rows = _assignments(EXAMPLE.read_text(encoding="utf-8"))
    active = {name for name, _value, is_active in rows if is_active}
    assert active == REQUIRED_ACTIVE_KEYS, (
        "a new .env should have to fill in the four endpoint groups and nothing else; "
        f"missing={sorted(REQUIRED_ACTIVE_KEYS - active)} "
        f"unexpected={sorted(active - REQUIRED_ACTIVE_KEYS)}"
    )


def test_the_template_parses_to_its_active_keys_and_carries_no_junk():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "[TEMPLATE]" not in text, "dotenv exports that junk key as an empty variable"
    parsed = set(dotenv_values(EXAMPLE))
    active = {name for name, _value, is_active in _assignments(text) if is_active}
    assert parsed == active
    assert len(parsed) == len(REQUIRED_ACTIVE_KEYS)


def test_the_guide_documents_every_variable_the_code_reads():
    guide = GUIDE.read_text(encoding="utf-8")
    stated = _guide_defaults()
    missing = sorted(name for name in _code_env_names() if name not in stated)
    assert not missing, f"docs/configuration.md has no row for: {missing}"
    for name in sorted(stated):
        assert f"`{name}`" in guide, name


def test_the_template_and_the_guide_state_the_code_defaults():
    """A commented tunable that lies about its default is worse than none."""
    defaults = _numeric_and_bool_defaults()
    assert defaults, "no numeric/bool default found in the code; this test needs updating"
    stated = _guide_defaults()

    def same(left: str, right: str) -> bool:
        try:
            return float(left) == float(right)
        except ValueError:
            return left == right

    wrong_guide = {
        name: (value, stated.get(name))
        for name, value in defaults.items()
        if name in stated and not same(value, stated[name])
    }
    assert not wrong_guide, f"guide disagrees with the code: {wrong_guide}"

    template = EXAMPLE.read_text(encoding="utf-8")
    wrong_template = {
        name: (line, defaults[name])
        for name, line, is_active in _assignments(template)
        if not is_active and name in defaults and not same(defaults[name], line)
    }
    assert not wrong_template, f"template disagrees with the code: {wrong_template}"


def test_the_guide_and_the_template_are_not_empty_lists_of_nothing():
    """Guard the guards: a regex that matches nothing would pass every test above."""
    assert len(_code_env_names()) > 50
    assert len(_guide_defaults()) > 50
    assert len(_assignments(EXAMPLE.read_text(encoding="utf-8"))) > 50
