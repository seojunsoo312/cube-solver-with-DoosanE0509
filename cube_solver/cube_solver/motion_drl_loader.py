"""Merge motions/*.drl body with cube_motion_preamble.drl when present."""
from __future__ import annotations

from pathlib import Path


def load_merged_drl(body_path: Path, prepend_path: str = "") -> tuple[bool, str]:
    """Return full DRL script: optional preamble + body file contents.

    If ``prepend_path`` is empty and ``cube_motion_preamble.drl`` exists next to
    ``body_path``, it is prepended. Otherwise ``prepend_path`` must point to a
    preamble file if you want one.
    """
    path = body_path.expanduser()
    if not path.exists():
        return False, f"missing file: {path}"
    if not path.is_file():
        return False, f"not a file: {path}"

    try:
        body = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"read body failed: {e}"

    preamble = ""
    prep_param = (prepend_path or "").strip()
    if prep_param:
        pp = Path(prep_param).expanduser()
        if not pp.is_file():
            return False, f"prepend_path not a file: {pp}"
        try:
            preamble = pp.read_text(encoding="utf-8").rstrip() + "\n\n"
        except OSError as e:
            return False, f"read prepend_path failed: {e}"
    else:
        auto = path.parent / "cube_motion_preamble.drl"
        if auto.is_file():
            try:
                preamble = auto.read_text(encoding="utf-8").rstrip() + "\n\n"
            except OSError as e:
                return False, f"read preamble failed ({auto}): {e}"

    code = f"{preamble}{body}"
    if not code.strip():
        return False, "empty script"
    return True, code
