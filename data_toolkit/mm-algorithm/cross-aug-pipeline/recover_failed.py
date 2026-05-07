'''
Recover templates that were quarantined to failed_templates/ due to the
faulty raw-string regex in has_input_output_placeholders().

Re-runs the validators on each quarantined raw-output file and, if it passes,
promotes it back to templates/<id>.py and updates the audit JSON.

Usage:
  conda run -n vpb python recover_failed_templates.py
'''

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from generate_templates import (
    REQUIRED_SYMBOLS,
    has_input_output_placeholders,
    has_required_symbols,
    strip_code_fence,
)

BASE = Path(__file__).resolve().parent


def main() -> None:
    promoted: list[str] = []
    still_bad: list[tuple[str, str]] = []
    fail_dir = BASE / "failed_templates"
    tpl_dir = BASE / "templates"
    audit_dir = BASE / "audit"

    for raw_path in sorted(fail_dir.glob("*.txt")):
        nid = raw_path.stem
        text = raw_path.read_text(encoding="utf-8")
        code = strip_code_fence(text)

        # Syntax
        try:
            compile(code, f"<{nid}>", "exec")
            syntax_ok = True
            syntax_msg = ""
        except SyntaxError as e:
            syntax_ok = False
            syntax_msg = f"SyntaxError: {e.msg} (line {e.lineno})"

        # Symbols + placeholders
        symbols_ok, missing = has_required_symbols(code)

        audit_path = audit_dir / f"{nid}.json"
        n_examples = 2
        if audit_path.exists():
            try:
                a = json.loads(audit_path.read_text(encoding="utf-8"))
                n_examples = int(a.get("n_examples", 2))
            except Exception:
                pass

        ph_ok, ph_msg = has_input_output_placeholders(code, n_examples)

        if syntax_ok and symbols_ok and ph_ok:
            (tpl_dir / f"{nid}.py").write_text(code, encoding="utf-8")
            # Update audit
            if audit_path.exists():
                a = json.loads(audit_path.read_text(encoding="utf-8"))
                a["syntax_ok"] = True
                a["symbols_ok"] = True
                a["placeholders_ok"] = True
                a["placeholders_msg"] = ""
                a["recovered_ts"] = datetime.now().isoformat(timespec="seconds")
                audit_path.write_text(
                    json.dumps(a, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            raw_path.unlink()
            promoted.append(nid)
        else:
            still_bad.append((
                nid,
                f"syntax={syntax_ok} symbols={symbols_ok}({missing}) ph={ph_ok}({ph_msg})",
            ))

    print(f"Promoted: {len(promoted)} templates back to templates/")
    print(f"Still failing: {len(still_bad)}")
    if still_bad:
        for nid, msg in still_bad[:8]:
            print(f"  {nid}: {msg}")


if __name__ == "__main__":
    main()
