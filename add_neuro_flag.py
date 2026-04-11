"""
add_neuro_flag.py — run from CORTEX root
Finds parse_args() however it appears and injects --neuro before it.
"""
from pathlib import Path
import re

path = Path("run_benchmark.py")
src  = path.read_text(encoding="utf-8")

if "--neuro" in src:
    print("--neuro already present — nothing to do")
else:
    lines = src.splitlines()

    # Find last add_argument line — insert --neuro after it
    insert_at = None
    for i, line in enumerate(lines):
        if "add_argument" in line:
            insert_at = i

    if insert_at is None:
        print("ERROR: no add_argument found")
        for i, l in enumerate(lines[:30]):
            print(f"  {i+1}: {l}")
    else:
        print(f"Inserting after line {insert_at+1}: {lines[insert_at].rstrip()}")
        # Match indentation of previous add_argument
        ref = lines[insert_at]
        indent = len(ref) - len(ref.lstrip())
        inject = " " * indent + 'parser.add_argument("--neuro", action="store_true", default=False, help="Enable 7-signal neuromodulator")'
        lines.insert(insert_at + 1, inject)
        path.write_text("\n".join(lines), encoding="utf-8")
        print("OK -- neuro flag added")
        print("Verify: python run_benchmark.py --help")
