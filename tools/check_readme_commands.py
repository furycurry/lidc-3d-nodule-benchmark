#!/usr/bin/env python3
"""Static check: every `python ...` / `bash ...` command in README.md points at a real file,
every --flag it uses appears in that script's source, and every flag the script declares
required=True is present in the command.

Run from the repo root:   python tools/check_readme_commands.py [README.md]
This does NOT run any script and does NOT prove the commands work end to end. It only catches
renamed/missing scripts and misspelled or removed flags.
"""
import re
import shlex
import sys
from pathlib import Path

readme = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md").read_text(encoding="utf-8")
cmds = []
for block in re.findall(r"```bash\n(.*?)```", readme, flags=re.S):
    text = re.sub(r"\\\n\s*", " ", block)            # join continuation lines
    for line in text.splitlines():
        line = line.split(" #", 1)[0].strip()         # drop trailing comments
        if line.startswith(("python ", "python3 ", "bash ")):
            cmds.append(line)

bad = 0
for cmd in cmds:
    try:
        toks = shlex.split(cmd)
    except ValueError:
        print(f"SKIP  (could not parse) {cmd}")
        continue
    script = Path(toks[1])
    flags = sorted({t.split("=")[0] for t in toks[2:] if t.startswith("--")})
    if not script.exists():
        print(f"FAIL  missing file: {script}   <- {cmd[:80]}")
        bad += 1
        continue
    if toks[0] == "bash":
        print(f"OK    {script} (exists)")
        continue
    src = script.read_text(encoding="utf-8", errors="ignore")
    missing = [f for f in flags if f'"{f}"' not in src and f"'{f}'" not in src]
    required = set(re.findall(r"add_argument\(\s*[\"'](--[\w-]+)[\"'][^)]*?required\s*=\s*True", src, flags=re.S))
    not_given = sorted(required - set(flags))
    if missing or not_given:
        if missing:
            print(f"FAIL  {script}: flag(s) not found in source: {missing}")
        if not_given:
            print(f"FAIL  {script}: required flag(s) missing from the README command: {not_given}   <- {cmd[:70]}")
        bad += 1
    else:
        print(f"OK    {script} {' '.join(flags)}")
print(f"\n{len(cmds)} commands checked, {bad} problem(s)")
sys.exit(1 if bad else 0)
