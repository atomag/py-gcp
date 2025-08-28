#!/usr/bin/env python3
"""
Extract gCP parameter arrays from the Fortran source into JSON.

Usage:
  python tools/extract_fortran_params.py /path/to/gcp.f90 -o py_gcp/data/fortran_params.json

Notes:
- Parses Fortran DATA statements like: data ZS / ... / across multiple lines.
- Merges into existing JSON if provided via -o; otherwise prints to stdout.
- Targets arrays: ZS, ZP, ZD, SHELL and any HF*/BAS* arrays present.

This script does not run at package import; it is a one-time data generator
to keep py-gcp Fortran-free at runtime.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

DATA_RE = re.compile(r"\bdata\s+([A-Za-z_][A-Za-z0-9_]*)\s*/", re.IGNORECASE)

def parse_fortran_data(filepath: Path) -> Dict[str, List[float]]:
    txt = filepath.read_text()
    # Remove Fortran-style comments: '!' to EOL
    lines = [re.split(r"!", ln, 1)[0] for ln in txt.splitlines()]
    src = "\n".join(lines)
    pos = 0
    arrays: Dict[str, List[float]] = {}
    while True:
        m = DATA_RE.search(src, pos)
        if not m:
            break
        name = m.group(1).upper()
        start = m.end()
        # Find closing '/' that terminates this DATA list; account for nested '/' pairs by scanning
        end = start
        depth = 1
        while end < len(src) and depth > 0:
            if src[end] == '/':
                depth -= 1
                if depth == 0:
                    break
            elif src[end] == '\n':
                pass
            elif src[end] == '"':
                # skip quoted strings if any
                end2 = src.find('"', end + 1)
                if end2 == -1:
                    end2 = end
                end = end2
            end += 1
        block = src[start:end]
        # Tokenize numbers (allow D/exponents)
        toks = re.findall(r"[-+]?\d*\.\d+(?:[DEde][-+]?\d+)?|[-+]?\d+(?:[DEde][-+]?\d+)?", block)
        vals: List[float] = []
        for t in toks:
            t2 = t.replace('D', 'E').replace('d', 'e')
            try:
                vals.append(float(t2))
            except Exception:
                pass
        arrays[name] = vals
        pos = end + 1
    return arrays

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fortran', type=Path, help='Path to Fortran gcp.f90 (or source with DATA statements)')
    ap.add_argument('-o', '--output', type=Path, default=None, help='Output JSON path (overwrites)')
    args = ap.parse_args()

    arrays = parse_fortran_data(args.fortran)
    # Merge into existing JSON if provided and exists
    out_obj = {'arrays': arrays}
    if args.output and args.output.exists():
        base = json.loads(args.output.read_text())
        if 'arrays' in base:
            base['arrays'].update(arrays)
        else:
            base['arrays'] = arrays
        out_obj = base

    out_txt = json.dumps(out_obj, indent=2)
    if args.output:
        args.output.write_text(out_txt)
        print(f'Wrote {args.output} with {len(arrays)} arrays')
    else:
        print(out_txt)

if __name__ == '__main__':
    main()

