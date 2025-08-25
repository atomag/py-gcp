#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from ase.io import read

from .gcp import gcp_energy_numpy, gcp_energy_pbc_numpy, _params_for_method, EH2KCAL, _method_options


ESYM = [
    'h','he','li','be','b','c','n','o','f','ne','na','mg','al','si','p','s','cl','ar',
    'k','ca','sc','ti','v','cr','mn','fe','co','ni','cu','zn','ga','ge','as','se','br','kr'
]


def print_header():
    print(" -------------------------------------------")
    print("|              **  g C P  **                |")
    print("|  a geometrical counterpoise correction    |")
    print("|             (py-gcp version)              |")
    print(" -------------------------------------------")
    print("py-gcp version 0.1.0\n")


def format_level(method: str) -> str:
    return method.lower().replace('-', '')


def print_param_table(method: str):
    params = _params_for_method(method)
    emiss = params.emiss
    nbas = params.nbas
    print(" ")
    print(f"element parameters {method.lower()}")
    print("  elem   emiss   nbas    elem   emiss   nbas    elem   emiss   nbas")
    for i in range(0, 36, 3):
        row = []
        for j in range(3):
            z = i + j
            if z < 36:
                row.append(f"  {ESYM[z]:>3s}   {emiss[z]:7.5f} {int(nbas[z]):4d}")
            else:
                row.append(" ")
        print(" ".join(row))
    print(" ")


def main() -> int:
    ap = argparse.ArgumentParser(description='Geometrical counterpoise correction (py-gcp)')
    ap.add_argument('-l', '--level', default='dft/def2-tzvp', help='method level (e.g., dft/def2-tzvp)')
    ap.add_argument('-i', '--input', default=None, help='hint for the input format (ignored; ASE autodetects)')
    ap.add_argument('-pbc', '--pbc', action='store_true', help='force periodic calculation')
    ap.add_argument('--no-pbc', action='store_true', help='force non-periodic calculation')
    ap.add_argument('--noprint', action='store_true', help='reduce printout')
    ap.add_argument('--version', action='store_true', help='print version and exit')
    ap.add_argument('geometry', nargs='?', help='input geometry file (xyz/cif/vasp/etc)')
    args = ap.parse_args()

    if args.version:
        print('py-gcp 0.1.0')
        return 0

    if not args.geometry:
        ap.error('missing <input> geometry file')

    method = args.level
    atoms = read(args.geometry)
    Z = atoms.get_atomic_numbers().astype(int)
    xyz = atoms.get_positions()

    # Determine PBC
    periodic = False
    if args.pbc and not args.no_pbc:
        periodic = True
    elif args.no_pbc:
        periodic = False
    else:
        try:
            periodic = bool(getattr(atoms, 'pbc', False)) and bool(atoms.pbc.any())
        except Exception:
            periodic = False

    if not args.noprint:
        print_header()
        params = _params_for_method(method)
        nb = int(np.sum(params.nbas[Z - 1]))
        print(f"  level    {format_level(method):12s}")
        print(f"  Nbf{nb:17d}")
        print(f"  Atoms{len(Z):15d}\n ")
        print("  Parameters: ")
        s, e, a, b = params.p
        print(f"  sigma{ s:11.4f}")
        print(f"  eta{ e:13.4f}")
        print(f"  alpha{ a:11.4f}")
        print(f"  beta{ b:12.4f}\n ")
        print_param_table(method)
        print("   ")
        print(" cutoffs: ")
        print("   distance [bohr] 60.0")
        print("   noise    [a.u.] 2.2E-16")
        damp = _method_options(method).damp
        print(f"   SR-damping      {'T' if damp else 'F'}\n   ")

    if periodic:
        lat = atoms.cell.array
        e = gcp_energy_pbc_numpy(Z, xyz, lat, method=method, units='Angstrom')
    else:
        e = gcp_energy_numpy(Z, xyz, method=method, units='Angstrom')

    if not args.noprint:
        print("** gCP correction ** ")
        print(f"  Egcp:  {e:14.10f} / (a.u.) || {e*EH2KCAL:9.4f} / (kcal/mol)")

    # Write scripting files like mctc-gcp
    Path('.CPC').write_text(f"{e}\n")
    Path('.CP').write_text(f"{e:22.16f}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

