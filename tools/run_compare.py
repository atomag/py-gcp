import subprocess
import sys
import json
from pathlib import Path

import numpy as np

from ase.build import molecule, surface
from ase.spacegroup import crystal
from ase.io import write

import py_gcp.gcp as gt


def run_cmd(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_gcp_stdout(stdout: str) -> float:
    # Find line starting with 'Egcp:' and parse the value (Hartree)
    for line in stdout.splitlines():
        if 'Egcp:' in line:
            # Example: "Egcp:  0.0016656920 / (a.u.) || 1.0452 / (kcal/mol)"
            # Extract the first float after 'Egcp:'
            try:
                after = line.split('Egcp:')[1]
                # Split on '/' to isolate the a.u. block, then split by spaces
                au_block = after.split('/')[0]
                for tok in au_block.strip().split():
                    try:
                        return float(tok)
                    except Exception:
                        continue
            except Exception:
                pass
    raise RuntimeError('Egcp not found in gcp64 output')


def ase_to_xyz(path: Path, atoms) -> None:
    # Save simple XYZ in Angstrom
    write(str(path), atoms, format='xyz')


def compute_python(atoms, method='dft/def2tzvp') -> float:
    Z = atoms.get_atomic_numbers().astype(np.int32)
    xyz = atoms.get_positions()  # Angstrom
    e = gt.gcp_energy_numpy(Z, xyz, method=method, units='Angstrom')
    return float(e)


def compute_gcp64(xyz_path: Path, level='DFT/def2-TZVP') -> float:
    # Run and then read the high-precision Egcp from .CP (written by gcp64)
    rc, out, err = run_cmd([str(Path.cwd() / 'gcp64'), '-l', level, str(xyz_path)])
    if rc != 0:
        sys.stderr.write(err)
        raise SystemExit(rc)
    cp_file = Path('.CP')
    if cp_file.exists():
        txt = cp_file.read_text().strip()
        try:
            return float(txt)
        except Exception:
            pass
    # Fallback to parsing stdout if .CP missing
    return parse_gcp_stdout(out)


def build_water() -> 'ase.Atoms':
    return molecule('H2O')


def build_tio2_slab() -> 'ase.Atoms':
    # Rutile TiO2 cell (a, a, c), Angstrom; O internal parameter u ~ 0.305
    a = 4.5937
    c = 2.9581
    u = 0.305
    # Fractional positions (P1) for 2 Ti and 4 O in rutile cell
    frac = [
        (0.0000, 0.0000, 0.0000),  # Ti
        (0.5000, 0.5000, 0.5000),  # Ti
        ( u,     u,     0.0000),   # O
        (-u % 1, -u % 1, 0.0000),  # O
        (0.5+u,  0.5-u, 0.5000),   # O
        (0.5-u,  0.5+u, 0.5000),   # O
    ]
    sym = 'Ti2O4'
    bulk_rutile = crystal(symbols=sym, basis=frac, spacegroup=1,
                          cellpar=(a, a, c, 90, 90, 90), primitive_cell=True)
    # Build a small (110) slab with 3 layers and 10 Å vacuum; treat as cluster for gcp
    slab = surface(bulk_rutile, (1, 1, 0), layers=3, vacuum=10.0)
    slab.center(vacuum=10.0, axis=2)
    return slab


def main():
    work = Path('.')
    out = {}

    # Water
    h2o = build_water()
    h2o_xyz = work / 'H2O.xyz'
    ase_to_xyz(h2o_xyz, h2o)
    out['H2O_py'] = compute_python(h2o, method='dft/def2tzvp')
    out['H2O_ref'] = compute_gcp64(h2o_xyz, level='DFT/def2-TZVP')

    # TiO2 slab (cluster via XYZ for non-PBC comparison)
    slab = build_tio2_slab()
    tio2_cif = work / 'TiO2.cif'
    write(str(tio2_cif), slab, format='cif')  # also produce CIF as requested
    slab_xyz = work / 'TiO2.xyz'
    ase_to_xyz(slab_xyz, slab)
    out['TiO2_py'] = compute_python(slab, method='dft/def2tzvp')
    out['TiO2_ref'] = compute_gcp64(slab_xyz, level='DFT/def2-TZVP')

    # Print and compare
    print(json.dumps(out, indent=2))
    tol = 0.0  # request exact match; relax if needed
    ok = (abs(out['H2O_py'] - out['H2O_ref']) <= tol) and (abs(out['TiO2_py'] - out['TiO2_ref']) <= tol)
    if not ok:
        print('Differences found:')
        print(f"H2O: py={out['H2O_py']:.15e}, ref={out['H2O_ref']:.15e}, diff={out['H2O_py'] - out['H2O_ref']:.3e}")
        print(f"TiO2: py={out['TiO2_py']:.15e}, ref={out['TiO2_ref']:.15e}, diff={out['TiO2_py'] - out['TiO2_ref']:.3e}")
        sys.exit(2)


if __name__ == '__main__':
    main()

