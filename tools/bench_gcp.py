
import time
from pathlib import Path
from statistics import median

import numpy as np
from ase.build import molecule
from ase.io import write

from py_gcp.gcp import gcp_energy_numpy


def time_func(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), median(ts)


def run_case(name, atoms, method='dft/def2tzvp', repeats_py=5, repeats_ref=3):
    print("\n--- %s [%s] ---" % (name, method))
    Z = atoms.get_atomic_numbers().astype(int)
    xyz = atoms.get_positions()
    # Warmup
    e_py = gcp_energy_numpy(Z, xyz, method=method, units='Angstrom')
    # Reference via gcp64
    xyzfile = Path(f'{name}.xyz')
    write(str(xyzfile), atoms, format='xyz')

    def run_py():
        gcp_energy_numpy(Z, xyz, method=method, units='Angstrom')

    def run_ref():
        import subprocess
        subprocess.run(['./gcp64', '-l', method, str(xyzfile)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    tmin_py, tmed_py = time_func(run_py, repeats=repeats_py)
    tmin_ref, tmed_ref = time_func(run_ref, repeats=repeats_ref)

    # Read exact ref from .CP
    try:
        e_ref = float(Path('.CP').read_text().strip())
    except Exception:
        e_ref = float('nan')
    print(f"py:  e={e_py:.12f}  t_min={tmin_py*1e3:.2f} ms  t_med={tmed_py*1e3:.2f} ms")
    print(f"ref: e={e_ref:.12f}  t_min={tmin_ref*1e3:.2f} ms  t_med={tmed_ref*1e3:.2f} ms")
    if np.isfinite(e_ref):
        print(f"|Δ| = {abs(e_py - e_ref):.3e}")
    if tmed_py > 0:
        print(f"speedup (vs ref median): {tmed_ref / tmed_py:.2f}x")


def build_tio2_slab():
    # Small TiO2 (rutile) slab via CIF → XYZ; not PBC for gCP
    from ase.spacegroup import crystal
    from ase.build import surface
    a = 4.5937; c = 2.9581; u = 0.305
    frac = [
        (0.0000, 0.0000, 0.0000),
        (0.5000, 0.5000, 0.5000),
        ( u,     u,     0.0000),
        (-u % 1, -u % 1, 0.0000),
        (0.5+u,  0.5-u, 0.5000),
        (0.5-u,  0.5+u, 0.5000),
    ]
    bulk_rutile = crystal(symbols='Ti2O4', basis=frac, spacegroup=1,
                          cellpar=(a, a, c, 90, 90, 90), primitive_cell=True)
    slab = surface(bulk_rutile, (1, 1, 0), layers=3, vacuum=10.0)
    slab.center(vacuum=10.0, axis=2)
    return slab


def main():
    # Small molecules
    h2o = molecule('H2O')
    ch4 = molecule('CH4')
    c6h6 = molecule('C6H6')

    # Surface cluster
    tio2 = build_tio2_slab()

    for atoms, name in [
        (h2o, 'H2O'),
        (ch4, 'CH4'),
        (c6h6, 'C6H6'),
        (tio2, 'TiO2_slab'),
    ]:
        run_case(name, atoms, method='dft/def2tzvp', repeats_py=7, repeats_ref=3)
        # Also test a damped 3c method on molecules
        if name != 'TiO2_slab':
            run_case(name + '_mTZVPP', atoms, method='def2mtzvpp', repeats_py=7, repeats_ref=3)


if __name__ == '__main__':
    main()

