import time
import json
from statistics import median
from pathlib import Path

import numpy as np
from ase.build import molecule, surface
from ase.spacegroup import crystal
from ase.io import write

import py_gcp.gcp as gt


def timeit(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), median(ts)


def run_cmd(cmd):
    import subprocess
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def gcp64_energy(path, method):
    rc, out, err = run_cmd([str(Path.cwd() / 'gcp64'), '-l', method, str(path)])
    if rc != 0:
        raise RuntimeError(err)
    cp = Path('.CP')
    if cp.exists():
        return float(cp.read_text().strip())
    # Fallback to parse stdout
    for line in out.splitlines():
        if 'Egcp:' in line:
            try:
                blk = line.split('Egcp:')[1].split('/')[0]
                for tok in blk.strip().split():
                    return float(tok)
            except Exception:
                pass
    raise RuntimeError('Egcp not found in gcp64 output')


def molecules():
    return {
        'H2O': molecule('H2O'),
        'CH4': molecule('CH4'),
        'C6H6': molecule('C6H6'),
    }


def tio2_slab():
    a = 4.5937; c = 2.9581; u = 0.305
    frac = [
        (0.0000, 0.0000, 0.0000),
        (0.5000, 0.5000, 0.5000),
        ( u,     u,     0.0000),
        (-u % 1, -u % 1, 0.0000),
        (0.5+u,  0.5-u, 0.5000),
        (0.5-u,  0.5+u, 0.5000),
    ]
    bulk = crystal(symbols='Ti2O4', basis=frac, spacegroup=1,
                    cellpar=(a, a, c, 90, 90, 90), primitive_cell=True)
    slab = surface(bulk, (1, 1, 0), layers=3, vacuum=10.0)
    slab.center(vacuum=10.0, axis=2)
    return slab


def rand_cell(N=128, box=20.0, seed=123, zmax=18):
    rng = np.random.default_rng(seed)
    Z = rng.integers(1, zmax + 1, size=N, dtype=np.int32)
    xyz = rng.random((N, 3)) * box
    lat = np.eye(3) * box
    return Z, xyz, lat


def bench_molecular(methods, reps_py=7, reps_ref=3):
    out = []
    sets = molecules()
    sets['TiO2_slab'] = tio2_slab()
    for name, atoms in sets.items():
        Z = atoms.get_atomic_numbers().astype(np.int32)
        xyz = atoms.get_positions()
        for m in methods:
            # Warm
            e_py = gt.gcp_energy_numpy(Z, xyz, method=m, units='Angstrom')
            # Prepare XYZ
            xyzfile = Path(f'{name}.xyz')
            write(str(xyzfile), atoms, format='xyz')

            def run_py():
                gt.gcp_energy_numpy(Z, xyz, method=m, units='Angstrom')

            def run_ref():
                gcp64_energy(xyzfile, m)

            tmin_py, tmed_py = timeit(run_py, reps=reps_py, warm=1)
            tmin_rf, tmed_rf = timeit(run_ref, reps=reps_ref, warm=1)
            e_ref = gcp64_energy(xyzfile, m)
            out.append({
                'system': name,
                'method': m,
                'E_py': float(e_py),
                'E_ref': float(e_ref),
                'dE': float(e_py - e_ref),
                't_py_ms': tmed_py * 1e3,
                't_ref_ms': tmed_rf * 1e3,
                'speedup': tmed_rf / tmed_py if tmed_py > 0 else float('inf')
            })
            print(f"mol {name:10s} {m:14s} | dE={out[-1]['dE']:.3e} | py={out[-1]['t_py_ms']:.2f} ms | ref={out[-1]['t_ref_ms']:.2f} ms | x{out[-1]['speedup']:.2f}")
    return out


def bench_pbc(methods, Ns=(64, 128), reps_py=5, reps_ref=3):
    out = []
    for N in Ns:
        Z, xyz, lat = rand_cell(N=N, box=20.0, seed=123, zmax=18)
        poscar = Path('POSCAR')
        from ase import Atoms
        from ase.data import chemical_symbols
        syms = [chemical_symbols[int(z)] for z in Z]
        at = Atoms(symbols=syms, positions=xyz, cell=lat, pbc=True)
        write(str(poscar), at, format='vasp')
        for m in methods:
            # Warm
            e_py = gt.gcp_energy_pbc_numpy(Z, xyz, lat, method=m, units='Angstrom')
            def run_py():
                gt.gcp_energy_pbc_numpy(Z, xyz, lat, method=m, units='Angstrom')
            def run_ref():
                gcp64_energy(poscar, m)
            tmin_py, tmed_py = timeit(run_py, reps=reps_py, warm=1)
            tmin_rf, tmed_rf = timeit(run_ref, reps=reps_ref, warm=1)
            e_ref = gcp64_energy(poscar, m)
            out.append({
                'N': N,
                'method': m,
                'E_py': float(e_py),
                'E_ref': float(e_ref),
                'dE': float(e_py - e_ref),
                't_py_ms': tmed_py * 1e3,
                't_ref_ms': tmed_rf * 1e3,
                'speedup': tmed_rf / tmed_py if tmed_py > 0 else float('inf')
            })
            print(f"pbc N={N:4d} {m:14s} | dE={out[-1]['dE']:.3e} | py={out[-1]['t_py_ms']:.2f} ms | ref={out[-1]['t_ref_ms']:.2f} ms | x{out[-1]['speedup']:.2f}")
    return out


def main():
    # Methods/basis to sweep (molecular)
    mol_methods = [
        'dft/sv', 'dft/sv(p)', 'dft/svp', 'dft/dz', 'dft/dzp', 'dft/631gd',
        'hf/sv', 'hf/sv(p)', 'hf/dz', 'hf/dzp', 'hf/631gd',
        'dft/def2tzvp', 'hf/def2tzvp',
        'def2mtzvpp',  # damped 3c
    ]
    # Methods for PBC benchmarks
    pbc_methods = [
        'dft/def2tzvp',  # general PBC gCP
        'b97-3c',        # SRB
        'hf3c',          # base + gCP
    ]

    print('=== Molecular benchmarks ===')
    mol_res = bench_molecular(mol_methods)
    print('=== PBC benchmarks ===')
    pbc_res = bench_pbc(pbc_methods)

    Path('bench_results.json').write_text(json.dumps({
        'molecular': mol_res,
        'pbc': pbc_res,
    }, indent=2))
    print('Results written to bench_results.json')


if __name__ == '__main__':
    main()

