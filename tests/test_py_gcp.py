import math
from pathlib import Path

import numpy as np

import py_gcp.gcp as gt


def build_h2o_xyz(tmp: Path) -> Path:
    from ase.build import molecule
    from ase.io import write
    at = molecule('H2O')
    xyz = tmp / 'H2O.xyz'
    write(str(xyz), at, format='xyz')
    return xyz


def test_params_available():
    a = gt._arr('HFtz')
    b = gt._arr('BAStz')
    r0 = gt._load_r0ab_matrix()
    assert a.shape[0] == 36
    assert b.shape[0] == 36
    assert r0.shape[0] >= 36 and r0.shape[0] == r0.shape[1]
    assert r0[0, 0] > 0


def test_h2o_def2tzvp_value(tmp_path: Path):
    # Known gcp64 reference (Hartree) from earlier runs
    e_ref = 0.0016656920076788
    xyz = build_h2o_xyz(tmp_path)
    from ase.io import read
    at = read(str(xyz))
    Z = at.get_atomic_numbers().astype(np.int32)
    xyzA = at.get_positions()
    e = gt.gcp_energy_numpy(Z, xyzA, method='dft/def2tzvp', units='Angstrom')
    assert abs(e - e_ref) < 1e-8


def test_cli_writes_cp(tmp_path: Path, monkeypatch):
    # Run CLI main and ensure .CP written and matches API
    xyz = build_h2o_xyz(tmp_path)
    from ase.io import read
    at = read(str(xyz))
    Z = at.get_atomic_numbers().astype(np.int32)
    xyzA = at.get_positions()
    e_api = gt.gcp_energy_numpy(Z, xyzA, method='dft/def2tzvp', units='Angstrom')
    from py_gcp import cli as gcp_cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(Path.cwd()))
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('NUMBA_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    monkeypatch.setenv('BLIS_NUM_THREADS', '1')
    monkeypatch.setenv('VECLIB_MAXIMUM_THREADS', '1')
    monkeypatch.setenv('NUMEXPR_NUM_THREADS', '1')
    monkeypatch.setenv('OMP_WAIT_POLICY', 'PASSIVE')
    monkeypatch.setenv('OMP_PROC_BIND', 'FALSE')
    monkeypatch.setattr('sys.argv', ['pygcp', '-l', 'dft/def2-tzvp', str(xyz)])
    rc = gcp_cli.main()
    assert rc == 0
    cp = (tmp_path / '.CP').read_text().strip()
    e_cli = float(cp)
    assert abs(e_cli - e_api) < 1e-12


def test_pbc_hf3c_finite():
    # Random small periodic cell; ensure we return a finite number
    rng = np.random.default_rng(42)
    N = 32
    Z = rng.integers(1, 19, size=N, dtype=np.int32)
    xyz = rng.random((N, 3)) * 15.0
    lat = np.eye(3) * 15.0
    e_py = gt.gcp_energy_pbc_numpy(Z, xyz, lat, method='hf3c', units='Angstrom')
    assert math.isfinite(e_py)


def test_ase_calculator_ev(tmp_path: Path):
    from ase.build import molecule
    from py_gcp.ase_calculator import PyGCP
    at = molecule('H2O')
    calc = PyGCP(method='dft/def2tzvp')
    at.calc = calc
    e_ev = at.get_potential_energy()
    # Compare to API (Hartree converted)
    e_h = gt.gcp_energy_numpy(at.get_atomic_numbers().astype(np.int32), at.get_positions(), method='dft/def2tzvp', units='Angstrom')
    EH2EV = 27.211386245988
    assert abs(e_ev - e_h * EH2EV) < 1e-6
