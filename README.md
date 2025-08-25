# py-gcp — Geometrical Counterpoise (gCP) in Python

py-gcp is a fast implementation of the Geometrical Counterpoise (gCP) correction for basis‑set superposition error (BSSE).
It provides:

- NumPy/Numba CPU kernels for molecular and periodic systems 
- Optional CUDA/Torch paths (molecular; SRB/base for PBC)
- An ASE calculator wrapper
- A gcp64‑like CLI (`pygcp`) that auto‑detects PBC

## Install

```
pip install py-gcp
# optional extras
pip install py-gcp[ase]
pip install py-gcp[torch]
```

## Quick start

- CLI:

```
pygcp -l dft/def2-TZVP H2O.xyz          # non-PBC
pygcp -l dft/def2-TZVP POSCAR           # auto PBC via ASE
pygcp -l hf3c --pbc POSCAR              # force PBC
pygcp -l dft/def2-TZVP --no-pbc H2O.xyz # force non-PBC
```

- Python API:

```
from py_gcp import gcp_energy_numpy
from ase.build import molecule

at = molecule('H2O')
Z = at.get_atomic_numbers()
xyz = at.get_positions()  # Angstrom
E = gcp_energy_numpy(Z, xyz, method='dft/def2tzvp', units='Angstrom')
```

- ASE calculator:

```
from ase.build import molecule
from py_gcp.ase_calculator import PyGCP

at = molecule('H2O')
at.calc = PyGCP(method='dft/def2tzvp')  # auto PBC from Atoms.pbc
E_eV = at.get_potential_energy()
```

## Supported methods (examples)

- Molecular/PBC gCP: `dft/sv`, `dft/sv(p)`, `dft/svp`, `dft/dz`, `dft/dzp`, `dft/631gd`, `dft/def2tzvp`
- HF analogues: `hf/sv`, `hf/sv(p)`, `hf/dz`, `hf/dzp`, `hf/631gd`, `hf/def2tzvp`
- 3c: `b97-3c` (SRB), `hf3c` (base), `def2mtzvpp` (damped)

## Brief theory

The gCP correction adds a pairwise term to approximate the residual BSSE of a finite basis calculation.
For a molecule, the energy is

\[
E_\mathrm{gCP} = p_1 \sum_i \epsilon(Z_i) \sum_{j\ne i}
\frac{\exp\big(-p_3\, r_{ij}^{\,p_4}\big)}{\sqrt{\,v_j\, s_{ab}(r_{ij}; Z_i,Z_j)\,}},
\]

where

- \(\epsilon(Z)\) is the per‑element emissivity (method dependent),
- \(v_j = n_\mathrm{bas}(Z_j) - \tfrac{1}{2}Z_j\) is the virtual count (with specific overrides for some 3c variants),
- \(s_{ab}(r; Z_i,Z_j)\) is the s‑type Slater overlap between the valence shells (1s/2s/3s) of \(Z_i\) and \(Z_j\) with exponents \(\zeta_a,\zeta_b\) from `setzet`,
- and \(p_1, p_2, p_3, p_4\) are the method parameters (\(p_2\) is the scaling for \(\zeta\)).

The Slater overlap \(s_{ab}\) is evaluated analytically with auxiliary functions \(A_k(x)\), \(B_k(x)\) and a robust small‑\(x\) series for \(B_k\). For PBC, the pair sum is extended over lattice images within a fixed cutoff (60 Bohr).

For damped 3c variants (e.g., PBEh‑3c, HSE‑3c, mTZVPP), we multiply the pair contribution by a short‑range damping function

\[\displaystyle
f_\mathrm{damp}(r) = 1 - \frac{1}{1 + d_\mathrm{scal}\, (r/r_0)^{d_\mathrm{exp}}}, \quad r_0 = r_0^{\,(Z_i,Z_j)}
\]

with element‑pair radii \(r_0\) tabulated and packed in the distribution.

Units: Coordinates are accepted in Angstrom or Bohr; energy is returned in Hartree (CLI also prints kcal/mol).

## Performance

- Molecular (NumPy/Numba): typically 10–15× faster than `gcp64` on small/medium systems; exact within ~1e−8 Ha.
- PBC (NumPy/Numba): 3–18× faster depending on method/size; hf‑3c, b97‑3c supported including SR terms.
- Torch/CUDA: available for molecular non‑damped methods; SRB/base PBC on GPU.  CPU fallback for damped.

## Development

- Run tests: `pip install -e .[ase] pytest` then `pytest -q`.
- Benchmarks: `python3 bench_suite.py` (NumPy vs gcp64), `python3 bench_torch_modes.py`.
