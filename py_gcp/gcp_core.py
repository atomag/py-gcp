"""GPU-accelerated Geometrical Counterpoise (gCP) correction (molecular only).

This module provides a PyTorch/Numpy implementation of the gCP energy for
common DFT/basis combinations (e.g., B3LYP/def2-SVP, def2-SV(P), def2-TZVP). It focuses
on the non-periodic, no-damping branch used for standard gCP corrections
as in Grimme's mctc-gcp for those methods.

Scope and notes:
- Implements energy only (no gradients); CPU reference path is faithful to
  the published formulas; a Torch backend is available but defaults to the
  CPU path for correctness when branching is heavy.
- Supported methods: 'b3lyp/def2svp', 'b3lyp/def2sv(p)', 'b3lyp/sv', 'b3lyp/def2tzvp'
  (synonyms 'dft/def2svp', 'dft/def2sv(p)', 'dft/sv', 'dft/def2tzvp', 'b3lyp/tz', 'dft/tz').
- Elements supported: Z = 1..36 (H..Kr), matching parameter tables here.
- Units: coordinates can be in 'Angstrom' or 'Bohr'; energy returned in Hartree.

If you need other methods/elements, extend the parameter tables as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import importlib
import importlib.resources as _res

import math
import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    # Torch not available – provide a minimal stub for decorators used at import time
    _HAVE_TORCH = False
    class _TorchStub:
        def no_grad(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator
    torch = _TorchStub()  # type: ignore

try:  # optional CPU JIT acceleration
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# Constants
# Match Fortran autoang = 0.5291772083 Bohr per Angstrom
ANG2BOHR = 1.0 / 0.5291772083  # 1 Angstrom = this many Bohr
EH2KCAL = 627.5094740631


# Parameter tables (Z=1..36). A minimal subset is embedded below for common
# methods. For full coverage, additional arrays are loaded on demand by parsing
# the Fortran source in fgcp/src/gcp.f90 to ensure numerical identity.
HFsv = np.array([
    0.009037, 0.008843,
    0.204189, 0.107747, 0.049530, 0.055482, 0.072823, 0.100847, 0.134029, 0.174222,
    0.315616, 0.261123, 0.168568, 0.152287, 0.146909, 0.168248, 0.187882, 0.211160,
    0.374252, 0.460972,
    0.444886, 0.404993, 0.378406, 0.373439, 0.361245, 0.360014, 0.362928, 0.243801, 0.405299, 0.396510,
    0.362671, 0.360457, 0.363355, 0.384170, 0.399698, 0.417307
], dtype=np.float64)

HFsvp = np.array([
    0.008107, 0.008045,
    0.113583, 0.028371, 0.049369, 0.055376, 0.072785, 0.100310, 0.133273, 0.173600,
    0.181140, 0.125558, 0.167188, 0.149843, 0.145396, 0.164308, 0.182990, 0.205668,
    0.200956, 0.299661,
    0.325995, 0.305488, 0.291723, 0.293801, 0.291790, 0.296729, 0.304603, 0.242041, 0.354186, 0.350715,
    0.350021, 0.345779, 0.349532, 0.367305, 0.382008, 0.399709
], dtype=np.float64)

# number of basis functions per element (1..36)
BASsv = np.array([
    2, 2, 3, 3, 9, 9, 9, 9, 9, 9, 7, 7, 13, 13, 13, 13, 13, 13,
    11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 24, 24, 24, 24, 24, 24
], dtype=np.int32)

# def2-TZVP emissivities (HFtz) and basis function counts (BAStz)
# Ported from gcp/src/gcp.f90 (data HFtz / ... / and data BAStz / ... /)
HFtz = np.array([
    0.007577, 0.003312,
    0.086763, 0.009962, 0.013964, 0.005997, 0.004731, 0.005687, 0.006367, 0.007511,
    0.077721, 0.050003, 0.068317, 0.041830, 0.025796, 0.025512, 0.023345, 0.022734,
    0.097241, 0.099167,
    0.219194, 0.189098, 0.164378, 0.147238, 0.137298, 0.127510, 0.118589, 0.0318653, 0.120985, 0.0568313,
    0.090996, 0.071820, 0.063562, 0.064241, 0.061848, 0.061021
], dtype=np.float64)

BAStz = np.array([
    6, 6,      # H, He
    14, 19,    # Li, Be
    31, 31, 31, 31, 31, 31,  # B..Ne
    32, 32,    # Na, Mg
    37, 37, 37, 37, 37, 37,  # Al..Ar
    33, 36,    # K, Ca
    45, 45, 45, 45, 45, 45, 45, 45, 45, 48,  # Sc..Zn (Zn has 48)
    48, 48, 48, 48, 48, 48   # Ga..Kr
], dtype=np.int32)

# Fortran: data BASsvp/2*5,9,9,6*14,15,18,6*18,24,24,10*31,6*32/
BASsvp = np.array([
    5, 5,   # H, He
    9, 9,   # Li, Be
    14, 14, 14, 14, 14, 14,  # B..Ne (6*14)
    15,     # Na
    18,     # Mg
    18, 18, 18, 18, 18, 18,  # Al..Ar (6*18)
    24, 24, # K, Ca
    31, 31, 31, 31, 31, 31, 31, 31, 31, 31,  # Sc..Zn (10*31)
    32, 32, 32, 32, 32, 32  # Ga..Kr (6*32)
], dtype=np.int32)


# Slater exponent base tables (ZA/ ZP/ ZD), Z=1..36 (from setzet)
ZS = np.array([
    1.2000, 1.6469, 0.6534, 1.0365, 1.3990, 1.7210, 2.0348, 2.2399, 2.5644, 2.8812,
    0.8675, 1.1935, 1.5143, 1.7580, 1.9860, 2.1362, 2.3617, 2.5796, 0.9362, 1.2112,
    1.2870, 1.3416, 1.3570, 1.3804, 1.4761, 1.5465, 1.5650, 1.5532, 1.5781, 1.7778,
    2.0675, 2.2702, 2.4546, 2.5680, 2.7523, 2.9299
], dtype=np.float64)

ZP = np.array([
    0.0000, 0.0000, 0.5305, 0.8994, 1.2685, 1.6105, 1.9398, 2.0477, 2.4022, 2.7421,
    0.6148, 0.8809, 1.1660, 1.4337, 1.6755, 1.7721, 2.0176, 2.2501, 0.6914, 0.9329,
    0.9828, 1.0104, 0.9947, 0.9784, 1.0641, 1.1114, 1.1001, 1.0594, 1.0527, 1.2448,
    1.5073, 1.7680, 1.9819, 2.0548, 2.2652, 2.4617
], dtype=np.float64)

ZD = np.array([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    2.4341, 2.6439, 2.7809, 2.9775, 3.2208, 3.4537, 3.6023, 3.7017, 3.8962, 2.0477,
    2.4022, 2.7421, 0.6148, 0.8809, 1.1660, 1.4337
], dtype=np.float64)


# Shell mapping per element (1s, 2s, 3s treated as 1,2,3)
def _default_shell(n: int) -> np.ndarray:
    arr = np.empty(n, dtype=np.int32)
    for i in range(n):
        z = i + 1
        if z <= 2:
            arr[i] = 1
        elif z <= 10:
            arr[i] = 2
        else:
            arr[i] = 3
    return arr


@dataclass
class GCPParams:
    emiss: np.ndarray  # (36,) per-element emissivity
    nbas: np.ndarray   # (36,) per-element #basis functions
    p: Tuple[float, float, float, float]  # sigma, eta, alpha, beta

_FORTRAN_FILE = None  # legacy
_PARAMS_RESOURCE = ('py_gcp.data', 'fortran_params.json')
_PARAM_CACHE: Dict[str, np.ndarray] = {}
_SAB_TABLE_CACHE: Dict[str, Tuple[np.ndarray, float, int]] = {}
_PARAM_JSON_CACHE: Dict[str, object] = {}


def _normalize_method(method: str) -> str:
    m = method.strip().lower()
    m = m.replace(' ', '')
    m = m.replace('-','')
    m = m.replace('def2sv(p)', 'def2sv(p)')
    m = m.replace('def2svp', 'def2svp')
    m = m.replace('def2tzvp', 'def2tzvp')
    m = m.replace('sv_p', 'sv(p)')
    m = m.replace('631gs', '631gd')
    # canonicalize aliases
    if m.startswith('b3lyp/'):
        m = 'dft/' + m.split('/', 1)[1]
    return m


def _load_param_json() -> dict:
    key = '/'.join(_PARAMS_RESOURCE)
    if key in _PARAM_JSON_CACHE:
        return _PARAM_JSON_CACHE[key]  # type: ignore
    pkg, fname = _PARAMS_RESOURCE
    try:
        with _res.as_file(_res.files(pkg).joinpath(fname)) as p:
            import json
            data = json.loads(p.read_text())
    except Exception as e:
        raise RuntimeError('Cannot load parameter JSON resource') from e
    _PARAM_JSON_CACHE[key] = data
    return data


def _load_fortran_array(name: str, dtype=float) -> np.ndarray:
    # Now loads from packaged JSON converted from Fortran
    key = name.upper()
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    data = _load_param_json()
    arrays = data.get('arrays', {})
    if key not in arrays:
        raise KeyError(f'Array {name} not found in params JSON')
    vals = arrays[key]
    arr = np.array(vals, dtype=np.float64)
    if dtype is int:
        arr = arr.astype(np.int32)
    _PARAM_CACHE[key] = arr
    return arr


def _arr(name: str) -> np.ndarray:
    # Helper to get array by name either from embedded subset or Fortran
    if name == 'HFsv':
        return HFsv.copy()
    if name == 'HFsvp':
        return HFsvp.copy()
    if name == 'HFtz':
        return HFtz.copy()
    if name == 'BASsv':
        return BASsv.copy()
    if name == 'BASsvp':
        return BASsvp.copy()
    if name == 'BAStz':
        return BAStz.copy().astype(np.int32)
    # Others: load from Fortran (return copies to avoid mutating cache)
    if name.upper().startswith('BAS'):
        return _load_fortran_array(name, dtype=int).copy()
    return _load_fortran_array(name, dtype=float).copy()


@dataclass
class MethodOptions:
    damp: bool = False
    base: bool = False
    srb: bool = False
    dmp_scal: float = 4.0
    dmp_exp: float = 6.0
    base_rscal: float = 0.7
    base_qscal: float = 0.03
    srb_rscal: float = 10.0
    srb_qscal: float = 0.08


def _method_options(method: str) -> MethodOptions:
    m = _normalize_method(method)
    if m in ('hf3c',):
        return MethodOptions(base=True)
    if m in ('pbeh3c', 'pbeh3cmod', 'hse3c'):
        return MethodOptions(damp=True)
    if m in ('b973c', 'b97-3c'):
        return MethodOptions(srb=True)
    if m in ('r2scan3c', 'def2mtzvpp', 'mtzvpp'):
        return MethodOptions(damp=True)
    return MethodOptions()


def _load_r0ab_matrix(max_elem: int | None = None) -> np.ndarray:
    data = _load_param_json()
    vals = data.get('r0ab_vals', [])
    # Deduce size N from triangular length if not provided
    if max_elem is None:
        L = int(len(vals))
        # Solve N(N+1)/2 = L
        import math as _math
        N = int((_math.isqrt(1 + 8 * L) - 1) // 2)
    else:
        N = int(max_elem)
    AUTOANG = 0.5291772083
    r = np.zeros((N, N), dtype=np.float64)
    k = 0
    for i in range(N):
        for j in range(i + 1):
            v = vals[k] / AUTOANG
            r[i, j] = v
            r[j, i] = v
            k += 1
    return r

# Initialize SHELL after loaders are available
try:
    _SHELL = _load_fortran_array('SHELL', dtype=int).astype(np.int32)
    ZS_len = int(_load_fortran_array('ZS').size)
    if _SHELL.ndim != 1 or _SHELL.size < ZS_len:
        raise ValueError
except Exception:
    ZS_len = int(_load_fortran_array('ZS').size) if 'ZS' in _load_param_json().get('arrays', {}) else 36
    _SHELL = _default_shell(ZS_len)


def _params_for_method(method: str) -> GCPParams:
    m = _normalize_method(method)
    # Map of method -> (emiss_name, nbas_name, p)
    MAP: Dict[str, Tuple[str, str, Tuple[float, float, float, float]]] = {
        # DFT section (synced with Fortran setparam cases)
        'dft/sv': ('HFsv', 'BASsv', (0.4048, 1.1626, 0.8652, 1.2375)),
        'dft/sv(p)': ('HFsvp', 'BASsvp', (0.2424, 1.2371, 0.6076, 1.4078)),
        'dft/svx': ('HFsvp', 'BASsvp', (0.1861, 1.3200, 0.6171, 1.4019)),
        'dft/svp': ('HFsvp', 'BASsvp', (0.2990, 1.2605, 0.6438, 1.3694)),
        'dft/dzp': ('HFdzp', 'BASdzp', (0.2687, 1.4634, 0.3513, 1.6880)),
        'dft/631gd': ('HF631gd', 'BAS631gd', (0.3405, 1.6127, 0.8589, 1.2830)),
        'dft/minis': ('HFminis', 'BASminis', (0.2059, 0.9722, 1.1961, 1.1456)),
        'dft/minix': ('HFminis', 'BASminis', (0.2059, 0.9722, 1.1961, 1.1456)),
        'dft/tz': ('HFtz', 'BAStz', (0.2905, 2.2495, 0.8120, 1.4412)),
        'dft/def2tzvp': ('HFtz', 'BAStz', (0.2905, 2.2495, 0.8120, 1.4412)),
        'dft/deftzvp': ('HFdef1tzvp', 'BASdef1tzvp', (0.2393, 2.2247, 0.8185, 1.4298)),
        'dft/tzvp': ('HFdef1tzvp', 'BASdef1tzvp', (0.2393, 2.2247, 0.8185, 1.4298)),
        'dft/ccdz': ('HFccdz', 'BASccdz', (0.5383, 1.6482, 0.6230, 1.4523)),
        'dft/accdz': ('HFaccdz', 'BASaccdz', (0.1465, 0.0500, 0.6003, 0.8761)),
        'dft/pobtz': ('HFpobtz', 'BASpobtz', (0.1300, 1.3743, 0.4792, 1.3962)),
        'dft/dz': ('HFdz', 'BASdz', (0.2687, 1.4634, 0.3513, 1.6880)),
        'dft/lanl': ('HF631gd', 'BAS631gd', (0.3405, 1.6127, 0.8589, 1.2830)),
        # HF section
        'hf/sv': ('HFsv', 'BASsv', (0.1724, 1.2804, 0.8568, 1.2342)),
        'hf/svp': ('HFsvp', 'BASsvp', (0.2054, 1.3157, 0.8136, 1.2572)),
        'hf/svp_old': ('oldHFsvp', 'oldBASsvp', (0.2054, 1.3157, 0.8136, 1.2572)),
        'hf/sv(p)': ('HFsvp', 'BASsvp', (0.1373, 1.4271, 0.8141, 1.2760)),
        'hf/dzp': ('HFdzp', 'BASdzp', (0.1443, 1.4547, 0.3711, 1.6300)),
        'hf/631gd': ('HF631gd', 'BAS631gd', (0.2048, 1.5652, 0.9447, 1.2100)),
        'hf/minis': ('HFminis', 'BASminis', (0.1290, 1.1526, 1.1549, 1.1763)),
        'hf/minix': ('HFminis', 'BASminis', (0.1290, 1.1526, 1.1549, 1.1763)),
        'hf/tz': ('HFtz', 'BAStz', (0.3127, 1.9914, 1.0216, 1.2833)),
        'hf/def2tzvp': ('HFtz', 'BAStz', (0.3127, 1.9914, 1.0216, 1.2833)),
        'hf/deftzvp': ('HFdef1tzvp', 'BASdef1tzvp', (0.2600, 2.2448, 0.7998, 1.4381)),
        'hf/tzvp': ('HFdef1tzvp', 'BASdef1tzvp', (0.2600, 2.2448, 0.7998, 1.4381)),
        'hf/ccdz': ('HFccdz', 'BASccdz', (0.4416, 1.5185, 0.6902, 1.3713)),
        'hf/accdz': ('HFaccdz', 'BASaccdz', (0.0748, 0.0663, 0.3811, 1.0155)),
        'hf/2g': ('HF2g', 'BAS2g', (0.2461, 1.1616, 0.7335, 1.4709)),
        'hf/dz': ('HFdz', 'BASdz', (0.1059, 1.4554, 0.3711, 1.6342)),
        # 3c-style built-ins
        'pbeh3c': ('HFmsvp', 'BASmsvp', (1.0000, 1.32492, 0.27649, 1.95600)),
        'hse3c': ('HFmsvp', 'BASmsvp', (1.0000, 1.32378, 0.28314, 1.94527)),
        'def2mtzvpp': ('HFdef2mtzvpp', 'BASdef2mtzvpp', (1.0000, 1.3150, 0.9410, 1.4636)),
        'mtzvpp': ('HFdef2mtzvpp', 'BASdef2mtzvpp', (1.0000, 1.3150, 0.9410, 1.4636)),
        'r2scan3c': ('HFdef2mtzvpp', 'BASdef2mtzvpp', (1.0000, 1.3150, 0.9410, 1.4636)),
        'hf3c': ('HFminis', 'BASminis', (0.1290, 1.1526, 1.1549, 1.1763)),
        'b3pbe3c': ('HFdef2mtzvp', 'BASdef2mtzvp', (1.0000, 2.98561, 0.3011, 2.4405)),
    }

    # Accept common synonyms
    alias = {
        'dft/def2svp': 'dft/svp',
        'dft/def2sv(p)': 'dft/sv(p)',
        'dft/tzvp': 'dft/deftzvp',
        'hf/def2svp': 'hf/svp',
        'hf/def2sv(p)': 'hf/sv(p)',
        'pbeh3cmod': 'pbeh3c',
        'b97-3c': 'b973c',
    }
    if m in alias:
        m = alias[m]

    if m not in MAP:
        raise ValueError(f'Unsupported gCP method: {method}')

    emiss_name, nbas_name, p = MAP[m]
    emiss = _arr(emiss_name)
    nbas = _arr(nbas_name).astype(np.int32)

    # Method-specific tweaks copied from Fortran
    if m in ('dft/sv(p)', 'dft/svx', 'hf/sv(p)'):
        emiss[0] = HFsv[0]
        nbas[0] = BASsv[0]
    if m == 'dft/svx':
        emiss[5] = HFsv[5]  # Z=6 (C) index 5
        nbas[5] = BASsv[5]
    if m in ('dft/minix', 'hf/minix'):
        # MINIS + historical tweaks used in Fortran for minix
        emiss[12 - 1] = 1.114110
        emiss[13 - 1] = 1.271150
        nbas[12 - 1] = 9
        nbas[13 - 1] = 9
        emiss[3 - 1] = 0.177871
        emiss[4 - 1] = 0.171596
        nbas[3 - 1] = 5
        nbas[4 - 1] = 5
    if m in ('hf3c',):
        # Reproduce Fortran setparam('hf/minix','hf3c') exactly for Z<=36
        # Allocate fresh arrays of length 36
        emiss = np.zeros(36, dtype=np.float64)
        nbas = np.zeros(36, dtype=np.int32)
        def _pad36(a: np.ndarray, dtype=float) -> np.ndarray:
            out = np.zeros(36, dtype=np.float64 if dtype is float else np.int32)
            n = min(36, int(a.size))
            out[:n] = a[:n]
            if dtype is int:
                return out.astype(np.int32)
            return out
        HFminis = _pad36(_arr('HFminis'), float)
        BASminis = _pad36(_arr('BASminis').astype(np.int32), int)
        HFminisd = _pad36(_arr('HFminisd'), float)
        BASminisd = _pad36(_arr('BASminisd').astype(np.int32), int)
        BASsv_arr = _arr('BASsv').astype(np.int32)
        BASsvp_arr = _arr('BASsvp').astype(np.int32)
        # H-Mg: MINIS
        emiss[0:12] = HFminis[0:12]
        nbas[0:12] = BASminis[0:12]
        # Al-Ar: MINIS+d
        emiss[12:18] = HFminisd[12:18]
        nbas[12:18] = BASminisd[12:18]
        # K-Zn: SV
        HFsv_arr = _arr('HFsv')
        emiss[18:30] = HFsv_arr[18:30]
        nbas[18:30] = BASsv_arr[18:30]
        # Ga-Kr: SVP
        HFsvp_arr = _arr('HFsvp')
        emiss[30:36] = HFsvp_arr[30:36]
        nbas[30:36] = BASsvp_arr[30:36]
        # Li,Be,Na,Mg MINIS + p-fkt; Na=11, Mg=12 in 1-based
        emiss[3 - 1] = 0.177871
        emiss[4 - 1] = 0.171596
        nbas[3 - 1] = 5
        nbas[4 - 1] = 5
        emiss[11 - 1] = 1.114110
        emiss[12 - 1] = 1.271150
        nbas[11 - 1] = 9
        nbas[12 - 1] = 9
    if m in ('pbeh3c', 'hse3c'):
        # HFmsvp with DZP emiss for 19..apar and zero for 36 (Fortran setparam)
        HFdzp = _arr('HFdzp')
        emiss[18:36] = HFdzp[18:36]
        emiss[35] = 0.0
    if m in ('dft/lanl', 'hf/lanl'):
        HFlanl2 = _arr('HFlanl2')
        BASlanl2 = _arr('BASlanl2').astype(np.int32)
        emiss[20:30] = HFlanl2[:10]
        nbas[20:30] = BASlanl2[:10]

    return GCPParams(emiss, nbas, p)


def _map_elements_for_params(Z: np.ndarray) -> np.ndarray:
    """Map elements >36 to lower homologues as in Fortran gcp.f90.

    Cases (zz original Z):
      37..54 -> zz-18
      55..57 -> zz-36
      58..71, 90..94 -> 21
      72..89 -> zz-50
      otherwise unchanged.
    """
    Z = np.asarray(Z, dtype=np.int32).copy()
    zz = Z.copy()
    mask = (zz >= 37) & (zz <= 54)
    Z[mask] = zz[mask] - 18
    mask = (zz >= 55) & (zz <= 57)
    Z[mask] = zz[mask] - 36
    mask = ((zz >= 58) & (zz <= 71)) | ((zz >= 90) & (zz <= 94))
    Z[mask] = 21
    mask = (zz >= 72) & (zz <= 89)
    Z[mask] = zz[mask] - 50
    return Z

def _setzet(eta: float, etaspec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Slater exponents za, zb per element.

    Uses arrays ZS/ZP/ZD from packaged data if available; falls back
    to embedded defaults. Applies etaspec to indices >= 11 (1-based),
    mirroring Fortran setzet behavior used in gcp.
    """
    ZS_arr = _arr('ZS')
    ZP_arr = _arr('ZP')
    ZD_arr = _arr('ZD') if 'ZD' in _load_param_json().get('arrays', {}) else np.zeros_like(ZS_arr)
    n = int(ZS_arr.shape[0])
    za = np.empty(n, dtype=np.float64)
    for i in range(n):
        z = i + 1
        if z <= 2:
            base = ZS_arr[i]
        elif 3 <= z <= 20 or z >= 31:
            base = 0.5 * (ZS_arr[i] + ZP_arr[i])
        else:  # 21..30
            base = (ZS_arr[i] + ZP_arr[i] + ZD_arr[i]) / 3.0
        za[i] = base
    if n > 10:
        za[10:] *= etaspec  # indices 11..n (Fortran 1-based 11..)
    za *= eta
    zb = za.copy()
    return za, zb

def _sab_table_key(p2: float, thrR: float, ngrid: int) -> str:
    return f"p2={p2:.6f}|R={thrR:.1f}|n={ngrid}"

def _build_sqrt_sab_table(p2: float, thrR: float = 60.0, ngrid: int = 2048) -> Tuple[np.ndarray, float, int]:
    """Precompute sqrt(sab(r)) tables for Z=1..36 pairs on a linear r-grid.

    Returns (sqrt_tab, dr, ngrid) and caches per (p2, thrR, ngrid).
    """
    key = _sab_table_key(p2, thrR, ngrid)
    if key in _SAB_TABLE_CACHE:
        return _SAB_TABLE_CACHE[key]
    za_tab, zb_tab = _setzet(p2, 1.0)
    shell = _SHELL
    nZ = int(min(za_tab.size, shell.size))
    dr = thrR / float(max(1, ngrid - 1))
    grid = np.linspace(0.0, thrR, ngrid, dtype=np.float64)
    if ngrid > 0:
        grid[0] = 1e-8  # avoid r=0 singularities in special functions
    sqrt_tab = np.zeros((nZ, nZ, ngrid), dtype=np.float64)
    for i in range(nZ):
        zi = i + 1
        si = int(shell[zi - 1])
        za = float(za_tab[zi - 1])
        for j in range(nZ):
            zj = j + 1
            sj = int(shell[zj - 1])
            zb = float(zb_tab[zj - 1])
            vals = np.empty(ngrid, dtype=np.float64)
            for k in range(ngrid):
                r = grid[k]
                s = _ssovl_one_nb(r, si, sj, za, zb)
                vals[k] = math.sqrt(s) if s > 0.0 else 0.0
            sqrt_tab[i, j, :] = vals
    _SAB_TABLE_CACHE[key] = (sqrt_tab, float(dr), int(ngrid))
    return _SAB_TABLE_CACHE[key]



# -------- Optional Numba-accelerated kernels (non-damped path) --------
if _HAVE_NUMBA:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _A0_nb(x): return math.exp(-x) / x
    @njit(cache=True, fastmath=True)
    def _A1_nb(x): return ((1.0 + x) * math.exp(-x)) / (x * x)
    @njit(cache=True, fastmath=True)
    def _A2_nb(x): return ((2.0 + 2.0 * x + x * x) * math.exp(-x)) / (x ** 3)
    @njit(cache=True, fastmath=True)
    def _A3_nb(x):
        x2 = x * x; x3 = x2 * x
        return ((6.0 + 6.0 * x + 3.0 * x2 + x3) * math.exp(-x)) / (x ** 4)
    @njit(cache=True, fastmath=True)
    def _A4_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x
        return ((24.0 + 24.0 * x + 12.0 * x2 + 4.0 * x3 + x4) * math.exp(-x)) / (x ** 5)
    @njit(cache=True, fastmath=True)
    def _A5_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
        return ((120.0 + 120.0 * x + 60.0 * x2 + 20.0 * x3 + 5.0 * x4 + x5) * math.exp(-x)) / (x ** 6)
    @njit(cache=True, fastmath=True)
    def _A6_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
        return ((720.0 + 720.0 * x + 360.0 * x2 + 120.0 * x3 + 30.0 * x4 + 6.0 * x5 + x6) * math.exp(-x)) / (x ** 7)

    @njit(cache=True, fastmath=True)
    def _B0_nb(x): return (math.exp(x) - math.exp(-x)) / x
    @njit(cache=True, fastmath=True)
    def _B1_nb(x):
        x2 = x * x
        return (((1.0 - x) * math.exp(x)) - ((1.0 + x) * math.exp(-x))) / (x2)
    @njit(cache=True, fastmath=True)
    def _B2_nb(x):
        x2 = x * x
        return (((2.0 - 2.0 * x + x2) * math.exp(x)) - ((2.0 + 2.0 * x + x2) * math.exp(-x))) / (x ** 3)
    @njit(cache=True, fastmath=True)
    def _B3_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x
        xx = (6.0 - 6.0 * x + 3.0 * x2 - x3) * math.exp(x) / x4
        yy = (6.0 + 6.0 * x + 3.0 * x2 + x3) * math.exp(-x) / x4
        return xx - yy
    @njit(cache=True, fastmath=True)
    def _B4_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
        xx = (24.0 - 24.0 * x + 12.0 * x2 - 4.0 * x3 + x4) * math.exp(x) / x5
        yy = (24.0 + 24.0 * x + 12.0 * x2 + 4.0 * x3 + x4) * math.exp(-x) / x5
        return xx - yy
    @njit(cache=True, fastmath=True)
    def _B5_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
        xx = (120.0 - 120.0 * x + 60.0 * x2 - 20.0 * x3 + 5.0 * x4 - x5) * math.exp(x) / x6
        yy = (120.0 + 120.0 * x + 60.0 * x2 + 20.0 * x3 + 5.0 * x4 + x5) * math.exp(-x) / x6
        return xx - yy
    @njit(cache=True, fastmath=True)
    def _B6_nb(x):
        x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x; x7 = x6 * x
        xx = (720.0 - 720.0 * x + 360.0 * x2 - 120.0 * x3 + 30.0 * x4 - 6.0 * x5 + x6) * math.exp(x) / x7
        yy = (720.0 + 720.0 * x + 360.0 * x2 + 120.0 * x3 + 30.0 * x4 + 6.0 * x5 + x6) * math.exp(-x) / x7
        return xx - yy

    @njit(cache=True, fastmath=True)
    def _Bint_series_nb(x: float, k: int) -> float:
        if abs(x) < 1e-6:
            # Match Fortran special-case: last-term value only
            return (1.0 + ((-1.0) ** k)) / (k + 1.0)
        s = 0.0
        fact = 1.0
        for i in range(0, 13):
            if i > 0:
                fact *= i
            num = 1.0 - ((-1.0) ** (k + i + 1))
            den = fact * (k + i + 1.0)
            s += num / den * ((-x) ** i)
        return s

    @njit(cache=True, fastmath=True)
    def _ssovl_one_nb(r: float, shell_i: int, shell_j: int, za: float, zb: float) -> float:
        R05 = 0.5 * r
        ax = (za + zb) * R05
        bx = (zb - za) * R05
        same = (za == zb) or (abs(za - zb) < 0.1)
        ii = shell_i * shell_j
        if same:
            if ii == 1:
                norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
                return norm * (_A2_nb(ax) * _Bint_series_nb(bx, 0) - _Bint_series_nb(bx, 2) * _A0_nb(ax))
            if ii == 2:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
                return math.sqrt(1.0 / 3.0) * norm * (_A3_nb(_ax) * _Bint_series_nb(_bx, 0) - _Bint_series_nb(_bx, 3) * _A0_nb(_ax) + _A2_nb(_ax) * _Bint_series_nb(_bx, 1) - _Bint_series_nb(_bx, 2) * _A1_nb(_ax))
            if ii == 4:
                norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
                return (norm / 3.0) * (_A4_nb(ax) * _Bint_series_nb(bx, 0) + _Bint_series_nb(bx, 4) * _A0_nb(ax) - 2.0 * _A2_nb(ax) * _Bint_series_nb(bx, 2))
            if ii == 3:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
                return (norm / math.sqrt(3.0)) * (_A4_nb(_ax) * _Bint_series_nb(_bx, 0) - _Bint_series_nb(_bx, 4) * _A0_nb(_ax) + 2.0 * (_A3_nb(_ax) * _Bint_series_nb(_bx, 1) - _Bint_series_nb(_bx, 3) * _A1_nb(_ax)))
            if ii == 6:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
                return (norm / 3.0) * (_A5_nb(_ax) * _Bint_series_nb(_bx, 0) + _A4_nb(_ax) * _Bint_series_nb(_bx, 1) - 2.0 * (_A3_nb(_ax) * _Bint_series_nb(_bx, 2) + _A2_nb(_ax) * _Bint_series_nb(_bx, 3)) + _A1_nb(_ax) * _Bint_series_nb(_bx, 4) + _A0_nb(_ax) * _Bint_series_nb(_bx, 5))
            if ii == 9:
                norm = math.sqrt((za * zb * r * r) ** 7) / 480.0
                return (norm / 3.0) * (_A6_nb(ax) * _Bint_series_nb(bx, 0) - 3.0 * (_A4_nb(ax) * _Bint_series_nb(bx, 2) - _A2_nb(ax) * _Bint_series_nb(bx, 4)) - _A0_nb(ax) * _Bint_series_nb(bx, 6))
        else:
            if ii == 1:
                norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
                return norm * (_A2_nb(ax) * _B0_nb(bx) - _B2_nb(bx) * _A0_nb(ax))
            if ii == 2:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
                return math.sqrt(1.0 / 3.0) * norm * (_A3_nb(_ax) * _B0_nb(_bx) - _B3_nb(_bx) * _A0_nb(_ax) + _A2_nb(_ax) * _B1_nb(_bx) - _B2_nb(_bx) * _A1_nb(_ax))
            if ii == 4:
                norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
                return (norm / 3.0) * (_A4_nb(ax) * _B0_nb(bx) + _B4_nb(bx) * _A0_nb(ax) - 2.0 * _A2_nb(ax) * _B2_nb(bx))
            if ii == 3:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
                return (norm / math.sqrt(3.0)) * (_A4_nb(_ax) * _B0_nb(_bx) - _B4_nb(_bx) * _A0_nb(_ax) + 2.0 * (_A3_nb(_ax) * _B1_nb(_bx) - _B3_nb(_bx) * _A1_nb(_ax)))
            if ii == 6:
                if shell_i < shell_j:
                    _za, _zb = za, zb
                else:
                    _za, _zb = zb, za
                _ax = (_za + _zb) * R05
                _bx = (_zb - _za) * R05
                norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
                return (norm / 3.0) * (_A5_nb(_ax) * _B0_nb(_bx) + _A4_nb(_ax) * _B1_nb(_bx) - 2.0 * (_A3_nb(_ax) * _B2_nb(_bx) + _A2_nb(_ax) * _B3_nb(_bx)) + _A1_nb(_ax) * _B4_nb(_bx) + _A0_nb(_ax) * _B5_nb(_bx))
            if ii == 9:
                norm = math.sqrt((za * zb * r * r) ** 7) / 1440.0
                return norm * (_A6_nb(ax) * _B0_nb(bx) - 3.0 * (_A4_nb(ax) * _B2_nb(bx) - _A2_nb(ax) * _B4_nb(bx)) - _A0_nb(ax) * _B6_nb(bx))
        return 0.0

    from numba import prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb(Z: np.ndarray, xyz_bohr: np.ndarray, emiss: np.ndarray, nbas: np.ndarray, p1: float, p2: float, p3: float, p4: float, shell: np.ndarray, za_tab: np.ndarray, zb_tab: np.ndarray, thrR: float, thrE: float) -> float:
        N = Z.shape[0]
        xva = np.empty(N, dtype=np.float64)
        for i in range(N):
            zi = int(Z[i])
            xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
        xvb = xva.copy()
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for j in range(N):
                if i == j:
                    continue
                zj = int(Z[j])
                vb = xvb[j]
                if vb < 0.5:
                    continue
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                sab = _ssovl_one_nb(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                if (sab == 0.0) or (math.sqrt(abs(sab)) < thrE):
                    continue
                expt = math.exp(-p3 * (r ** p4))
                if expt < EXPT_CUTOFF:
                    continue
                ene_old = expt / math.sqrt(vb * sab)
                if abs(ene_old) < thrE:
                    continue
                ea += emiss[zi - 1] * ene_old
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

# Auxiliary integrals A_k(x) and B_k(x)
def _A0(x): return np.exp(-x) / x
def _A1(x): return ((1 + x) * np.exp(-x)) / (x ** 2)
def _A2(x): return ((2 + 2 * x + x * x) * np.exp(-x)) / (x ** 3)
def _A3(x):
    x2 = x * x; x3 = x2 * x
    return ((6 + 6 * x + 3 * x2 + x3) * np.exp(-x)) / (x ** 4)
def _A4(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    return ((24 + 24 * x + 12 * x2 + 4 * x3 + x4) * np.exp(-x)) / (x ** 5)
def _A5(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
    return ((120 + 120 * x + 60 * x2 + 20 * x3 + 5 * x4 + x5) * np.exp(-x)) / (x ** 6)
def _A6(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
    return ((720 + 720 * x + 360 * x2 + 120 * x3 + 30 * x4 + 6 * x5 + x6) * np.exp(-x)) / (x ** 7)

def _B0(x): return (np.exp(x) - np.exp(-x)) / x
def _B1(x): return (((1 - x) * np.exp(x)) - ((1 + x) * np.exp(-x))) / (x ** 2)
def _B2(x): return (((2 - 2 * x + x * x) * np.exp(x)) - ((2 + 2 * x + x * x) * np.exp(-x))) / (x ** 3)
def _B3(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    xx = (6 - 6 * x + 3 * x2 - x3) * np.exp(x) / x4
    yy = (6 + 6 * x + 3 * x2 + x3) * np.exp(-x) / x4
    return xx - yy
def _B4(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
    xx = (24 - 24 * x + 12 * x2 - 4 * x3 + x4) * np.exp(x) / x5
    yy = (24 + 24 * x + 12 * x2 + 4 * x3 + x4) * np.exp(-x) / x5
    return xx - yy
def _B5(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
    xx = (120 - 120 * x + 60 * x2 - 20 * x3 + 5 * x4 - x5) * np.exp(x) / x6
    yy = (120 + 120 * x + 60 * x2 + 20 * x3 + 5 * x4 + x5) * np.exp(-x) / x6
    return xx - yy
def _B6(x):
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x; x7 = x6 * x
    xx = (720 - 720 * x + 360 * x2 - 120 * x3 + 30 * x4 - 6 * x5 + x6) * np.exp(x) / x7
    yy = (720 + 720 * x + 360 * x2 + 120 * x3 + 30 * x4 + 6 * x5 + x6) * np.exp(-x) / x7
    return xx - yy


def _Bint_series(x: float, k: int) -> float:
    # Series definition used in Fortran bint for general k (0..6), 12 terms
    if abs(x) < 1e-6:
        # Match Fortran special-case: last-term value only
        return (1.0 + ((-1.0) ** k)) / (k + 1.0)
    s = 0.0
    fact = 1.0
    for i in range(0, 13):
        if i > 0:
            fact *= i
        num = 1.0 - ((-1.0) ** (k + i + 1))
        den = fact * (k + i + 1.0)
        s += num / den * ((-x) ** i)
    return s


def _ssovl_one(r: float, shell_i: int, shell_j: int, za: float, zb: float) -> float:
    # s-type Slater overlap for 1s/2s/3s combos; mirrors Fortran ssovl
    R05 = 0.5 * r
    ax = (za + zb) * R05
    bx = (zb - za) * R05
    same = (za == zb) or (abs(za - zb) < 0.1)
    ii = shell_i * shell_j

    def A0(x): return _A0(x)
    def A1(x): return _A1(x)
    def A2(x): return _A2(x)
    def A3(x): return _A3(x)
    def A4(x): return _A4(x)
    def A5(x): return _A5(x)
    def A6(x): return _A6(x)

    def B0(x): return _B0(x)
    def B1(x): return _B1(x)
    def B2(x): return _B2(x)
    def B3(x): return _B3(x)
    def B4(x): return _B4(x)
    def B5(x): return _B5(x)
    def B6(x): return _B6(x)

    if same:
        if ii == 1:
            norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
            return norm * (A2(ax) * _Bint_series(bx, 0) - _Bint_series(bx, 2) * A0(ax))
        if ii == 2:
            # Ensure ordering corresponds to <1s|2s> vs <2s|1s> like Fortran
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
            return math.sqrt(1.0 / 3.0) * norm * (A3(_ax) * _Bint_series(_bx, 0) - _Bint_series(_bx, 3) * A0(_ax) + A2(_ax) * _Bint_series(_bx, 1) - _Bint_series(_bx, 2) * A1(_ax))
        if ii == 4:
            norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
            return (norm / 3.0) * (A4(ax) * _Bint_series(bx, 0) + _Bint_series(bx, 4) * A0(ax) - 2.0 * A2(ax) * _Bint_series(bx, 2))
        if ii == 3:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
            return (norm / math.sqrt(3.0)) * (A4(_ax) * _Bint_series(_bx, 0) - _Bint_series(_bx, 4) * A0(_ax) + 2.0 * (A3(_ax) * _Bint_series(_bx, 1) - _Bint_series(_bx, 3) * A1(_ax)))
        if ii == 6:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
            return (norm / 3.0) * (A5(_ax) * _Bint_series(_bx, 0) + A4(_ax) * _Bint_series(_bx, 1) - 2.0 * (A3(_ax) * _Bint_series(_bx, 2) + A2(_ax) * _Bint_series(_bx, 3)) + A1(_ax) * _Bint_series(_bx, 4) + A0(_ax) * _Bint_series(_bx, 5))
        if ii == 9:
            norm = math.sqrt((za * zb * r * r) ** 7) / 480.0
            return (norm / 3.0) * (A6(ax) * _Bint_series(bx, 0) - 3.0 * (A4(ax) * _Bint_series(bx, 2) - A2(ax) * _Bint_series(bx, 4)) - A0(ax) * _Bint_series(bx, 6))
    else:
        if ii == 1:
            norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
            return norm * (A2(ax) * B0(bx) - B2(bx) * A0(ax))
        if ii == 2:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
            return math.sqrt(1.0 / 3.0) * norm * (A3(_ax) * B0(_bx) - B3(_bx) * A0(_ax) + A2(_ax) * B1(_bx) - B2(_bx) * A1(_ax))
        if ii == 4:
            norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
            return (norm / 3.0) * (A4(ax) * B0(bx) + B4(bx) * A0(ax) - 2.0 * A2(ax) * B2(bx))
        if ii == 3:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
            return (norm / math.sqrt(3.0)) * (A4(_ax) * B0(_bx) - B4(_bx) * A0(_ax) + 2.0 * (A3(_ax) * B1(_bx) - B3(_bx) * A1(_ax)))
        if ii == 6:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
            return (norm / 3.0) * (A5(_ax) * B0(_bx) + A4(_ax) * B1(_bx) - 2.0 * (A3(_ax) * B2(_bx) + A2(_ax) * B3(_bx)) + A1(_ax) * B4(_bx) + A0(_ax) * B5(_bx))
        if ii == 9:
            norm = math.sqrt((za * zb * r * r) ** 7) / 1440.0
            return norm * (A6(ax) * B0(bx) - 3.0 * (A4(ax) * B2(bx) - A2(ax) * B4(bx)) - A0(ax) * B6(bx))
    # Fallback
    return 0.0
if _HAVE_NUMBA:
    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb_damped(Z, xyz_bohr, emiss, xva, xvb, p1, p2, p3, p4, shell, za_tab, zb_tab, thrR, thrE, r0ab, dmp_scal, dmp_exp):
        N = Z.shape[0]
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for j in range(N):
                if i == j:
                    continue
                zj = int(Z[j])
                vb = xvb[j]
                if vb < 0.5:
                    continue
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                sab = _ssovl_one_nb(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                if (sab == 0.0) or (math.sqrt(abs(sab)) < thrE):
                    continue
                expt = math.exp(-p3 * (r ** p4))
                if expt < EXPT_CUTOFF:
                    continue
                ene_old = expt / math.sqrt(vb * sab)
                if abs(ene_old) < thrE:
                    continue
                r0 = r0ab[zi - 1, zj - 1]
                rscal = r / r0
                dampval = (1.0 - 1.0 / (1.0 + dmp_scal * (rscal ** dmp_exp)))
                ea += emiss[zi - 1] * (ene_old * dampval)
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

    @njit(cache=True, fastmath=True)
    def _base_short_range_nb(Z, xyz_bohr, r0ab, rscal, qscal, thrR=30.0):
        N = Z.shape[0]
        e = 0.0
        for i in range(N - 1):
            zi = int(Z[i])
            if zi < 1 or zi > 18:
                continue
            for j in range(i + 1, N):
                zj = int(Z[j])
                if zj < 1 or zj > 18:
                    continue
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                r0 = rscal * (r0ab[zi - 1, zj - 1] ** 0.75)
                ff = -((float(zi) * float(zj)) ** 1.5)
                e += ff * math.exp(-r0 * r)
        return e * qscal

    @njit(cache=True, fastmath=True)
    def _srb_energy_nb(Z, xyz_bohr, r0ab, rscal, qscal, thrR=30.0):
        N = Z.shape[0]
        e = 0.0
        for i in range(N):
            zi = int(Z[i])
            for j in range(i + 1, N):
                zj = int(Z[j])
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                r0 = rscal / r0ab[zi - 1, zj - 1]
                ff = -((float(zi) * float(zj)) ** 0.5)
                e += qscal * ff * math.exp(-r0 * r)
        return e

    @njit(cache=True, parallel=True, fastmath=True)
    def _srb_energy_nb_pbc(Z, xyz_bohr, lat, r0ab, rscal, qscal, thrR, t1, t2, t3):
        """SRB energy in PBC using symmetric image set (no 0.5 factor).

        Counts each pair exactly once by:
        - Using half-lattice images: (a>0) or (a==0,b>0) or (a==0,b==0,c>0)
        - For the zero image (0,0,0), only j>i pairs
        """
        N = Z.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        for i in prange(N):
            zi = int(Z[i])
            ei = 0.0
            for a in range(-t1, t1+1):
                for b in range(-t2, t2+1):
                    for c in range(-t3, t3+1):
                        zero_img = (a == 0 and b == 0 and c == 0)
                        pos_img = (a > 0) or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0)
                        if (not zero_img) and (not pos_img):
                            continue  # skip negative images to avoid double counting
                        for j in range(N):
                            if zero_img and j <= i:
                                continue  # only upper triangle for zero image
                            zj = int(Z[j])
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            r0 = rscal / r0ab[zi - 1, zj - 1]
                            ff = -((float(zi) * float(zj)) ** 0.5)
                            ei += qscal * ff * math.exp(-r0 * r)
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e

    @njit(cache=True, parallel=True, fastmath=True)
    def _base_short_range_nb_pbc(Z, xyz_bohr, lat, r0ab, rscal, qscal, thrR, t1, t2, t3):
        """Base short-range correction in PBC matching Fortran counting.

        Loops over full symmetric image set (a,b,c in [-t..t]) and j<=i, with
        self-image (i==j and a==b==c==0) skipped and self-images weighted by 1/2.
        """
        N = Z.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        for i in prange(N):
            zi = int(Z[i])
            if zi < 1 or zi > 18:
                continue
            ei = 0.0
            for j in range(i + 1):
                zj = int(Z[j])
                if zj < 1 or zj > 18:
                    continue
                # Precompute pair factors
                zfac = (float(zi) * float(zj)) ** 1.5
                r0 = rscal * (r0ab[zi - 1, zj - 1] ** 0.75)
                for a in range(-t1, t1 + 1):
                    for b in range(-t2, t2 + 1):
                        for c in range(-t3, t3 + 1):
                            # Skip exact self for zero image
                            if (i == j) and (a == 0 and b == 0 and c == 0):
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            ex = math.exp(-r0 * r)
                            if i == j:
                                ei += -0.5 * zfac * ex
                            else:
                                ei += -zfac * ex
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e * qscal

    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb_pbc(Z, xyz_bohr, emiss, nbas, p1, p2, p3, p4, shell, za_tab, zb_tab, thrR, thrE, lat, t1, t2, t3):
        N = Z.shape[0]
        xva = np.empty(N, dtype=np.float64)
        for i in range(N):
            zi = int(Z[i])
            xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
        xvb = xva.copy()
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for a in range(-t1, t1+1):
                for b in range(-t2, t2+1):
                    for c in range(-t3, t3+1):
                        for j in range(N):
                            if a==0 and b==0 and c==0 and i==j:
                                continue
                            zj = int(Z[j])
                            vb = xvb[j]
                            if vb < 0.5:
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            expt = math.exp(-p3 * (r ** p4))
                            if expt < EXPT_CUTOFF:
                                continue
                            sab = _ssovl_one_nb(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                            if (sab == 0.0) or (math.sqrt(abs(sab)) < thrE):
                                continue
                            ene_old = expt / math.sqrt(vb * sab)
                            if abs(ene_old) < thrE:
                                continue
                            ea += emiss[zi - 1] * ene_old
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

    @njit(cache=True, fastmath=True)
    def _gcp_energy_nb_pbc_fortran(Z, xyz_bohr, emiss, nbas, p1, p2, p3, p4, shell, za_tab, zb_tab, thrR, thrE, lat, t1, t2, t3):
        """Literal Fortran-order PBC gCP summation (sequential, deterministic).

        Mirrors gcp.f90:gcp_egrad PBC loops: i over atoms, j over atoms,
        a,b,c over [-t..t], skipping only the exact self-image for (i==j,a==b==c==0).
        """
        N = Z.shape[0]
        # xva/xvb like Fortran
        xva = np.empty(N, dtype=np.float64)
        for i in range(N):
            zi = int(Z[i])
            xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
        xvb = xva.copy()
        ecp = 0.0
        EXPT_CUTOFF = 1e-17
        for i in range(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for j in range(N):
                zj = int(Z[j])
                vb = xvb[j]
                if vb < 0.5:
                    continue
                for a in range(-t1, t1 + 1):
                    for b in range(-t2, t2 + 1):
                        for c in range(-t3, t3 + 1):
                            if (a == 0 and b == 0 and c == 0 and i == j):
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            # Fortran computes expt first in many branches; keep order
                            expt = math.exp(-p3 * (r ** p4))
                            if expt < EXPT_CUTOFF:
                                continue
                            sab = _ssovl_one_nb(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                            if (sab == 0.0) or (math.sqrt(abs(sab)) < thrE):
                                continue
                            ene_old = expt / math.sqrt(vb * sab)
                            if abs(ene_old) < thrE:
                                continue
                            ea += emiss[zi - 1] * ene_old
            ecp += ea
        return ecp * p1

    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb_damped_pbc(Z, xyz_bohr, emiss, xva, xvb, p1, p2, p3, p4, shell, za_tab, zb_tab, thrR, thrE, lat, t1, t2, t3, r0ab, dmp_scal, dmp_exp):
        N = Z.shape[0]
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for a in range(-t1, t1+1):
                for b in range(-t2, t2+1):
                    for c in range(-t3, t3+1):
                        for j in range(N):
                            if a==0 and b==0 and c==0 and i==j:
                                continue
                            zj = int(Z[j])
                            vb = xvb[j]
                            if vb < 0.5:
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            expt = math.exp(-p3 * (r ** p4))
                            if expt < EXPT_CUTOFF:
                                continue
                            sab = _ssovl_one_nb(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                            if (sab == 0.0) or (math.sqrt(abs(sab)) < thrE):
                                continue
                            ene_old = expt / math.sqrt(vb * sab)
                            if abs(ene_old) < thrE:
                                continue
                            r0 = r0ab[zi - 1, zj - 1]
                            rscal = r / r0
                            dampval = (1.0 - 1.0 / (1.0 + dmp_scal * (rscal ** dmp_exp)))
                            ea += emiss[zi - 1] * (ene_old * dampval)
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

    # --- gCP via precomputed sqrt(sab) lookup tables (faster) ---
    @njit(cache=True, fastmath=True)
    def _interp_sqrt_sab_tab(sqrt_sab_tab, zi, zj, r, dr, ngrid):
        if r <= 0.0:
            return sqrt_sab_tab[zi - 1, zj - 1, 0]
        tmax = dr * (ngrid - 1)
        if r >= tmax:
            return 0.0
        x = r / dr
        i0 = int(x)
        t = x - i0
        v0 = sqrt_sab_tab[zi - 1, zj - 1, i0]
        v1 = sqrt_sab_tab[zi - 1, zj - 1, i0 + 1]
        return (1.0 - t) * v0 + t * v1

    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb_pbc_tab(Z, xyz_bohr, emiss, nbas, p1, p2, p3, p4, thrR, thrE, lat, t1, t2, t3, sqrt_sab_tab, dr, ngrid):
        N = Z.shape[0]
        xva = np.empty(N, dtype=np.float64)
        for i in range(N):
            zi = int(Z[i])
            xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
        xvb = xva.copy()
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for a in range(-t1, t1+1):
                for b in range(-t2, t2+1):
                    for c in range(-t3, t3+1):
                        for j in range(N):
                            if a==0 and b==0 and c==0 and i==j:
                                continue
                            zj = int(Z[j])
                            vb = xvb[j]
                            if vb < 0.5:
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            expt = math.exp(-p3 * (r ** p4))
                            if expt < EXPT_CUTOFF:
                                continue
                            sroot = _interp_sqrt_sab_tab(sqrt_sab_tab, zi, zj, r, dr, ngrid)
                            if sroot <= 0.0 or sroot < thrE:
                                continue
                            ene_old = expt / math.sqrt(vb) / sroot
                            if abs(ene_old) < thrE:
                                continue
                            ea += emiss[zi - 1] * ene_old
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

    @njit(cache=True, parallel=True, fastmath=True)
    def _gcp_energy_nb_damped_pbc_tab(Z, xyz_bohr, emiss, xva, xvb, p1, p2, p3, p4, thrR, thrE, lat, t1, t2, t3, r0ab, dmp_scal, dmp_exp, sqrt_sab_tab, dr, ngrid):
        N = Z.shape[0]
        ea_arr = np.zeros(N, dtype=np.float64)
        EXPT_CUTOFF = 1e-17
        for i in prange(N):
            zi = int(Z[i])
            va = xva[i]
            if emiss[zi - 1] < 1e-7:
                continue
            ea = 0.0
            for a in range(-t1, t1+1):
                for b in range(-t2, t2+1):
                    for c in range(-t3, t3+1):
                        for j in range(N):
                            if a==0 and b==0 and c==0 and i==j:
                                continue
                            zj = int(Z[j])
                            vb = xvb[j]
                            if vb < 0.5:
                                continue
                            dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + a*lat[0,0] + b*lat[1,0] + c*lat[2,0])
                            dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + a*lat[0,1] + b*lat[1,1] + c*lat[2,1])
                            dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + a*lat[0,2] + b*lat[1,2] + c*lat[2,2])
                            r = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if r > thrR:
                                continue
                            expt = math.exp(-p3 * (r ** p4))
                            if expt < EXPT_CUTOFF:
                                continue
                            sroot = _interp_sqrt_sab_tab(sqrt_sab_tab, zi, zj, r, dr, ngrid)
                            if sroot <= 0.0 or sroot < thrE:
                                continue
                            ene_old = expt / math.sqrt(vb) / sroot
                            if abs(ene_old) < thrE:
                                continue
                            r0 = r0ab[zi - 1, zj - 1]
                            rscal = r / r0
                            dampval = (1.0 - 1.0 / (1.0 + dmp_scal * (rscal ** dmp_exp)))
                            ea += emiss[zi - 1] * (ene_old * dampval)
            ea_arr[i] = ea
        ecp = 0.0
        for i in range(N):
            ecp += ea_arr[i]
        return ecp * p1

    # --- Vectorized PBC SRB/base using precomputed translation vectors and pair factors ---
    @njit(cache=True, parallel=True, fastmath=True)
    def _srb_energy_nb_pbc_vec(Z, xyz_bohr, tvecs, r0fac, ff, thrR):
        N = Z.shape[0]
        M = tvecs.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        for i in prange(N):
            zi = int(Z[i])
            r0fac_i = r0fac[zi - 1]
            ff_i = ff[zi - 1]
            ei = 0.0
            # Non-zero symmetric images
            for m in range(M):
                tx = tvecs[m,0]; ty = tvecs[m,1]; tz = tvecs[m,2]
                for j in range(N):
                    zj = int(Z[j])
                    dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + tx)
                    dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + ty)
                    dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + tz)
                    r = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if r > thrR:
                        continue
                    a = r0fac_i[zj - 1] * r
                    if a > 40.0:
                        continue
                    ei += ff_i[zj - 1] * math.exp(-a)
            # Zero image upper triangle
            for j in range(i + 1, N):
                zj = int(Z[j])
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                a = r0fac_i[zj - 1] * r
                if a > 40.0:
                    continue
                ei += ff_i[zj - 1] * math.exp(-a)
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e

    @njit(cache=True, parallel=True, fastmath=True)
    def _base_short_range_nb_pbc_vec(Z, xyz_bohr, tvecs, r0fac, ff, thrR):
        N = Z.shape[0]
        M = tvecs.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        for i in prange(N):
            zi = int(Z[i])
            if zi < 1 or zi > 18:
                continue
            r0fac_i = r0fac[zi - 1]
            ff_i = ff[zi - 1]
            ei = 0.0
            # Non-zero symmetric images
            for m in range(M):
                tx = tvecs[m,0]; ty = tvecs[m,1]; tz = tvecs[m,2]
                for j in range(N):
                    zj = int(Z[j])
                    if zj < 1 or zj > 18:
                        continue
                    dx = xyz_bohr[i,0] - (xyz_bohr[j,0] + tx)
                    dy = xyz_bohr[i,1] - (xyz_bohr[j,1] + ty)
                    dz = xyz_bohr[i,2] - (xyz_bohr[j,2] + tz)
                    r = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if r > thrR:
                        continue
                    a = r0fac_i[zj - 1] * r
                    if a > 40.0:
                        continue
                    ei += ff_i[zj - 1] * math.exp(-a)
            # Zero image upper triangle
            for j in range(i + 1, N):
                zj = int(Z[j])
                if zj < 1 or zj > 18:
                    continue
                dx = xyz_bohr[i,0] - xyz_bohr[j,0]
                dy = xyz_bohr[i,1] - xyz_bohr[j,1]
                dz = xyz_bohr[i,2] - xyz_bohr[j,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > thrR:
                    continue
                a = r0fac_i[zj - 1] * r
                if a > 40.0:
                    continue
                ei += ff_i[zj - 1] * math.exp(-a)
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e

    # SoA variants with split coords and translations
    @njit(cache=True, parallel=True, fastmath=True)
    def _srb_energy_nb_pbc_vec_soa(Z, x, y, z, tvx, tvy, tvz, r0fac, ff, thrR):
        N = Z.shape[0]
        M = tvx.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        thrR2 = thrR * thrR
        for i in prange(N):
            zi = int(Z[i])
            r0fac_i = r0fac[zi - 1]
            ff_i = ff[zi - 1]
            xi = x[i]; yi = y[i]; zi0 = z[i]
            ei = 0.0
            # Non-zero images
            for m in range(M):
                tx = tvx[m]; ty = tvy[m]; tz = tvz[m]
                for j in range(N):
                    zj = int(Z[j])
                    dx = xi - (x[j] + tx)
                    dy = yi - (y[j] + ty)
                    dz = zi0 - (z[j] + tz)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 > thrR2:
                        continue
                    r = math.sqrt(r2)
                    a = r0fac_i[zj - 1] * r
                    if a > 40.0:
                        continue
                    ei += ff_i[zj - 1] * math.exp(-a)
            # Zero image (upper triangle)
            for j in range(i + 1, N):
                zj = int(Z[j])
                dx = xi - x[j]
                dy = yi - y[j]
                dz = zi0 - z[j]
                r2 = dx*dx + dy*dy + dz*dz
                if r2 > thrR2:
                    continue
                r = math.sqrt(r2)
                a = r0fac_i[zj - 1] * r
                if a > 40.0:
                    continue
                ei += ff_i[zj - 1] * math.exp(-a)
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e

    @njit(cache=True, parallel=True, fastmath=True)
    def _base_short_range_nb_pbc_vec_soa(Z, x, y, z, tvx, tvy, tvz, validJ, r0fac, ff, thrR):
        N = Z.shape[0]
        M = tvx.shape[0]
        NJ = validJ.shape[0]
        e_arr = np.zeros(N, dtype=np.float64)
        thrR2 = thrR * thrR
        for i in prange(N):
            zi = int(Z[i])
            if zi < 1 or zi > 18:
                continue
            r0fac_i = r0fac[zi - 1]
            ff_i = ff[zi - 1]
            xi = x[i]; yi = y[i]; zi0 = z[i]
            ei = 0.0
            # Non-zero images
            for m in range(M):
                tx = tvx[m]; ty = tvy[m]; tz = tvz[m]
                for jj in range(NJ):
                    j = int(validJ[jj])
                    zj = int(Z[j])
                    dx = xi - (x[j] + tx)
                    dy = yi - (y[j] + ty)
                    dz = zi0 - (z[j] + tz)
                    r2 = dx*dx + dy*dy + dz*dz
                    if r2 > thrR2:
                        continue
                    r = math.sqrt(r2)
                    a = r0fac_i[zj - 1] * r
                    if a > 40.0:
                        continue
                    ei += ff_i[zj - 1] * math.exp(-a)
            # Zero image (upper triangle)
            for jj in range(NJ):
                j = int(validJ[jj])
                if j <= i:
                    continue
                zj = int(Z[j])
                dx = xi - x[j]
                dy = yi - y[j]
                dz = zi0 - z[j]
                r2 = dx*dx + dy*dy + dz*dz
                if r2 > thrR2:
                    continue
                r = math.sqrt(r2)
                a = r0fac_i[zj - 1] * r
                if a > 40.0:
                    continue
                ei += ff_i[zj - 1] * math.exp(-a)
            e_arr[i] = ei
        e = 0.0
        for i in range(N):
            e += e_arr[i]
        return e



def _srb_energy_only(Z: np.ndarray, xyz_bohr: np.ndarray, r0ab: np.ndarray, rscal: float, qscal: float, thrR: float = 30.0) -> float:
    n = int(Z.shape[0])
    e = 0.0
    for i in range(n):
        zi = int(Z[i])
        for j in range(i + 1, n):
            zj = int(Z[j])
            r = float(np.linalg.norm(xyz_bohr[i] - xyz_bohr[j]))
            if r > thrR:
                continue
            r0 = rscal / float(r0ab[zi - 1, zj - 1])
            ff = -((float(zi) * float(zj)) ** 0.5)
            e += qscal * ff * math.exp(-r0 * r)
    return e

def _base_short_range_energy(Z: np.ndarray, xyz_bohr: np.ndarray, r0ab: np.ndarray, rscal: float, qscal: float, thrR: float = 30.0) -> float:
    n = int(Z.shape[0])
    e = 0.0
    for i in range(n - 1):
        zi = int(Z[i])
        if zi < 1 or zi > 18:
            continue
        for j in range(i + 1, n):
            zj = int(Z[j])
            if zj < 1 or zj > 18:
                continue
            r = float(np.linalg.norm(xyz_bohr[i] - xyz_bohr[j]))
            if r > thrR:
                continue
            r0 = rscal * (float(r0ab[zi - 1, zj - 1]) ** 0.75)
            ff = -((float(zi) * float(zj)) ** 1.5)
            e += ff * math.exp(-r0 * r)
    return e * qscal

def _tau_max_from_lat(lat_bohr: np.ndarray, thrR: float) -> Tuple[int, int, int]:
    """Estimate translation extents (tau_max) such that |r| <= thrR is covered.

    Uses a conservative metric-based bound similar in spirit to Fortran criteria:
    tau_a >= ceil((thrR + |b|/2 + |c|/2)/|a|), and cyclic for b,c.
    This reduces images for skewed cells vs naive thrR/|a|.
    """
    a = lat_bohr[0]
    b = lat_bohr[1]
    c = lat_bohr[2]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    lc = float(np.linalg.norm(c))
    # avoid division by zero in pathological cells
    la = max(la, 1e-12)
    lb = max(lb, 1e-12)
    lc = max(lc, 1e-12)
    t1 = int(math.ceil((thrR + 0.5*lb + 0.5*lc) / la))
    t2 = int(math.ceil((thrR + 0.5*la + 0.5*lc) / lb))
    t3 = int(math.ceil((thrR + 0.5*la + 0.5*lb) / lc))
    return t1, t2, t3

def _build_half_tvecs(lat_bohr: np.ndarray, t1: int, t2: int, t3: int) -> np.ndarray:
    """Build symmetric half-lattice translation vectors (exclude zero).

    Includes vectors where (a>0) or (a==0 and b>0) or (a==0 and b==0 and c>0).
    Returns an array of shape (M,3) in Bohr.
    """
    tvecs = []
    # Ensure contiguous arrays for numba-friendly access
    a0 = lat_bohr[0].astype(np.float64)
    b0 = lat_bohr[1].astype(np.float64)
    c0 = lat_bohr[2].astype(np.float64)
    for a in range(-t1, t1+1):
        for b in range(-t2, t2+1):
            for c in range(-t3, t3+1):
                if a == 0 and b == 0 and c == 0:
                    continue
                pos_img = (a > 0) or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0)
                if not pos_img:
                    continue
                # t = a*a0 + b*b0 + c*c0
                tx = a*a0[0] + b*b0[0] + c*c0[0]
                ty = a*a0[1] + b*b0[1] + c*c0[1]
                tz = a*a0[2] + b*b0[2] + c*c0[2]
                tvecs.append((tx, ty, tz))
    if len(tvecs) == 0:
        return np.zeros((0,3), dtype=np.float64)
    return np.ascontiguousarray(np.array(tvecs, dtype=np.float64))

def _filter_tvecs_by_bound(tvecs: np.ndarray, xyz_bohr: np.ndarray, thrR: float) -> np.ndarray:
    """Cull translation vectors using a simple bounding-sphere criterion.

    If |T| - 2R > thrR, where R is the radius of the minimal bounding sphere
    approximation (around the centroid), then no atom pair across that image
    can be within thrR, and the image can be skipped.
    """
    if tvecs.shape[0] == 0:
        return tvecs
    ctr = np.mean(xyz_bohr, axis=0)
    R = float(np.max(np.linalg.norm(xyz_bohr - ctr, axis=1)))
    keep = []
    for k in range(tvecs.shape[0]):
        T = tvecs[k]
        if float(np.linalg.norm(T)) - 2.0 * R <= thrR:
            keep.append(k)
    if len(keep) == tvecs.shape[0]:
        return tvecs
    if len(keep) == 0:
        return np.zeros((0,3), dtype=np.float64)
    return np.ascontiguousarray(tvecs[np.array(keep, dtype=np.int64)])

def _split_coords(xyz_bohr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(xyz_bohr[:, 0].astype(np.float64))
    y = np.ascontiguousarray(xyz_bohr[:, 1].astype(np.float64))
    z = np.ascontiguousarray(xyz_bohr[:, 2].astype(np.float64))
    return x, y, z

def _split_tvecs(tvecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tvecs.shape[0] == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z
    tvx = np.ascontiguousarray(tvecs[:, 0])
    tvy = np.ascontiguousarray(tvecs[:, 1])
    tvz = np.ascontiguousarray(tvecs[:, 2])
    return tvx, tvy, tvz

def gcp_energy_pbc_numpy(Z: np.ndarray, xyz: np.ndarray, lat: np.ndarray, method: str = 'b3lyp/def2svp', units: Literal['Angstrom','Bohr'] = 'Angstrom') -> float:
    # Convert to Bohr
    if units.lower().startswith('ang'):
        xyzb = xyz * ANG2BOHR
        latb = lat * ANG2BOHR
    else:
        xyzb = xyz.copy()
        latb = lat.copy()
    # SRB-only short-circuit
    opts = _method_options(method)
    Zm = _map_elements_for_params(Z)
    if opts.srb:
        r0ab = _load_r0ab_matrix()
        t1, t2, t3 = _tau_max_from_lat(latb, 30.0)
        if _HAVE_NUMBA:
            # Build image set and evaluate with best kernel (heuristic)
            tvecs = _build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
            tvecs = _filter_tvecs_by_bound(tvecs, xyzb.astype(np.float64), 30.0)
            M = int(tvecs.shape[0])
            N = int(Z.shape[0])
            # Pair factors (cover all available r0ab elements)
            z = np.arange(1, r0ab.shape[0] + 1, dtype=np.float64)
            zz = np.sqrt(z[:,None] * z[None,:])
            r0fac = (float(opts.srb_rscal)) / r0ab.astype(np.float64)
            ff = -(zz * float(opts.srb_qscal))
            # Heuristic switch: small systems → loop; larger → SoA vec
            if M * N < 50000:
                return float(_srb_energy_nb_pbc(Zm.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), float(opts.srb_rscal), float(opts.srb_qscal), 30.0, int(t1), int(t2), int(t3)))
            tvx, tvy, tvz = _split_tvecs(tvecs)
            x, y, zc = _split_coords(xyzb.astype(np.float64))
            return float(_srb_energy_nb_pbc_vec_soa(Zm.astype(np.int32), x, y, zc, tvx, tvy, tvz, r0fac.astype(np.float64), ff.astype(np.float64), 30.0))
        # Fallback non-JIT using symmetric image set (no 0.5 factor)
        e = 0.0
        n = int(Z.shape[0])
        # Precompute half tvecs
        tvecs = _build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
        tvecs = _filter_tvecs_by_bound(tvecs, xyzb.astype(np.float64), 30.0)
        for i in range(n):
            zi = int(Zm[i])
            # Non-zero images
            for m in range(tvecs.shape[0]):
                t = tvecs[m]
                for j in range(n):
                    zj = int(Zm[j])
                    rij = xyzb[i] - (xyzb[j] + t)
                    r = float(np.linalg.norm(rij))
                    if r > 30.0:
                        continue
                    r0 = opts.srb_rscal / float(r0ab[zi-1, zj-1])
                    ff = -((float(zi) * float(zj)) ** 0.5)
                    e += opts.srb_qscal * ff * math.exp(-r0 * r)
            # Zero image (upper triangle)
            for j in range(i+1, n):
                zj = int(Zm[j])
                rij = xyzb[i] - xyzb[j]
                r = float(np.linalg.norm(rij))
                if r > 30.0:
                    continue
                r0 = opts.srb_rscal / float(r0ab[zi-1, zj-1])
                ff = -((float(zi) * float(zj)) ** 0.5)
                e += opts.srb_qscal * ff * math.exp(-r0 * r)
        return float(e)
    # General gCP E via JIT
    params = _params_for_method(method)
    emiss = params.emiss
    nbas = params.nbas
    p1, p2, p3, p4 = params.p
    # No hard error for Z>emiss.size; Fortran maps elements to lower homologues
    # Pad emiss/nbas so kernels can index safely for mapped Z
    maxZ = int(np.max(Zm)) if Zm.size else 0
    if emiss.size < maxZ:
        emiss = np.pad(emiss, (0, maxZ - emiss.size), mode='constant', constant_values=0.0)
    if nbas.size < maxZ:
        nbas = np.pad(nbas.astype(np.int32), (0, maxZ - nbas.size), mode='constant', constant_values=0).astype(np.int32)
    shell = _SHELL
    za_tab, zb_tab = _setzet(p2, 1.0)
    thrR = 60.0
    thrE = np.finfo(np.float64).eps
    t1, t2, t3 = _tau_max_from_lat(latb, thrR)
    if _HAVE_NUMBA:
        if not opts.damp:
            gcp = float(_gcp_energy_nb_pbc(Zm.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za_tab.astype(np.float64), zb_tab.astype(np.float64), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3)))
        else:
            N = int(Z.shape[0])
            xva = np.empty(N, dtype=np.float64)
            special_xva = _normalize_method(method) in ('def2mtzvpp', 'mtzvpp', 'r2scan3c')
            for i in range(N):
                zi = int(Zm[i])
                if special_xva:
                    val = 1.0
                    if zi == 6:
                        val = 3.0
                    elif zi in (7, 8):
                        val = 0.5
                    xva[i] = val
                else:
                    xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
            xvb = xva.copy()
            r0ab = _load_r0ab_matrix()
            gcp = float(_gcp_energy_nb_damped_pbc(Zm.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), xva.astype(np.float64), xvb.astype(np.float64), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za_tab.astype(np.float64), zb_tab.astype(np.float64), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3), r0ab.astype(np.float64), float(opts.dmp_scal), float(opts.dmp_exp)))
        if opts.base:
            # add base PBC short-range; prefer vectorized JIT
            if 'r0ab' not in locals():
                r0ab = _load_r0ab_matrix()
            tvecs = _build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
            tvecs = _filter_tvecs_by_bound(tvecs, xyzb.astype(np.float64), 30.0)
            M = int(tvecs.shape[0])
            Nn = int(Z.shape[0])
            # Precompute pair factors
            z = np.arange(1, r0ab.shape[0] + 1, dtype=np.float64)
            zz15 = (z[:,None] * z[None,:]) ** 1.5
            r0fac = (0.7) * (r0ab.astype(np.float64) ** 0.75)
            ff = -(zz15 * 0.03)
            # Heuristic switch: small → loop; large → SoA vec
            if M * Nn < 40000:
                gcp += float(_base_short_range_nb_pbc(Zm.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03, 30.0, int(t1), int(t2), int(t3)))
            else:
                tvx, tvy, tvz = _split_tvecs(tvecs)
                x, y, zc = _split_coords(xyzb.astype(np.float64))
                validJ = np.where((Zm >= 1) & (Zm <= 18))[0].astype(np.int32)
                gcp += float(_base_short_range_nb_pbc_vec_soa(Zm.astype(np.int32), x, y, zc, tvx, tvy, tvz, validJ.astype(np.int32), r0fac.astype(np.float64), ff.astype(np.float64), 30.0))
        return gcp
    # Fallback: not supported efficiently without numba; suggest enabling numba
    raise NotImplementedError('PBC gCP requires numba for performance in this build')

def gcp_energy_numpy(Z: np.ndarray, xyz: np.ndarray, method: str = 'b3lyp/def2svp', units: Literal['Angstrom','Bohr'] = 'Angstrom') -> float:
    """Compute gCP energy (Hartree) for a molecule using numpy reference implementation.

    Args:
      Z: (N,) atomic numbers (1..36)
      xyz: (N,3) coordinates in Angstrom or Bohr
      method: gCP parameterization (see supported above)
      units: 'Angstrom' or 'Bohr'
    """
    # Early SRB-only short-circuit (e.g., B97-3c)
    _opts__ = _method_options(method)
    if _opts__.srb:
        xyz_bohr = xyz * ANG2BOHR if units.lower().startswith('ang') else xyz
        r0ab = _load_r0ab_matrix()
        Zm = _map_elements_for_params(Z)
        if _HAVE_NUMBA:
            return float(_srb_energy_nb(Zm.astype(np.int32), xyz_bohr.astype(np.float64), r0ab.astype(np.float64), float(_opts__.srb_rscal), float(_opts__.srb_qscal)))
        else:
            return float(_srb_energy_only(Zm, xyz_bohr, r0ab, _opts__.srb_rscal, _opts__.srb_qscal))

    params = _params_for_method(method)
    opts = _method_options(method)
    emiss = params.emiss
    nbas = params.nbas
    p1, p2, p3, p4 = params.p
    # Map elements like Fortran does and pad arrays for safety
    Zm = _map_elements_for_params(Z)
    maxZ = int(np.max(Zm)) if Zm.size else 0
    if emiss.size < maxZ:
        emiss = np.pad(emiss, (0, maxZ - emiss.size), mode='constant', constant_values=0.0)
    if nbas.size < maxZ:
        nbas = np.pad(nbas.astype(np.int32), (0, maxZ - nbas.size), mode='constant', constant_values=0).astype(np.int32)
    if units.lower().startswith('ang'):
        xyz = xyz * ANG2BOHR
    N = int(Z.shape[0])
    # Fast numba path
    opts = _method_options(method)
    za_tab, zb_tab = _setzet(p2, 1.0)
    shell = _SHELL
    nZ = int(min(za_tab.size, shell.size))
    thrR = 60.0
    thrE = np.finfo(np.float64).eps
    if _HAVE_NUMBA and (not opts.base) and (not opts.srb):
        if not opts.damp:
            return float(_gcp_energy_nb(Zm.astype(np.int32), xyz.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za_tab.astype(np.float64), zb_tab.astype(np.float64), float(thrR), float(thrE)))
        else:
            # Precompute xva/xvb (handle special mTZVPP virtual counts)
            N = int(Z.shape[0])
            xva = np.empty(N, dtype=np.float64)
            special_xva = _normalize_method(method) in ('def2mtzvpp', 'mtzvpp', 'r2scan3c')
            for i in range(N):
                zi = int(Zm[i])
                if special_xva:
                    val = 1.0
                    if zi == 6:
                        val = 3.0
                    elif zi in (7, 8):
                        val = 0.5
                    xva[i] = val
                else:
                    xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
            xvb = xva.copy()
            r0ab = _load_r0ab_matrix()
            return float(_gcp_energy_nb_damped(Zm.astype(np.int32), xyz.astype(np.float64), emiss.astype(np.float64), xva.astype(np.float64), xvb.astype(np.float64), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za_tab.astype(np.float64), zb_tab.astype(np.float64), float(thrR), float(thrE), r0ab.astype(np.float64), float(opts.dmp_scal), float(opts.dmp_exp)))

    # xva/xvb per atom
    xva = np.empty(N, dtype=np.float64)
    special_xva = _normalize_method(method) in ('def2mtzvpp', 'mtzvpp', 'r2scan3c')
    for i in range(N):
        zi = int(Zm[i])
        if zi < 1:
            continue
        if special_xva:
            val = 1.0
            if zi == 6:
                val = 3.0
            elif zi in (7, 8):
                val = 0.5
            xva[i] = val
        else:
            nvirt = (float(nbas[zi - 1]) - 0.5 * float(zi)) if (zi - 1) < nbas.size else 0.0
            xva[i] = nvirt
    xvb = xva.copy()
    # Slater exponents
    za_tab, zb_tab = _setzet(p2, 1.0)
    # Shells per element
    shell = _SHELL
    thrR = 60.0  # Bohr cutoff
    thrE = np.finfo(np.float64).eps
    r0ab = None
    if opts.damp or opts.base or opts.srb:
        r0ab = _load_r0ab_matrix()

    ecp = 0.0
    for i in range(N):
        zi = int(Zm[i])
        va = xva[i]
        if (zi - 1) >= emiss.size or emiss[zi - 1] < 1e-7:
            continue
        ea = 0.0
        for j in range(N):
            if i == j:
                continue
            zj = int(Zm[j])
            vb = xvb[j]
            if vb < 0.5:
                continue
            rij = xyz[i] - xyz[j]
            r = float(np.linalg.norm(rij))
            if r > thrR:
                continue
            # Compute exponential and continue with Fortran-like thresholding
            expt = math.exp(-p3 * (r ** p4))
            if (zi - 1) >= za_tab.size or (zj - 1) >= za_tab.size:
                continue
            sab = _ssovl_one(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
            if abs(sab) <= 0.0 or math.sqrt(abs(sab)) < thrE:
                continue
            ene_old = expt / math.sqrt(vb * sab)
            if abs(ene_old) < thrE:
                continue
            if opts.damp:
                r0 = float(r0ab[zi - 1, zj - 1])
                rscal = r / r0
                dampval = (1.0 - 1.0 / (1.0 + opts.dmp_scal * (rscal ** opts.dmp_exp)))
                ea += emiss[zi - 1] * (ene_old * dampval)
            else:
                ea += emiss[zi - 1] * ene_old
        ecp += ea
    gcp = ecp * p1
    if opts.base:
        # HF-3c base short-range correction
        if _HAVE_NUMBA:
            gcp += float(_base_short_range_nb(Z.astype(np.int32), xyz.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03))
        else:
            gcp += _base_short_range_energy(Z, xyz, r0ab, 0.7, 0.03)
    if opts.srb:
        if _HAVE_NUMBA:
            return float(_srb_energy_nb(Z.astype(np.int32), xyz.astype(np.float64), r0ab.astype(np.float64), 10.0, 0.08))
        else:
            return float(_srb_energy_only(Z, xyz, r0ab, 10.0, 0.08))
    return float(gcp)


def _torch_sqrt_sab_table(p2: float, thrR: float, ngrid: int, device, dtype):
    # Build on CPU (NumPy), then move to torch on device
    sqrt_tab_np, dr, n2 = _build_sqrt_sab_table(float(p2), float(thrR), int(ngrid))
    import torch  # type: ignore
    return torch.as_tensor(sqrt_tab_np, device=device, dtype=dtype), float(dr), int(n2)

if _HAVE_TORCH:
    @torch.no_grad()  # type: ignore
    def gcp_energy_torch(Z: 'torch.Tensor', xyz: 'torch.Tensor', method: str = 'b3lyp/def2svp', units: Literal['Angstrom','Bohr'] = 'Angstrom') -> 'torch.Tensor':  # type: ignore
        """Torch front-end.

        - If tensors are CUDA, tries a true GPU implementation for molecular gCP
          (non-PBC) using sqrt(sab) lookup tables; supports damped and non-damped.
        - Otherwise falls back to NumPy reference and returns a torch scalar.
        """
        import torch  # type: ignore
        if Z.is_cuda and xyz.is_cuda:
            # GPU molecular path (same as before)
            dev = xyz.device
            dtype = torch.float64 if xyz.dtype != torch.float64 else xyz.dtype
            params = _params_for_method(method)
            opts = _method_options(method)
            if opts.damp:
                Z_np = Z.detach().cpu().to(torch.int32).numpy()
                xyz_np = xyz.detach().cpu().double().numpy()
                e = gcp_energy_numpy(Z_np, xyz_np, method=method, units=units)
                return torch.as_tensor(e, dtype=xyz.dtype, device=xyz.device)
            emiss_np = params.emiss
            nbas_np = params.nbas
            p1, p2, p3, p4 = params.p
            thrR = 60.0
            thrE = np.finfo(np.float64).eps
            xyzb = xyz.to(dtype) * float(ANG2BOHR) if units.lower().startswith('ang') else xyz.to(dtype)
            N = int(Z.shape[0])
            Zi = (Z.to(torch.int64) - 1)
            nbas_t = torch.as_tensor(nbas_np, device=dev, dtype=dtype)
            zvals = (Z.to(torch.int64)).to(dtype)
            xva = nbas_t.index_select(0, Zi) - 0.5 * zvals
            if _normalize_method(method) in ('def2mtzvpp', 'mtzvpp', 'r2scan3c'):
                xva = torch.where(Z == 6, torch.full_like(xva, 3.0), xva)
                xva = torch.where((Z == 7) | (Z == 8), torch.full_like(xva, 0.5), xva)
            xvb = xva.clone()
            emiss_t = torch.as_tensor(emiss_np, device=dev, dtype=dtype)
            emiss_i = emiss_t.index_select(0, Zi)
            sqrt_tab, dr, ngrid = _torch_sqrt_sab_table(float(p2), float(thrR), 2048, dev, dtype)
            x = xyzb[:, 0].view(N, 1) - xyzb[:, 0].view(1, N)
            y = xyzb[:, 1].view(N, 1) - xyzb[:, 1].view(1, N)
            zc = xyzb[:, 2].view(N, 1) - xyzb[:, 2].view(1, N)
            r2 = x*x + y*y + zc*zc
            iu = torch.triu_indices(N, N, offset=1, device=dev)
            r = torch.sqrt(torch.clamp_min(r2[iu[0], iu[1]], 0))
            maskR = (r <= thrR)
            r = r[maskR]
            I = iu[0][maskR]
            J = iu[1][maskR]
            if r.numel() == 0:
                return torch.zeros((), device=dev, dtype=dtype)
            xgrid = r / float(dr)
            i0 = torch.clamp(xgrid.floor().to(torch.int64), 0, ngrid - 2)
            t = (xgrid - i0.to(dtype))
            Zi_idx = Zi[I]
            Zj_idx = Zi[J]
            v0 = sqrt_tab[Zi_idx, Zj_idx, i0]
            v1 = sqrt_tab[Zi_idx, Zj_idx, i0 + 1]
            sroot = (1.0 - t) * v0 + t * v1
            expt = torch.exp(-float(p3) * (r**float(p4)))
            vb_j = xvb[J]
            vb_i = xvb[I]
            sroot = torch.where(sroot > 0, sroot, torch.zeros_like(sroot))
            mask_common = (sroot > thrE) & (expt > 1e-17)
            if mask_common.any():
                I2 = I[mask_common]; J2 = J[mask_common]
                Zi2 = Zi_idx[mask_common]; Zj2 = Zj_idx[mask_common]
                sroot2 = sroot[mask_common]; expt2 = expt[mask_common]
                vbj2 = vb_j[mask_common]; vbi2 = vb_i[mask_common]
                if opts.damp:
                    r0ab_np = _load_r0ab_matrix()
                    r0ab_t = torch.as_tensor(r0ab_np, device=dev, dtype=dtype)
                    r02 = r0ab_t[Zi2, Zj2]
                    rscal2 = r / r02; rscal2 = rscal2[mask_common]
                    damp2 = (1.0 - 1.0 / (1.0 + float(opts.dmp_scal) * (rscal2 ** float(opts.dmp_exp))))
                else:
                    damp2 = 1.0
                mask_j = (vbj2 > 0.5)
                term1 = torch.zeros_like(expt2)
                if mask_j.any():
                    term1 = expt2[mask_j] / torch.sqrt(vbj2[mask_j]) / sroot2[mask_j]
                    term1 = term1 * damp2 if isinstance(damp2, torch.Tensor) else term1 * damp2
                    term1 = term1 * emiss_i[I2[mask_j]]
                mask_i = (vbi2 > 0.5)
                term2 = torch.zeros_like(expt2)
                if mask_i.any():
                    emiss_j = emiss_t.index_select(0, Zj2[mask_i])
                    term2 = expt2[mask_i] / torch.sqrt(vbi2[mask_i]) / sroot2[mask_i]
                    term2 = term2 * damp2 if isinstance(damp2, torch.Tensor) else term2 * damp2
                    term2 = term2 * emiss_j
                e_sum = term1.sum() + term2.sum()
            else:
                e_sum = torch.zeros((), device=dev, dtype=dtype)
            return (e_sum * float(p1)).to(xyz.dtype)
        # CPU fallback
        import torch
        Z_np = Z.detach().cpu().to(torch.int32).numpy()
        xyz_np = xyz.detach().cpu().double().numpy()
        e = gcp_energy_numpy(Z_np, xyz_np, method=method, units=units)
        return torch.as_tensor(e, dtype=xyz.dtype, device=xyz.device)
else:
    def gcp_energy_torch(Z, xyz, method: str = 'b3lyp/def2svp', units: str = 'Angstrom'):
        raise RuntimeError('PyTorch not available')


__all__ = ['gcp_energy_numpy', 'gcp_energy_pbc_numpy', 'gcp_energy_torch']

def gcp_energy_pbc(Z, xyz, lat, method: str = 'b3lyp/def2svp', units: Literal['Angstrom','Bohr'] = 'Angstrom'):
    """Unified PBC API: dispatches to CPU (NumPy/Numba) or GPU (Torch CUDA).

    - If inputs are Torch tensors on CUDA and method is SRB or base-only, uses GPU path.
    - Otherwise uses CPU path. Torch tensors on CPU are converted to NumPy and back.
    Returns a scalar of the same type/device as the positional inputs.
    """
    # Torch path
    if _HAVE_TORCH:
        import torch
        if torch.is_tensor(xyz) or torch.is_tensor(Z) or torch.is_tensor(lat):
            # Normalize to tensors
            if not torch.is_tensor(Z):
                Z = torch.as_tensor(Z)
            if not torch.is_tensor(xyz):
                xyz = torch.as_tensor(xyz)
            if not torch.is_tensor(lat):
                lat = torch.as_tensor(lat)
            dev = xyz.device
            if dev.type == 'cuda':
                opts = _method_options(method)
                if opts.srb or opts.base:
                    return gcp_energy_pbc_torch(Z.to(torch.int64), xyz.to(torch.float64), lat.to(torch.float64), method=method, units=units)
            # CPU fallback
            Z_np = Z.detach().cpu().to(torch.int32).numpy()
            xyz_np = xyz.detach().cpu().double().numpy()
            lat_np = lat.detach().cpu().double().numpy()
            e = gcp_energy_pbc_numpy(Z_np, xyz_np, lat_np, method=method, units=units)
            return torch.as_tensor(e, dtype=xyz.dtype, device=dev)
    # NumPy path
    return gcp_energy_pbc_numpy(Z, xyz, lat, method=method, units=units)


def gcp_pbc_trace(Z: np.ndarray,
                  xyz: np.ndarray,
                  lat: np.ndarray,
                  method: str = 'hf3c',
                  units: Literal['Angstrom','Bohr'] = 'Angstrom',
                  i_sel: Tuple[int, int] | None = None,
                  j_sel: Tuple[int, int] | None = None,
                  tsel: Tuple[int, int, int] | None = None,
                  thrR: float = 60.0) -> list:
    """Trace per-(i,j,a,b,c) gCP PBC contributions for small cells.

    Returns a list of dict entries with keys:
      i,j,a,b,c, zi,zj, r, va, vb, expt, sab, ene_old, contrib

    Notes:
      - This uses the Fortran-order PBC summation and skips only (i==j,a==b==c==0).
      - Intended for tiny systems due to O(N^2 * images) output size.
    """
    params = _params_for_method(method)
    p1, p2, p3, p4 = params.p
    emiss = params.emiss
    nbas = params.nbas.astype(np.int32)
    if units.lower().startswith('ang'):
        xyzb = xyz * ANG2BOHR
        latb = lat * ANG2BOHR
    else:
        xyzb = xyz.copy()
        latb = lat.copy()
    shell = _SHELL
    za_tab, zb_tab = _setzet(p2, 1.0)
    N = int(Z.shape[0])
    # xva/xvb as in Fortran
    xva = np.empty(N, dtype=np.float64)
    for i in range(N):
        zi = int(Z[i])
        xva[i] = float(nbas[zi - 1]) - 0.5 * float(zi)
    xvb = xva.copy()
    # image extents
    if tsel is None:
        t1, t2, t3 = _tau_max_from_lat(latb.astype(np.float64), float(thrR))
    else:
        t1, t2, t3 = map(int, tsel)
    i0, i1 = (0, N) if i_sel is None else (max(0, int(i_sel[0])), min(N, int(i_sel[1])))
    j0, j1 = (0, N) if j_sel is None else (max(0, int(j_sel[0])), min(N, int(j_sel[1])))
    out = []
    EXPT_CUTOFF = 1e-17
    eps = np.finfo(np.float64).eps
    for i in range(i0, i1):
        zi = int(Z[i]); va = xva[i]
        if emiss[zi - 1] < 1e-7:
            continue
        for j in range(j0, j1):
            zj = int(Z[j]); vb = xvb[j]
            if vb < 0.5:
                continue
            for a in range(-t1, t1 + 1):
                for b in range(-t2, t2 + 1):
                    for c in range(-t3, t3 + 1):
                        if (a == 0 and b == 0 and c == 0 and i == j):
                            continue
                        dx = xyzb[i,0] - (xyzb[j,0] + a*latb[0,0] + b*latb[1,0] + c*latb[2,0])
                        dy = xyzb[i,1] - (xyzb[j,1] + a*latb[0,1] + b*latb[1,1] + c*latb[2,1])
                        dz = xyzb[i,2] - (xyzb[j,2] + a*latb[0,2] + b*latb[1,2] + c*latb[2,2])
                        r = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                        if r > thrR:
                            continue
                        expt = math.exp(-p3 * (r ** p4))
                        if expt < EXPT_CUTOFF:
                            continue
                        sab = _ssovl_one(r, int(shell[zi - 1]), int(shell[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
                        sroot = math.sqrt(abs(sab)) if sab > 0.0 else 0.0
                        if sroot < eps:
                            continue
                        ene_old = expt / math.sqrt(vb) / sroot
                        if abs(ene_old) < eps:
                            continue
                        contrib = emiss[zi - 1] * ene_old
                        out.append({'i': i, 'j': j, 'a': a, 'b': b, 'c': c,
                                    'zi': zi, 'zj': zj, 'r': r, 'va': va, 'vb': vb,
                                    'expt': expt, 'sab': sab, 'ene_old': ene_old, 'contrib': contrib})
    return out


# Optional: CUDA-accelerated PBC SRB/base energy
if _HAVE_TORCH:
    import torch

    def _torch_r0fac_ff(Z, method_opts, r0ab_np, device, dtype, kind='srb'):
        r0ab = torch.as_tensor(r0ab_np, device=device, dtype=dtype)
        nZ = int(r0ab.shape[0])
        z = torch.arange(1, nZ + 1, device=device, dtype=dtype)
        zz = torch.sqrt(z[:, None] * z[None, :])
        zz15 = (z[:, None] * z[None, :]) ** 1.5
        if kind == 'srb':
            ff = -(zz * float(method_opts.srb_qscal))
            r0fac = (float(method_opts.srb_rscal)) / r0ab
        else:
            ff = -(zz15 * float(0.03))
            r0fac = float(0.7) * (r0ab ** 0.75)
        return r0fac, ff

    @torch.no_grad()
    def gcp_energy_pbc_torch(Z: 'torch.Tensor', xyz: 'torch.Tensor', lat: 'torch.Tensor', method: str = 'b97-3c', units: str = 'Angstrom', image_chunk: int = 64) -> 'torch.Tensor':
        """PBC gCP energy on GPU for SRB (b97-3c) and base (hf3c).

        Uses half-lattice images and sums non-zero images over all (i,j) pairs,
        and the zero image over the upper triangle to avoid double counting.
        """
        if not (Z.is_cuda and xyz.is_cuda and lat.is_cuda):
            raise RuntimeError('Move Z, xyz, lat to CUDA device')
        device = xyz.device
        dtype = xyz.dtype
        # Convert units to Bohr
        if units.lower().startswith('ang'):
            xyzb = xyz * float(ANG2BOHR)
            latb = lat * float(ANG2BOHR)
        else:
            xyzb = xyz
            latb = lat
        opts = _method_options(method)
        if not (opts.srb or opts.base):
            raise NotImplementedError('GPU PBC supports b97-3c (SRB) and hf3c (base) only')
        # Build half-lattice images on CPU then copy
        lat_np = latb.detach().cpu().numpy()
        t1, t2, t3 = _tau_max_from_lat(lat_np, 30.0)
        tvecs_np = _build_half_tvecs(lat_np, int(t1), int(t2), int(t3))
        tvecs_np = _filter_tvecs_by_bound(tvecs_np, xyzb.detach().cpu().numpy(), 30.0)
        if tvecs_np.shape[0] == 0:
            # Zero image only; reduce to upper-tri pairs via CPU path for now
            e = gcp_energy_numpy(Z.detach().cpu().to(torch.int32).numpy(), xyz.detach().cpu().numpy(), method=method, units=units)
            return torch.as_tensor(e, dtype=dtype, device=device)
        tv = torch.as_tensor(tvecs_np, device=device, dtype=dtype)  # (M,3)

        # Pair factors on GPU
        r0ab_np = _load_r0ab_matrix()
        kind = 'srb' if opts.srb else 'base'
        r0fac, ff = _torch_r0fac_ff(Z, opts, r0ab_np, device, dtype, kind=kind)

        N = int(xyzb.shape[0])
        thrR = float(30.0)
        thrR2 = thrR * thrR

        Zi = (Z.to(torch.int64) - 1)
        # Precompute per-row selection tables of size (N,36)
        r0fac_i = r0fac.index_select(0, Zi)
        ff_i = ff.index_select(0, Zi)
        Zj = Zi
        Jidx = Zj.view(1, N).expand(N, N)
        # Final (N,N) pair tables
        r0fac_ij = torch.gather(r0fac_i, 1, Jidx)
        ff_ij = torch.gather(ff_i, 1, Jidx)

        x = xyzb[:, 0]; y = xyzb[:, 1]; zc = xyzb[:, 2]
        e_sum = torch.zeros((), device=device, dtype=dtype)

        # Non-zero images in chunks
        M = int(tv.shape[0])
        for start in range(0, M, int(image_chunk)):
            stop = min(start + int(image_chunk), M)
            T = tv[start:stop]
            K = T.shape[0]
            tx = T[:, 0].view(K, 1, 1)
            ty = T[:, 1].view(K, 1, 1)
            tz = T[:, 2].view(K, 1, 1)
            dx = x.view(1, N, 1) - (x.view(1, 1, N) + tx)
            dy = y.view(1, N, 1) - (y.view(1, 1, N) + ty)
            dz = zc.view(1, N, 1) - (zc.view(1, 1, N) + tz)
            r2 = dx*dx + dy*dy + dz*dz
            maskR = (r2 <= thrR2)
            r = torch.sqrt(torch.clamp_min(r2, 0))
            a = r0fac_ij.view(1, N, N) * r
            mask = maskR & (a <= 40.0)
            contrib = ff_ij.view(1, N, N) * torch.exp(-a)
            e_sum = e_sum + torch.where(mask, contrib, torch.zeros((), device=device, dtype=dtype)).sum()

        # Zero image (upper triangle only)
        dx0 = x.view(N, 1) - x.view(1, N)
        dy0 = y.view(N, 1) - y.view(1, N)
        dz0 = zc.view(N, 1) - zc.view(1, N)
        r20 = dx0*dx0 + dy0*dy0 + dz0*dz0
        iu = torch.triu_indices(N, N, offset=1, device=device)
        r20 = r20[iu[0], iu[1]]
        maskR0 = (r20 <= thrR2)
        r0 = torch.sqrt(torch.clamp_min(r20, 0))
        r0f0 = r0fac_i[iu[0], Zj[iu[1]]]
        ff0 = ff_i[iu[0], Zj[iu[1]]]
        a0 = r0f0 * r0
        mask0 = maskR0 & (a0 <= 40.0)
        e_sum = e_sum + torch.where(mask0, ff0 * torch.exp(-a0), torch.zeros((), device=device, dtype=dtype)).sum()

        return e_sum
else:
    def gcp_energy_pbc_torch(Z, xyz, lat, method: str = 'b97-3c', units: str = 'Angstrom', image_chunk: int = 64):
        raise RuntimeError('PyTorch not available')


def _ssovl_debug(r: float, shell_i: int, shell_j: int, za: float, zb: float) -> dict:
    R05 = 0.5 * r
    ax = (za + zb) * R05
    bx = (zb - za) * R05
    same = (za == zb) or (abs(za - zb) < 0.1)
    ii = shell_i * shell_j
    d = {'same': same, 'ii': ii, 'ax': ax, 'bx': bx}

    if same:
        if ii == 1:
            norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
            d.update({'norm': norm, 'A2': _A2(ax), 'Bint0': _Bint_series(bx, 0), 'Bint2': _Bint_series(bx, 2), 'A0': _A0(ax)})
            d['sab'] = norm * (d['A2'] * d['Bint0'] - d['Bint2'] * d['A0'])
            return d
        if ii == 2:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A3': _A3(_ax), 'Bint0': _Bint_series(_bx, 0), 'Bint3': _Bint_series(_bx, 3), 'A0': _A0(_ax), 'A2': _A2(_ax), 'Bint1': _Bint_series(_bx, 1), 'Bint2': _Bint_series(_bx, 2), 'A1': _A1(_ax)})
            d['sab'] = math.sqrt(1.0 / 3.0) * norm * (d['A3'] * d['Bint0'] - d['Bint3'] * d['A0'] + d['A2'] * d['Bint1'] - d['Bint2'] * d['A1'])
            return d
        if ii == 4:
            norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
            d.update({'norm': norm, 'A4': _A4(ax), 'Bint0': _Bint_series(bx, 0), 'Bint4': _Bint_series(bx, 4), 'A0': _A0(ax), 'A2': _A2(ax), 'Bint2': _Bint_series(bx, 2)})
            d['sab'] = (norm / 3.0) * (d['A4'] * d['Bint0'] + d['Bint4'] * d['A0'] - 2.0 * d['A2'] * d['Bint2'])
            return d
        if ii == 3:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A4': _A4(_ax), 'Bint0': _Bint_series(_bx, 0), 'Bint4': _Bint_series(_bx, 4), 'A0': _A0(_ax), 'A3': _A3(_ax), 'Bint1': _Bint_series(_bx, 1), 'Bint3': _Bint_series(_bx, 3), 'A1': _A1(_ax)})
            d['sab'] = (norm / math.sqrt(3.0)) * (d['A4'] * d['Bint0'] - d['Bint4'] * d['A0'] + 2.0 * (d['A3'] * d['Bint1'] - d['Bint3'] * d['A1']))
            return d
        if ii == 6:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A5': _A5(_ax), 'Bint0': _Bint_series(_bx, 0), 'A4': _A4(_ax), 'Bint1': _Bint_series(_bx, 1), 'A3': _A3(_ax), 'Bint2': _Bint_series(_bx, 2), 'A2': _A2(_ax), 'Bint3': _Bint_series(_bx, 3), 'A1': _A1(_ax), 'Bint4': _Bint_series(_bx, 4), 'A0': _A0(_ax), 'Bint5': _Bint_series(_bx, 5)})
            d['sab'] = (norm / 3.0) * (d['A5'] * d['Bint0'] + d['A4'] * d['Bint1'] - 2.0 * (d['A3'] * d['Bint2'] + d['A2'] * d['Bint3']) + d['A1'] * d['Bint4'] + d['A0'] * d['Bint5'])
            return d
        if ii == 9:
            norm = math.sqrt((za * zb * r * r) ** 7) / 480.0
            d.update({'norm': norm, 'A6': _A6(ax), 'Bint0': _Bint_series(bx, 0), 'A4': _A4(ax), 'Bint2': _Bint_series(bx, 2), 'A2': _A2(ax), 'Bint4': _Bint_series(bx, 4), 'A0': _A0(ax), 'Bint6': _Bint_series(bx, 6)})
            d['sab'] = (norm / 3.0) * (d['A6'] * d['Bint0'] - 3.0 * (d['A4'] * d['Bint2'] - d['A2'] * d['Bint4']) - d['A0'] * d['Bint6'])
            return d
    else:
        if ii == 1:
            norm = 0.25 * math.sqrt((za * zb * r * r) ** 3)
            d.update({'norm': norm, 'A2': _A2(ax), 'B0': _B0(bx), 'B2': _B2(bx), 'A0': _A0(ax)})
            d['sab'] = norm * (d['A2'] * d['B0'] - d['B2'] * d['A0'])
            return d
        if ii == 2:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 5)) * (r ** 4) * 0.125
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A3': _A3(_ax), 'B0': _B0(_bx), 'B3': _B3(_bx), 'A0': _A0(_ax), 'A2': _A2(_ax), 'B1': _B1(_bx), 'B2': _B2(_bx), 'A1': _A1(_ax)})
            d['sab'] = math.sqrt(1.0 / 3.0) * norm * (d['A3'] * d['B0'] - d['B3'] * d['A0'] + d['A2'] * d['B1'] - d['B2'] * d['A1'])
            return d
        if ii == 4:
            norm = math.sqrt((za * zb) ** 5) * (r ** 5) * 0.0625
            d.update({'norm': norm, 'A4': _A4(ax), 'B0': _B0(bx), 'B4': _B4(bx), 'A0': _A0(ax), 'A2': _A2(ax), 'B2': _B2(bx)})
            d['sab'] = (norm / 3.0) * (d['A4'] * d['B0'] + d['B4'] * d['A0'] - 2.0 * d['A2'] * d['B2'])
            return d
        if ii == 3:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 3) * (_zb ** 7) / 7.5) * (r ** 5) * 0.0625
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A4': _A4(_ax), 'B0': _B0(_bx), 'B4': _B4(_bx), 'A0': _A0(_ax), 'A3': _A3(_ax), 'B1': _B1(_bx), 'B3': _B3(_bx), 'A1': _A1(_ax)})
            d['sab'] = (norm / math.sqrt(3.0)) * (d['A4'] * d['B0'] - d['B4'] * d['A0'] + 2.0 * (d['A3'] * d['B1'] - d['B3'] * d['A1']))
            return d
        if ii == 6:
            if shell_i < shell_j:
                _za, _zb = za, zb
            else:
                _za, _zb = zb, za
            _ax = (_za + _zb) * R05
            _bx = (_zb - _za) * R05
            norm = math.sqrt((_za ** 5) * (_zb ** 7) / 7.5) * (r ** 6) * 0.03125
            d.update({'swap': shell_i >= shell_j, 'norm': norm, 'A5': _A5(_ax), 'B0': _B0(_bx), 'A4': _A4(_ax), 'B1': _B1(_bx), 'A3': _A3(_ax), 'B2': _B2(_bx), 'A2': _A2(_ax), 'B3': _B3(_bx), 'A1': _A1(_ax), 'B4': _B4(_bx), 'A0': _A0(_ax), 'B5': _B5(_bx)})
            d['sab'] = (norm / 3.0) * (d['A5'] * d['B0'] + d['A4'] * d['B1'] - 2.0 * (d['A3'] * d['B2'] + d['A2'] * d['B3']) + d['A1'] * d['B4'] + d['A0'] * d['B5'])
            return d
        if ii == 9:
            norm = math.sqrt((za * zb * r * r) ** 7) / 1440.0
            d.update({'norm': norm, 'A6': _A6(ax), 'B0': _B0(bx), 'A4': _A4(ax), 'B2': _B2(bx), 'A2': _A2(ax), 'B4': _B4(bx), 'A0': _A0(ax), 'B6': _B6(bx)})
            d['sab'] = norm * (d['A6'] * d['B0'] - 3.0 * (d['A4'] * d['B2'] - d['A2'] * d['B4']) - d['A0'] * d['B6'])
            return d
    return d


def gcp_debug_two_atom(Z: np.ndarray, xyz: np.ndarray, method: str = 'b3lyp/def2svp', units: str = 'Angstrom') -> dict:
    assert Z.shape[0] == 2 and xyz.shape[0] == 2
    params = _params_for_method(method)
    opts = _method_options(method)
    emiss = params.emiss
    nbas = params.nbas
    p1, p2, p3, p4 = params.p
    if units.lower().startswith('ang'):
        xyz_bohr = xyz * ANG2BOHR
    else:
        xyz_bohr = xyz.copy()
    # va, vb
    xva = float(nbas[int(Z[0]) - 1]) - 0.5 * float(Z[0])
    xvb = float(nbas[int(Z[1]) - 1]) - 0.5 * float(Z[1])
    za_tab, zb_tab = _setzet(p2, 1.0)
    sh = _SHELL
    r = float(np.linalg.norm(xyz_bohr[0] - xyz_bohr[1]))
    # i->j term
    zi, zj = int(Z[0]), int(Z[1])
    di = _ssovl_debug(r, int(sh[zi - 1]), int(sh[zj - 1]), float(za_tab[zi - 1]), float(zb_tab[zj - 1]))
    dj = _ssovl_debug(r, int(sh[zj - 1]), int(sh[zi - 1]), float(za_tab[zj - 1]), float(zb_tab[zi - 1]))
    # ene_old and ea per atom
    ene_i = math.exp(-p3 * (r ** p4)) / math.sqrt(xvb * max(1e-300, di.get('sab', 0.0)))
    ene_j = math.exp(-p3 * (r ** p4)) / math.sqrt(xva * max(1e-300, dj.get('sab', 0.0)))
    ea_i = emiss[zi - 1] * ene_i
    ea_j = emiss[zj - 1] * ene_j
    Egcp = (ea_i + ea_j) * p1
    return {
        'Z': (int(Z[0]), int(Z[1])),
        'r_bohr': r,
        'p': (p1, p2, p3, p4),
        'shells': (int(sh[zi - 1]), int(sh[zj - 1])),
        'za': (float(za_tab[zi - 1]), float(za_tab[zj - 1])),
        'vbasis': (xva, xvb),
        'i_to_j': di,
        'j_to_i': dj,
        'ene_old': (ene_i, ene_j),
        'ea': (ea_i, ea_j),
        'Egcp': Egcp,
        'BSSE_kcal': (ea_i * p1 * EH2KCAL, ea_j * p1 * EH2KCAL),
    }
