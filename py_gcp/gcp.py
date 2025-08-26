from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import importlib.resources as _res
import math
import numpy as np

try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

try:  # optional CPU JIT acceleration
    from numba import njit  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

# Constants
ANG2BOHR = 1.0 / 0.5291772083
EH2KCAL = 627.5094740631

_PARAM_CACHE: Dict[str, np.ndarray] = {}
_SAB_TABLE_CACHE: Dict[str, Tuple[np.ndarray, float, int]] = {}
_PARAM_JSON_CACHE: Dict[str, object] = {}
_PARAMS_RESOURCE = ('py_gcp.data', 'fortran_params.json')


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


def _load_r0ab_matrix(max_elem: int | None = None) -> np.ndarray:
    data = _load_param_json()
    vals = data.get('r0ab_vals', [])
    # Deduce size from triangular list length if not provided
    if max_elem is None:
        L = int(len(vals))
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

# Public API: import the in-package core implementation.
# gcp_core contains the full implementation; this module re-exports its API
# and provides lightweight helpers for loading packaged parameter data.

from .gcp_core import (  # type: ignore
    gcp_energy_numpy,
    gcp_energy_pbc_numpy,
    gcp_energy_pbc,
    gcp_energy_torch,
    _arr,
    _params_for_method,
    _method_options,
    _SHELL,
    _build_sqrt_sab_table,
    _setzet,
)
