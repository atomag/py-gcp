from __future__ import annotations

from ase.calculators.calculator import Calculator, all_changes
import numpy as np

from .gcp import gcp_energy_numpy, gcp_energy_pbc_numpy

EH2EV = 27.211386245988  # Hartree to eV


class PyGCP(Calculator):
    """ASE calculator wrapper for py-gcp.

    Parameters
    ----------
    method : str
        gCP parameterization (e.g., 'dft/def2tzvp', 'b97-3c', 'hf3c').
    units : {'Angstrom','Bohr'}
        Units of input coordinates passed by ASE (default 'Angstrom').
    pbc : {None, True, False}
        If None (default), auto-detect from Atoms.pbc.any(). Otherwise force.
    """

    implemented_properties = ['energy']

    def __init__(self, method: str = 'dft/def2tzvp', units: str = 'Angstrom', pbc: bool | None = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.method = method
        self.units = units
        self._force_pbc = pbc

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        Z = at.get_atomic_numbers().astype(np.int32)
        xyz = at.get_positions()
        use_pbc = bool(at.pbc.any()) if self._force_pbc is None else bool(self._force_pbc)
        if use_pbc:
            lat = at.cell.array
            e_h = gcp_energy_pbc_numpy(Z, xyz, lat, method=self.method, units=self.units)
        else:
            e_h = gcp_energy_numpy(Z, xyz, method=self.method, units=self.units)
        self.results['energy'] = float(e_h) * EH2EV
