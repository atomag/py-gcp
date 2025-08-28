import time
from statistics import median

import numpy as np

import py_gcp.gcp as gt


def time_func(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), median(ts)


def rand_cell(N=96, box=15.0, seed=0, zmax=18):
    rng = np.random.default_rng(seed)
    Z = rng.integers(1, zmax + 1, size=N, dtype=np.int32)
    xyz = rng.random((N, 3)) * box
    lat = np.eye(3) * box
    return Z, xyz, lat


def bench_tab_vs_exact(N=96, method='dft/def2tzvp', repeats=5):
    Z, xyz, lat = rand_cell(N=N, box=15.0, seed=123, zmax=18)
    # Build params like the API does
    params = gt._params_for_method(method)
    emiss = params.emiss
    nbas = params.nbas
    p1, p2, p3, p4 = params.p
    thrR = 60.0
    thrE = np.finfo(np.float64).eps
    xyzb = xyz * gt.ANG2BOHR
    latb = lat * gt.ANG2BOHR
    t1, t2, t3 = gt._tau_max_from_lat(latb, thrR)
    za, zb = gt._setzet(p2, 1.0)
    shell = gt._SHELL

    # Tab data
    sqrt_tab, dr, ngrid = gt._build_sqrt_sab_table(float(p2), thrR, 2048)

    def run_exact():
        gt._gcp_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za.astype(np.float64), zb.astype(np.float64), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3))

    def run_tab():
        gt._gcp_energy_nb_pbc_tab(Z.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3), sqrt_tab.astype(np.float64), float(dr), int(ngrid))

    # Times
    tmin_e, tmed_e = time_func(run_exact, warmup=1, repeats=repeats)
    tmin_t, tmed_t = time_func(run_tab, warmup=1, repeats=repeats)

    # Energies
    E_exact = gt._gcp_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), shell.astype(np.int32), za.astype(np.float64), zb.astype(np.float64), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3))
    E_tab = gt._gcp_energy_nb_pbc_tab(Z.astype(np.int32), xyzb.astype(np.float64), emiss.astype(np.float64), nbas.astype(np.int32), float(p1), float(p2), float(p3), float(p4), float(thrR), float(thrE), latb.astype(np.float64), int(t1), int(t2), int(t3), sqrt_tab.astype(np.float64), float(dr), int(ngrid))

    print(f"{method} N={N} | exact={tmed_e*1e3:.2f} ms | tab={tmed_t*1e3:.2f} ms | speedup={tmed_e/tmed_t:.2f}x | |ΔE|={abs(E_tab-E_exact):.3e}")


def main():
    for N in [48, 96, 160]:
        bench_tab_vs_exact(N=N, method='dft/def2tzvp', repeats=5)
        bench_tab_vs_exact(N=N, method='pbeh3c', repeats=5)


if __name__ == '__main__':
    main()

