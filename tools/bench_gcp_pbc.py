import time
from statistics import median

import numpy as np

import py_gcp.gcp as gt


def time_func(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), median(ts)


def rand_cell(N=64, box=15.0, seed=0, zmax=8):
    rng = np.random.default_rng(seed)
    Z = rng.integers(1, zmax + 1, size=N, dtype=np.int32)
    xyz = rng.random((N, 3)) * box
    lat = np.eye(3) * box
    return Z, xyz, lat


def bench_case(N=64, method='b97-3c', repeats=5):
    Z, xyz, lat = rand_cell(N=N, box=15.0, seed=42)
    # Warmups
    e_vec = gt.gcp_energy_pbc_numpy(Z, xyz, lat, method=method, units='Angstrom')

    # Build shared precomputes
    latb = lat * gt.ANG2BOHR
    xyzb = xyz * gt.ANG2BOHR
    t1, t2, t3 = gt._tau_max_from_lat(latb, 30.0)

    # SRB vs Base paths
    if method.lower() in ('b97-3c', 'b973c'):
        opts = gt._method_options(method)
        r0ab = gt._load_r0ab_matrix()
        # Vectorized pieces (already used in gcp_energy_pbc_numpy)
        tvecs = gt._build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
        z = np.arange(1, 37, dtype=np.float64)
        zz = np.sqrt(z[:, None] * z[None, :])
        r0fac = (float(opts.srb_rscal)) / r0ab.astype(np.float64)
        ff = -(zz * float(opts.srb_qscal))

        def run_vec():
            gt._srb_energy_nb_pbc_vec(Z.astype(np.int32), xyzb.astype(np.float64), tvecs, r0fac, ff, 30.0)

        def run_loop():
            gt._srb_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), float(opts.srb_rscal), float(opts.srb_qscal), 30.0, int(t1), int(t2), int(t3))

    else:  # hf3c base
        r0ab = gt._load_r0ab_matrix()
        tvecs = gt._build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
        z = np.arange(1, 37, dtype=np.float64)
        zz15 = (z[:, None] * z[None, :]) ** 1.5
        r0fac = (0.7) * (r0ab.astype(np.float64) ** 0.75)
        ff = -(zz15 * 0.03)

        def run_vec():
            gt._base_short_range_nb_pbc_vec(Z.astype(np.int32), xyzb.astype(np.float64), tvecs, r0fac, ff, 30.0)

        def run_loop():
            gt._base_short_range_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03, 30.0, int(t1), int(t2), int(t3))

    # Time
    tmin_v, tmed_v = time_func(run_vec, repeats=repeats)
    tmin_l, tmed_l = time_func(run_loop, repeats=repeats)

    # Compare energies from both paths for sanity
    if method.lower() in ('b97-3c', 'b973c'):
        e_loop = gt._srb_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), float(opts.srb_rscal), float(opts.srb_qscal), 30.0, int(t1), int(t2), int(t3))
        e_vec2 = gt._srb_energy_nb_pbc_vec(Z.astype(np.int32), xyzb.astype(np.float64), tvecs, r0fac, ff, 30.0)
    else:
        e_loop = gt._base_short_range_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03, 30.0, int(t1), int(t2), int(t3))
        e_vec2 = gt._base_short_range_nb_pbc_vec(Z.astype(np.int32), xyzb.astype(np.float64), tvecs, r0fac, ff, 30.0)

    return {
        'N': N,
        'method': method,
        't_vec_min_ms': tmin_v * 1e3,
        't_vec_med_ms': tmed_v * 1e3,
        't_loop_min_ms': tmin_l * 1e3,
        't_loop_med_ms': tmed_l * 1e3,
        'vec_speedup_x': (tmed_l / tmed_v) if tmed_v > 0 else float('inf'),
        'E_vec': float(e_vec2),
        'E_loop': float(e_loop),
        'E_diff': float(abs(e_vec2 - e_loop)),
        'E_gcp_api': float(e_vec),
    }


def main():
    for method in ['b97-3c', 'hf3c']:
        for N in [16, 48, 96, 160]:
            res = bench_case(N=N, method=method, repeats=5)
            print(f"{method:7s} N={res['N']:3d} | vec_med={res['t_vec_med_ms']:.2f} ms | loop_med={res['t_loop_med_ms']:.2f} ms | speedup={res['vec_speedup_x']:.2f}x | |ΔE|={res['E_diff']:.3e}")


if __name__ == '__main__':
    main()

