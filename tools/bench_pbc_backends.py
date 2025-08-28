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


def srb_py(Z, xyzb, latb, r0ab, rscal=10.0, qscal=0.08, thrR=30.0):
    t1, t2, t3 = gt._tau_max_from_lat(latb, thrR)
    tvecs = gt._build_half_tvecs(latb, int(t1), int(t2), int(t3))
    tvecs = gt._filter_tvecs_by_bound(tvecs, xyzb, thrR)
    e = 0.0
    n = int(Z.shape[0])
    for i in range(n):
        zi = int(Z[i])
        for m in range(tvecs.shape[0]):
            t = tvecs[m]
            for j in range(n):
                zj = int(Z[j])
                rij = xyzb[i] - (xyzb[j] + t)
                r = float(np.linalg.norm(rij))
                if r > thrR:
                    continue
                r0 = rscal / float(r0ab[zi-1, zj-1])
                ff = -((float(zi) * float(zj)) ** 0.5)
                a = r0 * r
                if a > 40.0:
                    continue
                e += qscal * ff * np.exp(-a)
        for j in range(i+1, n):
            zj = int(Z[j])
            rij = xyzb[i] - xyzb[j]
            r = float(np.linalg.norm(rij))
            if r > thrR:
                continue
            r0 = rscal / float(r0ab[zi-1, zj-1])
            ff = -((float(zi) * float(zj)) ** 0.5)
            a = r0 * r
            if a > 40.0:
                continue
            e += qscal * ff * np.exp(-a)
    return float(e)


def base_py(Z, xyzb, latb, r0ab, rscal=0.7, qscal=0.03, thrR=30.0):
    t1, t2, t3 = gt._tau_max_from_lat(latb, thrR)
    tvecs = gt._build_half_tvecs(latb, int(t1), int(t2), int(t3))
    tvecs = gt._filter_tvecs_by_bound(tvecs, xyzb, thrR)
    e = 0.0
    n = int(Z.shape[0])
    for i in range(n):
        zi = int(Z[i])
        if zi < 1 or zi > 18:
            continue
        for m in range(tvecs.shape[0]):
            t = tvecs[m]
            for j in range(n):
                zj = int(Z[j])
                if zj < 1 or zj > 18:
                    continue
                rij = xyzb[i] - (xyzb[j] + t)
                r = float(np.linalg.norm(rij))
                if r > thrR:
                    continue
                r0 = rscal * (float(r0ab[zi-1, zj-1]) ** 0.75)
                ff = -((float(zi) * float(zj)) ** 1.5)
                a = r0 * r
                if a > 40.0:
                    continue
                e += qscal * ff * np.exp(-a)
        for j in range(i+1, n):
            zj = int(Z[j])
            if zj < 1 or zj > 18:
                continue
            rij = xyzb[i] - xyzb[j]
            r = float(np.linalg.norm(rij))
            if r > thrR:
                continue
            r0 = rscal * (float(r0ab[zi-1, zj-1]) ** 0.75)
            ff = -((float(zi) * float(zj)) ** 1.5)
            a = r0 * r
            if a > 40.0:
                continue
            e += qscal * ff * np.exp(-a)
    return float(e)


def bench_backends(N=96, method='b97-3c', repeats=5):
    Z, xyz, lat = rand_cell(N=N, box=15.0, seed=42, zmax=18)
    xyzb = xyz * gt.ANG2BOHR
    latb = lat * gt.ANG2BOHR
    r0ab = gt._load_r0ab_matrix()
    t1, t2, t3 = gt._tau_max_from_lat(latb, 30.0)
    tvecs = gt._build_half_tvecs(latb.astype(np.float64), int(t1), int(t2), int(t3))
    tvecs = gt._filter_tvecs_by_bound(tvecs, xyzb.astype(np.float64), 30.0)
    tvx, tvy, tvz = gt._split_tvecs(tvecs)
    x, y, zc = gt._split_coords(xyzb.astype(np.float64))
    # Pair tables
    z = np.arange(1, 37, dtype=np.float64)
    if method.lower() in ('b97-3c', 'b973c'):
        opts = gt._method_options(method)
        r0fac = (float(opts.srb_rscal)) / r0ab.astype(np.float64)
        zz = np.sqrt(z[:, None] * z[None, :])
        ff = -(zz * float(opts.srb_qscal))
        # Define callables
        def run_py():
            srb_py(Z, xyzb, latb, r0ab, rscal=float(opts.srb_rscal), qscal=float(opts.srb_qscal))
        def run_numba_loop():
            gt._srb_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), float(opts.srb_rscal), float(opts.srb_qscal), 30.0, int(t1), int(t2), int(t3))
        def run_numba_vec():
            gt._srb_energy_nb_pbc_vec_soa(Z.astype(np.int32), x, y, zc, tvx, tvy, tvz, r0fac.astype(np.float64), ff.astype(np.float64), 30.0)
        E_py = srb_py(Z, xyzb, latb, r0ab, rscal=float(opts.srb_rscal), qscal=float(opts.srb_qscal))
        E_loop = gt._srb_energy_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), float(opts.srb_rscal), float(opts.srb_qscal), 30.0, int(t1), int(t2), int(t3))
        E_vec = gt._srb_energy_nb_pbc_vec_soa(Z.astype(np.int32), x, y, zc, tvx, tvy, tvz, r0fac.astype(np.float64), ff.astype(np.float64), 30.0)
    else:
        r0fac = (0.7) * (r0ab.astype(np.float64) ** 0.75)
        zz15 = (z[:, None] * z[None, :]) ** 1.5
        ff = -(zz15 * 0.03)
        validJ = np.where((Z >= 1) & (Z <= 18))[0].astype(np.int32)
        def run_py():
            base_py(Z, xyzb, latb, r0ab)
        def run_numba_loop():
            gt._base_short_range_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03, 30.0, int(t1), int(t2), int(t3))
        def run_numba_vec():
            gt._base_short_range_nb_pbc_vec_soa(Z.astype(np.int32), x, y, zc, tvx, tvy, tvz, validJ.astype(np.int32), r0fac.astype(np.float64), ff.astype(np.float64), 30.0)
        E_py = base_py(Z, xyzb, latb, r0ab)
        E_loop = gt._base_short_range_nb_pbc(Z.astype(np.int32), xyzb.astype(np.float64), latb.astype(np.float64), r0ab.astype(np.float64), 0.7, 0.03, 30.0, int(t1), int(t2), int(t3))
        E_vec = gt._base_short_range_nb_pbc_vec_soa(Z.astype(np.int32), x, y, zc, tvx, tvy, tvz, validJ.astype(np.int32), r0fac.astype(np.float64), ff.astype(np.float64), 30.0)

    # Torch GPU backend
    try:
        import torch
        if torch.cuda.is_available():
            Zt = torch.as_tensor(Z, device='cuda', dtype=torch.int64)
            xyzt = torch.as_tensor(xyz, device='cuda', dtype=torch.float64)
            latt = torch.as_tensor(lat, device='cuda', dtype=torch.float64)
            def run_torch():
                gt.gcp_energy_pbc_torch(Zt, xyzt, latt, method=method, units='Angstrom')
            tmin_t, tmed_t = time_func(run_torch, warmup=1, repeats=repeats)
            E_torch = float(gt.gcp_energy_pbc_torch(Zt, xyzt, latt, method=method, units='Angstrom'))
        else:
            tmin_t = tmed_t = float('nan')
            E_torch = float('nan')
    except Exception:
        tmin_t = tmed_t = float('nan')
        E_torch = float('nan')

    # Timings
    tmin_py, tmed_py = time_func(run_py, warmup=0, repeats=repeats)
    tmin_lp, tmed_lp = time_func(run_numba_loop, warmup=1, repeats=repeats)
    tmin_vc, tmed_vc = time_func(run_numba_vec, warmup=1, repeats=repeats)

    out = {
        'N': N,
        'method': method,
        'E_py': float(E_py),
        'E_loop': float(E_loop),
        'E_vec': float(E_vec),
        'E_torch': float(E_torch),
        't_py_ms': tmed_py * 1e3,
        't_loop_ms': tmed_lp * 1e3,
        't_vec_ms': tmed_vc * 1e3,
        't_torch_ms': (tmed_t * 1e3) if np.isfinite(tmed_t) else float('nan'),
        'speedup_loop_vs_py': (tmed_py / tmed_lp) if tmed_lp > 0 else float('inf'),
        'speedup_vec_vs_loop': (tmed_lp / tmed_vc) if tmed_vc > 0 else float('inf'),
        'speedup_torch_vs_loop': (tmed_lp / tmed_t) if (tmed_t > 0 and np.isfinite(tmed_t)) else float('nan'),
        'd_py_loop': abs(E_py - E_loop),
        'd_vec_loop': abs(E_vec - E_loop),
        'd_torch_loop': abs(E_torch - E_loop) if np.isfinite(E_torch) else float('nan'),
    }
    return out


def main():
    for method in ['b97-3c', 'hf3c']:
        for N in [32, 64, 128, 192]:
            res = bench_backends(N=N, method=method, repeats=5)
            print(f"{method:7s} N={res['N']:3d} | py={res['t_py_ms']:.2f} ms | loop={res['t_loop_ms']:.2f} ms | vec={res['t_vec_ms']:.2f} ms | torch={res['t_torch_ms']:.2f} ms | spd(vec/loop)={res['speedup_vec_vs_loop']:.2f}x | spd(torch/loop)={res['speedup_torch_vs_loop']:.2f}x | dE(vec-loop)={res['d_vec_loop']:.2e} | dE(torch-loop)={res['d_torch_loop']:.2e}")


if __name__ == '__main__':
    main()

