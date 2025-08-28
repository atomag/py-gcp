import time
import json
from statistics import median
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.io import write

import py_gcp.gcp as gt


def timeit(fn, reps=3, warm=1):
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
    for line in out.splitlines():
        if 'Egcp:' in line:
            blk = line.split('Egcp:')[1].split('/')[0]
            for tok in blk.strip().split():
                try:
                    return float(tok)
                except Exception:
                    pass
    raise RuntimeError('Egcp not found in gcp64 output')


def bench_molecule(name='H2O', methods=None, reps=3):
    import torch
    if methods is None:
        methods = ['dft/sv', 'dft/sv(p)', 'dft/def2tzvp', 'def2mtzvpp']
    at = molecule(name)
    Z_np = at.get_atomic_numbers().astype(np.int32)
    xyz_np = at.get_positions()
    xyzfile = Path(f'{name}.xyz')
    write(str(xyzfile), at, format='xyz')
    res = []
    for m in methods:
        # numpy
        def run_numpy():
            gt.gcp_energy_numpy(Z_np, xyz_np, method=m, units='Angstrom')
        tmin_n, tmed_n = timeit(run_numpy, reps=reps, warm=1)
        E_numpy = gt.gcp_energy_numpy(Z_np, xyz_np, method=m, units='Angstrom')
        # torch cpu
        Zt = torch.as_tensor(Z_np, dtype=torch.int64)
        xyzt = torch.as_tensor(xyz_np, dtype=torch.float64)
        def run_tcpu():
            gt.gcp_energy_torch(Zt, xyzt, method=m, units='Angstrom')
        tmin_tc, tmed_tc = timeit(run_tcpu, reps=reps, warm=1)
        E_tcpu = float(gt.gcp_energy_torch(Zt, xyzt, method=m, units='Angstrom'))
        # torch gpu (if available and method non-damped)
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        E_tgpu = float('nan'); tmed_tg = float('nan')
        if cuda_ok and gt._method_options(m).damp is False:
            Zg = Zt.to('cuda')
            xyzg = xyzt.to('cuda')
            def run_tgpu():
                gt.gcp_energy_torch(Zg, xyzg, method=m, units='Angstrom')
            tmin_tg, tmed_tg = timeit(run_tgpu, reps=reps, warm=1)
            E_tgpu = float(gt.gcp_energy_torch(Zg, xyzg, method=m, units='Angstrom'))
        # gcp64
        def run_ref():
            gcp64_energy(xyzfile, m)
        tmin_rf, tmed_rf = timeit(run_ref, reps=reps, warm=1)
        E_ref = gcp64_energy(xyzfile, m)
        item = {
            'system': name,
            'method': m,
            'E_numpy': float(E_numpy),
            'E_torch_cpu': float(E_tcpu),
            'E_torch_gpu': float(E_tgpu),
            'E_ref': float(E_ref),
            'd_numpy': float(E_numpy - E_ref),
            'd_torch_cpu': float(E_tcpu - E_ref),
            'd_torch_gpu': float(E_tgpu - E_ref) if np.isfinite(E_tgpu) else float('nan'),
            't_numpy_ms': tmed_n*1e3,
            't_torch_cpu_ms': tmed_tc*1e3,
            't_torch_gpu_ms': tmed_tg*1e3 if np.isfinite(E_tgpu) else float('nan'),
            't_ref_ms': tmed_rf*1e3,
        }
        res.append(item)
        print(f"{name:6s} {m:12s} | dN={item['d_numpy']:.2e} dTC={item['d_torch_cpu']:.2e} dTG={item['d_torch_gpu']:.2e} | N={item['t_numpy_ms']:.2f}ms TC={item['t_torch_cpu_ms']:.2f}ms TG={item['t_torch_gpu_ms'] if np.isfinite(item['t_torch_gpu_ms']) else float('nan'):.2f}ms REF={item['t_ref_ms']:.2f}ms")
    return res


def main():
    out = {'molecular': {}}
    for name in ['H2O', 'CH4', 'C6H6']:
        out['molecular'][name] = bench_molecule(name=name)
    Path('bench_torch_results.json').write_text(json.dumps(out, indent=2))
    print('Wrote bench_torch_results.json')


if __name__ == '__main__':
    main()

