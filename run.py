#!/usr/bin/env python3
# env -S python3 -u

from utils import compute_df, indicator, search_insts, set_verbose
from bounded_pool import BoundedPool
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
import scipy.io as sio
import simulate_igraph
import settings as st
from glob import glob
import numpy as np
import samples
import mle_run
import fire
import re
import os
import gc

# NOTE: DATA STRUCTURES

# Instance
# ├── 1.00
# │   ├── generations.csv
# │   ├── ig_data.mat
# │   ├── res_cnt_size.mat
# │   ├── ...
# │   └── sub
# │       ├── 0
# │       │   ├── generations.csv
# │       │   ├── ig_data.mat -> ../../ig_data.mat
# │       │   ├── res_cnt_size.mat
# │       │   └── ...
# │       ├── ...
# │       └── 9 -> ..
# ├── ...
# └── 1.09
#     └── ...

# Results
# ├── 1.00
# │   └── sub
# │       ├── 0
# │       │   ├── 1.00 (test basis)
# │       │   │   ├── cntz_our_m_0_t.npz
# │       │   │   └── pmat_m_0_t.mtx
# │       │   ├── ...
# │       │   ├── 1.09 (test basis)
# │       │   │   └── ...
# │       │   │
# │       │   ├── cntz_our_m_0.npz
# │       │   ├── pmat_m_0.mtx
# │       │   │
# │       │   ├── fmat_0.pkl
# │       │   ├── ptrace_0.npy
# │       │   ├── rank_0.npy
# │       │   └── theta_0.npy
# │       ├── ...
# │       └── 9 -> ..
# ├── ...
# └── 1.09
#     └── ...

# NOTE: For igraph simulation, we set to Monte Carlo with probability of 1/len(P)

def extract_init_failures(instance):
    genfile = f"{instance}/generations.csv"
    init_failures = []
    if os.path.exists(genfile):
        print("Extract initial failures")
        with open(genfile, 'r') as f:
            last = '-1'
            for ls in f:
                ls = ls.strip()
                if ls:
                    if ls[0] == ',':
                        init_failures.append([])
                    elif last[:2] == '-1':
                        init_failures.append([int(x) - 1 for x in ls.split(',') if x.strip()])
                    last = ls
    return init_failures

def reformat_vcnt(mat, precode=None, offset=1):
    vcnt = {}
    for k, d in mat.items():
        # remove precoded inst no.
        k = k.replace(f'_{precode}', '')
        if 'counts' in k:
            k = k.replace('counts_', '')
            # NOTE: format: gens x brches
            vcnt[k] = d[offset:, :].sum(axis=0, dtype=float)
    return vcnt

# NOTE: manually call gc before MC (with multiprocess) after MLE (with GPU) to avoid following error:
# Abort was called at 91 line in file:
# /usr/src/debug/intel-compute-runtime/compute-runtime-22.43.24595.30/shared/source/os_interface/linux/drm_buffer_object.cpp
def run_mle_mc(input_m):
    res, scaler = mle_run.run(**input_m)
    gc.collect()
    # NOTE: Monte Carlo or N-k contingencies
    no_mc = input_m['no_mc']
    p0 = no_mc <= 0
    if p0:
        no_mc = samples.cnt_cascades(f"{input_m['instance']}/generations.csv")
    init = input_m['init_failures']
    if init is True:
        init = extract_init_failures(input_m['instance'])
    stat = simulate_igraph.run(
        res, init, no_mc=no_mc, p0=p0, k=input_m['k'],
        fp=input_m['fp'], seed=None, max_workers=0, verbose=input_m['verbose']
    )
    key = ('l', 'm')[input_m['pfunc']=='cloglog'] + ('', '_t')[bool(input_m['test'])]
    pmat = {key: res[1]}
    return stat, pmat, scaler

def run_base(input_b):
    inst = input_b['instance']
    print(('Test' if input_b['test'] else 'Run') + f' instance {inst}...')
    stat, pmat, scaler = run_mle_mc(input_b)
    vcnt = reformat_vcnt(stat, os.path.basename(inst))

    matfile = f'{inst}/res_cnt_size.mat'
    if os.path.exists(matfile):
        vcnt.update(reformat_vcnt(sio.loadmat(matfile)))
    else:
        print(f'{matfile} not exists!')
    return (pmat, vcnt), scaler

# NOTE: only for instances in format x/sub/y
def load_results(instance, output):
    pmat, vcnt = {}, {}
    # training and testing results of basis
    ptn = output + '/%s/' + st.F_PMAT % '*'
    basis = ''.join(re.findall(r'/([^/]+)/sub', instance)[:1])
    for f in glob(ptn % '.') + glob(ptn % basis):
        label = ''.join(re.findall(st.F_PMAT % '(.*)', f)[:1])
        # e.g., pmat_m_0_t.mtx
        grp = label.split('_')
        key = grp[0] + ('', '_t')[len(grp) > 2]
        pmat[key] = sio.mmread(f).tocsr()

    matfile = f'{instance}/res_cnt_size.mat'
    if os.path.exists(matfile):
        # benchmark results of basis
        mat = sio.loadmat(matfile)
        vcnt.update(reformat_vcnt(mat))
        ptn = output + '/%s/' + st.F_CNTZ % '*'
        for f in glob(ptn % '.') + glob(ptn % basis):
            stat = np.load(f)
            vcnt.update(reformat_vcnt(stat, os.path.basename(instance)))
    return (pmat, vcnt)

# NOTE: (test=.../theta.npy) and (test=0,1,) have different meanings:
# i.e., (test with an instance result) and (instances to be tested)

def _update(data, ind, results):
    for d, r in zip(data, results):
        d[ind].update(r)

def run_only_test(input_t):
    instance, test = input_t['instance'], input_t['test']
    if not os.path.exists(test) or not re.search(st.F_THETA % '.*', test):
        raise SystemExit(f'Test {test} is not valid!')
    # pmat, vcnt
    data = [defaultdict(dict), defaultdict(dict)]
    if '/sub/' not in test:
        # matfile cannot be inferred in this case
        basis = os.path.basename(os.path.dirname(test))
    else:
        basis = ''.join(re.findall(r'/([^/]+)/sub', test)[:1])
        inst_b = re.sub(r'/[^/]+/sub', f'/{basis}/sub', instance)
        res = load_results(inst_b, os.path.dirname(test))
        _update(data, basis, res)

    input_t['output'] += f'/{basis}'
    res, _ = run_base(input_t)
    if '/sub/' not in instance:
        inst = os.path.basename(instance)
    else:
        inst = ''.join(re.findall(r'/([^/]+)/sub', instance)[:1])
    _update(data, inst, res)
    return compute_df(data, [input_t['fp'], input_t['fc']], basis)

def run_one_test_many(input_m):
    input_m = input_m.copy()
    instance, test_m, output_m = [input_m.get(k, None) for k in ['instance', 'test', 'output']]
    if test_m:
        input_m['test'] = None
        if isinstance(test_m, bool):
            # e.g., grid/1.00/sub/9
            grid = ''.join(re.findall(r'(.*)/[^/]+/sub', instance)[:1])
            test_m = search_insts(grid)
        elif isinstance(test_m, int):
            test_m = (test_m,)
        elif not isinstance(test_m, (list, tuple)):
            raise SystemExit(f'Test {test_m} is not valid!')

    if not input_m['rerun'] and glob(f'{output_m}/{st.F_CNTZ}' % '*'):
        res = load_results(instance, output_m)
        scaler = None
    else:
        res, scaler = run_base(input_m)

    data = [defaultdict(dict), defaultdict(dict)]
    basis = ''.join(re.findall(r'/([^/]+)/sub', instance)[:1])
    inst = os.path.basename(instance)
    key = basis if basis else inst
    _update(data, key, res)
    if not test_m or not output_m or not basis:
        compute_df(data[1:], [input_m['fc']], key, reverse=True)
        raise SystemExit()

    theta_t = f'{output_m}/{st.F_THETA % inst}'
    if not os.path.exists(theta_t):
        raise SystemExit(f'{theta_t} not exists!')

    input_m['test'] = theta_t
    # or load later automatically if test is valid
    input_m['scaler'] = scaler
    # NOTE: hardcode test
    grids = {*test_m, basis}
    max_workers = min(input_m['max_workers'], len(grids))
    parallel = max_workers > 1
    input_m['gpu'] = not parallel
    verbose = input_m['verbose']
    input_m['verbose'] = 0
    with BoundedPool(max_workers) if parallel else nullcontext() as pool:
        def _update_res(fut, pg):
            _update(data, pg, fut.result()[0] if parallel else fut[0])
        # power grids with different demand settings
        for pg in sorted(grids):
            input_t = input_m.copy()
            input_t['instance'] = re.sub(r'/[^/]+/sub', f'/{pg}/sub', instance)
            input_t['output'] = re.sub(r'/[^/]+/sub', f'/{pg}/sub', output_m) + f'/{basis}'
            if 'sub' not in input_t['instance']:
                print(f'No test case {pg}!')
                continue
            ufunc = partial(_update_res, pg=pg)
            if parallel:
                pool.submit(run_base, input_t).add_done_callback(ufunc)
            else:
                ufunc(run_base(input_t))

    set_verbose(verbose)
    return compute_df(data, [input_m['fp'], input_m['fc']], basis)

# NOTE: a < x1 < x2 < b
def golden_section_search(f, a, b, tol=1e-2):
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    best = (x1, f1)
    it = 0
    while b - a > 2 * tol:
        it += 1
        if f1 < f2:
            if f1 < best[1]:
                best = (x1, f1)
            b, x2 = x2, x1
            x1 = a + (b - x2)
            f1, f2 = f(x1), f1
        else:
            if f2 < best[1]:
                best = (x2, f2)
            a, x1 = x1, x2
            x2 = a + (b - x1)
            f1, f2 = f2, f(x2)
        print(f'Iteration: {it}, new range: {a, b}')
    xs = (a + b) / 2
    fs = f(xs)
    if fs > best[1]:
        xs, fs = best
    return xs, fs

# NOTE: dtype=0: pmat, dtype=1: vcnt
def evaluate(r, input_m, w=np.ones(4)/4, dtype=1):
    print(f'Try regularization {r}...')
    input_m['a2'], input_m['rerun'] = r, True
    df = run_one_test_many(input_m)[dtype]
    key = indicator(df.columns, dtype)
    score = df[key].values.dot(w)
    print(f'Score: {score}')
    return score

def run(instance, method='lbfgsb', gpu=False, poly_fts=False, dist=False, corr=0.9, sel=-1, fr=1,
        rank=None, a1=0.001, a2=0.01, B='inf', tol=1e-6, maxiter=300, weight=[], test=None, pfunc='logistic',
        theta=0, resample=False, precision=6, output='', build_only=False, scaler='max_abs', block=1, verbose=0,
        fc=0.05, no_mc=-1, fp=1, rerun=False, init_failures='', k=1, max_workers=1):
    input_m = locals().copy()
    set_verbose(input_m['verbose'])

    # initial theta / warm start
    if isinstance(theta, str):
        if not os.path.exists(theta):
            raise SystemExit(f'Invalid theta {theta}!')
        input_m['rerun'] = True
        input_m['theta'] = np.load(theta)

    if isinstance(a2, (int, float)):
        func = run_only_test if isinstance(test, str) else run_one_test_many
        dfs = func(input_m)
    elif isinstance(a2, (list, tuple)):
        eva = partial(evaluate, input_m=input_m)
        xs, fs = golden_section_search(eva, *a2)
    else:
        raise SystemExit(f'Invalid regularization {a2}!')


if __name__ == '__main__':
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(run, serialize=lambda results: None)
