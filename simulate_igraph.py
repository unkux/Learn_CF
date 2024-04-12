#!/usr/bin/env python3
# env -S python3 -u

from utils import vfilter, load_mat, progress, set_verbose, vprint, Timer
from bounded_pool import BoundedPool
from itertools import combinations
from contextlib import nullcontext
from functools import partial
import settings as st
from scipy import io
import numpy as np
import fire
import re
import os

def parse(genfile, n_brches):
    cnt_gen_br = np.zeros((int(0.01*n_brches), n_brches), dtype=int)
    cas_size = []
    gen_no, gen_max, outages = 0, 0, 0
    with open(genfile, 'r') as f:
        no_lines = 0
        for _ in f:
            no_lines += 1
        vprint(f'Processing file ({no_lines} lines)...')
        f.seek(0)
        pg = progress(no_lines)
        for line in f:
            pg.update()
            if line and line[0] != ',':
                items = [int(x) - 1 for x in line.split(',') if x.strip()]
                if items[0] == -2:
                    if gen_no > gen_max:
                        gen_max = gen_no
                    cas_size.append(outages)
                    gen_no, outages = 0, 0
                else:
                    if gen_no == cnt_gen_br.shape[0]:
                        vprint(f'Extend row of cnt matrix {cnt_gen_br.shape}.')
                        cnt_gen_br = np.vstack([cnt_gen_br, np.zeros(n_brches)])
                    cnt_gen_br[gen_no, items] += 1
                    gen_no += 1
                    outages += len(items)

    cnt_gen_br = cnt_gen_br[:gen_max, :]
    cas_size = np.array(cas_size)
    vprint('Sample size:', len(cas_size))
    vprint('Maximum generation:', gen_max)
    vprint('Total cascading size:', sum(cas_size))
    np.savez('/tmp/res_cnt_size.npz', counts_real=cnt_gen_br, sizes_real=cas_size)
    return cnt_gen_br, cas_size


#@Timer()
def compute_stat(output, key, data, n_brches, gen_max, repeat, genfile='/tmp/generations.txt'):
    cnt_gen_br = np.zeros((gen_max, n_brches), dtype=int)
    cas_size = []
    gen_no, outages = 0, 0
    with open(genfile, 'w') if genfile else nullcontext() as f:
        n = len(data)
        vprint(f'No. of batches: {n}', level=1)
        pg = progress(n)
        for i in range(n):
            pg.update()
            for r in data[i]:
                if len(r) > 0:
                    if genfile:
                        f.write(','.join(map(str, r+1))+'\n')
                    cnt_gen_br[gen_no, r] += 1
                    gen_no += 1
                    outages += len(r)
                else:
                    if genfile:
                        f.write('-1\n')
                    cas_size.append(outages)
                    gen_no, outages = 0, 0

    cnt_gen_br = cnt_gen_br/repeat
    cas_size = np.array(cas_size)/repeat
    vprint('Total cascading size:', sum(cas_size))
    stat = {f'counts_{key}': cnt_gen_br, f'sizes_{key}': cas_size}
    if output:
        np.savez(f"{output}/{st.F_CNTZ}" % key, **stat)
    return stat


# NOTE: p0>0 indicate Monte Carlo
def simulate_igraph(Pt, samples, p0=-1, seed=0, show=False, timing=False):
    rng = np.random.default_rng(seed)
    n = Pt.shape[0]
    active = np.zeros(n, dtype=bool)
    data, gen_max = [], 0
    pg = progress(len(samples), timing=timing)
    for s in samples:
        if show:
            pg.update()
        active[:] = False
        parents = (rng.random(n)<p0).nonzero()[0] if p0 > 0 else np.array(s)
        gen_no = 0
        while len(parents) > 0:
            active[parents] = True
            data.append(parents)
            children = (~active) & (rng.random((len(parents), n))<Pt[parents]).any(axis=0)
            parents = children.nonzero()[0]
            gen_no += 1
            if gen_no > gen_max:
                gen_max = gen_no
        data.append(parents)
    return data, gen_max

@Timer()
def run_para(Pt, samples, p0=-1, seed=0, bs=10000, max_workers=8, timing=False):
    batches = range(0, len(samples), bs)
    pg = progress(len(batches), timing=timing)
    data, gen_max = {}, {}
    with BoundedPool(max_workers) as pool:
        def _update_res(fut, i):
            pg.update()
            data[i], gen_max[i] = fut.result()
        for i, k in enumerate(batches):
            ufunc = partial(_update_res, i=i)
            pool.submit(simulate_igraph,
                        Pt, samples[k:k+bs], p0, seed=[i, seed]).add_done_callback(ufunc)

    return data, max(gen_max.values())

def run(inf_matrix, init_failures='', no_mc=1, p0=-1, k=1, repeat=1, scale=1, fp=1, seed=None, genfile='', bs=20000, max_workers=8, verbose=0):
    set_verbose(verbose)
    pt_fname, Pt = '', None
    if isinstance(inf_matrix, str):
        pt_fname = inf_matrix
        if os.path.exists(pt_fname):
            Pt = io.mmread(pt_fname).toarray()
    elif isinstance(inf_matrix, (tuple, list)):
        pt_fname, Pt = inf_matrix

    if Pt is None:
        raise SystemExit('No valid influence matrix!')

    vprint('Run Monte Carlo...')

    if scale != 1:
        Pt *= scale

    f = vfilter(Pt, fp=fp)
    Pt = Pt.copy()
    Pt[~f] = 0

    repeat = max(1, repeat * (p0<=0))
    if init_failures:
        if isinstance(init_failures, str):
            print(f'Load init failures: {init_failures}')
            samples = load_mat(
                init_failures, [int],
                # note matlab index
                lambda x: [] if x[0]==-1 else [v-1 for v in x]
            )
        elif hasattr(init_failures, '__iter__'):
            samples = init_failures
        else:
            raise SystemExit(f'Invalid init_failures {init_failures}!')
        p0 = -1
    elif p0 > 0:
        samples = range(no_mc)
        if p0 >= 1:
            p0 = 1.0/Pt.shape[0]
        vprint(f'Probability: {p0}')
    else:
        # n-k contingencies
        samples = list(combinations(range(Pt.shape[0]), k)) * no_mc * repeat
        p0 = -1
        vprint('Note that no_mc * repeat is applied!')
    vprint('Sample size:', len(samples))
    vprint('Simulate igraph...')
    data = {}
    if max_workers > 1 or max_workers == 0:
        if not isinstance(seed, int):
            seed = 0
        data, gen_max = run_para(Pt, samples, p0, seed, bs, max_workers)
    else:
        data[0], gen_max = simulate_igraph(Pt, samples, p0, seed, verbose)

    vprint('Maximum generation:', gen_max)
    vprint('Compute statistics...')
    output, key = None, None
    if pt_fname:
        output = os.path.dirname(pt_fname)
        inst = ''.join(re.findall(st.F_PMAT % '(.*)', pt_fname)[:1])
        key = ('stat', 'our')['_m_' in pt_fname or '_l_' in pt_fname] + f'_{inst}'
    return compute_stat(output, key, data, Pt.shape[0], gen_max, repeat, genfile)


if __name__ == '__main__':
    # Not use pager for printing help text
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(dict(run=run, parse=parse), serialize=lambda results: None)
