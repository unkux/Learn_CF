#!/usr/bin/env python3

from utils import dev_ctx, progress, set_verbose, vprint
from scipy.sparse import csr_matrix
from collections import defaultdict
from itertools import zip_longest

from utils import _use_dpt_arr
if _use_dpt_arr:
    import dpctl.tensor as dpt

import numba_dpex as dpex
import numpy as np
import fire
import os


# NOTE:
#   - parse 'generations' file from branching process, and build samples for training ML model
#   - the interactions forms a complete graph


MTX_GRP = [[0], [1, 2], [3], [4, 5]]


def cnt_cascades(genfile):
    n_cas = 0
    with open(genfile, 'r') as f:
        for line in f:
            if line[:2] == '-1':
                n_cas += 1
    return n_cas

def parse_data(filename, n_brches, delimiter=',', dtype=np.int32):
    # NOTE: segment data into different parts based on (cascading, [other_]active) due to sparsity
    y0_p0 = defaultdict(int)
    y0_p1 = defaultdict(int)
    y0_p2 = defaultdict(lambda: defaultdict(int))

    y1_p0 = defaultdict(lambda: defaultdict(int))
    y1_p1 = defaultdict(int)
    y1_p2 = defaultdict(lambda: defaultdict(int))

    o_active = []
    n_act = np.zeros((n_brches, n_brches), dtype=dtype)
    n_tot = np.zeros((n_brches, n_brches), dtype=dtype)

    with open(filename, 'r') as f:
        no_lines = 0
        for _ in f:
            no_lines += 1
        vprint('No. of lines (of generations file):', no_lines)

        vprint('Processing data...')
        f.seek(0)

        # idx: -1 (MATLAB) - 1
        last = [-2]
        pg = progress(no_lines)
        for line in f:
            pg.update()
            if line and line[0] != ',':
                items = [int(x) - 1 for x in line.split(delimiter) if x.strip()]
                if last[0] != -2:
                    k = tuple(last)
                    n_tot[k, :] += 1
                    # end of a cascade
                    if items[0] == -2:
                        # data for no cascading branches
                        if not o_active:
                            y0_p0[k] += 1
                        else:
                            # data for cascading end
                            y0_p1[k] += 1
                            for i in o_active:
                                y0_p2[k][i] -= 1

                        n_tot[np.ix_(k, last+o_active)] -= 1

                        o_active = []
                    else:
                        # data for cascading branches
                        for i in items:
                            y1_p0[k][i] += 1

                        # data for no cascading branches
                        y1_p1[k] += 1
                        for i in items + o_active:
                            y1_p2[k][i] -= 1

                        o_active.extend(last)

                        # NOTE: for each v in items, which u in k that activates v is unknown
                        # and here we just count all nodes in k, overestimating the probability
                        n_act[np.ix_(k, items)] += 1
                        n_tot[np.ix_(k, o_active)] -= 1

                last = items

    return ((y0_p0, y0_p1, y0_p2),
            (y1_p0, y1_p1, y1_p2),
            n_act, n_tot)

def max_kv_len(data):
    max_kl, max_vl = 0, 0
    for k, d in data.items():
        length = len(k)
        if max_kl < length:
            max_kl = length
        if hasattr(d, '__iter__'):
            length = len(d)
            if max_vl < length:
                max_vl = length
    return max_kl, max_vl


def make_array(data, dtype=np.int32):
    R = len(data)
    # offsets for the data
    C0, C1 = max_kv_len(data)
    vdict = C1 != 0
    # for 2 * C1: half keys, harf values
    M = -np.ones((R, C0 + (2*C1 if vdict else 1)), dtype=dtype)
    # NOTE: ensure the same order for (y0_p1, y0_p2) and (y1_p1, y1_p2) to have consistent sampling
    for i, (k, v) in enumerate(sorted(data.items())):
        M[i, :len(k)] = k
        # NOTE: keys and values are in same order
        if vdict:
            s, t = C0, C0+len(v)
            M[i, s:t] = list(v.keys())
            s, t = s+C1, t+C1
            M[i, s:t] = list(v.values())
        else:
            M[i, -1] = v
    vprint('Data shape:', M.shape, level=1)
    # C1 can be deduced
    return M, C0


def encode_data(data_y0, data_y1, dtype=np.int32):
    M, C = {}, np.zeros(6, dtype=dtype)
    for i, d in enumerate([*data_y0, *data_y1]):
        M[i], C[i] = make_array(d)
    return M, C


def build(instance, n_brches, output=True, rerun=False, cnt=True, p_stat=False, verbose=0):
    set_verbose(verbose)
    # branch no. is consecutive
    datafile, statfile = instance + '/samples.npz', instance + '/stats.npy'
    stat = None
    if not rerun and os.path.exists(datafile):
        vprint(f'Load existing samples {datafile}', level=1)
        with np.load(datafile) as data:
            C = data['C']
            M = {int(k[1:]): m for k, m in data.items() if k != 'C'}
        data = {'M': M, 'C': C}
        if p_stat and os.path.exists(statfile):
            vprint(f'Load existing stats {statfile}', level=1)
            stat = np.load(statfile, allow_pickle=True)[()]
        if verbose > 1:
            print('Data shapes:', [M[k].shape for k in sorted(M)])
            print('Data offsets:', C)
    else:
        infile = instance + '/generations.csv'
        if not os.path.exists(infile):
            raise SystemExit('Error: %s not existed!' % infile)

        data_y0, data_y1, n_act, n_tot = parse_data(infile, n_brches)
        M, C = encode_data(data_y0, data_y1)
        data = {'M': M, 'C': C}
        if p_stat:
            stat = {'n_act': csr_matrix(n_act), 'n_tot': csr_matrix(n_tot)}

        if output:
            M = {f'M{i}': m for i, m in M.items()}
            np.savez_compressed(datafile, C=C, **M)
            if p_stat:
                np.save(statfile, stat)

    if cnt:
        size = dev_cnt_samples(M, C, n_brches, verbose)
        print("No. of samples:", size)

    Ps = None
    if p_stat and stat:
        vprint("Statistic interactions (inaccurate):", stat['n_act'].count_nonzero())
        n_act, n_tot = stat['n_act'].toarray(), stat['n_tot'].toarray()
        np.fill_diagonal(n_tot, 1)
        Ps = n_act/n_tot
        Ps[Ps<1e-6] = 0

    return data, Ps

@dev_ctx
def dev_cnt_samples(M, C, n_brches, verbose, gpu):
    return cnt_samples(M, C, n_brches, ({}, np.ones(6)), gpu, verbose)

def resample(resize, seed):
    rs = {}
    rng = np.random.default_rng(seed)
    for i, (m, s) in enumerate(resize):
        si = rng.choice(m, size=s, replace=False) if s > 0 else s
        for j in MTX_GRP[i]:
            rs[j] = si
    return rs

def gen_rand_idx(msize, rng):
    rand_idx = []
    for i, m in enumerate(msize):
        rand_idx.append(rng.permutation(m))
    return rand_idx

def sampler(msize, bp, rand_idx):
    # NOTE: sample at least 1
    mb = [(m, max(int(bp*m), 1)) for m in msize]
    for idx in zip_longest(*(range(0, *t) for t in mb), fillvalue=-1):
        ss = map(lambda r, i, t: r[i:i+t[1]] if i != -1 else 0, rand_idx, idx, mb)
        yield {k: s for i, s in enumerate(ss) for k in MTX_GRP[i]}

def select(M, i, rs):
    if len(M) == 0:
        M = None
    elif i in rs:
        if isinstance(rs[i], int):
            if rs[i] == 0:
                M = None
        else:
            M = M[rs[i]]
    return M

#@Timer()
def cnt_samples(Ma, Ca, n_brches, rdata, gpu=True, verbose=False):
    if gpu:
        Rc, ndrange = init_cnt_kernel()

    rs, ws = rdata
    cnt = np.zeros(6, dtype=int)
    for j, M in Ma.items():

        M = select(M, j, rs)
        if M is None:
            continue

        M_ROW, M_COL = M.shape
        C0 = Ca[j]
        if j in [0, 1, 4]:
            if gpu:
                if _use_dpt_arr:
                    M = dpt.asarray(M)
                _cnt_samples_kernel[ndrange](M, M_ROW, C0, n_brches, Rc)
                cnt[j] = sum(Rc)
            else:
                for i in range(M_ROW):
                    k = 1  # at least 1
                    while k < C0 and M[i, k] >= 0:
                        k += 1
                    cnt[j] += M[i, -1] * (n_brches - k)
        else:
            C1 = int((M_COL-C0)/2)
            for i in range(M_ROW):
                for k in range(C0, C0 + C1):
                    if M[i, k] < 0:
                        break
                    cnt[j] += M[i, k+C1]

    if verbose > 1:
        t = (cnt*ws).astype(int)
        print('Data group size:', [sum(t[m]) for m in MTX_GRP])
    return int(cnt.dot(ws))


def init_cnt_kernel(grp_size=128):
    if _use_dpt_arr:
        Rc = dpt.zeros(grp_size, dtype=int)
    else:
        Rc = np.zeros(grp_size, dtype=int)
    return Rc, dpex.Range(grp_size)

@dpex.kernel
def _cnt_samples_kernel(M, M_ROW, C0, BR_SIZE, Rc):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)

    cnt = 0
    for i in range(gi, M_ROW, gis):
        k = 1  # at least 1
        while k < C0 and M[i, k] >= 0:
            k += 1
        cnt += M[i, -1] * (BR_SIZE - k)
    Rc[gi] = cnt


#def init_cnt_v1_kernel(grp0_size=256, grp1_size=256, gpu=None):
#    Rc = np.zeros(grp0_size * grp1_size, dtype=np.int64)
#    return Rc, dpex.Range(grp0_size, grp1_size)
#
#@dpex.kernel
#def _cnt_samples_v1_kernel(M, M_ROW, C0, BR_SIZE, Rc):
#    gi = dpex.get_global_id(0)
#    gis = dpex.get_global_size(0)
#    gv = dpex.get_global_id(1)
#    gvs = dpex.get_global_size(1)
#
#    cnt = 0
#    for i in range(gi, M_ROW, gis):
#        for v in range(gv, BR_SIZE, gvs):
#            # exclude v in M[i,:C0]
#            valid = True
#            for j in range(C0):
#                u = M[i, j]
#                if u == v:
#                    valid = False
#                    break
#                if u < 0:
#                    break
#            if valid:
#                # compute one group of samples
#                cnt += M[i, -1]
#    Rc[gi * gvs + gv] = cnt


if __name__ == '__main__':
    # Not use pager for printing help text
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(build, serialize=lambda results: None)
