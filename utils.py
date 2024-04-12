from contextlib import ContextDecorator
from collections.abc import Iterable
from multiprocessing import Value
from pprint import pprint
import pandas as pd
import numpy as np
import functools
import time
import os

_use_dpt_arr = False
try:
    from pkg_resources import parse_version, get_distribution
    # NOTE: from v0.21.0, numpy array as kernel argument is not supported,
    # dpt performs better than numpy for large instances, worse for small ones
    _use_dpt_arr = parse_version(get_distribution('numba_dpex').version) >= parse_version('0.21.0')
except Exception as e:
    print(e)

if True:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

# cloglog / logistic
CLOGLOG = True
def set_ktype(v):
    global CLOGLOG
    CLOGLOG = v

VERBOSE = 0
def set_verbose(v):
    global VERBOSE
    VERBOSE = v

def vprint(*arg, level=0, **kwargs):
    if VERBOSE > level:
        if arg:
            if len(arg) > 1 or isinstance(arg[0], str) or 'flush' in kwargs:
                print(*arg, **kwargs)
            else:
                pprint(*arg, **kwargs, width=200)
        else:
            print(*arg, **kwargs)

class Timer(ContextDecorator):
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.run_time = time.perf_counter() - self._start_time
        vprint(f'Elapsed time: {self.run_time:.6f} seconds')
        return False

def dev_ctx(_func=None, *, gpu=True, verbose=False):
    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if gpu and not _use_dpt_arr:
                import dpctl
                device = dpctl.select_default_device()
                with dpctl.device_context(device):
                    if verbose:
                        print('\nUse device:')
                        device.print_device_info()
                    val = func(*args, gpu=gpu, **kwargs)
            else:
                val = func(*args, gpu=gpu, **kwargs)
            return val
        return wrapper
    if _func is None:
        return _decorator
    else:
        return _decorator(_func)

# process-safe
class progress:
    def __init__(self, length, precent=100, timing=False):
        self.precent = precent
        self.timing = timing
        self.cnt = Value('i', 0)
        self.set_progress(length)

    def set_progress(self, length):
        self.length, self.blk = length, max(1, int(length/self.precent))
        self.cnt.value, self.blk_cnt = 0, 0
        self.t1 = time.time()

    def update(self):
        with self.cnt.get_lock():
            self.cnt.value += 1
            if self.cnt.value % self.blk == 0:
                self.blk_cnt += 1
                vprint('.', end='', flush=True)
                if self.timing:
                    t2 = time.time()
                    vprint(t2 - self.t1)
                    self.t1 = t2
            if self.cnt.value == self.length:
                self.cnt.value, self.blk_cnt = 0, 0
                vprint()

def flatten(xs):
    # NOTE: recursive
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# NOTE: avoid exp(x) overflow when testing a pre-trained model
XMAX = 100
def clip_max(*arrs, xmax=XMAX):
    for a in arrs:
        f = a > xmax
        if f.any():
            a[f] = xmax

def vfilter(X, fp=1):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X >= np.quantile(X, 1-fp)

def _error(x1, x2, relative=True):
    err = x1 - x2
    if relative:
        mx = np.maximum(x1, x2)
        mx[mx==0] = 1
        err /= mx
    return err

METHODS = {
    'our_m': 'Our ($C_{nn}$)',
    'our_m_t': 'Our ($C_{n1}$)',
    'our_l': 'Our ($L_{nn}$)',
    'our_l_t': 'Our ($L_{n1}$)',
    'sim_cor': 'Hines (Train each)',
    'sim_cor_1st': 'Hines (Train $1^{st}$)',
    'mk': 'Wu (Train each)',
    'mk_1st': 'Wu (Train $1^{st}$)',
}
def cnt_err(cnt, cref=None, fc=1, relative=True):
    F, derr = {}, {}
    if 'real' in cnt:
        F['real'] = vfilter(cnt['real'], fc)
        for k, n in METHODS.items():
            if k in cnt:
                if k not in F:
                    F[k] = vfilter(cnt[k], fc)
                if F[k].shape[0] != F['real'].shape[0]:
                    continue
                f = F[k] + F['real']
                err = _error(cnt[k][f], cnt['real'][f], relative)
                derr[METHODS[k]] = np.abs(err).mean()
                if cref and k in ['sim_cor', 'mk']:
                    if cref[k].shape[0] != F['real'].shape[0]:
                        continue
                    f = vfilter(cref[k], fc) + F['real']
                    err = _error(cref[k][f], cnt['real'][f], relative)
                    derr[METHODS[f'{k}_1st']] = np.abs(err).mean()
    return derr

def pm_err(pm, pref, fp=1, relative=True):
    derr = {}
    D = {'m': 'C', 'l': 'L'}
    pairs = [('{nn}', '{n1}'), ('{n1}', '{11}'), ('{nn}', '{11}')]
    for key, val in D.items():
        P, F = {}, {}
        if key in pref:
            P['{11}'] = pref[key]
            if key in pm:
                P['{nn}'] = pm[key]
            if f'{key}_t' in pm:
                P['{n1}'] = pm[f'{key}_t']
            for k, p in P.items():
                if hasattr(p, 'tocsr'):
                    P[k] = p.tocsr()
                F[k] = vfilter(P[k], fp)
            for k1, k2 in pairs:
                if k1 in P and k2 in P:
                    # skip cross-grid pmat err check
                    if F[k1].shape[0] != F[k2].shape[0]:
                        continue
                    f = F[k1] + F[k2]
                    err = _error(P[k1][f], P[k2][f], relative)
                    derr[f'${val}_{k1}-{val}_{k2}$'] = np.abs(err).mean()
    return derr

def compute_err(data, func, fx=1, ref=1.00):
    err = {}
    xref = data[f'{ref:.2f}' if isinstance(ref, float) else ref]
    for key, x in sorted(data.items()):
        err[key] = func(x, xref, fx)
    return pd.DataFrame(err).transpose()

def compute_df(data, fx, basis, funcs=[pm_err, cnt_err], reverse=False):
    dfs = []
    for d, f, p in zip(data, funcs[::-1] if reverse else funcs, fx):
        df = compute_err(d, f, fx=p, ref=basis)
        if df.empty:
            raise SystemExit('No valid test case!')
        print(df)
        df_stat = df.agg(['mean', 'std', 'min', 'max'])
        if len(df) > 1:
            print(df_stat)
        dfs.append(df_stat)
    return dfs

def indicator(keys, dtype=1):
    for k in keys:
        if '{n1}' in k and (dtype == 1 or '{nn}' in k):
            return k
    return None

# return [[] if m[0]==-1 else m for m in mat]
def load_mat(filename, dtypes=[float], vfilter=lambda x: x, comment='#', delimiter=','):
    if not os.path.exists(filename):
        raise SystemExit(f'{filename} not exists!')
    data, n = [], len(dtypes)
    with open(filename, 'r') as f:
        for line in f:
            if line and line[0] != comment:
                ls = [x for x in line.split(delimiter) if x.strip()]
                if dtypes:
                    item = list(map(lambda x: dtypes[x[0] % n](x[1]), enumerate(ls)))
                else:
                    item = ls
                data.append(vfilter(item))
    return data

def str2num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

def num2str(v, f_digit=2):
    return f'{v:.{f_digit}f}' if isinstance(v, float) else f'{v}'

def search_insts(path, to_num=False):
    insts = []
    if os.path.exists(path):
        for d in os.listdir(path):
            v = str2num(d)
            if v is not None:
                insts.append(v if to_num else d)
    return sorted(insts)
