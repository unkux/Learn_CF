#!/usr/bin/env python3
# env -S python3 -u

from utils import dev_ctx, set_verbose, vprint, Timer, vfilter, set_ktype
from scipy import optimize, sparse, io
from kernel_api import Kernel
import settings as st
import numpy as np
import features
import samples
import fire
import re
import os


def build_dataset(instance, poly_fts=True, corr=0.9, test=None, scaler='max_abs', output=None, verbose=0):
    F, D, scaler = features.build(instance, poly_fts=poly_fts, corr=corr, test=test, scaler=scaler, output=output, verbose=verbose)
    data, Ps = samples.build(instance, len(F), cnt=False, p_stat=False, verbose=verbose)
    Ma, Ca = data['M'], data['C']
    return F, D, scaler, Ma, Ca, Ps

class MLE:
    def __init__(self, F, D, Ma, Ca, method='lbfgsb', **kwargs):
        self.F, self.D, self.Ma, self.Ca, self.method = F, D, Ma, Ca, method
        self.BR_SIZE, self.F_SIZE = self.F.shape
        self.F2_SIZE, self.D_SIZE = 2*self.F_SIZE, len(self.D)
        self.FT_SIZE = self.F2_SIZE + self.D_SIZE

        self.sel = kwargs.get('sel', -1)
        self.mask = None
        rank = kwargs.get('rank', None)
        if isinstance(rank, np.ndarray):
            if len(rank) < self.FT_SIZE or self.sel < self.FT_SIZE:
                self.mask = np.ones(self.FT_SIZE, bool)
                if self.sel >= 1 and self.sel < len(rank):
                    rank = rank[:self.sel]
                self.mask[rank] = False
                vprint(f'Feature dim (filtered): {(~self.mask).sum()}')
        elif self.sel >= 2 and self.sel < self.F2_SIZE:
            self.F = self.F[:, :int(self.sel/2)].copy()
            self.F_SIZE = self.F.shape[1]
            self.F2_SIZE = 2*self.F_SIZE
            self.FT_SIZE = self.F2_SIZE + self.D_SIZE
            vprint(f'Feature dim: {self.FT_SIZE}')

        self.msize = [len(self.Ma[m[0]]) for m in samples.MTX_GRP]
        self.seed = kwargs.get('seed', 0)

        self.gpu = kwargs.get('gpu', False)
        self.block = kwargs.get('block', 1)
        self.test = kwargs.get('test', False)
        self.verbose = kwargs.get('verbose', False)
        self.kn = Kernel(self.F, self.D, self.Ma, self.Ca, self.gpu, verbose=self.verbose)

        self.resample = kwargs.get('resample', False)
        self.weight = kwargs.get('weight', [])
        self.rs, self.ws = {}, np.ones(6)
        # NOTE: resample on row, -1: no resampling, 0: resampling 0 (bypass data)
        if self.resample and self.method != 'sgd':
            # NOTE: e.g., resampling 10%, except part 2 (positive/outage data)
            mb = [(m, int(m/10) if i != 2 else m) for i, m in enumerate(self.msize)]
            self.rs = samples.resample(mb, self.seed)

        # NOTE: resample on column, weights on gradient and loglike,
        # if weighting category 0 (i.e., [1,2,4]) with the same value,
        # it's equivalent to weighting directly on the likelihood, i.e.,
        # w1*y*log(f) + w1*y*log(1-f), keeping original no. of samples
        rdata = [self.rs, self.ws]
        if self.weight:
            if isinstance(self.weight, (float, int)):
                self.weight = [self.weight if i==2 else 1-self.weight for i in range(4)]
                rdata[1] = np.ones(6)
            for i, v in enumerate(self.weight[:4]):
                self.ws[samples.MTX_GRP[i]] = v

        self.S_SIZE = samples.cnt_samples(self.Ma, self.Ca, self.BR_SIZE, rdata, self.gpu, verbose=True)
        vprint("No. of samples:", self.S_SIZE)

        # NOTE: regularization (l1, l2)
        self.a1 = kwargs.get('a1', 0)/self.S_SIZE
        self.a2 = kwargs.get('a2', 0)/self.S_SIZE

        # only for lbfgsb
        self.B = kwargs.get('B', 20)

        # record param updates
        self.param_trace = []

        self.it = 0

    # NOTE: recnt for accurate computation, more time-consuming
    def loglike(self, params, rs={}, recnt=True):
        n_samples = self.S_SIZE
        if rs and self.method == 'sgd':
            if recnt:
                n_samples = samples.cnt_samples(self.Ma, self.Ca, self.BR_SIZE, (rs, self.ws), self.gpu)
        else:
            rs = self.rs
        self.g, self.ll = self.kn.compute_gl(params, n_samples, (rs, self.ws), self.test, self.block)

        if self.mask is not None:
            self.g[self.mask], params[self.mask] = 0, 0

        self.param_trace.append(params)
        self.ll -= self.a1*np.abs(params).sum() + self.a2*(params**2).sum()
        return self.ll

    def gradient(self, params):
        self.g -= self.a1*np.sign(params) + 2*self.a2*params
        self.it += 1
        if self.verbose:
            print(f"{self.it:<5}", "max |gradient|:", f"{max(abs(self.g)):<25}", 'loglike:', self.ll, flush=(self.it % 1 == 0))
        # NOTE: the sign of gradient due to negative loglike
        return -self.g

    def sgd(self, theta, rate=2e-2, decay=0.2, bp=0.2, recnt=True, maxiter=1000, eps=1e-6):
        rng = np.random.default_rng(self.seed)
        diff = 0
        for i in range(maxiter):
            rand_idx = samples.gen_rand_idx(self.msize, rng)
            gm = 0
            for rs in samples.sampler(self.msize, bp, rand_idx):
                ll = self.loglike(theta, rs, recnt=recnt)
                g = self.gradient(theta)

                diff = decay * diff - (1-decay) * rate*g
                theta += diff

                gm = max(gm, *abs(diff))

            if self.verbose and i % 100 == 0:
                ll = self.loglike(theta)
                print(f"{i:<5}", "max |gradient|:", f"{gm:<25}", 'loglike:', ll, flush=True)
            if gm <= eps:
                break

        ll = self.loglike(theta)
        return theta, -ll

    def prob_matrix(self, theta=None):
        return self.kn.prob_matrix(theta)

    @Timer()
    def run(self, params, gtol=1e-5, norm=np.Inf, maxiter=100, **kwargs):
        if hasattr(params, '__iter__'):
            params = np.array(params)
        elif params is None:
            rng = np.random.default_rng(self.seed)
            params = rng.random(self.FT_SIZE)
        else:
            params = np.zeros(self.FT_SIZE)

        # NOTE: negative loglike
        def f(params, *args):
            return -self.loglike(params, *args)

        if self.method == 'bfgs':
            xopt, fopt, *info = optimize.fmin_bfgs(
                f, params, self.gradient, args=(),
                gtol=gtol, norm=norm, maxiter=maxiter,
                full_output=True, disp=True, callback=None)
        elif self.method == 'lbfgsb':
            xopt, fopt, info = optimize.fmin_l_bfgs_b(
                f, params, self.gradient, args=(),
                bounds=[(-self.B, self.B)]*self.FT_SIZE,
                pgtol=gtol, maxiter=maxiter, maxfun=maxiter-1,
                disp=False, callback=None)
        elif self.method == 'sgd':
            xopt, fopt = self.sgd(params, eps=gtol, maxiter=maxiter)
        else:
            xopt, fopt = params, -self.loglike(params)

        vprint(f'No. of iterations: {self.it}')

        Pm = self.prob_matrix()
        return xopt, -fopt, Pm, np.stack(self.param_trace, axis=0)

def run(instance, method='lbfgsb', gpu=False, poly_fts=False, dist=False, corr=0.9, sel=-1, fr=1,
        rank=None, a1=0, a2=0, B='inf', tol=1e-6, maxiter=200, weight=[], test=None, pfunc='cloglog',
        theta=0, resample=False, precision=6, output='', build_only=False, scaler='max_abs', block=1, verbose=0,
        **_):
    set_verbose(verbose)
    genfile = f'{instance}/generations.csv'
    if not os.path.exists(genfile):
        raise SystemExit(f'{genfile} not exists!')

    if test:
        res = re.findall(st.F_THETA % '(.*)', test)
        if res and os.path.exists(str(test)):
            theta = np.load(test)
            # retrieve the corresponding feature filter and scale
            test = f"{os.path.dirname(test)}/{st.F_FT_SC}" % res[0]
        else:
            raise SystemExit(f'No valid test file (input: {st.F_THETA})!')

    inst = os.path.basename(instance)

    if rank:
        if os.path.exists(str(rank)):
            vprint(f'Load parameter rank {rank}')
            perr = np.load(rank)
            rank = np.argsort(-perr)
            if fr > 0:
                f = vfilter(perr[rank], fr)
                rank = rank[f]
        else:
            raise SystemExit(f'{rank} not exists!')

    if method not in ['lbfgsb', 'bfgs', 'sgd']:
        raise SystemExit(f'Unknown method: {method}')

    F, D, scaler, Ma, Ca, Ps = build_dataset(instance, poly_fts, corr, test=test, scaler=scaler, output=output, verbose=verbose)
    if build_only:
        raise SystemExit()

    # NOTE: use D0 for the dimension of theta_0
    D0 = np.ones((1, F.shape[0], F.shape[0]))
    D = np.concatenate((D, D0)) if dist else D0

    @dev_ctx(gpu=gpu, verbose=verbose)
    def mle_run(F, D, Ma, Ca, method, theta=None, **kwargs):
        set_ktype(pfunc == 'cloglog')
        mle = MLE(F, D, Ma, Ca, method, sel=sel, rank=rank, test=test, block=block,
                  a1=a1, a2=a2, B=float(B), weight=weight, resample=resample, **kwargs)
        # NOTE: consider the scale of features when testing
        if test:
            vprint('Test hyperparametric model...')
            return theta, mle.loglike(theta), mle.prob_matrix(), None, mle
        else:
            vprint(f'Run optimization ({method})...')
            res = mle.run(theta, gtol=tol, maxiter=maxiter)
            return *res, mle

    theta, ll, Pm, param_trace, mle = mle_run(F, D, Ma, Ca, method, theta=theta, verbose=verbose)

    if output:
        if isinstance(output, bool):
            output = f"{instance}/{st.D_OUTPUT}"
        os.makedirs(output, exist_ok=True)
        if not test:
            np.save(f"{output}/{st.F_THETA}" % inst, theta)
            np.save(f"{output}/{st.F_PARAM_TR}" % inst, param_trace)
            if rank is False:
                perr = np.zeros_like(theta)
                for i, t in enumerate(theta):
                    theta[i] = 0
                    perr[i] = np.abs(mle.prob_matrix(theta) - Pm).sum()
                    theta[i] = t
                np.save(f"{output}/{st.F_RANK}" % inst, perr)

    np.fill_diagonal(Pm, 0)
    Pm[Pm<10**(-precision)] = 0

    vprint(f"\nParameters: (dim: {len(theta)}, range: {min(theta), max(theta)}, mean: {theta.mean()})")
    vprint("\nLikelihood:", ll)
    vprint(f"\nEstimated matrix: {Pm.shape}, nonzero: {np.count_nonzero(Pm)}")
    if Ps is not None:
        err = np.abs(Pm-Ps).mean()
        vprint('\nEstimation error (inaccurate):', err)

    pm_fname = ''
    if output:
        ps_fname = f"{output}/{st.F_PMAT}" % f"s_{inst}"
        if Ps is not None and not os.path.exists(ps_fname):
            io.mmwrite(ps_fname, sparse.csr_matrix(Ps))
        pf = ('l', 'm')[pfunc == 'cloglog']
        pm_fname = f"{output}/{st.F_PMAT}" % f"{pf}_{inst}{'_t' if test else ''}"
        io.mmwrite(pm_fname, sparse.csr_matrix(Pm))

    return (pm_fname, Pm), scaler

if __name__ == '__main__':
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(run, serialize=lambda results: None)
