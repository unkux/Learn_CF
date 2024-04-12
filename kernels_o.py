from utils import clip_max
import numba_dpex as dpex
import numpy as np
import math

F_SIZE, F2_SIZE = 0, 0

@dpex.kernel
def _prod_kernel(Ft1, Ft2, X):
    i = dpex.get_global_id(0)
    j = dpex.get_global_id(1)
    X[i, j] = Ft1[i] * Ft2[j]

def compute_aux_mats(F, theta, knargs, test=False, ndrange=None, extra={}):
    Ft1, Ft2, Ft2_s1, Ft2_s2 = knargs
    if theta is not None:
        t1, t2 = F.dot(theta[:F_SIZE]), F.dot(theta[F_SIZE:])
        if test:
            clip_max(t1, t2)
        Ft1[:], Ft2[:] = np.exp(t1), np.exp(t2)
        knargs[2], Ft2_s2[:] = Ft2.sum(), Ft2.T.dot(F)

    if 'Ft' in extra:
        Ft = extra['Ft']
        if ndrange is None:
            Ft[:] = np.outer(Ft1, Ft2)
        else:
            _prod_kernel[ndrange](Ft1, Ft2, Ft)


# compute partial gradient, partial loglike
# category c0, y=0
def compute_gl_c0(F, Ft1, Ft2, Ft2_s1, Ft2_s2, M):
    M_ROW, M_COL = M.shape
    g, ll = np.zeros(F2_SIZE), 0
    for i in range(M_ROW):
        fs1, fs2, ts1, ts2 = Ft2_s1, 0, 0, 0
        for j in range(M_COL-1):
            v = M[i, j]
            if v < 0:
                break
            fs1 -= Ft2[v]
            fs2 -= Ft2[v]*F[v]
            ts1 += Ft1[v]*F[v]
            ts2 += Ft1[v]
        g[:F_SIZE] -= (fs1*M[i, -1]) * ts1
        g[F_SIZE:] -= (ts2*M[i, -1]) * (Ft2_s2 + fs2)
        ll -= ts2 * fs1 * M[i, -1]
    return g, ll

@dpex.kernel
def _gl_c0_kernel(F, Ft1, Ft2, Ft2_s1, Ft2_s2, M, M_ROW, C0, R):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)
    gj = dpex.get_global_id(1)

    gl = 0
    for i in range(gi, M_ROW, gis):
        if gj < F_SIZE or gj == F2_SIZE:
            fs = Ft2_s1
        else:
            fs = Ft2_s2[gj-F_SIZE]
        ts = 0
        for j in range(C0):
            v = M[i, j]
            if v < 0:
                break
            if gj < F_SIZE:
                fs -= Ft2[v]
                ts += Ft1[v]*F[v, gj]
            elif gj < F2_SIZE:
                fs -= Ft2[v]*F[v, gj-F_SIZE]
                ts += Ft1[v]
            else:
                fs -= Ft2[v]
                ts += Ft1[v]
        gl -= ts * fs * M[i, -1]
    R[gi, gj] = gl

# category c10
def compute_gl_c10(F, Ft1, Ft2, M, C):
    M_ROW, M_COL = M.shape
    C0, C1 = C, int((M_COL-C)/2)
    g, ll = np.zeros(F2_SIZE), 0
    for i in range(M_ROW):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            t = M[i, k+C1] * Ft2[v]
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                p = Ft1[u] * t
                g[:F_SIZE] -= p * F[u]
                g[F_SIZE:] -= p * F[v]
                ll -= p
    return g, ll

@dpex.kernel
def _gl_c10_kernel(F, Ft1, Ft2, M, M_ROW, C0, C1, R):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)
    gj = dpex.get_global_id(1)

    gl = 0
    for i in range(gi, M_ROW, gis):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            t = M[i, k+C1] * Ft2[v]
            ps = 0
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                if gj < F_SIZE:
                    ps += Ft1[u] * F[u, gj]
                else:
                    ps += Ft1[u]
            if gj < F_SIZE or gj == F2_SIZE:
                gl -= ps * t
            else:
                gl -= ps * t * F[v, gj-F_SIZE]
    R[gi, gj] = gl

# category c11
def compute_gl_c11(F, Ft1, Ft2, M, C):
    M_ROW, M_COL = M.shape
    C0, C1 = C, int((M_COL-C)/2)
    g, ll = np.zeros(F2_SIZE), 0
    x = np.zeros(F2_SIZE)
    for i in range(M_ROW):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            t = M[i, k+C1]
            x[:], ps = 0, 0
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                p = Ft1[u] * Ft2[v]
                x[:F_SIZE] += p * F[u]
                x[F_SIZE:] += p * F[v]
                ps += p
            pb = 1-math.exp(-ps)
            if pb < 1e-9:
                pb = 1e-9
            g += x * ((1/pb - 1) * t)
            ll += math.log(pb) * t
    return g, ll

@dpex.kernel
def _gl_c11_kernel(F, Ft1, Ft2, M, M_ROW, C0, C1, R):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)
    gj = dpex.get_global_id(1)

    gl = 0
    for i in range(gi, M_ROW, gis):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            t = M[i, k+C1]
            xs, ps = 0, 0
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                p = Ft1[u]
                if gj < F_SIZE:
                    xs += p * F[u, gj]
                ps += p
            ps *= Ft2[v]
            if gj < F_SIZE:
                xs *= Ft2[v]
            else:
                xs = ps * F[v, gj-F_SIZE]

            pb = 1-math.exp(-ps)
            if pb < 1e-9:
                pb = 1e-9
            if gj == F2_SIZE:
                gl += math.log(pb) * t
            else:
                gl += xs * ((1/pb - 1) * t)
    R[gi, gj] = gl
