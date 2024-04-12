from utils import clip_max, XMAX, CLOGLOG
import numba_dpex as dpex
import numpy as np
import math

F_SIZE, F2_SIZE, FT_SIZE, D_SIZE = 0, 0, 0, 0

@dpex.kernel
def _prod_kernel(Ft1, Ft2, Dw, R):
    i = dpex.get_global_id(0)
    j = dpex.get_global_id(1)
    t = Ft1[i] + Ft2[j] + Dw[i, j]
    t = XMAX if t > XMAX else t
    R[i, j] = math.exp(t) if CLOGLOG else 1/(1+math.exp(-t))

def compute_aux_mats(F, theta, knargs, test=False, ndrange=None, extra={}):
    D, FtDt, T = knargs
    Ft1, Ft2 = F.dot(theta[:F_SIZE]), F.dot(theta[F_SIZE:F2_SIZE])
    Dw = np.sum(theta[F2_SIZE:, None, None] * D, axis=0)
    if ndrange is None:
        # arr: theta^T x_{uv}
        arr = np.add.outer(Ft1, Ft2) + Dw
        if test:
            clip_max(arr)
        FtDt[:] = np.exp(arr) if CLOGLOG else 1/(1+np.exp(-arr))
    else:
        _prod_kernel[ndrange](Ft1, Ft2, Dw, FtDt)

    if extra.get('T', True):
        # T: forall u: sum_{v in V} w(theta^T x_{uv}) x_{uv}
        T[:, F_SIZE:F2_SIZE] = FtDt.dot(F)
        for i in range(D_SIZE):
            T[:, F2_SIZE+i] = (FtDt * D[i]).sum(axis=1)
        T[:, -1] = FtDt.sum(axis=1)
        T[:, :F_SIZE] = T[:, [-1]] * F
        if not CLOGLOG:
            T[:, -1] = -np.log(1-FtDt).sum(axis=1)


# compute partial gradient, partial loglike
# category c0, y=0
def compute_gl_c0(F, D, FtDt, T, M):
    M_ROW, M_COL = M.shape
    gl = np.zeros(FT_SIZE+1)
    for i in range(M_ROW):
        ps = 0
        for j in range(M_COL-1):
            u = M[i, j]
            if u < 0:
                break
            ps += T[u]
            for k in range(M_COL-1):
                v = M[i, k]
                if v < 0:
                    break
                ps[:F_SIZE] -= FtDt[u, v]*F[u]
                ps[F_SIZE:F2_SIZE] -= FtDt[u, v]*F[v]
                for n in range(D_SIZE):
                    ps[F2_SIZE+n] -= FtDt[u, v]*D[n, u, v]
                ps[-1] -= FtDt[u, v] if CLOGLOG else -math.log(1-FtDt[u, v])
        gl -= ps * M[i, -1]
    return gl[:-1], gl[-1]

@dpex.kernel
def _gl_c0_kernel(F, D, FtDt, T, M, M_ROW, C0, R):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)
    gj = dpex.get_global_id(1)

    gl = 0
    for i in range(gi, M_ROW, gis):
        ps = 0
        for j in range(C0):
            u = M[i, j]
            if u < 0:
                break
            ps += T[u, gj]
            for k in range(C0):
                v = M[i, k]
                if v < 0:
                    break
                if gj < F_SIZE:
                    ps -= FtDt[u, v]*F[u, gj]
                elif gj < F2_SIZE:
                    ps -= FtDt[u, v]*F[v, gj-F_SIZE]
                elif gj < FT_SIZE:
                    ps -= FtDt[u, v]*D[gj-F2_SIZE, u, v]
                else:
                    ps -= FtDt[u, v] if CLOGLOG else -math.log(1-FtDt[u, v])
        gl -= ps * M[i, -1]
    R[gi, gj] = gl

# category c10
def compute_gl_c10(F, D, FtDt, M, C):
    M_ROW, M_COL = M.shape
    C0, C1 = C, int((M_COL-C)/2)
    gl = np.zeros(FT_SIZE+1)
    ps = gl.copy()
    for i in range(M_ROW):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            ps[:] = 0
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                ps[:F_SIZE] += FtDt[u, v]*F[u]
                ps[F_SIZE:F2_SIZE] += FtDt[u, v]*F[v]
                for n in range(D_SIZE):
                    ps[F2_SIZE+n] += FtDt[u, v]*D[n, u, v]
                ps[-1] += FtDt[u, v] if CLOGLOG else -math.log(1-FtDt[u, v])
            gl -= ps * M[i, k+C1]
    return gl[:-1], gl[-1]

@dpex.kernel
def _gl_c10_kernel(F, D, FtDt, M, M_ROW, C0, C1, R):
    gi = dpex.get_global_id(0)
    gis = dpex.get_global_size(0)
    gj = dpex.get_global_id(1)

    gl = 0
    for i in range(gi, M_ROW, gis):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            ps = 0
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                if gj < F_SIZE:
                    ps += FtDt[u, v]*F[u, gj]
                elif gj < F2_SIZE:
                    ps += FtDt[u, v]*F[v, gj-F_SIZE]
                elif gj < FT_SIZE:
                    ps += FtDt[u, v]*D[gj-F2_SIZE, u, v]
                else:
                    ps += FtDt[u, v] if CLOGLOG else -math.log(1-FtDt[u, v])
            gl -= ps * M[i, k+C1]
    R[gi, gj] = gl

# category c11
def compute_gl_c11(F, D, FtDt, M, C):
    M_ROW, M_COL = M.shape
    C0, C1 = C, int((M_COL-C)/2)
    gl = np.zeros(FT_SIZE+1)
    ps = gl.copy()
    for i in range(M_ROW):
        for k in range(C0, C0 + C1):
            v = M[i, k]
            if v < 0:
                break
            t = M[i, k+C1]
            ps[:-1], ps[-1] = 0, 0 if CLOGLOG else 1
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                ps[:F_SIZE] += FtDt[u, v]*F[u]
                ps[F_SIZE:F2_SIZE] += FtDt[u, v]*F[v]
                for n in range(D_SIZE):
                    ps[F2_SIZE+n] += FtDt[u, v]*D[n, u, v]
                if CLOGLOG:
                    ps[-1] += FtDt[u, v]
                else:
                    ps[-1] *= (1-FtDt[u, v])
            pb = 1 - math.exp(-ps[-1]) if CLOGLOG else 1 - ps[-1]
            if pb < 1e-9:
                pb = 1e-9
            gl[:-1] += ps[:-1] * ((1/pb - 1) * t)
            gl[-1] += math.log(pb) * t
    return gl[:-1], gl[-1]

@dpex.kernel
def _gl_c11_kernel(F, D, FtDt, M, M_ROW, C0, C1, R):
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
            xs, ps = 0, 0 if CLOGLOG else 1
            for j in range(C0):
                u = M[i, j]
                if u < 0:
                    break
                if gj < F_SIZE:
                    xs += FtDt[u, v]*F[u, gj]
                elif gj < F2_SIZE:
                    xs += FtDt[u, v]*F[v, gj-F_SIZE]
                elif gj < FT_SIZE:
                    xs += FtDt[u, v]*D[gj-F2_SIZE, u, v]
                if CLOGLOG:
                    ps += FtDt[u, v]
                else:
                    ps *= (1-FtDt[u, v])
            pb = 1 - math.exp(-ps) if CLOGLOG else 1 - ps
            if pb < 1e-9:
                pb = 1e-9
            if gj < FT_SIZE:
                gl += xs * ((1/pb - 1) * t)
            else:
                gl += math.log(pb) * t
    R[gi, gj] = gl
