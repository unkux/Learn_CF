from importlib import import_module
import numba_dpex as dpex
import numpy as np

from utils import _use_dpt_arr, get_ktype
if _use_dpt_arr:
    import dpctl.tensor as dpt
    import dpctl

from numba.core.errors import NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import samples

class Kernel:
    def __init__(self, F, D, Ma, Ca, gpu, **kwargs):
        self.F, self.D, self.Ma, self.Ca, self.gpu = F, D, Ma, Ca, gpu
        self.BR_SIZE, self.F_SIZE = F.shape
        self.F2_SIZE, self.D_SIZE = 2*self.F_SIZE, len(D)

        self.kn = import_module('kernels_' + ('o', 'cl')[self.D_SIZE > 0])
        self.kn.F_SIZE, self.kn.F2_SIZE = self.F_SIZE, self.F2_SIZE
        self.FT_SIZE = self.F2_SIZE + self.D_SIZE

        self.FtDt = np.zeros((self.BR_SIZE, self.BR_SIZE))
        if self.D_SIZE > 0:
            self.kn.FT_SIZE, self.kn.D_SIZE = self.FT_SIZE, self.D_SIZE
            self.T = np.zeros((self.BR_SIZE, self.FT_SIZE+1))
            self.knargs = [self.D, self.FtDt, self.T]
        else:
            self.knargs = [np.empty(self.BR_SIZE), np.empty(self.BR_SIZE), 0, np.empty(self.F_SIZE)]

        self.Fg = self.F
        self.device = None
        if self.gpu:
            if _use_dpt_arr:
                self.device = dpctl.select_default_device()
                if kwargs.get('verbose', 0) > 0:
                    print('\nUse device:')
                    self.device.print_device_info()
                self.Fg = dpt.asarray(self.F, device=self.device)
            self.R, self.ndrange = self.init_gl_kernel(glb_g0=128)

        self.gl = np.zeros((6, self.FT_SIZE+1))

    def init_gl_kernel(self, glb_g0=128*8):
        gl_size = self.FT_SIZE + 1
        if _use_dpt_arr:
            R = dpt.zeros((glb_g0, gl_size), device=self.device)
        else:
            R = np.zeros((glb_g0, gl_size))
        ndrange = dpex.Range(glb_g0, gl_size)
        return R, ndrange

    def _compute_gl_c0(self, M, knargs):
        if not self.gpu:
            return self.kn.compute_gl_c0(self.F, *knargs, M)
        else:
            M_ROW, M_COL = M.shape
            self.kn._gl_c0_kernel[self.ndrange](self.Fg, *knargs, M, M_ROW, M_COL-1, self.R)
            if _use_dpt_arr:
                Rs = dpt.asnumpy(self.R).sum(axis=0)
            else:
                Rs = self.R.sum(axis=0)
            return Rs

    def _compute_gl_c1(self, M, knargs, C0, y):
        if not self.gpu:
            gl_func = self.kn.compute_gl_c11 if y else self.kn.compute_gl_c10
            return gl_func(self.F, *knargs[:2], M, C0)
        else:
            M_ROW, M_COL = M.shape
            C1 = int((M_COL-C0)/2)
            kernel = self.kn._gl_c11_kernel if y else self.kn._gl_c10_kernel
            kernel[self.ndrange](self.Fg, *knargs[:2], M, M_ROW, C0, C1, self.R)
            if _use_dpt_arr:
                Rs = dpt.asnumpy(self.R).sum(axis=0)
            else:
                Rs = self.R.sum(axis=0)
            return Rs

    #@Timer()
    def compute_gl(self, theta, n_samples, rdata, test, block):
        self.kn.compute_aux_mats(self.F, theta, self.knargs, test, ndrange=None)
        knargs = self.knargs
        if self.gpu and _use_dpt_arr:
            knargs = []
            for v in self.knargs:
                knargs.append(dpt.asarray(v, device=self.device) if isinstance(v, np.ndarray) else v)
        rs, ws = rdata

        self.gl[:] = 0
        for i, M in self.Ma.items():
            Ms = samples.select(M, i, rs)
            if Ms is None:
                continue
            for k in range(block):
                M = Ms[k::block, :]
                if self.gpu and _use_dpt_arr:
                    M = dpt.asarray(M, device=self.device)
                if i in [0, 1, 4]:
                    self.gl[i] += self._compute_gl_c0(M, knargs)
                else:
                    self.gl[i] += self._compute_gl_c1(M, knargs, self.Ca[i], i==3)

        gl = self.gl.T.dot(ws)/n_samples
        return gl[:-1], gl[-1]

    def prob_matrix(self, theta=None):
        extra = {}
        if self.D_SIZE == 0:
            extra = {'Ft': self.FtDt}
        elif theta is not None:
            extra['T'] = False
        if extra:
            self.kn.compute_aux_mats(self.F, theta, self.knargs, test=True, ndrange=None, extra=extra)
        return (1 - np.exp(-self.FtDt)) if get_ktype() else self.FtDt
