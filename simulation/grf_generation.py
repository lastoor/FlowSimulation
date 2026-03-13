"""
Fractional Brownian motion (fBm) field generation.
"""

import warnings
import numpy as np
from scipy.fft import fftn, dctn, next_fast_len
import time


class GaussianRandomFieldCircEmbed:
    """
    Gaussian random field via circulant embedding.
    """
    def __init__(self, mean, cov, pts,
                 minpadding=0):
        self.mean = mean
        self.cov  = cov
        self.pts  = pts
        self._deterministic = bool(getattr(cov, "is_degenerate", False))

        if self._deterministic:
            self.data = None
            return

        normedpts = [p - p[0] for p in pts]

        # compute minimal circulant‑embedding size in each dim
        pad = minpadding
        if np.isscalar(pad):
            pads = [pad]*len(pts) # if minpadding is scalar, use same in each dim
        else:
            pads = pad
        dims = [
            self.circulant_minsize(cov, normedpts[i], pads[i])
            for i in range(len(pts))
        ]

        # compute the embedded covariance eigenvalues
        Λ = self.circulant_eigvals(cov, normedpts, dims)

        # zero out any small negative λ due to numerical error,
        #    track how many and how negative, then scale by sqrt(λ/M)
        M = np.prod(dims)
        neg_mask = Λ < 0
        n_neg = int(neg_mask.sum())
        λ_min = None
        if n_neg:
            λ_min = float(Λ[neg_mask].min())
            Λ[neg_mask] = 0.0
        pos_mask = ~neg_mask
        Λ[pos_mask] = np.sqrt(Λ[pos_mask] / M)

        if n_neg:
            warnings.warn(
                f"{n_neg} negative eigenvalues ≥ {λ_min:.3e} were set to zero; "
                "consider increasing padding."
            )

        # store data:  v = sqrt(eigs/M)  and  shape_emb = tuple(dims)
        self.data = (Λ, tuple(dims))

    def sample(self):
        """
        Single realization.  xi must be shape == self.data[1].
        """
        if self._deterministic:
            return np.array(self.mean, copy=True)

        Λ, shape_emb = self.data
        xi = np.random.randn(*shape_emb)
        # 1) spectral weights
        w = Λ * xi
        # 2) forward FFT over all dims
        w = fftn(w)
        # 3) crop back to original grid and extract real+imag
        orig_shape = tuple(len(p) for p in self.pts)
        slices = tuple(slice(0, n) for n in orig_shape)
        w_crop = w[slices]
        return self.mean + (w_crop.real + w_crop.imag)

    @staticmethod
    def circulant_minsize(cov, pts, minpadding):
        n = len(pts)
        base = n + minpadding - (1 if cov.is_even else 0)
        return 2 * next_fast_len(base) # SciPy’s next_fast_len returns the smallest ≥ base with 2,3,5 factors

    @staticmethod
    def circulant_eigvals(cov, pts, dims):
        pts = [np.asarray(p) for p in pts]
        starts = [p[0] for p in pts]
        steps = [p[1] - p[0] for p in pts]
        N = len(dims)
        is_even = getattr(cov, 'is_even', False)

        # Vectorized covariance evaluation
        cov_vec = np.vectorize(cov, signature='(n)->()')

        if is_even:
            half_shape = tuple(d//2 + 1 for d in dims)
            coords = [starts[d] + np.arange(half_shape[d]) * steps[d] for d in range(N)]
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            C = cov_vec(mesh)
            C = dctn(C, type=1, norm=None)
            idx = [np.minimum(np.arange(d), d - np.arange(d)) for d in dims]
            mesh_idx = np.ix_(*idx)
            return C[mesh_idx]
        else:
            C = np.zeros(dims, dtype=float)
            dims2 = [d + 2 for d in dims]
            mids  = [d2 // 2 for d2 in dims2]
            grid = [np.arange(d) for d in dims]
            # Build signed offsets
            coords = []
            for d in range(N):
                idx = grid[d]
                pos = np.where(idx < mids[d], idx, idx - dims[d])
                coords.append(starts[d] + pos * steps[d])
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            mask = np.ones(dims, dtype=bool)
            for d in range(N):
                mask &= ((grid[d] + 1) != mids[d]).reshape(
                    *(dims[j] if j == d else 1 for j in range(N))
                )
            C[mask] = cov_vec(mesh[mask])
            return np.real(np.fft.fftn(C))


class PrefbmCov:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_even = True
        self.c0 = None
        self.c2 = None
    def __call__(self, x):
        r = np.linalg.norm(x)
        α = self.alpha
        if 0 < α <= 1.5:
            self.c0 = 1 - α/2
            self.c2 = α/2
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0- r**α + self.c2*r**2
            else:
                return 0.0
        elif 1.5 < α < 2:
            self.c0 = 1 - α/6 - α**2/6
            self.c2 = α*(5+2*α)/18
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**α + self.c2*r**2
            elif r <= 2:
                return α*(2-α)/18 * (2-r)**3 / r
            else:
                return 0.0
        else:
            raise ValueError("α must lie in (0, 2)")


class fBmGenerator:
    """
    Generate one realization of a d‑dimensional fBm field on pts via:
      1) circulant‑embedded GRF (mean zero, PrefbmCov)
      2) additive linear trend ∑ x_i·X_i
      3) rescaling

    Usage:
      fbm = fBmGenerator(c=2.0, alpha=0.5, pts=(x_grid, y_grid), minpadding=128)
      field = fbm.sample()
    """
    def __init__(self, *, c, alpha, num_pt, dim=2, minpadding=0):
        self.c          = c
        self.alpha      = alpha
        self.num_pt     = num_pt
        self.dim        = dim
        self.pts        = (np.arange(num_pt) / num_pt,) * dim
        self.minpadding = minpadding if alpha <= 1.5 else minpadding+num_pt
        self.cov_c0 = None
        self.cov_c2 = None
        self._deterministic = (c == 0)

        if self._deterministic:
            self._prefbm_gen = None
            self._σ_trend = 0.0
            return

        # 1) set up zero‑mean GRF with your Cov class
        mean = np.zeros(tuple(len(p) for p in self.pts))
        cov  = PrefbmCov(alpha)
        self._prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, self.pts, minpadding=self.minpadding)
        self.cov_c0 = self._prefbm_gen.cov.c0
        self.cov_c2 = self._prefbm_gen.cov.c2
        self._σ_trend = np.sqrt(2*self.cov_c2)

    def sample(self):
        if self._deterministic:
            return np.zeros((self.num_pt,) * self.dim)

        prefbm = self._prefbm_gen.sample()

        # build the linear trend ∑ x_i·X_i
        coords = np.meshgrid(*self.pts, indexing="ij")
        Xs     = np.random.normal(scale=self._σ_trend, size=len(coords))
        trend  = sum(coord * X for coord, X in zip(coords, Xs))

        # Add trend and rescale
        fbm_c0_unit = prefbm + trend # Var(Z(x)) = c0 + 2*c2*|x|^2
        origin = tuple([0] * fbm_c0_unit.ndim)
        fbm = np.sqrt(self.c) * self.num_pt ** (self.alpha/2) * (fbm_c0_unit - fbm_c0_unit[origin])

        return fbm


def compute_c(l, H, std_target):
    """
    Compute scale factor c needed so that a 2D fBm field of size l×l
    with Var[X] = 2*c*r^(2H) has marginal std = std_target.
    """
    if std_target == 0:
        return 0.0

    coords = np.arange(l) - (l - 1) / 2
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    r = np.sqrt(X**2 + Y**2)

    # compute average r^(2H)
    avg_r2H = np.mean(r**(2 * H))

    # solve std_target^2 = 2*c*avg_r2H  =>  c = std_target^2 / (2*avg_r2H)
    return std_target**2 / (2 * avg_r2H)


def generate_logk_field(fbm_gen):
    """Generate the logk field using the given fBm generator."""
    start_time_fbm = time.time()
    try:
        logk_field = fbm_gen.sample()
    except Exception as e:
        elapsed_time_fbm = time.time() - start_time_fbm
        return_msg = f'fBm field generated unsuccessfully {np.round(elapsed_time_fbm, 4)} seconds: {e}'
        return None, return_msg

    elapsed_time_fbm = time.time() - start_time_fbm
    return_msg = f'fBm field generated successfully after {np.round(elapsed_time_fbm, 4)} seconds.'
    return logk_field, return_msg


class ExpCov:
    def __init__(self, sigma=1.0, length_scale=1.0):
        self.sigma = sigma
        self.length_scale = length_scale
        self.is_even = True
        self.is_degenerate = (sigma == 0)
    def __call__(self, x):
        r = np.linalg.norm(x)
        return self.sigma**2 * np.exp(-np.abs(r) / self.length_scale)
