import argparse
import multiprocessing as mp
from multiprocessing import get_context
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.grf_generation import fBmGenerator, GaussianRandomFieldCircEmbed, ExpCov


# ─── ExpCov wrapper ────────────────────────────────────────────────────────────
class ExpCovGenerator:
    """Thin wrapper around GaussianRandomFieldCircEmbed with ExpCov, matching the
    fBmGenerator interface (.pts, .sample())."""
    def __init__(self, *, sigma, length_scale, num_pt, dim=2, minpadding=0):
        self.sigma        = sigma
        self.length_scale = length_scale
        self.num_pt       = num_pt
        self.dim          = dim
        # pts in [0, 1) with spacing 1/num_pt — same convention as fBmGenerator
        self.pts = (np.arange(num_pt) / num_pt,) * dim
        mean = np.zeros(tuple(len(p) for p in self.pts))
        cov  = ExpCov(sigma=sigma, length_scale=length_scale)
        self._grf = GaussianRandomFieldCircEmbed(mean, cov, self.pts,
                                                 minpadding=minpadding)

    def sample(self):
        return self._grf.sample()


# ─── worker shim ───────────────────────────────────────────────────────────────
_gen = None

def init_worker(gen):
    global _gen
    _gen = gen

def _sample(_i):
    return _gen.sample()


# ─── empirical isotropic variogram ─────────────────────────────────────────────
def empirical_variogram(fields, max_lag=None):
    """Return (lags, gamma_iso) using all axis directions for isotropy."""
    num_pt = fields.shape[1]
    if max_lag is None:
        max_lag = num_pt - 1
    lags = np.arange(1, max_lag + 1)
    gamma_iso = np.zeros(len(lags), dtype=float)
    for idx, h in enumerate(lags):
        diffs_all = []
        for d in range(fields.ndim - 1):   # skip ensemble axis 0
            s1 = [slice(None)] * fields.ndim
            s2 = [slice(None)] * fields.ndim
            s1[d + 1] = slice(0, -h)
            s2[d + 1] = slice(h, None)
            diffs_all.append((fields[tuple(s1)] - fields[tuple(s2)]).ravel())
        gamma_iso[idx] = np.var(np.concatenate(diffs_all), ddof=1) / 2
    return lags, gamma_iso

def estimate_cov_full(gen, n_real=2000):
    """
    gen.sample(): returns a (N,N) field from your fBmGenerator.
    n_real: number of realizations for Monte Carlo.
    
    Returns:
        pts: (M, 2) array of grid coordinates
        Sigma_hat: (M, M) empirical covariance matrix
    """
    Z0 = gen.sample()
    N = Z0.shape[0]
    M = N * N

    # stack flattened realizations column-wise
    X = np.empty((M, n_real), dtype=float)
    X[:, 0] = Z0.ravel()

    for k in range(1, n_real):
        Z = gen.sample()
        X[:, k] = Z.ravel()

    # subtract mean at each point (row-wise)
    X -= X.mean(axis=1, keepdims=True)

    # empirical covariance Σ̂ ≈ E[Z Z^T]
    Sigma_hat = (X @ X.T) / n_real  # shape (M, M)

    # construct coordinates corresponding to each index
    N = Z0.shape[0]
    x = np.arange(N) / N
    y = np.arange(N) / N
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # shape (M,2)

    return pts, Sigma_hat

def fBs_cov_matrix(H, L, N, c):
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (M,2)
    M = N * N

    diff = pts[:, None, :] - pts[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    p_norm2H = np.linalg.norm(pts, axis=1)**(2*H)

    Sigma_th = c * (p_norm2H[:, None] + p_norm2H[None, :] - r**(2*H))
    return Sigma_th

# ─── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRF ensemble statistics test")
    parser.add_argument("--mode", choices=["fbm", "exp"], default="fbm",
                        help="Generator type: 'fbm' (fractional Brownian motion) "
                             "or 'exp' (exponential covariance)")
    parser.add_argument("--num_pt",      type=int,   default=512)
    parser.add_argument("--num_mc",      type=int,   default=10)
    parser.add_argument("--minpadding",  type=int,   default=0)
    # fBm params
    parser.add_argument("--alpha",       type=float, default=0.1,
                        help="fBm: variogram exponent (Hurst = alpha/2)")
    parser.add_argument("--c",           type=float, default=1.0,
                        help="fBm: amplitude scale")
    # ExpCov params
    parser.add_argument("--sigma",       type=float, default=1.0,
                        help="ExpCov: marginal std")
    parser.add_argument("--length_scale",type=float, default=0.1,
                        help="ExpCov: correlation length in [0,1] physical units")
    # covariance comparison params
    parser.add_argument("--cov_num_pt",  type=int,   default=32,
                        help="Grid size for covariance comparison (matrix is cov_num_pt^2 x cov_num_pt^2)")
    parser.add_argument("--n_real",      type=int,   default=500,
                        help="MC realizations for estimate_cov_full")
    args = parser.parse_args()

    num_pt      = args.num_pt
    num_mc      = args.num_mc
    num_process = min(num_mc, mp.cpu_count())

    # ── build generator and theoretical variogram ──────────────────────────────
    if args.mode == "fbm":
        alpha, c = args.alpha, args.c
        gen = fBmGenerator(c=c, alpha=alpha, num_pt=num_pt,
                           minpadding=args.minpadding)
        # variogram in grid-point lags: γ(h) = c·h^α  (fBmGenerator rescales by
        # num_pt^(α/2) so the grid-unit variogram has this exact form)
        def gamma_theory_fn(lags):
            return c * lags ** alpha
        theory_label = (fr"Theoretical: $\gamma(h)={c}\,h^{{{alpha}}}$"
                        fr"  (Hurst $= {alpha/2}$)")
        suptitle = (f"fBm Ensemble Statistics  |  "
                    f"Hurst={alpha/2}, c={c}, size={num_pt}, MC={num_mc}")
        filename = f"stats_fbm_hurst{alpha/2}_size{num_pt}.pdf"
        cov_num_pt = args.cov_num_pt
        small_gen  = fBmGenerator(c=c, alpha=alpha, num_pt=cov_num_pt,
                                   minpadding=args.minpadding)
        def sigma_theory_fn(pts):
            return fBs_cov_matrix(H=alpha/2, L=1.0, N=cov_num_pt, c=c)

    else:  # exp
        sigma, ell = args.sigma, args.length_scale
        gen = ExpCovGenerator(sigma=sigma, length_scale=ell, num_pt=num_pt,
                              minpadding=args.minpadding)
        # physical lag for h grid steps = h / num_pt
        def gamma_theory_fn(lags):
            return sigma**2 * (1 - np.exp(-lags / (ell * num_pt)))
        theory_label = (fr"Theoretical: $\gamma(h)=\sigma^2(1-e^{{-h/\ell}})$"
                        fr"  $\sigma={sigma}$, $\ell={ell}$")
        suptitle = (f"ExpCov Ensemble Statistics  |  "
                    f"σ={sigma}, ℓ={ell}, size={num_pt}, MC={num_mc}")
        filename = f"stats_exp_sigma{sigma}_l{ell}_size{num_pt}.pdf"
        cov_num_pt = args.cov_num_pt
        small_gen  = ExpCovGenerator(sigma=sigma, length_scale=ell, num_pt=cov_num_pt,
                                     minpadding=args.minpadding)
        def sigma_theory_fn(pts):
            diff = pts[:, None, :] - pts[None, :, :]
            r    = np.linalg.norm(diff, axis=2)
            return sigma**2 * np.exp(-r / ell)

    # ── parallel sampling ──────────────────────────────────────────────────────
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_process,
                  initializer=init_worker,
                  initargs=(gen,)) as pool:
        fields_list = pool.map(_sample, range(num_mc))

    orig_shape = tuple(len(p) for p in gen.pts)
    fields = np.stack(fields_list, axis=0).reshape((num_mc, *orig_shape))

    # ── statistics ─────────────────────────────────────────────────────────────
    mean_field = fields.mean(axis=0)
    var_field  = fields.var(axis=0, ddof=1)

    lags, gamma_iso = empirical_variogram(fields)
    gamma_theory    = gamma_theory_fn(lags)
    lags_normed     = lags / (num_pt - 1)

    # ── covariance comparison ──────────────────────────────────────────────────
    pts_cov, Sigma_hat = estimate_cov_full(small_gen, n_real=args.n_real)
    Sigma_th  = sigma_theory_fn(pts_cov)
    cov_vmin  = min(Sigma_hat.min(), Sigma_th.min())
    cov_vmax  = max(Sigma_hat.max(), Sigma_th.max())

    # ── visualization ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 12))
    gs  = gridspec.GridSpec(2, 3, height_ratios=[1, 3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    im0 = ax1.imshow(Sigma_hat, cmap='RdBu_r', vmin=cov_vmin, vmax=cov_vmax)
    ax1.set_title(r"Estimated Cov $\hat{\Sigma}$")
    fig.colorbar(im0, ax=ax1, fraction=0.046)

    im1 = ax2.imshow(mean_field, cmap='viridis')
    ax2.set_title("Ensemble Mean")
    fig.colorbar(im1, ax=ax2, fraction=0.046)

    im2 = ax3.imshow(Sigma_th, cmap='RdBu_r', vmin=cov_vmin, vmax=cov_vmax)
    ax3.set_title(r"Theoretical Cov $\Sigma$")
    fig.colorbar(im2, ax=ax3, fraction=0.046)

    ax4.loglog(lags_normed, gamma_iso,    'o-',  label="Empirical (isotropic)")
    ax4.loglog(lags_normed, gamma_theory, 'k--', label=theory_label)
    ax4.set_xlabel("Normalized lag $h / (N-1)$")
    ax4.set_ylabel(r"$\hat{\gamma}(h)$")
    ax4.set_title("Empirical vs Theoretical Variogram")
    ax4.legend()
    ax4.grid(True, which="both", ls="--")

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format='pdf')
    print(f"Saved {filename}")
    plt.show()
