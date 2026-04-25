import argparse
import multiprocessing as mp
from multiprocessing import get_context
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmcrameri.cm as cmc

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


def estimate_cov_from_fields(fields, stride=1):
    """
    Compute empirical covariance from pre-generated fields, optionally subsampled.

    fields: (n_real, N, N)
    stride: take every stride-th grid point along each axis for efficiency
    Returns pts (M, 2), Sigma_hat (M, M)
    """
    n_real, N, _ = fields.shape
    sub = fields[:, ::stride, ::stride]   # (n_real, n_sub, n_sub)
    n_sub = sub.shape[1]
    M = n_sub * n_sub

    X = sub.reshape(n_real, M).T          # (M, n_real)
    X -= X.mean(axis=1, keepdims=True)
    Sigma_hat = (X @ X.T) / n_real

    # grid coords consistent with generator convention: arange(N)/N, subsampled
    idx = np.arange(0, N, stride)
    coords = idx / N
    Xg, Yg = np.meshgrid(coords, coords, indexing='ij')
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (M, 2)

    return pts, Sigma_hat


# ─── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRF ensemble statistics test")
    parser.add_argument("--mode", choices=["fbm", "exp"], default="fbm",
                        help="Generator type: 'fbm' (fractional Brownian motion) "
                             "or 'exp' (exponential covariance)")
    parser.add_argument("--num_pt",      type=int,   default=512)
    parser.add_argument("--num_mc",      type=int,   default=10,
                        help="MC realizations (shared by variogram and covariance estimation)")
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
    parser.add_argument("--cov_stride",  type=int,   default=16,
                        help="Subsample stride for covariance estimation; "
                             "covariance grid has (num_pt // cov_stride)^2 points")
    args = parser.parse_args()

    num_pt      = args.num_pt
    num_mc      = args.num_mc
    num_process = min(num_mc, mp.cpu_count())
    cov_stride  = args.cov_stride

    # ── build generator and theoretical variogram/covariance ──────────────────
    if args.mode == "fbm":
        alpha, c = args.alpha, args.c
        gen = fBmGenerator(c=c, alpha=alpha, num_pt=num_pt,
                           minpadding=args.minpadding)
        def gamma_theory_fn(lags):
            return c * lags ** alpha
        theory_label = (fr"Theoretical: $\gamma(h)={c}\,h^{{{alpha}}}$"
                        fr"  (Hurst $= {alpha/2}$)")
        suptitle = (f"fBm Ensemble Statistics  |  "
                    f"Hurst={alpha/2}, c={c}, size={num_pt}, MC={num_mc}")
        filename = f"stats_fbm_hurst{alpha/2}_size{num_pt}.png"
        def sigma_theory_fn(pts):
            diff = pts[:, None, :] - pts[None, :, :]
            r = np.linalg.norm(diff, axis=2)
            p_norm2H = np.linalg.norm(pts, axis=1) ** alpha
            return c * (p_norm2H[:, None] + p_norm2H[None, :] - r ** alpha)
    else:  # exp
        sigma, ell = args.sigma, args.length_scale
        gen = ExpCovGenerator(sigma=sigma, length_scale=ell, num_pt=num_pt,
                              minpadding=args.minpadding)
        def gamma_theory_fn(lags):
            return sigma**2 * (1 - np.exp(-lags / (ell * num_pt)))
        # theory_label = (fr"Theoretical: $\gamma(h)=\sigma^2(1-e^{{-h/\ell}})$"
        #                 fr"  $\sigma={sigma}$, $\ell={ell}$")
        theory_label = "Theoretical"
        suptitle = (f"ExpCov Ensemble Statistics  |  "
                    f"σ={sigma}, ℓ={ell}, size={num_pt}, MC={num_mc}")
        filename = f"stats_exp_sigma{sigma}_l{ell}_size{num_pt}.png"
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

    lags, gamma_iso = empirical_variogram(fields)
    gamma_theory    = gamma_theory_fn(lags)
    lags_normed     = lags / (num_pt - 1)

    # ── covariance estimation (reuses the same MC fields, subsampled) ───────────
    pts_cov, Sigma_hat = estimate_cov_from_fields(fields, stride=cov_stride)
    Sigma_th = sigma_theory_fn(pts_cov)

    # ── errors ────────────────────────────────────────────────────────────────
    std_field = fields.std(axis=0, ddof=1)
    rel_err_mean = np.linalg.norm(mean_field) / np.linalg.norm(std_field)
    rel_err_cov  = np.linalg.norm(Sigma_hat - Sigma_th, 'fro') / np.linalg.norm(Sigma_th, 'fro')
    rel_err_vario = np.linalg.norm(gamma_iso - gamma_theory) / np.linalg.norm(gamma_theory)

    # ── visualization ──────────────────────────────────────────────────────────
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    vmin = min(mean_field.min(), Sigma_hat.min())
    vmax = max(mean_field.max(), Sigma_hat.max())

    im0 = ax1.imshow(mean_field, cmap=cmc.batlowW_r, vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(rf"Ensemble Mean ($\epsilon = {rel_err_mean:.3f}$)")
    div0 = make_axes_locatable(ax1)
    fig.colorbar(im0, cax=div0.append_axes("right", size="5%", pad=0.05))

    im1 = ax2.imshow(Sigma_hat, cmap=cmc.batlowW_r, vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title(rf"Ensemble Covariance ($\epsilon = {rel_err_cov:.3f}$)")
    ax2.set_xlabel(r"Spatial location $\mathbf{s}'$")
    ax2.set_ylabel(r"Spatial location $\mathbf{s}$")
    div1 = make_axes_locatable(ax2)
    fig.colorbar(im1, cax=div1.append_axes("right", size="5%", pad=0.05))

    ax3.plot(lags_normed, gamma_iso, color='#444444', linewidth=3, label="Estimated")
    line, = ax3.plot(lags_normed, gamma_theory, color='#FFFFFF', linewidth=1, linestyle='--', label=theory_label)
    line.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")
    
    ax3.set_xlabel("Normalized lag $h / (N-1)$")
    ax3.set_ylabel(r"$\hat{\gamma}(h)$")
    ax3.set_title(rf"Ensemble Variogram ($\epsilon = {rel_err_vario:.3f}$)")
    ax3.legend()
    ax3.grid(True, which="both", ls="--")

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.show()
