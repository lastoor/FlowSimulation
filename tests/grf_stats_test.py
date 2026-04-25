import argparse
import multiprocessing as mp
from multiprocessing import get_context
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
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
    """Return (lags, gamma_iso) using FFT autocorrelation; O(K·N²·logN) vs O(K·N³)."""
    K, N, _ = fields.shape
    if max_lag is None:
        max_lag = N - 1
    lags = np.arange(1, max_lag + 1)

    f_sq     = fields ** 2
    sq_total = f_sq.sum()

    # ── direction 0 (axis 1) ──────────────────────────────────────────────────
    # Σ_{k,j,i=0}^{N-h-1} (f[k,i+h,j] - f[k,i,j])²
    #   = sq_sum_0(h) - 2 * corr_0(h)
    # sq_sum_0(h) = Σ_{i=h}^{N-1} f²(i) + Σ_{i=0}^{N-h-1} f²(i)  (summed over k,j)
    # corr_0(h)   = Σ_{k,j,i=0}^{N-h-1} f[k,i,j]*f[k,i+h,j]  ← linear autocorr via FFT
    sq_cs_0  = np.cumsum(f_sq.sum(axis=(0, 2)))          # (N,)
    sq_sum_0 = sq_cs_0[N - lags - 1] + sq_total - sq_cs_0[lags - 1]
    F0 = np.fft.rfft(fields, n=2 * N, axis=1)            # zero-pad → linear autocorr
    A0 = np.fft.irfft(F0 * F0.conj(), n=2 * N, axis=1)
    del F0
    SSQ_0 = sq_sum_0 - 2 * A0[:, lags, :].sum(axis=(0, 2))
    del A0

    # ── direction 1 (axis 2) ──────────────────────────────────────────────────
    sq_cs_1  = np.cumsum(f_sq.sum(axis=(0, 1)))          # (N,)
    sq_sum_1 = sq_cs_1[N - lags - 1] + sq_total - sq_cs_1[lags - 1]
    F1 = np.fft.rfft(fields, n=2 * N, axis=2)
    A1 = np.fft.irfft(F1 * F1.conj(), n=2 * N, axis=2)
    del F1
    SSQ_1 = sq_sum_1 - 2 * A1[:, :, lags].sum(axis=(0, 1))
    del A1

    # Matheron estimator: mean squared increment / 2, pooled over both directions
    n_pairs   = 2 * K * (N - lags) * N
    gamma_iso = (SSQ_0 + SSQ_1) / (2 * n_pairs)

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

def configure_style():
    import glob
    import matplotlib.font_manager as fm

    family: str = "CMU Sans Serif"
    mathtext_fontset: str = "cm"
    unicode_minus: bool = False
    cmu_font_glob: list = ["/home1/binhaoli/.local/share/fonts/cmun*.ttf", "/Users/binhaoli/Library/Fonts/cmun*.ttf"]

    for paths in cmu_font_glob:
        for path in glob.glob(paths):
            fm.fontManager.addfont(path)
            
    plt.rcParams["font.family"] = family
    plt.rcParams["mathtext.fontset"] = mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = unicode_minus

    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12

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

    configure_style()

    # ── build generator and theoretical variogram/covariance ──────────────────
    if args.mode == "fbm":
        alpha, c = args.alpha, args.c
        gen = fBmGenerator(c=c, alpha=alpha, num_pt=num_pt,
                           minpadding=args.minpadding)
        def gamma_theory_fn(lags):
            return c * lags ** alpha
        theory_label = "Theoretical"
        # theory_label = (fr"Theoretical: $\gamma(h)={c}\,h^{{{alpha}}}$"
        #                 fr"  (Hurst $= {alpha/2}$)")
        suptitle = (f"fBm Ensemble Statistics  |  "
                    f"Hurst={alpha/2}, c={c}, size={num_pt}, MC={num_mc}")
        filename = f"stats_fbm_c{c}_hurst{alpha/2}_size{num_pt}_mc{num_mc}_stride{cov_stride}.png"
        def sigma_theory_fn(pts):
            diff = pts[:, None, :] - pts[None, :, :]
            r = np.linalg.norm(diff, axis=2)
            p_norm2H = np.linalg.norm(pts, axis=1) ** alpha
            return c * num_pt**alpha * (p_norm2H[:, None] + p_norm2H[None, :] - r ** alpha)
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
        filename = f"stats_exp_sigma{sigma}_l{ell}_size{num_pt}_mc{num_mc}_stride{cov_stride}.png"
        def sigma_theory_fn(pts):
            diff = pts[:, None, :] - pts[None, :, :]
            r    = np.linalg.norm(diff, axis=2)
            return sigma**2 * np.exp(-r / ell)

    # ── covariance grid (computed once, independent of realizations) ──────────
    N          = num_pt
    cov_idx    = np.arange(0, N, cov_stride)
    cov_coords = cov_idx / N
    Xg, Yg    = np.meshgrid(cov_coords, cov_coords, indexing='ij')
    pts_cov    = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (M, 2)
    M          = len(cov_idx) ** 2
    lags       = np.arange(1, N)

    # ── online accumulators — O(N²) memory, no full ensemble stored ───────────
    acc_field    = np.zeros((N, N))
    acc_field_sq = np.zeros((N, N))
    acc_sq_sum_0 = np.zeros(N - 1)
    acc_corr_0   = np.zeros(N - 1)
    acc_sq_sum_1 = np.zeros(N - 1)
    acc_corr_1   = np.zeros(N - 1)
    acc_x        = np.zeros(M)
    acc_outer    = np.zeros((M, M))

    # ── parallel sampling with online accumulation ────────────────────────────
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_process,
                  initializer=init_worker,
                  initargs=(gen,)) as pool:
        for f_k in pool.imap(_sample, range(num_mc)):
            acc_field    += f_k
            acc_field_sq += f_k ** 2

            # variogram: per-realization FFT autocorrelation
            fk2  = f_k[np.newaxis]          # (1, N, N) — axis-aware rfft
            f_sq = f_k ** 2
            sq_t = f_sq.sum()

            sq_cs_0       = np.cumsum(f_sq.sum(axis=1))
            acc_sq_sum_0 += sq_cs_0[N - lags - 1] + sq_t - sq_cs_0[lags - 1]
            F0            = np.fft.rfft(fk2, n=2 * N, axis=1)
            corr_0        = np.fft.irfft(F0 * F0.conj(), n=2 * N, axis=1)[0]
            acc_corr_0   += np.take(corr_0, lags, axis=0).sum(axis=1)
            del F0

            sq_cs_1       = np.cumsum(f_sq.sum(axis=0))
            acc_sq_sum_1 += sq_cs_1[N - lags - 1] + sq_t - sq_cs_1[lags - 1]
            F1            = np.fft.rfft(fk2, n=2 * N, axis=2)
            corr_1        = np.fft.irfft(F1 * F1.conj(), n=2 * N, axis=2)[0]
            acc_corr_1   += np.take(corr_1, lags, axis=1).sum(axis=0)
            del F1

            # covariance: rank-1 outer-product update
            x_k        = f_k[::cov_stride, ::cov_stride].ravel()
            acc_x     += x_k
            acc_outer += np.outer(x_k, x_k)

            del f_k, fk2, f_sq, x_k

    # ── finalize statistics ───────────────────────────────────────────────────
    mean_field = acc_field / num_mc
    std_field  = np.sqrt(np.maximum(
        (acc_field_sq / num_mc - mean_field ** 2) * num_mc / (num_mc - 1), 0.0
    ))

    SSQ_0        = acc_sq_sum_0 - 2 * acc_corr_0
    SSQ_1        = acc_sq_sum_1 - 2 * acc_corr_1
    n_pairs      = 2 * num_mc * (N - lags) * N
    gamma_iso    = (SSQ_0 + SSQ_1) / (2 * n_pairs)
    gamma_theory = gamma_theory_fn(lags)

    mean_x    = acc_x / num_mc
    Sigma_hat = acc_outer / num_mc - np.outer(mean_x, mean_x)
    Sigma_th  = sigma_theory_fn(pts_cov)

    # ── errors ────────────────────────────────────────────────────────────────
    rel_err_mean  = np.linalg.norm(mean_field) / np.linalg.norm(std_field)
    rel_err_cov   = np.linalg.norm(Sigma_hat - Sigma_th, 'fro') / np.linalg.norm(Sigma_th, 'fro')
    rel_err_vario = np.linalg.norm(gamma_iso - gamma_theory) / np.linalg.norm(gamma_theory)

    # ── visualization ──────────────────────────────────────────────────────────
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    vmin = min(mean_field.min(), Sigma_hat.min())
    vmax = max(mean_field.max(), Sigma_hat.max())

    im0 = ax1.imshow(mean_field, cmap=cmc.batlow_r, vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(rf"Ensemble Mean ($\epsilon = {rel_err_mean:.3f}$)")
    ax1.set_xlabel(r"$x / \delta$")
    ax1.set_ylabel(r"$y / \delta$")
    div0 = make_axes_locatable(ax1)
    fig.colorbar(im0, cax=div0.append_axes("right", size="5%", pad=0.05))
    ax1.set_aspect('equal')

    im1 = ax2.imshow(Sigma_hat, cmap=cmc.batlow_r, vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_title(rf"Ensemble Covariance ($\epsilon = {rel_err_cov:.3f}$)")
    ax2.set_xlabel(r"$s / \delta$")
    ax2.set_ylabel(r"$s' / \delta$")
    div1 = make_axes_locatable(ax2)
    fig.colorbar(im1, cax=div1.append_axes("right", size="5%", pad=0.05))
    ax2.set_aspect('equal')

    ax3.plot(lags, gamma_theory, color='#000000', linewidth=3, linestyle='-', label=theory_label)
    line, = ax3.plot(lags, gamma_iso, color='#FFFFFF', linewidth=1, linestyle=':', label="Numerical")
    line.set_path_effects([pe.Stroke(linewidth=2, foreground='#444444'), pe.Normal()])
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")
    div2 = make_axes_locatable(ax3)
    div2.append_axes("right", size="5%", pad=0.05).set_visible(False)

    ax3.set_xlabel(r"$r / \delta$")
    ax3.set_ylabel(r"$\gamma(r / \delta)$")
    ax3.set_title(rf"Ensemble Variogram ($\epsilon = {rel_err_vario:.3f}$)")
    ax3.legend()
    ax3.grid(True, which="both", ls="--")
    ax3.set_aspect('auto')

    # plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.show()
