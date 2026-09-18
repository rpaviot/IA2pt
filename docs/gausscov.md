# Gaussian covariance

`ia2pt.gausscov` predicts the Gaussian (disconnected) covariance of the two
data vectors the model can be fitted to:

* `GaussianCov` — the multipoles $[\xi_0, \tilde\xi_{2,2}]$, following
  Grieb et al. 2016 ([arXiv:1509.04293](https://arxiv.org/abs/1509.04293))
  generalised to the spin-2 IA quadrupole;
* `GaussianCovProjected` — the projected statistics $[w_p, w_{g+}]$, with the
  exact finite-$\Pi_{\max}$ line-of-sight window and annulus-averaged Bessel
  kernels.

Each gives the clustering block, the IA block and the clustering × IA cross
block. The survey enters only through its footprint area and the redshift
distribution and weights of the two samples; the number density is allowed to
vary with redshift (integrated shell by shell), and the shape noise is the
per-component ellipticity dispersion of the shape sample.

## Workflow

Four steps, the same for both estimators:

```python
from ia2pt.gausscov import (density_profile, gg_coeffs, gp_coeffs, cross_coeffs,
                            GaussianCov, GaussianCovProjected)

# 1. shell densities of the density sample (g) and the shape sample (s)
ng_w, ng_eff, v = density_profile(z_g, w_g, z_edges, area_sr, chi_of_z)
ns_w, ns_eff, _ = density_profile(z_s, w_s, z_edges, area_sr, chi_of_z)

# 2. per-mode variance coefficients (dV integrals over the shells)
cgg = gg_coeffs(ng_w, ng_eff, v)
cgp = gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sigma_gamma2)
cx  = cross_coeffs(ng_w, ng_eff, ns_w, v)

# 3. the estimator geometry and the P(k, mu) model at each block's z_eff
G = GaussianCov(s_edges, area_sr, kmax=10.0)          # or GaussianCovProjected(rp_edges, area_sr, pimax=100)
pk_gg = G.pk_model(cosmo_ccl, z_eff_clustering, b1, a1)
pk_ia = G.pk_model(cosmo_ccl, z_eff_ia, b1, a1)

# 4. the blocks
cov_00, cov_22, cov_02, cov_combined = G.blocks(pk_gg, pk_ia, cgg, cgp, cx)
```

### Step 1 — `density_profile(z, w, z_edges, area_sr, chi_of_z)`

| argument | meaning |
|---|---|
| `z`, `w` | redshifts and weights of the objects; the weights must be the ones the pair-count estimator uses (e.g. completeness × FKP on the density side, shape weights on the shape side) |
| `z_edges` | shell edges (0.01 wide shells are plenty) |
| `area_sr` | footprint solid angle [sr] |
| `chi_of_z` | callable returning the comoving distance [$h^{-1}$ Mpc] of an array of redshifts |

Returns three arrays over the shells: `nw` $= \sum w / V_{\rm shell}$ (the
pair-weighting density), `neff` $= (\sum w)^2 / \sum w^2 / V_{\rm shell}$
(the shot-noise density; weighted counts raise the shot noise) and the shell
volumes `v` [$(h^{-1}$ Mpc$)^3$].

### Step 2 — coefficient dicts

The per-mode covariance is integrated over the survey with the local FKP-style
weighting, which reduces to scalars multiplying the powers of $P$:

| function | returns | used for |
|---|---|---|
| `gg_coeffs(nw, neff, v)` | `c4, c3, c2, N` — $Q_{gg} = 2[c_4 P_{gg}^2 + 2 c_3 P_{gg} + c_2]/N^2$ | clustering block |
| `gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sig2)` | `dPP, dgn, dsn, dwhite, M` — $Q_{g+} = [d_{PP}(P_{gg}P_{EE} + P_{gE}^2) + d_{gn} P_{gg} + d_{sn} P_{EE} + d_{\rm white}]/M^2$ | IA block |
| `cross_coeffs(ng_w, ng_eff, ns_w, v)` | `x1, x0` — $Q_\times = 2[x_1 P_{gg} P_{gE} + x_0 P_{gE}]/(NM)$ | cross block |

`sig2` is $\sigma_\gamma^2$, the per-component ellipticity variance of the
shape sample weighted like the estimator: $\sum w^2 (e_1^2 + e_2^2) / (2\sum w^2)$.
For a constant density $\bar n$ over a volume $V_s$ these reduce to Grieb's
$[P + 1/\bar n]^2 / V_s$.

### Step 3 — the classes

```python
GaussianCov(s_edges, area_sr, kmax=10.0, n_mu=60)
GaussianCovProjected(rp_edges, area_sr, kmax=10.0, pimax=100.0)
```

| argument | meaning |
|---|---|
| `s_edges` / `rp_edges` | bin edges of the measurement [$h^{-1}$ Mpc], shared by the two statistics |
| `area_sr` | footprint solid angle (stored for bookkeeping) |
| `kmax` | upper limit of the $k$ integrals [$h$ Mpc$^{-1}$]; the white-noise terms are analytic so this only truncates the $P$-dependent ones |
| `n_mu` | Gauss-Legendre nodes of the $\mu$ integral (multipoles) |
| `pimax` | line-of-sight limit of the projected estimator |

`pk_model(cosmo_ccl, z_eff, b1, a1)` builds the $P(k,\mu)$ grids the blocks
need, from a `pyccl.Cosmology`: non-linear $P_{\delta\delta}$ at `z_eff`,
Kaiser $(b_1 + f\mu^2)^2$ on the density side, NLA
$P_{gE} = (b_1 + f\mu^2)\,F\,(1-\mu^2)\,P_{\delta\delta}$ and
$P_{EE} = F^2 (1-\mu^2)^2 P_{\delta\delta}$ with
$F = -a_1\, C_1\rho_{\rm crit}\,\Omega_m / D(z)$. It returns a dict with
`gg`, `gE`, `EE` on the class's grid plus `f` and `D`. The IA signal terms are
negligible next to the shape noise for typical samples, so `a1 = 0` is a safe
default; `b1` matters (it sets the clustering sample variance) and can be
taken from a fit or from `fit_b1`.

### Step 4 — `blocks(pk_gg, pk_ia, cgg, cgp, cx)`

Takes the `pk_model` dicts at the clustering and IA effective redshifts and
the three coefficient dicts, and returns four matrices: the clustering block,
the IA block, the cross block (clustering rows × IA columns) and the combined
$2N \times 2N$ matrix in `[clustering, IA]` order — i.e. `cov_xi0, cov_xi2,
cov_cross, cov_combined` for the multipoles and `cov_wp, cov_wgp, cov_cross,
cov_combined` for the projected statistics.

### Helpers

| function | what it does |
|---|---|
| `jbar(ell, k, s_lo, s_hi)` | volume-averaged $j_\ell$ over the $s$ shells (Grieb eq. 19, closed forms, $\ell = 0, 2$) |
| `Jbar(m, kp, rp_lo, rp_hi)` | area-averaged $J_m$ over the $r_p$ annuli ($m = 0, 2$) |
| `nonlin_pdd(cosmo_ccl, k, z_eff)` | pyccl non-linear $P_{\delta\delta}$ in $(h^{-1}$ Mpc$)^3$ at $k$ in $h$ Mpc$^{-1}$ |
| `kaiser_nla_pk(cosmo_ccl, z_eff, b1, a1, pdd, mu2)` | the Kaiser + NLA factors on arbitrary `pdd`, `mu2` grids |
| `GaussianCov.xi0_template(cosmo_ccl, z_eff)` | bin-averaged $\xi_0$ of $P_{\delta\delta}$ ($b = 1$, no RSD) and $f$ |
| `fit_b1(xi0_meas, var, template, f, s_mid, s_range)` | closed-form $b_1$ from the Kaiser amplitude $b_1^2 + 2b_1 f/3 + f^2/5$ of that template fitted to a measured $\xi_0$ |

## Formalism in brief

Multipoles (Grieb et al. 2016 eqs. 16–19):

$$C_{\ell_1\ell_2}(s_i, s_j) = \frac{i^{\ell_1+\ell_2}}{2\pi^2}\int dk\,k^2\,
\sigma^2_{\ell_1\ell_2}(k)\,\bar j_{\ell_1}(k s_i)\,\bar j_{\ell_2}(k s_j),
\qquad
\sigma^2_{\ell_1\ell_2}(k) = \frac{1}{2}\int_{-1}^{1} d\mu\,
c_{\ell_1}(\mu)\,c_{\ell_2}(\mu)\,Q(k,\mu),$$

with $c_0 = 1$ for the monopole and, because the IA estimator is the
associated-Legendre spin-2 multipole
$\tilde\xi_{2,2} = \tfrac{5}{2}\tfrac{1}{24}\int d\mu\,3(1-\mu^2)\,\xi_{g+}$,
$c_2 = \tfrac{5}{8}(1-\mu^2)$ with $\bar j_2$. The per-mode covariances are
$2[P_{gg} + 1/\bar n_g]^2$, $[P_{gg} + 1/\bar n_g][P_{EE} + \sigma_\gamma^2/\bar n_s] + P_{gE}^2$
and $2[P_{gg} + 1/\bar n_g]\,P_{gE}$ (with $i^2 = -1$ on the cross block).

Projected statistics: writing $w_A(r_p) = \int_{-\Pi}^{\Pi} d\pi\,\xi_A$ in
Fourier space factorises the covariance into a transverse Hankel pair and a
line-of-sight integral,

$$C_{AB}(r_{p,i}, r_{p,j}) = \pm\int_0^\infty \frac{k_\perp dk_\perp}{2\pi}\,
\bar J_a(k_\perp; i)\,\bar J_b(k_\perp; j)\int\frac{dk_z}{2\pi}\,W(k_z)^2\,Q_{AB}(k,\mu),$$

with $W(k_z) = 2\sin(k_z\Pi)/k_z$, $\bar J_0$ for the density side and
$-\bar J_2$ for the shear side (hence the $-$ sign of the cross block). White
terms are analytic in both cases ($\int k^2 \bar j_\ell^2 dk = 2\pi^2/V_{\rm bin}$,
$\int k_\perp \bar J_m^2 dk_\perp / 2\pi = 1/A_{\rm bin}$), so they carry no
$k$-truncation error.

## Validation

`tests/test_gausscov.py` checks the closure identities of both kernels, the
shot-noise-only analytic limit of every block, a `fit_b1` round trip and a
synthetic characterisation fixture. For the LRG sample of the UNIONS × DESI
analysis the Gaussian errors are 0.65–0.85× the jackknife ones (medians over
the bins; the remainder is the non-Gaussian and patch-scale part of the
jackknife) and the cross-block sign matches the jackknife.
