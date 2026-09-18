# Model

`ia2pt.TwoPointModel` turns a cosmology, an IA model (NLA or TATT) and a set
of bias / alignment parameters into two-point statistics. Under the hood it
uses pyccl for the non-linear matter power spectrum (CAMB + HMCode-2020) and
pyccl's `EulerianPTCalculator` (FAST-PT) for the one-loop bias and TATT
spectra $P_{gg}(k)$, $P_{gI}(k)$, $P_{EE}(k)$, $P_{BB}(k)$.

Everything downstream of the spectra is done in configuration space via
Hankel transforms: the *signed multipoles* $\xi^{\ell,s}(r)$ of each
correlation are computed once, and every statistic below is an angular
integral of the reconstructed $\xi(r, \mu)$. A multipole fit and a projected
fit are therefore the same physical model.

## Building a model

```python
TwoPointModel(cosmology, config, do_rsd=True, pimax=100, dpi=0.1,
              bin_avg=False, bin_factor=20, evolve_bias=False,
              rp_min_wedge=5.0, n_mu_wedge=101)
```

| argument | meaning |
|---|---|
| `cosmology` | `[Omega_c, Omega_b, sum m_nu (eV), A_s, n_s, h, z_eff]`; `z_eff` is the redshift at which single-redshift predictions are made |
| `config` | `'NLA'` or `'TATT'` |
| `do_rsd` | include the Kaiser redshift-space distortions ($\beta = f/b_1$) in the multipoles and projections; `False` gives real-space statistics |
| `pimax`, `dpi` | line-of-sight integration limit and step of the projected statistics [$h^{-1}$ Mpc] |
| `bin_avg` | if `True`, every statistic takes bin **edges** and returns bin averages (annular for $w$, spherical for the multipoles) instead of point values at the given separations |
| `bin_factor` | density of the internal integration grids (`n_bins * bin_factor` points) |
| `evolve_bias` | $b_1(z) = b_1 D(z_{\rm ref}) / D(z)$ inside the PT tracer |
| `rp_min_wedge` | minimum $r_p$ of the wedge cut applied to the multipoles [$h^{-1}$ Mpc]; `0` / `None` disables it |
| `n_mu_wedge` | number of $\mu$ nodes of the wedge integral |

### Redshift integration

By default predictions are made at the single redshift `z_eff`. Calling

```python
model.set_nz(zedge, z, dn1, dn2)
```

switches to $n(z)$-integrated predictions: `dn1` is the density-tracer
$n(z)$ and `dn2` the shape-sample $n(z)$, both tabulated at `z`; `zedge` sets
the integration range. Each statistic is then averaged over the Gauss-Legendre
nodes with the window $W(z) \propto n_i(z)\,n_j(z) / (\chi^2\, d\chi/dz)$ of
the pair of samples it correlates ($n_1 n_1$ for clustering, $n_1 n_2$ for
galaxy–shape, $n_2 n_2$ for shape–shape).

To model the IA block at a different effective redshift than the clustering
block, build a second `TwoPointModel` with that `z_eff` and use its methods for
the IA statistics.

## Statistics

All statistics share the same signature,

```python
model.compute_<statistic>(r, b1, b2, a1, a2, bTA)
```

| argument | meaning |
|---|---|
| `r` | separations [$h^{-1}$ Mpc]: points (default) or bin edges (`bin_avg=True`); $s$ for the multipoles, $r_p$ for the projected statistics |
| `b1`, `b2` | linear and second-order bias of the density tracer; in TATT the tidal and third-order biases follow $b_s = -\tfrac{4}{7}(b_1-1)$, $b_{3nl} = b_1 - 1$ (zero in NLA) |
| `a1` | linear alignment amplitude ($A_{\rm IA}$ of the NLA model, pyccl `translate_IA_norm` normalisation) |
| `a2` | tidal-torquing amplitude (TATT only; ignored in NLA) |
| `bTA` | TATT density weighting, $a_{1\delta} = a_1\, b_{\rm TA}$ (TATT only) |

and return a 1D array of the same length as the points (or `len(r) - 1` bin
averages).

| method | returns | what it is |
|---|---|---|
| `compute_xi_gg_wedge_monopole` | $\tilde\xi_{0,0}(s)$ | clustering monopole of $\xi_{gg}(s,\mu)$ restricted to $r_p \ge$ `rp_min_wedge` |
| `compute_xi_gi_wedge_quadrupole` | $\tilde\xi_{2,2}(s)$ | galaxy–shape quadrupole with the spin-2 associated-Legendre filter $L_{2,2} \propto 1-\mu^2$, same wedge |
| `compute_wgg` | $w_{gg}(r_p)$ | projected clustering, $2\int_0^{\Pi_{\max}} d\Pi\, \xi_{gg}(r_p, \Pi)$ |
| `compute_wgp` | $w_{g+}(r_p)$ | projected galaxy–shape correlation, same integral of $\xi_{g+}$ |
| `compute_xi_pp_wedge_monopole` | $\tilde\xi^{++}_{0,0}(s)$ | shape–shape monopole, same wedge |
| `compute_xi_pp_wedge_hexadecapole` | $\tilde\xi^{++}_{4,4}(s)$ | shape–shape hexadecapole with the spin-4 filter $L_{4,4} \propto (1-\mu^2)^2$ (Singh et al. 2023, $s_{ab}=4$) |
| `compute_wpp` | $w_{++}(r_p)$ | projected shape–shape correlation |
| `compute_wxx` | $w_{\times\times}(r_p)$ | projected cross-shape correlation |

The wedged multipoles are eq. 18 of Singh et al. 2023,

$$\tilde\xi_{\ell,s}(r) = \frac{2\ell+1}{2}\frac{(\ell-s)!}{(\ell+s)!}
\int_{-1}^{1} d\mu\, \Theta(r_p \ge r_{p,\min})\, L_{\ell,s}(\mu)\, \xi(r,\mu),$$

with $L_{\ell,s}$ the associated Legendre polynomials. The projected
statistics are eq. 19: the same $\xi(r,\mu)$, integrated along the line of
sight up to $\Pi_{\max}$ — not a direct Hankel transform of $P(k)$, so the
finite $\Pi_{\max}$ and the residual RSD are in.

### How $\xi(r, \mu)$ is built

Each correlation is written as a sum of signed multipoles $\xi^{\ell,s}(r)$
(Hankel transforms of the relevant $P(k)$) times $L_{\ell,s}(\mu)$:

* clustering: $\ell = 0, 2, 4$, spin 0, with the Kaiser coefficients
  $\alpha_\ell(\beta)$ of Singh et al. 2023 eqs. 13–15;
* galaxy–shape: $\ell = 2, 4$, spin 2, coefficients of eqs. 16–17 — the
  $(1-\mu^2)$ sky-projection factor of $P_{gE}(k,\mu)$ enters exactly through
  the associated Legendre expansion;
* shape–shape: with $e_+$ measured along the projected separation, the
  Fourier kernel is $P_{EE}\cos^2 2\Delta\phi + P_{BB}\sin^2 2\Delta\phi$,
  both spectra carrying the sky-projection factor $(1-\mu_k^2)^2$. The
  addition theorem gives

  $$\xi_{++}(r,\mu) = \sum_{\ell=0,2,4} (-1)^{\ell/2}\, \xi_A^{\ell,0}(r)\, L_{\ell,0}(\mu)
  + \xi_B^{4,4}(r)\, L_{4,4}(\mu),$$

  with $\xi_A^{\ell,0}$ the Hankel transforms of $\alpha_\ell P_A$,
  $P_A = (P_{EE}+P_{BB})/2$, $\alpha = (8/15, -16/21, 8/35)$ the Legendre
  coefficients of $(1-\mu^2)^2$, and $\xi_B^{4,4} = \tfrac{1}{105}\,
  \mathcal{H}_4[P_B]$, $P_B = (P_{EE}-P_{BB})/2$, $L_{4,4} = 105(1-\mu^2)^2$.
  $\xi_{\times\times}$ is the same with $P_B \to -P_B$. There is no Kaiser
  factor (neither side is a density field). In the $\Pi_{\max}\to\infty$
  limit $w_{++}$ reduces to the familiar Limber form
  $\tfrac{1}{2\pi}\int k\,dk\,[P_A J_0 + P_B J_4]$, which the test suite
  checks.

The $(1-\mu^2)^2$ factor is exact for the linear-alignment part of
$P_{EE}$ and is an approximation for the TATT one-loop terms, which FAST-PT
computes at $\mu = 0$; see [Conventions](conventions.md).

### Cost

The PT spectra are rebuilt only when the parameters change (the class caches
them); the E/B-mode spectra are built only when a shape–shape statistic is
requested, so galaxy–shape fits pay nothing for them. Typical single-redshift
evaluations take a few tens of milliseconds, $n(z)$-integrated ones ~20× more.
