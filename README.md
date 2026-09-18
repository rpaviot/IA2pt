# ia2pt

NLA / TATT two-point intrinsic-alignment model, likelihood and fit drivers —
the modelling layer of the UNIONS × DESI intrinsic-alignment analysis
(Paviot et al.).

## What it computes

`TwoPointModel` predicts, from the pyccl 1-loop perturbation-theory spectra
(CAMB + HMCode-2020 matter power, `EulerianPTCalculator` / FAST-PT for the bias
and TATT terms):

| method | statistic | notes |
|---|---|---|
| `compute_xi_gg_wedge_monopole(s, ...)` | ξ̃₀,₀(s) | clustering monopole of ξ(s, μ) restricted to r_p ≥ `rp_min_wedge` |
| `compute_xi_gi_wedge_quadrupole(s, ...)` | ξ̃₂,₂(s) | galaxy–shape (spin-2) quadrupole, same wedge |
| `compute_wgg(rp, ...)` | w_gg(r_p) | Singh et al. (2023) eq. 19: Π-integral of the signed multipoles up to `pimax` |
| `compute_wgp(rp, ...)` | w_g+(r_p) | same, from ξ^{2,2} and ξ^{4,2} |
| `compute_xi_pp_wedge_monopole(s, ...)` | ξ̃₀,₀^{++}(s) | shape–shape monopole, same wedge |
| `compute_xi_pp_wedge_hexadecapole(s, ...)` | ξ̃₄,₄^{++}(s) | shape–shape spin-4 filter (Singh et al. 2023, s_ab = 4) |
| `compute_wpp(rp, ...)` / `compute_wxx(rp, ...)` | w_++(r_p), w_××(r_p) | eq. 19 on the ++ multipoles |

All share the signature `(r, b1, b2, a1, a2, bTA)` and the same
redshift-space multipoles (Kaiser β = f / b1 through the α_ℓ coefficients of
Singh et al. 2023, eqs. 13–17), so a multipole fit and a projected fit are the
same physical model.

The shape–shape statistics use the E/B-mode spectra of pyccl. Their
line-of-sight dependence is exact only for the linear-alignment term,
(1 − μ²)²; the TATT one-loop terms come out of FAST-PT at μ = 0 and are given
the same factor (Maion et al. 2024, arXiv:2307.13754, sect. 3.2.3). The
expansion of ξ₊₊(r, μ) into Hankel transforms is documented in
`TwoPointModel._multipoles_pp` and checked against the Limber limit in the tests. Predictions can be made at a single effective redshift
(the constructor's `z_eff`) or integrated over the density × shape window built
from two n(z) (`set_nz`), as point values or as bin averages between the given
edges (`bin_avg=True`: annular for w, spherical for the multipoles).

The projected statistics are **not** direct Hankel transforms of P(k): the
finite Π_max and the RSD are in.

Conventions: h⁻¹ Mpc throughout; `a1, a2` in the pyccl `translate_IA_norm`
normalisation (a1 = A_IA of the NLA model); `bTA` the TATT density weighting;
in TATT b_s = −4/7 (b1 − 1) and b_3nl = b1 − 1 (zero in NLA). e_+ is positive
for radial alignment, so a positive a1 gives positive ξ̃₂,₂ **and** positive
w_g+; a measurement built on the lensing tangential shear (e.g. treecorr NG)
has the opposite sign and must be flipped before fitting.

`IALikelihood` stacks data vectors + joint covariance, applies per-statistic
radial cuts, optional Hartlap correction / covariance tapering / zeroing of the
cross-statistic blocks, and exposes χ² for iminuit and `log_like` for nautilus
and emcee. `run_minuit`, `run_nautilus`, `run_emcee` drive it.

`ia2pt.gausscov` computes the Gaussian covariance of both data vectors —
`GaussianCov` for [ξ₀, ξ̃₂,₂] (Grieb et al. 2016 generalised to the spin-2
quadrupole) and `GaussianCovProjected` for [w_p, w_g+] (exact finite-Π_max
window, annulus-averaged Bessel kernels) — including the clustering × IA cross
block, with the varying number density and the shape noise integrated over the
survey's redshift shells.

## Documentation

https://ia2pt.readthedocs.io/en/latest/ — the model statistics, the fitting pipeline and
the Gaussian covariance, each with the arguments and outputs of every public
function.

## Install

```
git clone https://github.com/rpaviot/IA2pt.git && cd IA2pt
pip install -e .            # from the repository root
pip install -e .[samplers]  # + nautilus-sampler, emcee
```

Dependencies: numpy, scipy, pyccl ≥ 3, fast-pt ≥ 3, iminuit. pyccl needs CAMB
(installed with pyccl from conda-forge; with pip, `pip install camb`).

## Minimal example

```python
import numpy as np
from ia2pt import TwoPointModel, IALikelihood, run_minuit

# [Omega_c, Omega_b, sum m_nu (eV), A_s, n_s, h, z_eff]
cosmo = [0.2607, 0.0490, 0.0600, 2.105e-9, 0.9665, 0.6766, 0.60]
model = TwoPointModel(cosmo, "NLA", rp_min_wedge=5.0)

s = np.geomspace(6, 100, 20); s_mid = np.sqrt(s[1:] * s[:-1])
xi0 = model.compute_xi_gg_wedge_monopole(s_mid, b1=1.9, b2=0, a1=1.2, a2=0, bTA=0)
xi2 = model.compute_xi_gi_wedge_quadrupole(s_mid, 1.9, 0, 1.2, 0, 0)

lik = IALikelihood({"xi0": model.compute_xi_gg_wedge_monopole,
                    "xi2": model.compute_xi_gi_wedge_quadrupole},
                   [s_mid, s_mid], [xi0_data, xi2_data], cov,
                   n_realisations=70, jack=False, zero_cross_cov=True)
lik.set_cut([30, 10], [100, 100])
lik.set_prior({"b1": (0.1, 4.0), "b2": 0.0, "a1": (-10, 10), "a2": 0.0, "bTA": 0.0})
res = run_minuit(lik)          # res["bestfit"], res["errors"], res["chi2"]
```

To model the IA block at a different effective redshift than the clustering
block (the paper's z_pair convention), build a second `TwoPointModel` with that
`z_eff` and pass its `compute_xi_gi_wedge_quadrupole` in the dict.

## Provenance and tests

`tests/test_model_regression.py` pins every gg / g+ statistic to the
predictions of the analysis' original `scripts/ia_model.py` (fixture generated
2026-09-17 on the LRG 0.40 < z < 0.75 grids and n(z); rtol 1e-10), pins the
++ statistics to their 2026-09-18 implementation and checks them against the
Limber limit; `tests/test_gausscov.py` checks the covariance kernels, the
shot-noise analytic limit and a synthetic fixture (the module itself was
validated bit-for-bit against the analysis' stored covariances). Relative to that file the
package drops the superseded direct-Hankel projections (w_gg / w_g+ without
Π_max and RSD, w_++, W_+; never used by the fits), renames
`compute_wgg_v2 / compute_wgp_v2` to `compute_wgg / compute_wgp`, makes
`do_rsd=False` actually switch the RSD off (β = 0) in the multipoles and eq. 19
paths, and invalidates the spectrum cache when `set_nz` is called.

## Citation

Paviot et al., *UNIONS × DESI: intrinsic alignments of DESI BGS, LRG and ELG
galaxies* (in prep.); Singh et al. 2023 (arXiv:2307.02545) for the multipole /
projection formalism; pyccl (Chisari et al. 2019) and FAST-PT (McEwen et al.
2016; Fang et al. 2017).
