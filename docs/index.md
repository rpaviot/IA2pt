# IA2pt

NLA / TATT two-point intrinsic-alignment model, likelihood, fit drivers and
Gaussian covariance — the modelling layer of the UNIONS × DESI
intrinsic-alignment analysis (Paviot et al., in prep.).

The package answers three questions, one page each:

1. [Model](model.md) — given bias and alignment parameters, what are the
   clustering, galaxy–shape and shape–shape two-point functions? Wedged
   multipoles $\tilde\xi_{0,0}$, $\tilde\xi_{2,2}$, $\tilde\xi^{++}_{0,0}$,
   $\tilde\xi^{++}_{4,4}$ and projected $w_{gg}$, $w_{g+}$, $w_{++}$,
   $w_{\times\times}$, at one effective redshift or integrated over $n(z)$.
2. [Fitting](fitting.md) — given measurements and a covariance, what are
   the best-fit parameters? A $\chi^2$ likelihood with radial cuts and the
   usual covariance corrections, driven by iminuit, nautilus or emcee.
3. [Gaussian covariance](gausscov.md) — what covariance should the
   measurements have? Analytic (disconnected) covariance of both data
   vectors, $[\xi_0, \tilde\xi_{2,2}]$ and $[w_p, w_{g+}]$, including the
   clustering × IA cross block.

[Conventions](conventions.md) collects the units, signs and approximations
that every page relies on; [API](api.md) is the full docstring reference.

## Install

```
git clone https://github.com/rpaviot/IA2pt.git && cd IA2pt
pip install -e .            # numpy, scipy, pyccl >= 3, fast-pt >= 3, iminuit
pip install -e .[samplers]  # + nautilus-sampler, emcee
pip install -e .[test] && pytest
```

pyccl needs CAMB (installed with pyccl from conda-forge; with pip,
`pip install camb`).

## Minimal example

```python
import numpy as np
from ia2pt import TwoPointModel, IALikelihood, run_minuit

# [Omega_c, Omega_b, sum m_nu (eV), A_s, n_s, h, z_eff]
cosmo = [0.2607, 0.0490, 0.0600, 2.105e-9, 0.9665, 0.6766, 0.60]
model = TwoPointModel(cosmo, "NLA", rp_min_wedge=5.0)

s_edges = np.geomspace(6, 100, 20); s_mid = np.sqrt(s_edges[1:] * s_edges[:-1])
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

## Citation

Paviot et al., *UNIONS × DESI: intrinsic alignments of DESI BGS, LRG and ELG
galaxies* (in prep.). The formalism follows Singh et al. 2023
([arXiv:2307.02545](https://arxiv.org/abs/2307.02545)) for the multipoles and
projections, Grieb et al. 2016 ([arXiv:1509.04293](https://arxiv.org/abs/1509.04293))
for the Gaussian covariance, and uses pyccl (Chisari et al. 2019) and FAST-PT
(McEwen et al. 2016; Fang et al. 2017) for the spectra.

```{toctree}
:hidden:
:maxdepth: 2

model
fitting
gausscov
conventions
api
```
