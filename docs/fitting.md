# Fitting

Two objects: `IALikelihood` holds the data, the covariance and the model
functions and evaluates $\chi^2$; the three drivers `run_minuit`,
`run_nautilus` and `run_emcee` run it and return the same kind of result
dict.

## `IALikelihood`

```python
IALikelihood(model_funcs, r_list, data_list, cov, n_realisations,
             jack=True, edges_list=None, bin_avg=False,
             taper_scale=None, zero_cross_cov=False)
```

| argument | meaning |
|---|---|
| `model_funcs` | ordered dict `{name: callable(r, b1, b2, a1, a2, bTA)}` — normally bound methods of a `TwoPointModel`, e.g. `{'xi0': m.compute_xi_gg_wedge_monopole, 'xi2': m.compute_xi_gi_wedge_quadrupole}` or `{'wgg': m.compute_wgg, 'wgp': m.compute_wgp}`; any callable with that signature works |
| `r_list`, `data_list` | one array of bin centres and one data vector per statistic, in `model_funcs` order |
| `cov` | joint covariance of the stacked data vector (same order) |
| `n_realisations` | number of jackknife realisations behind `cov` (used by the Hartlap-style correction) |
| `jack` | apply the correction $(n - n_{\rm data} - 2)/(n-1)$ to the inverse covariance (Hartlap et al. 2007) |
| `edges_list`, `bin_avg` | when the model is built with `bin_avg=True`, pass the bin edges (one `N+1` array per statistic) so the model returns bin averages; data, covariance and cuts stay on the centres |
| `taper_scale` | covariance tapering (Paz & Sánchez 2015): Hadamard product with a compact Wendland kernel of the separation lag, in $h^{-1}$ Mpc |
| `zero_cross_cov` | zero the covariance blocks between different statistics, fitting them as independent |

Then, in order:

```python
lik.set_cut(rmin_list, rmax_list)   # per-statistic [rmin, rmax] in h^-1 Mpc
lik.set_prior({"b1": (0.1, 4.0), "b2": 0.0, "a1": (-10, 10), "a2": 0.0, "bTA": 0.0})
```

`set_cut` selects the bins with `rmin <= r <= rmax` for each statistic,
slices the covariance accordingly and inverts it (with the Hartlap-style
factor when `jack=True`). It must be called before any evaluation; the
constructor applies a no-cut default.

`set_prior` takes a dict over **exactly** `("b1", "b2", "a1", "a2", "bTA")`
in that order: a scalar fixes the parameter, a `(lo, hi)` tuple frees it
with a flat prior. This one dict drives all three back-ends.

What you can evaluate afterwards:

| call | returns |
|---|---|
| `lik(b1, b2, a1, a2, bTA)` | $\chi^2$ at the full parameter vector (iminuit's cost function) |
| `lik.log_like({"b1": .., ...})` | $-\chi^2/2$ from a dict of all five parameters (nautilus) |
| `lik.log_prob_emcee(theta)` | log-probability over the **free** parameters only, $-\infty$ outside the prior box (emcee) |
| `lik.get_bestfit(b1, b2, a1, a2, bTA, r_list=None, edges_list=None)` | stacked model prediction at those parameters, on the cut separations by default (or on the given ones) |
| `lik.ndata` | number of data points after the cut |
| `lik.free_params`, `lik.fixed_params`, `lik.ndim` | what the prior freed / fixed |

## Drivers

```python
run_minuit(lik, start=None)
run_nautilus(lik, n_live=3000, n_eff=50000, pool=1)
run_emcee(lik, nwalkers=32, nsteps=5000, seed=42)
```

| driver | what it does | extra arguments |
|---|---|---|
| `run_minuit` | Migrad over the free parameters, prior tuples as limits | `start`: dict of starting values overriding the defaults `b1=1, b2=0, a1=1, a2=0, bTA=0` |
| `run_nautilus` | nested sampling (nautilus), fixed parameters as delta priors | `n_live` live points, `n_eff` effective posterior samples, `pool` processes |
| `run_emcee` | affine-invariant MCMC, walkers started uniformly in the prior box, first half of the chain discarded | `nwalkers`, `nsteps`, `seed` |

All three return a dict with:

| key | content |
|---|---|
| `bestfit` | `{name: value}` over all five parameters (fixed ones carry their fixed value) — Migrad minimum, or the posterior median for the samplers |
| `errors` | `{name: error}`, zero for fixed parameters — Migrad (Hesse) errors, half the 16–84 percentile range (nautilus), or the standard deviation (emcee) |
| `chi2` | $\chi^2$ at the minimum, or $-2 \max \log\mathcal{L}$ for the samplers |
| `valid` | `run_minuit` only: Migrad's convergence flag; `minuit` holds the `Minuit` object |
| `chain`, `log_w`, `log_l` | samplers only: posterior points over the free parameters (in `lik.free_params` order), log-weights (nautilus) and log-likelihoods |

`weighted_quantile(values, weights, q)` is exported for post-processing
nautilus chains.

## Recommended settings

These are the conventions of the UNIONS × DESI analysis, worth knowing before
changing them:

* Fit with `jack=False` and `zero_cross_cov=True` when the covariance is a
  delete-one jackknife over patches: the Hartlap factor over-corrects there,
  and the jackknife cross blocks between statistics are noise-dominated.
  Both flags must match across any fits being compared, otherwise a superset
  model (TATT) can appear to fit worse than a subset (NLA).
* Radial cuts: $s \ge 25\,h^{-1}$ Mpc for the clustering monopole (the
  model is linear bias + Kaiser), $s \ge 10$ for the IA quadrupole; $r_p \ge
  6$ for both projected statistics; upper limit $100\,h^{-1}$ Mpc.
* TATT: fix `b2 = 0`. It is not constrained by these statistics and, left
  free, rails against its prior bound while buying no improvement in fit,
  which corrupts the error budget of the other parameters. A TATT fit is thus
  `b1, a1, a2, bTA` (or `b1, a1, a2` with `bTA` fixed as well).
* Projected $w_{g+}$ and the multipole $\tilde\xi_{2,2}$ share the same sign
  convention in this package (see [Conventions](conventions.md)); a
  measurement built on the lensing tangential shear has the opposite sign and
  must be flipped before fitting.
