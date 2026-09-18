"""Drivers running an :class:`ia2pt.IALikelihood` through iminuit, nautilus or emcee.

Each returns a dict with ``bestfit`` and ``errors`` over all of ``PARAM_NAMES``
(fixed parameters carry their fixed value and a zero error), the minimum
``chi2`` (= -2 max log-likelihood for the samplers) and, for the samplers, the
posterior samples. Best-fit values from the samplers are posterior medians,
errors half the 16-84 percentile range (nautilus) or the standard deviation
(emcee).
"""

import numpy as np

from .likelihood import PARAM_NAMES

__all__ = ["run_minuit", "run_nautilus", "run_emcee", "weighted_quantile"]

# Migrad start values for the free parameters
MINUIT_START = {"b1": 1.0, "b2": 0.0, "a1": 1.0, "a2": 0.0, "bTA": 0.0}


def weighted_quantile(values, weights, q):
    """Weighted quantile(s) ``q`` of ``values``."""
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(q) * c[-1])]]


def run_minuit(lik, start=None):
    """Migrad over the free parameters (flat priors as limits)."""
    from iminuit import Minuit
    init = dict(MINUIT_START, **(start or {}))
    start = {k: (init[k] if isinstance(v, tuple) else v) for k, v in lik.prior.items()}
    m = Minuit(lik, **start)
    for name, value in lik.prior.items():
        if isinstance(value, tuple):
            m.limits[name] = value
        else:
            m.fixed[name] = True
    m.migrad()
    bestfit = {k: m.values[k] for k in PARAM_NAMES}
    errors = {k: (m.errors[k] if k in lik.free_params else 0.0) for k in PARAM_NAMES}
    return {"bestfit": bestfit, "errors": errors, "chi2": float(m.fval),
            "valid": bool(m.valid), "minuit": m}


def run_nautilus(lik, n_live=3000, n_eff=50000, pool=1):
    """Nested sampling with nautilus; returns the weighted posterior as well."""
    points, log_w, log_l = lik.call_sampler(n_eff=n_eff, n_live=n_live, pool=pool)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    bestfit, errors = {}, {}
    free = list(lik.free_params)
    for name in PARAM_NAMES:
        if name in free:
            j = free.index(name)  # points columns follow free_params order
            bestfit[name] = float(weighted_quantile(points[:, j], w, 0.5))
            errors[name] = float(0.5 * (weighted_quantile(points[:, j], w, 0.84)
                                        - weighted_quantile(points[:, j], w, 0.16)))
        else:
            bestfit[name] = float(lik.fixed_params[name])
            errors[name] = 0.0
    chi2 = float(-2.0 * log_l.max())
    return {"bestfit": bestfit, "errors": errors, "chi2": chi2,
            "chain": points, "log_w": log_w, "log_l": log_l}


def run_emcee(lik, nwalkers=32, nsteps=5000, seed=42):
    """Affine-invariant MCMC with emcee; the first half of the chain is discarded."""
    import emcee
    ndim = lik.ndim
    rng = np.random.default_rng(seed)
    p0 = np.array([[rng.uniform(*lik.prior[name]) for name in lik.free_params]
                   for _ in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lik.log_prob_emcee)
    sampler.run_mcmc(p0, nsteps, progress=False)
    burn = nsteps // 2
    flat = sampler.get_chain(discard=burn, flat=True)
    logp = sampler.get_log_prob(discard=burn, flat=True)
    bestfit, errors = {}, {}
    for name in PARAM_NAMES:
        if name in lik.free_params:
            j = lik.free_params.index(name)
            bestfit[name] = float(np.median(flat[:, j]))
            errors[name] = float(np.std(flat[:, j]))
        else:
            bestfit[name] = float(lik.fixed_params[name])
            errors[name] = 0.0
    return {"bestfit": bestfit, "errors": errors, "chi2": float(-2.0 * logp.max()),
            "chain": flat, "log_l": logp}
