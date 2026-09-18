"""Chi^2 / log-likelihood for a stack of two-point statistics.

``IALikelihood`` wraps the prediction functions of a :class:`ia2pt.TwoPointModel`
(or any callables with the signature ``f(r, b1, b2, a1, a2, bTA)``), stacks the
data vectors and their joint covariance, applies per-statistic radial cuts,
optionally the Hartlap et al. (2007) correction of the inverse covariance
(n = number of jackknife realisations), covariance tapering (Paz & Sanchez
2015) and the zeroing of the cross-statistic covariance blocks, and exposes
a chi^2 ``__call__`` for iminuit plus ``log_like`` for nautilus / emcee.

The priors dict uses exactly the names in ``PARAM_NAMES``, with a scalar =
fixed value and a ``(lo, hi)`` tuple = free parameter with a flat prior.
"""

import numpy as np
from iminuit import Minuit

__all__ = ["IALikelihood", "PARAM_NAMES"]

# canonical model-parameter order shared by every TwoPointModel wrapper
PARAM_NAMES = ("b1", "b2", "a1", "a2", "bTA")


class IALikelihood:
    """Chi^2 / log-likelihood for a stack of two-point statistics.

    Parameters
    ----------
    model_funcs : dict
        Ordered {stat_name: callable(r, b1, b2, a1, a2, bTA)} -- e.g.
        {'xi0e': model.compute_xi_gg_wedge_monopole, 'xi2p': ...} or
        {'WGG': model.compute_wgg, 'WGP': model.compute_wgp}.
    r_list, data_list : list of arrays
        Separations (bin centres) and data vector per statistic, in
        model_funcs order.
    cov : 2D array
        Joint covariance of the stacked data vector.
    n_realisations : int
        Number of jackknife patches (drives the Hartlap-style correction).
    jack : bool
        Apply the Hartlap-style inverse-covariance correction (default True).
    edges_list : list of arrays, optional
        Bin edges (length N+1 per statistic) matching r_list/data_list. Required
        when ``bin_avg`` is True; the model is then evaluated on the edges so it
        can return the volume-averaged statistic per bin instead of a point value.
    bin_avg : bool
        Feed bin edges to the model (which must itself be built with bin_avg=True)
        and compare its per-bin averages against the centre-indexed data. The
        data vector, covariance and radial cuts stay on the bin centres.
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, model_funcs, r_list, data_list, cov, n_realisations,
                 jack=True, edges_list=None, bin_avg=False, taper_scale=None,
                 zero_cross_cov=False):
        if len(data_list) != len(model_funcs) or len(r_list) != len(model_funcs):
            raise ValueError("r_list/data_list must match model_funcs length")
        total = sum(len(d) for d in data_list)
        if cov.shape != (total, total):
            raise ValueError(f"covariance shape {cov.shape} != data length {total}")

        self.model_funcs = dict(model_funcs)
        self.r_list = [np.asarray(r) for r in r_list]
        self.data_list = [np.asarray(d) for d in data_list]
        self.Cov = np.asarray(cov).copy()
        self.n_realisations = int(n_realisations)
        self.jack = jack
        self.bin_avg = bool(bin_avg)
        # covariance tapering (Paz & Sanchez 2015): Hadamard-multiply the covariance
        # by a compact Wendland kernel of the separation lag |s_i - s_j|, suppressing
        # the noisy long-lag off-diagonals while preserving positive-definiteness.
        self.taper_scale = taper_scale
        if taper_scale is not None:
            s_full = np.concatenate(self.r_list)
            x = np.abs(s_full[:, None] - s_full[None, :]) / float(taper_scale)
            taper = np.where(x < 1.0, (1.0 - x) ** 4 * (1.0 + 4.0 * x), 0.0)
            self.Cov = self.Cov * taper
        # zero the cross-covariance blocks between statistics (e.g. monopole vs
        # quadrupole), fitting them as independent -- the old_codes minimization_NRV
        # zero_cross_cov diagnostic. Keep only the per-statistic diagonal blocks.
        self.zero_cross_cov = bool(zero_cross_cov)
        if self.zero_cross_cov:
            block = np.zeros_like(self.Cov, dtype=bool)
            start = 0
            for d in self.data_list:
                end = start + len(d)
                block[start:end, start:end] = True
                start = end
            self.Cov = self.Cov * block
        if self.bin_avg:
            if edges_list is None or len(edges_list) != len(self.r_list):
                raise ValueError("bin_avg requires one edges array per statistic")
            self.edges_list = [np.asarray(e) for e in edges_list]
            for r, e in zip(self.r_list, self.edges_list):
                if len(e) != len(r) + 1:
                    raise ValueError("each edges array must have len(centres)+1 entries")
        else:
            self.edges_list = None
        # default: no cut
        self.set_cut([r.min() for r in self.r_list], [r.max() for r in self.r_list])

    # ------------------------------------------------------------- radial cuts
    def set_cut(self, rmin_list, rmax_list):
        """Apply per-statistic [rmin, rmax] cuts to the data vector and covariance,
        then invert with the Hartlap-style jackknife correction
        (n - n_data - 2)/(n - 1), n = n_realisations."""
        if len(rmin_list) != len(self.r_list) or len(rmax_list) != len(self.r_list):
            raise ValueError("cut lists must match the number of statistics")

        filtered_data, filtered_r, filtered_edges, cut_indices = [], [], [], []
        current_idx = 0
        for k, (r, d, rmin, rmax) in enumerate(
                zip(self.r_list, self.data_list, rmin_list, rmax_list)):
            mask = (r >= rmin) & (r <= rmax)
            filtered_data.append(d[mask])
            filtered_r.append(r[mask])
            if self.bin_avg:
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    filtered_edges.append(np.empty(0))
                else:
                    j0, j1 = idx[0], idx[-1]
                    if not np.array_equal(idx, np.arange(j0, j1 + 1)):
                        raise ValueError("bin_avg needs a contiguous radial cut "
                                         "to map centres to bounding edges")
                    # the N retained centres [j0..j1] are bounded by N+1 edges
                    filtered_edges.append(self.edges_list[k][j0:j1 + 2])
            cut_indices.append(current_idx + np.where(mask)[0])
            current_idx += len(r)

        self.xi_fit = np.concatenate(filtered_data)
        self.r_fit = filtered_r
        self.edges_fit = filtered_edges if self.bin_avg else None
        all_indices = np.concatenate(cut_indices)
        self.Covfit = self.Cov[np.ix_(all_indices, all_indices)]

        # rp_cut measurements zero out the small-s bins (no pairs with rp >= rp_cut),
        # so a cut that includes them is singular; fall back to a pseudo-inverse with a
        # warning. The real fit cut (rmin above rp_cut) is non-singular and uses true inv.
        try:
            base_inv = np.linalg.inv(self.Covfit)
        except np.linalg.LinAlgError:
            print(f"  WARNING: singular cut covariance ({len(all_indices)} bins); "
                  "using pseudo-inverse", flush=True)
            base_inv = np.linalg.pinv(self.Covfit)

        if self.jack:
            n, size = self.n_realisations, len(all_indices)
            correction_factor = (n - size - 2) / (n - 1)
            self.invcov = base_inv * correction_factor
        else:
            self.invcov = base_inv

    @property
    def ndata(self):
        return len(self.xi_fit)

    # ------------------------------------------------------------------ priors
    def set_prior(self, dict_prior):
        """Priors over PARAM_NAMES: scalar = fixed, (lo, hi) tuple = free (flat).
        Builds the iminuit parameter spec and the nautilus Prior."""
        if tuple(dict_prior) != PARAM_NAMES:
            raise ValueError(f"prior keys must be {PARAM_NAMES} in order, "
                             f"got {tuple(dict_prior)}")
        self.prior = dict_prior
        self.ndim = 0
        self.fixed_params = {}
        self.free_params = []
        prior_iminuit = {}
        for parameter, value in dict_prior.items():
            if isinstance(value, tuple):
                self.ndim += 1
                self.free_params.append(parameter)
                prior_iminuit[parameter] = value
            else:
                self.fixed_params[parameter] = value
                prior_iminuit[parameter] = None
        self._parameters = prior_iminuit  # read by iminuit.util.describe

    def nautilus_prior(self):
        """nautilus Prior matching the dict: fixed params as delta distributions."""
        from nautilus import Prior
        prior = Prior()
        for parameter, value in self.prior.items():
            if isinstance(value, tuple):
                prior.add_parameter(parameter, dist=(value[0], value[1]))
            else:
                prior.add_parameter(parameter, dist=value)
        return prior

    # -------------------------------------------------------------- evaluation
    def _model_vector(self, dict_par, r_list):
        """Stacked model prediction at the given parameters and separations."""
        args = [dict_par[k] for k in PARAM_NAMES]
        return np.hstack([model(r, *args)
                          for (name, model), r in zip(self.model_funcs.items(), r_list)])

    def __call__(self, *par):
        """Chi^2 at the full parameter vector (PARAM_NAMES order) -- for iminuit."""
        dict_par = dict(zip(PARAM_NAMES, par))
        model_r = self.edges_fit if self.bin_avg else self.r_fit
        diff = self.xi_fit - self._model_vector(dict_par, model_r)
        return float(np.dot(diff, np.dot(self.invcov, diff)))

    def log_like(self, dict_par):
        """Log-likelihood -chi^2/2 from a {name: value} dict (all 5 params) --
        for nautilus / emcee."""
        par = [dict_par[k] for k in PARAM_NAMES]
        return -self(*par) / 2.0

    def log_prob_emcee(self, theta):
        """emcee log-probability over the FREE parameters only (flat priors)."""
        dict_par = dict(self.fixed_params)
        for name, val in zip(self.free_params, theta):
            lo, hi = self.prior[name]
            if not (lo <= val <= hi):
                return -np.inf
            dict_par[name] = val
        return self.log_like(dict_par)

    def get_bestfit(self, *par, r_list=None, edges_list=None):
        """Stacked model prediction at the full parameter vector; defaults to the
        cut separations (pass r_list=self.r_list for the uncut binning). In bin_avg
        mode the model is evaluated on bin edges: the cut edges by default, or
        ``edges_list`` (one N+1 array per statistic) for an arbitrary binning."""
        dict_par = dict(zip(PARAM_NAMES, par))
        if self.bin_avg:
            model_r = self.edges_fit if edges_list is None else edges_list
        else:
            model_r = self.r_fit if r_list is None else r_list
        return self._model_vector(dict_par, model_r)

    # ---------------------------------------------------------------- samplers
    def call_sampler(self, n_eff=400000, n_live=3000, pool=30):
        """Run the nautilus nested sampler; returns (points, log_w, log_l)."""
        from nautilus import Sampler
        sampler = Sampler(self.nautilus_prior(), self.log_like,
                          n_live=n_live, pool=pool)
        sampler.run(verbose=False, discard_exploration=True, n_eff=n_eff)
        return sampler.posterior()
