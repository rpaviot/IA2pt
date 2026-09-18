"""Two-point NLA / TATT intrinsic-alignment model.

``TwoPointModel`` predicts the galaxy-clustering and galaxy-intrinsic-alignment
two-point statistics used in the UNIONS x DESI analysis (Paviot et al.), from the
1-loop perturbation-theory spectra of pyccl (CAMB / HMCode-2020 non-linear matter
power, EulerianPTCalculator FAST-PT for the bias and TATT terms):

* the wedged multipoles ``xi~_{0,0}(s)`` (clustering monopole) and ``xi~_{2,2}(s)``
  (galaxy-shape quadrupole), i.e. the Legendre / associated-Legendre moments of
  xi(s, mu) restricted to r_p >= rp_min_wedge, including Kaiser redshift-space
  distortions through the alpha_l coefficients of Singh et al. (2023, eqs. 13-17);
* the projected statistics ``w_gg(r_p)`` and ``w_g+(r_p)``, obtained from the same
  signed multipoles by the finite-Pi_max line-of-sight integral of Singh et al.
  (2023) eq. 19 -- NOT by a direct Hankel transform of P(k);
* the shape-shape statistics ``xi~_{0,0}^{++}``, ``xi~_{4,4}^{++}`` (Singh et al.
  2023 spin-4 filter), ``w_++`` and ``w_xx``, from the E- and B-mode IA spectra.
  Only the linear-alignment part of P_EE has an exact line-of-sight dependence,
  (1 - mu^2)^2; the TATT one-loop terms are computed by FAST-PT at mu = 0 and
  are given the same (1 - mu^2)^2 factor (Maion et al. 2024, arXiv:2307.13754,
  sect. 3.2.3), for both P_EE and P_BB. See ``_multipoles_pp`` for the expansion.

Both paths share the multipoles, so a wedge fit and a projected fit are the same
physical model. Predictions can be made at a single effective redshift or
integrated over the density x shape redshift window built from two n(z)
(Gauss-Legendre in z), as point values at the given separations or as
annular (2D) / spherical (3D) bin averages between the given bin edges.

Conventions: separations in h^-1 Mpc, spectra in (h^-1 Mpc)^3; the IA amplitudes
``a1, a2`` follow the pyccl ``translate_IA_norm`` normalisation (a1 = A_IA of the
NLA model, C1 rho_crit = 0.0134); ``bTA`` is the TATT density-weighting
(a1delta = a1 * bTA). Linear bias ``b1``, second-order ``b2``; in TATT the tidal
and third-order biases follow the local-Lagrangian relations bs = -4/7 (b1 - 1),
b3nl = b1 - 1 (they are zero in NLA).

Sign convention: e_+ is positive for RADIAL alignment (e_+ = -e_t of the
lensing convention), so a positive a1 gives a positive ``xi~_{2,2}`` and a
positive ``w_g+`` (as in Singh et al. 2015, 2023). Estimators built on the
lensing tangential shear (e.g. treecorr NG correlations) return w_g+ with the
opposite sign: flip such a measurement before fitting, or negate the fitted a1.
"""

import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pyccl.background as pb
from fastpt.misc import HT
from scipy.interpolate import CubicSpline as CS, interp1d
from scipy.integrate import trapezoid

__all__ = ["TwoPointModel", "L00", "L20", "L40", "L22", "L42", "L44", "derivative"]


def derivative(func, x, h):
    """4th-order 5-point-stencil derivative df/dx at x."""
    return (-func(x + 2*h) + 8*func(x + h) - 8*func(x - h) + func(x - 2*h)) / (12*h)


# ---------------------------------------------------------------------------
# (Associated) Legendre polynomials for the wedged multipoles
# ---------------------------------------------------------------------------
def L00(mu):
    """L_{0,0}(mu) = 1 (monopole, spin-0)."""
    return np.ones_like(mu)


def L20(mu):
    """L_{2,0}(mu) = (3 mu^2 - 1)/2 (quadrupole, spin-0)."""
    return (3*mu**2 - 1) / 2


def L40(mu):
    """L_{4,0}(mu) = (35 mu^4 - 30 mu^2 + 3)/8 (hexadecapole, spin-0)."""
    return (35*mu**4 - 30*mu**2 + 3) / 8


def L22(mu):
    """L_{2,2}(mu) = 3 (1 - mu^2) (quadrupole, spin-2)."""
    return 3.0 * (1.0 - mu**2)


def L42(mu):
    """L_{4,2}(mu) = (15/2)(1 - mu^2)(7 mu^2 - 1) (spin-2)."""
    return 15.0/2.0 * (1.0 - mu**2) * (7.0 * mu**2 - 1.0)


def L44(mu):
    """L_{4,4}(mu) = 105 (1 - mu^2)^2 (hexadecapole, spin-4)."""
    return 105.0 * (1.0 - mu**2)**2


def _has_wedge_cut(rp_min):
    """True if a transverse wedge cut should be applied (rp_min positive)."""
    return rp_min is not None and rp_min > 0


class TwoPointModel:
    """NLA / TATT model for clustering + intrinsic-alignment two-point statistics.

    Parameters
    ----------
    cosmology : sequence
        Fiducial cosmology ``[Omega_c, Omega_b, sum m_nu (eV), A_s, n_s, h, z_eff]``.
    config : {'NLA', 'TATT'}
        IA model.
    do_rsd : bool
        Include the Kaiser redshift-space distortions (beta = f / b1) in the
        multipoles and in the eq. 19 projections (default True). With False
        beta = 0, i.e. real-space statistics.
    pimax : float
        Line-of-sight integration limit for w_gg / w_g+ [h^-1 Mpc] (default 100).
    dpi : float
        Line-of-sight step of the eq. 19 trapezoid integration (default 0.1).
    bin_avg : bool
        Treat input separations as bin EDGES and return bin averages: annular for
        w_gg / w_g+, spherical for the multipoles (default False = point values).
    bin_factor : int
        Integration-grid density: ``n_bins * bin_factor`` coarse points (default 20).
    evolve_bias : bool
        b1(z) = b1_ref * D(z_ref) / D(z) in the PT tracer (default False).
    rp_min_wedge : float or None
        Minimum r_p of the wedge cut for the multipoles [h^-1 Mpc]; 0 / None
        disables it (default 5.0).
    n_mu_wedge : int
        Number of mu points for the wedge integration (default 101).

    Notes
    -----
    All statistics share the parameter signature ``(r, b1, b2, a1, a2, bTA)``.
    """

    def __init__(self, cosmology, config, do_rsd=True, pimax=100, dpi=0.1,
                 bin_avg=False, bin_factor=20, evolve_bias=False,
                 rp_min_wedge=5.0, n_mu_wedge=101):
        if config not in ("NLA", "TATT"):
            raise ValueError(f"config must be 'NLA' or 'TATT', got {config!r}")
        self.Omc, self.Omb, self.mnu, self.As, self.ns, self.h, self.zeff = cosmology
        self.config = config
        self.do_rsd = do_rsd
        self.pimax = pimax
        self.dpi = dpi
        self.bin_avg = bin_avg
        self.bin_factor = bin_factor
        self.evolve_bias = evolve_bias
        self.rp_min_wedge = rp_min_wedge
        self.n_mu_wedge = n_mu_wedge
        self.unique_z = True

        self.params = None
        self._cached_params = None  # set_pks2D cache

        self._init_cosmology()
        self._init_fastpt()

    # ------------------------------------------------------------------ setup
    def _init_cosmology(self):
        """Initialize the CCL cosmology and the growth rate at zeff."""
        self.cosmo = ccl.Cosmology(
            Omega_c=self.Omc, Omega_b=self.Omb, m_nu=self.mnu, h=self.h,
            A_s=self.As, n_s=self.ns,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='camb',
            mass_split='single',
            extra_parameters={"camb": {"halofit_version": "mead2020"}})

        self.f = pb.growth_rate(self.cosmo, 1./(1. + self.zeff))

        # redshift array for the perturbation-theory tracers
        if self.zeff > 0.01:
            self.zpt = np.linspace(self.zeff - 0.001, self.zeff + 0.001, 100)
        else:
            self.zpt = np.linspace(self.zeff, self.zeff + 0.001, 50)
        self.apt = 1./(1. + self.zpt)

    def _init_fastpt(self):
        """Initialize the FAST-PT calculator and the k arrays."""
        nk = 512
        kmin, kmax = -5, 2
        self.ks = np.logspace(kmin, kmax, nk)
        self.k = self.ks / self.h
        self.ptc = pt.EulerianPTCalculator(
            with_NC=True, with_IA=True, cosmo=self.cosmo,
            log10k_min=kmin, log10k_max=kmax, nk_per_decade=20)
        self.Pklin = self.h**3 * ccl.power.linear_power(
            self.cosmo, self.ks, 1./(1. + self.zeff))

    # ------------------------------------------------------------------- n(z)
    def set_nz(self, zedge, z, dn1, dn2):
        """Switch to n(z)-integrated predictions.

        ``dn1`` is the density-tracer n(z) and ``dn2`` the shape-sample n(z),
        tabulated at ``z``; ``zedge`` sets the integration range and the grid on
        which both are normalised. Builds the Gauss-Legendre (20-node) windows
        W ~ n_i n_j / (chi^2 dchi/dz) for the clustering (n1 n1), the cross
        (n1 n2) and the shape-shape (n2 n2) statistics."""
        self.unique_z = False
        self.z_hist = zedge
        z_min, z_max = zedge[0], zedge[-1]
        self.zmin, self.zmax = z_min, z_max

        self.spline_nz1 = self._normalize_nz(z, dn1, zedge)
        self.spline_nz2 = self._normalize_nz(z, dn2, zedge)

        self.n_gauss = 20
        nodes, weights = np.polynomial.legendre.leggauss(self.n_gauss)
        self.z_gauss = 0.5 * (z_max - z_min) * nodes + 0.5 * (z_max + z_min)
        self.w_gauss = 0.5 * (z_max - z_min) * weights

        self.sf = 1./(1. + self.z_gauss)
        self.f_gauss = np.array([pb.growth_rate(self.cosmo, a) for a in self.sf])

        z_extend = np.linspace(z_min - 0.01, z_max + 0.01, self.n_gauss + 4)
        self.zpt = z_extend
        self.apt = 1./(1. + self.zpt)

        z_chi = np.linspace(0, 10, 10000)
        sf_chi = 1./(1. + z_chi)
        dist = ccl.comoving_radial_distance(self.cosmo, sf_chi)
        self.spline_chi = interp1d(z_chi, dist, kind='cubic',
                                   fill_value=(0, 0), bounds_error=False)

        self._compute_window_functions()
        self._cached_params = None  # the PT tracers depend on zpt

    def _normalize_nz(self, z, dn, zedge):
        """Normalize and spline-interpolate an n(z)."""
        from scipy.interpolate import splrep, splev
        spline = splrep(z, dn, k=3, s=0)
        dn_interp = splev(zedge, spline, ext=1)
        dn_norm = dn_interp / np.trapezoid(dn_interp, zedge)
        return CS(zedge, dn_norm)

    def _compute_window_functions(self):
        """Cross / clustering window functions at the GL points."""
        chi = self.spline_chi(self.z_gauss)
        h = 0.001
        dchi = derivative(self.spline_chi, self.z_gauss, h)

        nz1 = self.spline_nz1(self.z_gauss)
        nz2 = self.spline_nz2(self.z_gauss)

        Wz_unnorm = nz1 * nz2 / (chi**2 * dchi)
        self.Wz = Wz_unnorm / np.sum(Wz_unnorm * self.w_gauss)

        Wz_clust_unnorm = nz1 * nz1 / (chi**2 * dchi)
        self.Wz_clustering = Wz_clust_unnorm / np.sum(Wz_clust_unnorm * self.w_gauss)

        Wz_shapes_unnorm = nz2 * nz2 / (chi**2 * dchi)
        self.Wz_shapes = Wz_shapes_unnorm / np.sum(Wz_shapes_unnorm * self.w_gauss)

    # ---------------------------------------------------------- power spectra
    def set_pks2D(self, params):
        """Compute the 1-loop Pk2D for [b1, b2, bs, b3nl, a1, a2, bTA]."""
        b1, b2, bs, b3nl, a1, a2, bTA = params

        if self.evolve_bias:
            a_ref = 1.0 / (1.0 + self.zeff)
            D_ref = ccl.growth_factor(self.cosmo, a_ref)
            D_z = ccl.growth_factor(self.cosmo, self.apt)
            b1_tracer = (self.zpt, b1 * D_ref / D_z)
        else:
            b1_tracer = b1

        if self.config == 'TATT':
            c1, cd, c2 = pt.translate_IA_norm(
                self.cosmo, z=self.zpt, a1=a1, a1delta=a1*bTA, a2=a2,
                Om_m2_for_c2=False)
        else:  # NLA
            c1, cd, c2 = pt.translate_IA_norm(
                self.cosmo, z=self.zpt, a1=a1, a1delta=0., a2=0.,
                Om_m2_for_c2=False)

        ptt_g = pt.PTNumberCountsTracer(b1=b1_tracer, b2=b2, bs=bs, b3nl=b3nl)
        ptt_i = pt.PTIntrinsicAlignmentTracer(
            c1=(self.zpt, c1), c2=(self.zpt, c2), cdelta=(self.zpt, cd))

        self.pk_gi = self.ptc.get_biased_pk2d(ptt_g, tracer2=ptt_i)
        self.pk_gg = self.ptc.get_biased_pk2d(ptt_g)
        # E/B-mode shape auto-spectra are only built when a ++ statistic asks
        # for them (they are not needed by the gg / g+ fits)
        self._ptt_i = ptt_i
        self.pk_ii_ee = None
        self.pk_ii_bb = None

    def _pk_ii(self):
        """(P_EE, P_BB) Pk2D of the shape field at the current parameters."""
        if self.pk_ii_ee is None:
            self.pk_ii_ee = self.ptc.get_biased_pk2d(self._ptt_i, return_ia_bb=False)
            self.pk_ii_bb = self.ptc.get_biased_pk2d(self._ptt_i, return_ia_bb=True)
        return self.pk_ii_ee, self.pk_ii_bb

    def _needs_pks2d_update(self, params):
        if self._cached_params is None:
            return True
        return not np.allclose(params, self._cached_params, rtol=1e-10, atol=1e-12)

    def _update_pks2d_if_needed(self, b1, b2, a1, a2, bTA):
        """Recompute the Pk2Ds only if the bias/IA parameters changed."""
        if self.config == 'TATT':
            bs = -(4./7.) * (b1 - 1.)
            b3nl = b1 - 1.
        else:
            bs = 0.
            b3nl = 0.
        params = [b1, b2, bs, b3nl, a1, a2, bTA]
        if self._needs_pks2d_update(params):
            self.params = params
            self.set_pks2D(self.params)
            self._cached_params = np.array(params)

    def _beta(self, f, b1):
        """Kaiser beta = f / b1, or 0 when RSD are switched off."""
        return f / b1 if self.do_rsd else 0.0

    # ----------------------------------------------------------- bin averaging
    def _bin_average_from_coarse(self, r_bins, r_coarse, xi_coarse, dim=2):
        """Bin-average xi from a coarse-grid evaluation; dim=2 annular, dim=3 spherical."""
        n_bins = len(r_bins) - 1
        xi_binned = np.zeros(n_bins)
        xi_spline = CS(r_coarse, xi_coarse)

        for i in range(n_bins):
            r1, r2 = r_bins[i], r_bins[i+1]
            mask = (r_coarse >= r1) & (r_coarse <= r2)
            r_bin = r_coarse[mask]
            if len(r_bin) == 0 or r_bin[0] != r1:
                r_bin = np.concatenate([[r1], r_bin])
            if r_bin[-1] != r2:
                r_bin = np.concatenate([r_bin, [r2]])
            xi_bin = xi_spline(r_bin)
            if dim == 2:
                integral = trapezoid(xi_bin * r_bin, r_bin)
                xi_binned[i] = 2.0 * integral / (r2**2 - r1**2)
            elif dim == 3:
                integral = trapezoid(xi_bin * r_bin**2, r_bin)
                xi_binned[i] = 3.0 * integral / (r2**3 - r1**3)
        return xi_binned

    def bin_averaged_2D(self, r_bins, xi_func):
        """2D bin-averaged xi: evaluate xi_func on a coarse grid, annular-average."""
        r_coarse = np.logspace(np.log10(r_bins[0]), np.log10(r_bins[-1]),
                               len(r_bins) * self.bin_factor)
        return self._bin_average_from_coarse(r_bins, r_coarse, xi_func(r_coarse), dim=2)

    def bin_averaged_3D(self, r_bins, xi_func):
        """3D bin-averaged xi: evaluate xi_func on a coarse grid, spherical-average."""
        r_coarse = np.logspace(np.log10(r_bins[0]), np.log10(r_bins[-1]),
                               len(r_bins) * self.bin_factor)
        return self._bin_average_from_coarse(r_bins, r_coarse, xi_func(r_coarse), dim=3)

    # ------------------------------------------------- signed multipoles xi^{l,s}
    def _multipoles_gg(self, Pgg, beta):
        """xi^{l,0}_{gg}(r) for l = 0, 2, 4 from P_gg with the Kaiser alpha_l
        (Singh et al. 2023 eqs. 13-15): raw Hankel output on the fastpt r grid."""
        a0 = 1.0 + (2.0 / 3.0) * beta + (1.0 / 5.0) * beta**2
        a2 = (4.0 / 3.0) * beta + (4.0 / 7.0) * beta**2
        a4 = (8.0 / 35.0) * beta**2
        r, xi0 = HT.k_to_r(self.k, a0 * Pgg, 1.5, -1.5, 0.5)
        _, xi2 = HT.k_to_r(self.k, a2 * Pgg, 1.5, -1.5, 2.5)
        _, xi4 = HT.k_to_r(self.k, a4 * Pgg, 1.5, -1.5, 4.5)
        return r, xi0, xi2, xi4

    def _multipoles_gp(self, PgI, beta_D):
        """xi^{l,2}_{g+}(r) for l = 2, 4 from P_gI with the Kaiser alpha_l of the
        density tracer (Singh et al. 2023 eqs. 16-17)."""
        a2 = (1.0 / 3.0) * (1.0 + beta_D / 7.0)
        a4 = (2.0 / 105.0) * beta_D
        r, xi2 = HT.k_to_r(self.k, a2 * PgI, 1.5, -1.5, 2.5)
        _, xi4 = HT.k_to_r(self.k, a4 * PgI, 1.5, -1.5, 4.5)
        return r, xi2, xi4

    # Legendre coefficients of the sky-projection factor (1 - mu^2)^2
    _ALPHA_PP = (8.0 / 15.0, -16.0 / 21.0, 8.0 / 35.0)   # l = 0, 2, 4

    def _multipoles_pp(self, P_EE, P_BB):
        """Signed multipoles of the shape-shape correlation.

        With e_+ measured along the projected separation, the Fourier kernel
        of <e_+ e_+> is P_EE cos^2(2 dphi) + P_BB sin^2(2 dphi), dphi the azimuth
        between k_perp and r_p, and both spectra carry the sky-projection factor
        (1 - mu_k^2)^2 (exact for the linear-alignment term, assumed for the
        TATT loops). Writing cos^2 = (1 + cos 4 dphi)/2 and using the addition
        theorem, the spin-0 part expands the (1 - mu_k^2)^2 factor in Legendre
        polynomials and the spin-4 part is a single associated-Legendre term::

            xi_++(r, mu) = sum_{l=0,2,4} (-1)^{l/2} xi_A^{l,0}(r) L_{l,0}(mu)
                           + xi_B^{4,4}(r) L_{4,4}(mu)
            xi_A^{l,0} = (1/2 pi^2) Int k^2 dk alpha_l P_A j_l(kr),
                         P_A = (P_EE + P_BB)/2, alpha = (8/15, -16/21, 8/35)
            xi_B^{4,4} = (1/105)(1/2 pi^2) Int k^2 dk P_B j_4(kr),
                         P_B = (P_EE - P_BB)/2

        <e_x e_x> is the same with P_B -> -P_B. No Kaiser factor: neither side
        is a density field. Returns the raw Hankel outputs on the fastpt r grid.
        """
        a0, a2, a4 = self._ALPHA_PP
        P_A = 0.5 * (P_EE + P_BB)
        P_B = 0.5 * (P_EE - P_BB)
        r, xiA0 = HT.k_to_r(self.k, a0 * P_A, 1.5, -1.5, 0.5)
        _, xiA2 = HT.k_to_r(self.k, a2 * P_A, 1.5, -1.5, 2.5)
        _, xiA4 = HT.k_to_r(self.k, a4 * P_A, 1.5, -1.5, 4.5)
        _, xiB4 = HT.k_to_r(self.k, P_B / 105.0, 1.5, -1.5, 4.5)
        return r, xiA0, xiA2, xiA4, xiB4

    def _pp_terms(self, a, cross=False):
        """(sign, L, spline) terms of xi_++ (or xi_xx when ``cross``) at scale factor a."""
        pk_ee, pk_bb = self._pk_ii()
        P_EE = self.h**3 * pk_ee(self.ks, a, self.cosmo)
        P_BB = self.h**3 * pk_bb(self.ks, a, self.cosmo)
        r, xiA0, xiA2, xiA4, xiB4 = self._multipoles_pp(P_EE, P_BB)
        return [(+1.0, L00, CS(r, xiA0)),
                (-1.0, L20, CS(r, xiA2)),
                (+1.0, L40, CS(r, xiA4)),
                (-1.0 if cross else +1.0, L44, CS(r, xiB4))]

    # ----------------------------------------------- 3D wedged multipoles
    def compute_xi_gg_wedge_monopole(self, r, b1, b2, a1, a2, bTA):
        """Wedged clustering monopole xi~_{0,0}(s); ``r`` = points, or edges when
        bin_avg (spherical bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        return self.wedge_gg_monopole(r, self.rp_min_wedge, self.n_mu_wedge)

    def compute_xi_gi_wedge_quadrupole(self, r, b1, b2, a1, a2, bTA):
        """Wedged galaxy-shape quadrupole xi~_{2,2}(s); ``r`` = points, or edges
        when bin_avg (spherical bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        return self.wedge_gi_quadrupole(r, self.rp_min_wedge, self.n_mu_wedge)

    def wedge_gg_monopole(self, r_vals, rp_min=5.0, n_mu=101):
        """Wedged clustering monopole xi~_{0,0}(r) at the current parameters."""
        if self.bin_avg:
            def wedge_func(r):
                return self._compute_wedge_gg_at_points(r, rp_min, n_mu)
            return self.bin_averaged_3D(r_vals, wedge_func)
        return self._compute_wedge_gg_at_points(r_vals, rp_min, n_mu)

    def _compute_wedge_gg_at_points(self, r_vals, rp_min=5.0, n_mu=101):
        if self.unique_z:
            return self._compute_wedge_gg_single_z(r_vals, self.params[0],
                                                   self.f, rp_min, n_mu)
        return self._compute_wedge_gg_nz_integrated(r_vals, rp_min, n_mu)

    def _compute_wedge_gg_single_z(self, r_vals, b1, f, rp_min=5.0, n_mu=101,
                                   a=None):
        """Wedged monopole at single z (point evaluation). ``a`` is the scale
        factor at which P(k) is evaluated (default: the model's z_eff); the
        n(z)-integrated path passes each Gauss-Legendre node."""
        if a is None:
            a = 1./(1. + self.zeff)
        Pkgg = self.h**3 * self.pk_gg(self.ks, a, self.cosmo)

        r_full, xi_0, xi_2, xi_4 = self._multipoles_gg(Pkgg, self._beta(f, b1))
        xi_0_interp, xi_2_interp, xi_4_interp = CS(r_full, xi_0), CS(r_full, xi_2), CS(r_full, xi_4)

        r_vals_min = r_vals.min() if hasattr(r_vals, 'min') else r_vals[0]
        r_vals_max = r_vals.max() if hasattr(r_vals, 'max') else r_vals[-1]
        r_min = min(rp_min, r_vals_min) if _has_wedge_cut(rp_min) else r_vals_min
        r_max = r_vals_max

        r_grid_coarse = np.logspace(np.log10(r_min), np.log10(r_max),
                                    len(r_vals) * self.bin_factor)
        xi_0_coarse = xi_0_interp(r_grid_coarse)
        xi_2_coarse = xi_2_interp(r_grid_coarse)
        xi_4_coarse = xi_4_interp(r_grid_coarse)

        mu = np.linspace(0, 1, n_mu)
        r_grid, mu_grid = np.meshgrid(r_grid_coarse, mu, indexing='ij')
        rp_grid = r_grid * np.sqrt(1 - mu_grid**2)

        if _has_wedge_cut(rp_min):
            wedge_filter = (rp_grid >= rp_min).astype(float)
        else:
            wedge_filter = np.ones_like(rp_grid)

        L00_vals, L20_vals, L40_vals = L00(mu_grid), L20(mu_grid), L40(mu_grid)
        # (-1)^{l/2} signs of the Hankel multipoles: +, -, +
        xi_r_mu = (xi_0_coarse[:, np.newaxis] * L00_vals -
                   xi_2_coarse[:, np.newaxis] * L20_vals +
                   xi_4_coarse[:, np.newaxis] * L40_vals)

        # (2l+1)/2 * int_{-1}^{1} = (1/2) * 2 int_0^1 for l = 0
        integrand = wedge_filter * L00_vals * xi_r_mu
        xi_wedge_coarse = 2.0 * 0.5 * np.trapezoid(integrand, mu, axis=1)

        return CS(r_grid_coarse, xi_wedge_coarse)(r_vals)

    def _compute_wedge_gg_nz_integrated(self, r_vals, rp_min=5.0, n_mu=101):
        """Wedged monopole with n(z) integration."""
        wedges_at_z = [self._compute_wedge_gg_single_z(r_vals, self.params[0],
                                                       f_val, rp_min, n_mu, a=a_sf)
                       for a_sf, f_val in zip(self.sf, self.f_gauss)]
        wedges_at_z = np.array(wedges_at_z)
        return np.sum(wedges_at_z * self.Wz_clustering[:, None] *
                      self.w_gauss[:, None], axis=0)

    def wedge_gi_quadrupole(self, r_vals, rp_min=5.0, n_mu=101):
        """Wedged galaxy-shape quadrupole xi~_{2,2}(r) at the current parameters."""
        if self.bin_avg:
            def wedge_func(r):
                return self._compute_wedge_gi_at_points(r, rp_min, n_mu)
            return self.bin_averaged_3D(r_vals, wedge_func)
        return self._compute_wedge_gi_at_points(r_vals, rp_min, n_mu)

    def _compute_wedge_gi_at_points(self, r_vals, rp_min=5.0, n_mu=101):
        if self.unique_z:
            return self._compute_wedge_gi_single_z(r_vals, self.params[0],
                                                   self.f, rp_min, n_mu)
        return self._compute_wedge_gi_nz_integrated(r_vals, rp_min, n_mu)

    def _compute_wedge_gi_single_z(self, r_vals, b1, f, rp_min=5.0, n_mu=101,
                                   a=None):
        """Wedged quadrupole at single z (point evaluation). ``a`` as in
        _compute_wedge_gg_single_z."""
        if a is None:
            a = 1./(1. + self.zeff)
        p_gI = self.h**3 * self.pk_gi(self.ks, a, self.cosmo)

        r_full, xi_2, xi_4 = self._multipoles_gp(p_gI, self._beta(f, b1))
        xi_2_interp, xi_4_interp = CS(r_full, xi_2), CS(r_full, xi_4)

        r_vals_min = r_vals.min() if hasattr(r_vals, 'min') else r_vals[0]
        r_vals_max = r_vals.max() if hasattr(r_vals, 'max') else r_vals[-1]
        r_grid_coarse = np.logspace(np.log10(r_vals_min), np.log10(r_vals_max),
                                    len(r_vals) * self.bin_factor)

        xi_2_coarse = xi_2_interp(r_grid_coarse)
        xi_4_coarse = xi_4_interp(r_grid_coarse)

        mu = np.linspace(0, 1, n_mu)
        r_grid, mu_grid = np.meshgrid(r_grid_coarse, mu, indexing='ij')
        rp_grid = r_grid * np.sqrt(1 - mu_grid**2)

        if _has_wedge_cut(rp_min):
            wedge_filter = (rp_grid >= rp_min).astype(float)
        else:
            wedge_filter = np.ones_like(rp_grid)

        L22_vals, L42_vals = L22(mu_grid), L42(mu_grid)
        xi_r_mu = (-xi_2_coarse[:, np.newaxis] * L22_vals +
                   xi_4_coarse[:, np.newaxis] * L42_vals)

        # (2l+1)/2 * (l-2)!/(l+2)! * int_{-1}^{1} = (5/48) * 2 int_0^1 for l = 2
        integrand = wedge_filter * L22_vals * xi_r_mu
        xi_wedge_coarse = 2.0 * (5.0/48.0) * np.trapezoid(integrand, mu, axis=1)

        return CS(r_grid_coarse, xi_wedge_coarse)(r_vals)

    def _compute_wedge_gi_nz_integrated(self, r_vals, rp_min=5.0, n_mu=101):
        """Wedged quadrupole with n(z) integration."""
        wedges_at_z = [self._compute_wedge_gi_single_z(r_vals, self.params[0],
                                                       f_val, rp_min, n_mu, a=a_sf)
                       for a_sf, f_val in zip(self.sf, self.f_gauss)]
        wedges_at_z = np.array(wedges_at_z)
        return np.sum(wedges_at_z * self.Wz[:, None] * self.w_gauss[:, None], axis=0)

    def compute_xi_pp_wedge_monopole(self, r, b1, b2, a1, a2, bTA):
        """Wedged shape-shape monopole xi~_{0,0}^{++}(s) (L_{0,0} filter);
        ``r`` = points, or edges when bin_avg (spherical bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        return self.wedge_pp_multipole(r, 0, 0, self.rp_min_wedge, self.n_mu_wedge)

    def compute_xi_pp_wedge_hexadecapole(self, r, b1, b2, a1, a2, bTA):
        """Wedged shape-shape hexadecapole xi~_{4,4}^{++}(s), the spin-4
        matched filter of Singh et al. (2023, eq. 18 with s_ab = 4); ``r`` =
        points, or edges when bin_avg (spherical bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        return self.wedge_pp_multipole(r, 4, 4, self.rp_min_wedge, self.n_mu_wedge)

    def wedge_pp_multipole(self, r_vals, ell, spin, rp_min=5.0, n_mu=101):
        """Wedged xi~_{ell,spin}^{++}(r) at the current parameters, (ell, spin)
        in {(0, 0), (4, 4)}."""
        if self.bin_avg:
            def wedge_func(r):
                return self._compute_wedge_pp_at_points(r, ell, spin, rp_min, n_mu)
            return self.bin_averaged_3D(r_vals, wedge_func)
        return self._compute_wedge_pp_at_points(r_vals, ell, spin, rp_min, n_mu)

    def _compute_wedge_pp_at_points(self, r_vals, ell, spin, rp_min=5.0, n_mu=101):
        if self.unique_z:
            return self._compute_wedge_pp_single_z(r_vals, ell, spin, rp_min, n_mu)
        wedges_at_z = np.array([
            self._compute_wedge_pp_single_z(r_vals, ell, spin, rp_min, n_mu, a=a_sf)
            for a_sf in self.sf])
        return np.sum(wedges_at_z * self.Wz_shapes[:, None] * self.w_gauss[:, None],
                      axis=0)

    _PP_FILTERS = {(0, 0): (L00, 1.0 / 2.0),          # (2l+1)/2 (l-s)!/(l+s)!
                   (4, 4): (L44, 9.0 / 2.0 / 40320.0)}

    def _compute_wedge_pp_single_z(self, r_vals, ell, spin, rp_min=5.0, n_mu=101,
                                   a=None):
        """Wedged ++ multipole at single z (point evaluation): rebuild
        xi_++(r, mu) from the signed multipoles and apply the L_{ell,spin}
        filter of Singh et al. (2023) eq. 18 with the r_p >= rp_min cut."""
        if (ell, spin) not in self._PP_FILTERS:
            raise ValueError(f"(ell, spin) must be (0, 0) or (4, 4), got {(ell, spin)}")
        if a is None:
            a = 1./(1. + self.zeff)
        terms = self._pp_terms(a)

        r_vals_min = r_vals.min() if hasattr(r_vals, 'min') else r_vals[0]
        r_vals_max = r_vals.max() if hasattr(r_vals, 'max') else r_vals[-1]
        r_grid_coarse = np.logspace(np.log10(r_vals_min), np.log10(r_vals_max),
                                    len(r_vals) * self.bin_factor)

        mu = np.linspace(0, 1, n_mu)
        r_grid, mu_grid = np.meshgrid(r_grid_coarse, mu, indexing='ij')
        rp_grid = r_grid * np.sqrt(1 - mu_grid**2)

        if _has_wedge_cut(rp_min):
            wedge_filter = (rp_grid >= rp_min).astype(float)
        else:
            wedge_filter = np.ones_like(rp_grid)

        xi_r_mu = np.zeros_like(r_grid)
        for sign, filt, xi_spline in terms:
            xi_r_mu += sign * xi_spline(r_grid_coarse)[:, np.newaxis] * filt(mu_grid)

        filt, norm = self._PP_FILTERS[(ell, spin)]
        # norm * int_{-1}^{1} = norm * 2 int_0^1 (xi_++ is even in mu)
        integrand = wedge_filter * filt(mu_grid) * xi_r_mu
        xi_wedge_coarse = 2.0 * norm * np.trapezoid(integrand, mu, axis=1)
        return CS(r_grid_coarse, xi_wedge_coarse)(r_vals)

    # ---------------------------------------------------------------------
    # Projected w_gg / w_g+ via eq. 19 of Singh et al. 2023 (arXiv:2307.02545):
    # reconstruct xi(r, mu) from the signed multipoles on a coarse (rp, Pi) grid
    # and trapezoid-integrate along Pi up to pimax. The (-1)^{l/2} sign from
    # Eq. 9 is applied explicitly: +1 (l=0), -1 (l=2), +1 (l=4).
    # ---------------------------------------------------------------------
    def _eq19_project(self, rp_vals, multipole_terms):
        """Eq. 19 projection: trapezoidal Pi integration on a coarse rp grid.
        ``multipole_terms`` = list of (sign, legendre_fn, xi_spline)."""
        rp_vals = np.asarray(rp_vals, dtype=float)
        rp_min = rp_vals.min() if hasattr(rp_vals, 'min') else rp_vals[0]
        rp_max = rp_vals.max() if hasattr(rp_vals, 'max') else rp_vals[-1]
        rp_grid_coarse = np.logspace(np.log10(rp_min), np.log10(rp_max),
                                     len(rp_vals) * self.bin_factor)

        dpi = self.dpi
        npt = int(self.pimax / dpi) + 1
        pi = np.linspace(0, self.pimax, npt)

        rp_grid, pi_grid = np.meshgrid(rp_grid_coarse, pi, indexing='ij')
        s = np.sqrt(rp_grid**2 + pi_grid**2)
        mu = pi_grid / s

        xi_rppi = np.zeros_like(s)
        for sign, filt, xi_spline in multipole_terms:
            xi_rppi += sign * xi_spline(s) * filt(mu)

        # factor of 2 for the [-Pimax, 0] side
        w_coarse = 2.0 * trapezoid(xi_rppi, pi, axis=1)
        return CS(rp_grid_coarse, w_coarse)(rp_vals)

    def _wgg_terms(self, a, f):
        Pgg = self.h**3 * self.pk_gg(self.ks, a, self.cosmo)
        r, xi0, xi2, xi4 = self._multipoles_gg(Pgg, self._beta(f, self.params[0]))
        return [(+1.0, L00, CS(r, xi0)),
                (-1.0, L20, CS(r, xi2)),
                (+1.0, L40, CS(r, xi4))]

    def _wgp_terms(self, a, f):
        PgI = self.h**3 * self.pk_gi(self.ks, a, self.cosmo)
        r, xi2, xi4 = self._multipoles_gp(PgI, self._beta(f, self.params[0]))
        return [(-1.0, L22, CS(r, xi2)),
                (+1.0, L42, CS(r, xi4))]

    def _compute_wgg_at_points(self, rp):
        """Point evaluation of w_gg(rp) via eq. 19."""
        if self.unique_z:
            return np.asarray(self._eq19_project(rp, self._wgg_terms(1.0 / (1.0 + self.zeff), self.f)))
        w_of_z = np.asarray([self._eq19_project(rp, self._wgg_terms(a_sf, f_z))
                             for a_sf, f_z in zip(self.sf, self.f_gauss)])
        return np.sum(w_of_z * self.Wz_clustering[:, None] * self.w_gauss[:, None], axis=0)

    def _compute_wgp_at_points(self, rp):
        """Point evaluation of w_g+(rp) via eq. 19."""
        if self.unique_z:
            return np.asarray(self._eq19_project(rp, self._wgp_terms(1.0 / (1.0 + self.zeff), self.f)))
        w_of_z = np.asarray([self._eq19_project(rp, self._wgp_terms(a_sf, f_z))
                             for a_sf, f_z in zip(self.sf, self.f_gauss)])
        return np.sum(w_of_z * self.Wz[:, None] * self.w_gauss[:, None], axis=0)

    def compute_wgg(self, rp, b1, b2, a1, a2, bTA):
        """Projected galaxy clustering w_gg(r_p) (eq. 19 of Singh et al. 2023);
        ``rp`` = points, or edges when bin_avg (annular bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        if self.bin_avg:
            return self.bin_averaged_2D(rp, self._compute_wgg_at_points)
        return self._compute_wgg_at_points(rp)

    def compute_wgp(self, rp, b1, b2, a1, a2, bTA):
        """Projected galaxy-IA cross-correlation w_g+(r_p) (eq. 19 of Singh et al.
        2023); ``rp`` = points, or edges when bin_avg (annular bin average)."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        if self.bin_avg:
            return self.bin_averaged_2D(rp, self._compute_wgp_at_points)
        return self._compute_wgp_at_points(rp)

    def _compute_wpp_at_points(self, rp, cross=False):
        """Point evaluation of w_++(rp) (or w_xx when ``cross``) via eq. 19."""
        if self.unique_z:
            return np.asarray(self._eq19_project(
                rp, self._pp_terms(1.0 / (1.0 + self.zeff), cross)))
        w_of_z = np.asarray([self._eq19_project(rp, self._pp_terms(a_sf, cross))
                             for a_sf in self.sf])
        return np.sum(w_of_z * self.Wz_shapes[:, None] * self.w_gauss[:, None], axis=0)

    def compute_wpp(self, rp, b1, b2, a1, a2, bTA):
        """Projected shape-shape correlation w_++(r_p) (eq. 19 of Singh et al.
        2023 on the ++ multipoles of ``_multipoles_pp``); ``rp`` = points, or
        edges when bin_avg (annular bin average). In the Pi_max -> infinity
        limit this is the Limber form (1/2pi) Int k dk [P_A J_0 + P_B J_4]."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        if self.bin_avg:
            return self.bin_averaged_2D(rp, self._compute_wpp_at_points)
        return self._compute_wpp_at_points(rp)

    def compute_wxx(self, rp, b1, b2, a1, a2, bTA):
        """Projected cross-shape correlation w_xx(r_p): as ``compute_wpp`` with
        the sign of the spin-4 (P_EE - P_BB) term flipped."""
        self._update_pks2d_if_needed(b1, b2, a1, a2, bTA)
        if self.bin_avg:
            return self.bin_averaged_2D(rp, lambda r: self._compute_wpp_at_points(r, True))
        return self._compute_wpp_at_points(rp, True)
