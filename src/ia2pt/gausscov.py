"""Gaussian (disconnected) covariance of the clustering + IA two-point statistics.

Two estimators are covered, each with its clustering block, its IA block and
the cross block between the two:

* ``GaussianCov`` -- the multipoles ``[xi_0, xi~_{2,2}]``: Grieb et al. (2016,
  arXiv:1509.04293) generalised to the spin-2 IA quadrupole (see also Kurita &
  Takada 2022, arXiv:2202.11839 for the IA power-spectrum covariance);
* ``GaussianCovProjected`` -- the projected statistics ``[w_p, w_g+]`` with the
  exact finite-Pi_max line-of-sight window and annulus-averaged Bessel kernels.

Formalism
---------
Grieb et al. eq. (16)+(18): the multipole covariance is

    C_{l1,l2}(s_i, s_j) = i^{l1+l2}/(2 pi^2) Int dk k^2 sigma2_{l1,l2}(k)
                          jbar_{l1}(k s_i) jbar_{l2}(k s_j),

with ``jbar_l`` the VOLUME-averaged spherical Bessel function over the s bin
(eq. 19, closed forms in ``jbar``) and sigma2 the per-mode power-spectrum
variance, written uniformly as

    sigma2_{l1,l2}(k) = 1/2 Int_{-1}^{1} dmu c_{l1}(mu) c_{l2}(mu) Q(k, mu),

where ``c_l(mu)`` is the per-side estimator kernel and ``Q`` the per-mode
covariance divided by the volume:

* clustering monopole: c_0 = 1, per-mode cov 2 [P_gg + 1/nbar_g]^2 (Grieb eq. 15);
* IA quadrupole: the estimator is the associated-Legendre spin-2 multipole
  xi~_{2,2} = (5/2)(1/24) Int dmu 3(1-mu^2) xi_g+(s, mu), so the per-side kernel
  is c_2 = (5/24) P_2^2 = (5/8)(1-mu^2) with jbar_2, and the per-mode cov is
  [P_gg + 1/nbar_g][P_EE + sigma_gamma^2/nbar_s] + P_gE^2 (sigma_gamma the
  per-component ellipticity dispersion, weighted like the estimator);
* cross block: per-mode cov 2 [P_gg + 1/nbar_g] P_gE, kernels c_0 x c_2,
  prefactor i^2 = -1.

Varying number density: rather than a constant nbar and an effective volume,
the FKP-style local approximation is integrated over the survey shells
(``density_profile`` + ``gg_coeffs`` / ``gp_coeffs`` / ``cross_coeffs``), e.g.

    Q_gg(k,mu) = Int dV [nbar_w^2(z) (P_gg + 1/nbar_eff(z))]^2 / [Int dV nbar_w^2]^2

with nbar_w = Sum w / V_shell the pair-weighting density and the SHOT-NOISE
density nbar_eff = (Sum w)^2 / Sum w^2 / V_shell. The weights must mirror the
estimator's (e.g. completeness x FKP on the density side, shape weights on the
shape side). Pure white (P-independent) terms are integrated analytically
(Int k^2 jbar_l jbar_l dk = 2 pi^2 delta_ij / V_bin), so they carry no
k-truncation error.

Projected statistics: writing w_A(rp_i) = Int_{-Pi}^{Pi} dpi xi_A(rp, pi)
(annulus-averaged) in Fourier space,

    C_AB(rp_i, rp_j) = sign_AB Int_0^inf dkp kp/(2 pi) Jbar_a(kp; i) Jbar_b(kp; j)
                       x Int dkz/(2 pi) W(kz)^2 Q_AB(k, mu)

with W(kz) = 2 sin(kz Pi) / kz the finite-Pi_max window, ``Jbar_m`` the
AREA-averaged Bessel kernels (J_0 for the density side, -J_2 for the spin-2
shear side, whence sign -1 on the cross block) and the same Q_AB as above.

P(k, mu) model (``kaiser_nla_pk``): pyccl non-linear P_dd at the block's
effective redshift, Kaiser RSD (b1 + f mu^2)^2 on the density side, NLA
P_gE = (b1 + f mu^2) F (1 - mu^2) P_dd and P_EE = F^2 (1 - mu^2)^2 P_dd with
F = -a1 C1 rho_crit Omega_m / D(z). The signal terms are negligible next to
the shape noise for typical samples, so a1 = 0 is a safe default.

Units: h^-1 Mpc, h Mpc^-1, (h^-1 Mpc)^3 throughout.
"""

import numpy as np
from scipy.special import j0, j1, sici

__all__ = ["jbar", "Jbar", "density_profile", "gg_coeffs", "gp_coeffs",
           "cross_coeffs", "nonlin_pdd", "kaiser_nla_pk", "GaussianCov",
           "GaussianCovProjected", "fit_b1", "C1RHOC"]

C1RHOC = 0.0134  # C1 rho_crit of the NLA amplitude (pyccl translate_IA_norm)


# =============================================================================
# BIN-AVERAGED BESSEL KERNELS (closed forms)
# =============================================================================
def _int_x2j0(x):
    """Int x^2 j0(x) dx = sin x - x cos x."""
    return np.sin(x) - x * np.cos(x)


def _int_x2j2(x):
    """Int x^2 j2(x) dx = x cos x - 4 sin x + 3 Si(x)."""
    si, _ = sici(x)
    return x * np.cos(x) - 4.0 * np.sin(x) + 3.0 * si


def jbar(ell, k, s_lo, s_hi):
    """Volume-averaged spherical Bessel j_ell over the [s_lo, s_hi] shells
    (Grieb et al. 2016 eq. 19).

    Parameters
    ----------
    ell : {0, 2}
    k : array (Nk,)
    s_lo, s_hi : arrays (Nbin,)

    Returns
    -------
    array (Nk, Nbin)
    """
    x_lo = np.outer(k, s_lo)
    x_hi = np.outer(k, s_hi)
    anti = {0: _int_x2j0, 2: _int_x2j2}[ell]
    return 3.0 * (anti(x_hi) - anti(x_lo)) / (x_hi**3 - x_lo**3)


def _int_xj0(x):
    """Int x J0(x) dx = x J1(x)."""
    return x * j1(x)


def _int_xj2(x):
    """Int x J2(x) dx = -2 J0(x) - x J1(x)   (since x J2 = 2 J1 - x J0)."""
    return -2.0 * j0(x) - x * j1(x)


def Jbar(m, kp, rp_lo, rp_hi):
    """Area-averaged Bessel J_m over the [rp_lo, rp_hi] annuli.

    Parameters
    ----------
    m : {0, 2}
    kp : array (Nk,) transverse wavenumbers
    rp_lo, rp_hi : arrays (Nbin,)

    Returns
    -------
    array (Nk, Nbin)
    """
    x_lo = np.outer(kp, rp_lo)
    x_hi = np.outer(kp, rp_hi)
    anti = {0: _int_xj0, 2: _int_xj2}[m]
    return 2.0 * (anti(x_hi) - anti(x_lo)) / (x_hi**2 - x_lo**2)


# =============================================================================
# SURVEY GEOMETRY: n(z) shells -> density integrals
# =============================================================================
def density_profile(z, w, z_edges, area_sr, chi_of_z):
    """Per-shell weighted density and shot-noise density of a sample.

    Parameters
    ----------
    z, w : arrays
        Redshifts and weights of the objects (weights as in the estimator).
    z_edges : array
        Shell edges.
    area_sr : float
        Footprint solid angle [sr].
    chi_of_z : callable
        Comoving distance [h^-1 Mpc] of an array of redshifts.

    Returns
    -------
    nw, neff, v : arrays (Nshell,)
        nbar_w = Sum w / V_shell (pair-weighting density), nbar_eff =
        (Sum w)^2 / Sum w^2 / V_shell (shot-noise density) and the shell
        volumes [(h^-1 Mpc)^3]. Empty shells get 0 and contribute nothing.
    """
    chi = np.asarray(chi_of_z(np.asarray(z_edges, dtype=float)), dtype=float)
    v = area_sr * np.diff(chi**3) / 3.0
    sw, _ = np.histogram(z, bins=z_edges, weights=w)
    sw2, _ = np.histogram(z, bins=z_edges, weights=w**2)
    nw = np.where(v > 0, sw / v, 0.0)
    neff = np.divide(sw**2, sw2 * v, out=np.zeros_like(sw), where=sw2 * v > 0)
    return nw, neff, v


# =============================================================================
# PER-MODE VARIANCE Q(k, mu) COEFFICIENTS (dV integrals over the shells)
# =============================================================================
def gg_coeffs(nw, neff, v):
    """Density-density coefficients: Q_gg = 2 [c4 P^2 + 2 c3 P + c2] / N^2
    (c2 is the analytic white term). Arguments from ``density_profile``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(neff > 0, 1.0 / neff, 0.0)
    return {
        "c4": np.sum(nw**4 * v),
        "c3": np.sum(nw**4 * inv * v),
        "c2": np.sum(nw**4 * inv**2 * v),
        "N": np.sum(nw**2 * v),
    }


def gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sig2):
    """Density-shape coefficients: Q_g+ = [dPP (P_gg P_EE + P_gE^2) + dgn P_gg
    + dsn P_EE + dwhite] / M^2. ``sig2`` is the per-component ellipticity
    variance sigma_gamma^2 of the shape sample (weighted like the estimator)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_g = np.where(ng_eff > 0, 1.0 / ng_eff, 0.0)
        inv_s = np.where(ns_eff > 0, sig2 / ns_eff, 0.0)
    b2 = ng_w**2 * ns_w**2
    return {
        "dPP": np.sum(b2 * v),
        "dgn": np.sum(b2 * inv_s * v),          # P_gg x shape noise
        "dsn": np.sum(b2 * inv_g * v),          # P_EE x g shot noise
        "dwhite": np.sum(b2 * inv_g * inv_s * v),
        "M": np.sum(ng_w * ns_w * v),
    }


def cross_coeffs(ng_w, ng_eff, ns_w, v):
    """Clustering x IA cross coefficients: Q_x = 2 [x1 P_gg P_gE + x0 P_gE] / (N M)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_g = np.where(ng_eff > 0, 1.0 / ng_eff, 0.0)
    return {
        "x1": np.sum(ng_w**3 * ns_w * v),
        "x0": np.sum(ng_w**3 * ns_w * inv_g * v),
    }


# =============================================================================
# P(k, mu) MODEL
# =============================================================================
def nonlin_pdd(cosmo_ccl, k, z_eff):
    """pyccl non-linear matter power spectrum [(h^-1 Mpc)^3] at k [h Mpc^-1]."""
    import pyccl as ccl
    h = cosmo_ccl["h"]
    return h**3 * ccl.nonlin_power(cosmo_ccl, np.asarray(k, dtype=float) * h,
                                   1.0 / (1.0 + z_eff))


def kaiser_nla_pk(cosmo_ccl, z_eff, b1, a1, pdd, mu2):
    """Kaiser + NLA P(k, mu) grids at ``z_eff``.

    Parameters
    ----------
    cosmo_ccl : pyccl.Cosmology
    z_eff : float
    b1 : float
        Linear galaxy bias of the density sample.
    a1 : float
        NLA amplitude (pyccl ``translate_IA_norm`` convention).
    pdd, mu2 : arrays
        Non-linear matter power [(h^-1 Mpc)^3] (see ``nonlin_pdd``) and mu^2
        on grids broadcastable against each other.

    Returns
    -------
    dict
        ``gg`` = (b1 + f mu^2)^2 P_dd, ``gE`` = (b1 + f mu^2) F (1 - mu^2) P_dd,
        ``EE`` = F^2 (1 - mu^2)^2 P_dd with F = -a1 C1 rho_crit Omega_m / D(z),
        plus the growth rate ``f`` and growth factor ``D`` at z_eff.
    """
    import pyccl as ccl
    import pyccl.background as pb
    a = 1.0 / (1.0 + z_eff)
    f = pb.growth_rate(cosmo_ccl, a)
    D = ccl.growth_factor(cosmo_ccl, a)
    om = ccl.omega_x(cosmo_ccl, 1.0, "matter")
    F = -a1 * C1RHOC * om / D
    kai = b1 + f * mu2
    e = 1.0 - mu2
    return {"gg": kai**2 * pdd, "gE": kai * F * e * pdd,
            "EE": F**2 * e**2 * pdd, "f": f, "D": D}


# =============================================================================
# MULTIPOLES [xi0, xi~_{2,2}]
# =============================================================================
def k_grid(kmax):
    """Hybrid log+linear k grid [h/Mpc]: log to 0.1, then dk=0.002 to kmax."""
    return np.concatenate([np.geomspace(1e-4, 0.1, 400, endpoint=False),
                           np.arange(0.1, kmax, 2e-3)])


def cov_from_sigma2(k, sigma2_k, jb1, jb2, sign=1.0):
    """C_ij = sign/(2 pi^2) Int dk k^2 sigma2(k) jbar1(k,s_i) jbar2(k,s_j)."""
    dk = np.gradient(k)
    w = k**2 * sigma2_k * dk / (2.0 * np.pi**2)
    return sign * (jb1 * w[:, None]).T @ jb2


class GaussianCov:
    """Gaussian covariance of the multipole data vector [xi_0, xi~_{2,2}].

    Parameters
    ----------
    s_edges : array (Nbin+1,)
        Separation bin edges [h^-1 Mpc] (shared by the two statistics).
    area_sr : float
        Footprint solid angle [sr] (only stored; the density integrals are
        done by ``density_profile`` with the same area).
    kmax : float
        Upper limit of the k integrals [h Mpc^-1] (default 10).
    n_mu : int
        Gauss-Legendre nodes of the mu integrals (default 60).

    Notes
    -----
    Usage: ``pk_model`` at the clustering and IA effective redshifts, the
    coefficient dicts from ``gg_coeffs`` / ``gp_coeffs`` / ``cross_coeffs``,
    then ``blocks``.
    """

    def __init__(self, s_edges, area_sr, kmax=10.0, n_mu=60):
        s_edges = np.asarray(s_edges, dtype=float)
        self.s_lo, self.s_hi = s_edges[:-1], s_edges[1:]
        self.v_bin = 4.0 * np.pi * (self.s_hi**3 - self.s_lo**3) / 3.0
        self.area_sr = area_sr
        self.k = k_grid(kmax)
        self.mu, self.mu_w = np.polynomial.legendre.leggauss(n_mu)
        self.jb0 = jbar(0, self.k, self.s_lo, self.s_hi)
        self.jb2 = jbar(2, self.k, self.s_lo, self.s_hi)

    def pk_model(self, cosmo_ccl, z_eff, b1, a1):
        """P(k, mu) grids (Nk, Nmu) at z_eff -- see ``kaiser_nla_pk``."""
        pdd = nonlin_pdd(cosmo_ccl, self.k, z_eff)[:, None]
        return kaiser_nla_pk(cosmo_ccl, z_eff, b1, a1, pdd, (self.mu**2)[None, :])

    def _mu_int(self, kern, Q):
        """1/2 Int dmu kern(mu) Q(k,mu) via Gauss-Legendre: (Nk,)."""
        return 0.5 * np.einsum("m,km->k", kern * self.mu_w, Q)

    def blocks(self, pk_gg, pk_ia, cgg, cgp, cx):
        """All covariance blocks.

        Parameters
        ----------
        pk_gg, pk_ia : dict
            ``pk_model`` outputs at the clustering and IA effective redshifts.
        cgg, cgp, cx : dict
            ``gg_coeffs``, ``gp_coeffs`` and ``cross_coeffs`` outputs.

        Returns
        -------
        cov00, cov22, cov02, comb : arrays
            xi0 x xi0, xi~22 x xi~22, xi0 x xi~22 (Nbin x Nbin) and the
            combined [xi0, xi~22] matrix (2 Nbin x 2 Nbin).
        """
        one = np.ones_like(self.mu)
        c0 = one                                   # (2l+1) L_0
        c2 = (5.0 / 8.0) * (1.0 - self.mu**2)      # (5/24) P_2^2

        # xi0 x xi0: per-mode 2 [P_gg + 1/n]^2 -> Q = [c4 P^2 + 2 c3 P + c2w]/N^2
        P = pk_gg["gg"]
        Q = 2.0 * (cgg["c4"] * P**2 + 2.0 * cgg["c3"] * P) / cgg["N"] ** 2
        s2 = self._mu_int(c0 * c0, Q)
        cov00 = cov_from_sigma2(self.k, s2, self.jb0, self.jb0)
        white00 = 2.0 * cgg["c2"] / cgg["N"] ** 2          # sigma2_B (mu-int of 1 = 2 -> 1/2*2*2c2)
        cov00 += np.diag(white00 / self.v_bin)

        # xi2 x xi2: per-mode [P_gg + 1/n_g][P_EE + sig^2/n_s] + P_gE^2
        Pg, PE, PgE = pk_ia["gg"], pk_ia["EE"], pk_ia["gE"]
        Q = (cgp["dPP"] * (Pg * PE + PgE**2) + cgp["dgn"] * Pg
             + cgp["dsn"] * PE) / cgp["M"] ** 2
        s2 = self._mu_int(c2 * c2, Q)
        cov22 = cov_from_sigma2(self.k, s2, self.jb2, self.jb2)
        white22 = (5.0 / 24.0) * cgp["dwhite"] / cgp["M"] ** 2
        cov22 += np.diag(white22 / self.v_bin)

        # xi0 x xi2 cross: per-mode 2 [P_gg + 1/n_g] P_gE, prefactor i^2 = -1
        Q = 2.0 * (cx["x1"] * pk_ia["gg"] * pk_ia["gE"]
                   + cx["x0"] * pk_ia["gE"]) / (cgg["N"] * cgp["M"])
        s2 = self._mu_int(c0 * c2, Q)
        cov02 = cov_from_sigma2(self.k, s2, self.jb0, self.jb2, sign=-1.0)

        n = len(self.v_bin)
        comb = np.zeros((2 * n, 2 * n))
        comb[:n, :n] = cov00
        comb[n:, n:] = cov22
        comb[:n, n:] = cov02
        comb[n:, :n] = cov02.T
        return cov00, cov22, cov02, comb

    def xi0_template(self, cosmo_ccl, z_eff):
        """Bin-averaged xi0 of P_dd (b = 1, no RSD) and the growth rate at
        z_eff -- the template of ``fit_b1``."""
        import pyccl as ccl
        import pyccl.background as pb
        h = cosmo_ccl["h"]
        a = 1.0 / (1.0 + z_eff)
        pdd = h**3 * ccl.nonlin_power(cosmo_ccl, self.k * h, a)
        dk = np.gradient(self.k)
        x = (self.jb0 * (self.k**2 * pdd * dk / (2 * np.pi**2))[:, None]).sum(axis=0)
        return x, pb.growth_rate(cosmo_ccl, a)


def fit_b1(xi0_meas, var, template, f, s_mid, s_range):
    """Closed-form linear bias from the measured monopole.

    Fits the Kaiser amplitude K = b1^2 + 2 b1 f / 3 + f^2 / 5 of the
    ``xi0_template`` to ``xi0_meas`` over ``s_range`` (inverse-variance
    weighted by ``var``) and solves for b1.
    """
    m = (s_mid >= s_range[0]) & (s_mid <= s_range[1]) & (var > 0)
    K = np.sum(template[m] * xi0_meas[m] / var[m]) / np.sum(template[m] ** 2 / var[m])
    disc = f**2 / 9.0 - f**2 / 5.0 + K
    if disc <= 0:
        raise RuntimeError(f"b1 fit failed: Kaiser amplitude K={K:.3f} too small")
    return -f / 3.0 + np.sqrt(disc)


# =============================================================================
# PROJECTED [w_p, w_g+]
# =============================================================================
def kp_grid(kmax):
    """Transverse k grid [h/Mpc]: log to 0.1, then dk=2.5e-3 to kmax."""
    return np.concatenate([np.geomspace(1e-4, 0.1, 300, endpoint=False),
                           np.arange(0.1, kmax, 2.5e-3)])


def kz_grid(pimax, kmax, kz_split=1.0):
    """LOS grid + integration weights for (1/2pi) Int_{-inf}^{inf} dkz W(kz)^2.

    Fine linear grid to kz_split resolving the sin^2(kz Pi) oscillation
    (~40 nodes per period), then a log tail with W^2 -> its envelope mean
    2/kz^2 (sin^2 -> 1/2). Weights include the factor 2 for +-kz."""
    dkz = np.pi / pimax / 40.0
    kz_f = np.arange(dkz / 2.0, kz_split, dkz)
    w_f = 2.0 * dkz * (2.0 * np.sin(kz_f * pimax) / kz_f) ** 2 / (2.0 * np.pi)
    kz_t = np.geomspace(kz_split, kmax, 200)
    w_t = 2.0 * np.gradient(kz_t) * (2.0 / kz_t**2) / (2.0 * np.pi)
    return np.concatenate([kz_f, kz_t]), np.concatenate([w_f, w_t])


class GaussianCovProjected:
    """Gaussian covariance of the projected data vector [w_p, w_g+].

    Parameters
    ----------
    rp_edges : array (Nbin+1,)
        Transverse bin edges [h^-1 Mpc] (shared by the two statistics).
    area_sr : float
        Footprint solid angle [sr] (stored only, see ``GaussianCov``).
    kmax : float
        Upper limit of the k integrals [h Mpc^-1] (default 10).
    pimax : float
        Line-of-sight integration limit of the estimator [h^-1 Mpc].

    Notes
    -----
    Usage as ``GaussianCov``: ``pk_model`` at the two effective redshifts,
    coefficient dicts, ``blocks``.
    """

    def __init__(self, rp_edges, area_sr, kmax=10.0, pimax=100.0):
        rp_edges = np.asarray(rp_edges, dtype=float)
        self.rp_lo, self.rp_hi = rp_edges[:-1], rp_edges[1:]
        self.a_bin = np.pi * (self.rp_hi**2 - self.rp_lo**2)
        self.area_sr = area_sr
        self.pimax = pimax
        self.kp = kp_grid(kmax)
        self.kz, self.kz_w = kz_grid(pimax, kmax)
        # 2D geometry (Nkp, Nkz)
        self.k2d = np.sqrt(self.kp[:, None] ** 2 + self.kz[None, :] ** 2)
        self.mu2 = (self.kz[None, :] / self.k2d) ** 2
        self.jb0 = Jbar(0, self.kp, self.rp_lo, self.rp_hi)
        self.jb2 = Jbar(2, self.kp, self.rp_lo, self.rp_hi)

    def pk_model(self, cosmo_ccl, z_eff, b1, a1):
        """P(k, mu) grids (Nkp, Nkz) at z_eff -- see ``kaiser_nla_pk``."""
        # log-log interpolation of the 1D spectrum onto the 2D (kp, kz) grid
        k1d = np.geomspace(self.k2d.min() * 0.99, self.k2d.max() * 1.01, 2048)
        p1d = nonlin_pdd(cosmo_ccl, k1d, z_eff)
        pdd = np.exp(np.interp(np.log(self.k2d), np.log(k1d), np.log(p1d)))
        return kaiser_nla_pk(cosmo_ccl, z_eff, b1, a1, pdd, self.mu2)

    def _cov(self, Q2d, jba, jbb, sign=1.0):
        """kz-integrate Q, then the transverse Hankel pair sum."""
        I = Q2d @ self.kz_w                       # (Nkp,)
        w = self.kp * np.gradient(self.kp) * I / (2.0 * np.pi)
        return sign * (jba * w[:, None]).T @ jbb

    def blocks(self, pk_gg, pk_ia, cgg, cgp, cx):
        """All covariance blocks ([w_p, w_g+] ordering); arguments and
        returns as ``GaussianCov.blocks``."""
        two_pi_los = 2.0 * self.pimax             # analytic Int dkz W^2 / 2pi

        # wp x wp
        P = pk_gg["gg"]
        Q = 2.0 * (cgg["c4"] * P**2 + 2.0 * cgg["c3"] * P) / cgg["N"] ** 2
        cov_gg = self._cov(Q, self.jb0, self.jb0)
        cov_gg += np.diag(2.0 * cgg["c2"] / cgg["N"] ** 2 * two_pi_los / self.a_bin)

        # wgp x wgp
        Pg, PE, PgE = pk_ia["gg"], pk_ia["EE"], pk_ia["gE"]
        Q = (cgp["dPP"] * (Pg * PE + PgE**2) + cgp["dgn"] * Pg
             + cgp["dsn"] * PE) / cgp["M"] ** 2
        cov_gp = self._cov(Q, self.jb2, self.jb2)
        cov_gp += np.diag(cgp["dwhite"] / cgp["M"] ** 2 * two_pi_los / self.a_bin)

        # wp x wgp cross (one -J2 side -> sign -1, like the multipoles' i^2)
        Q = 2.0 * (cx["x1"] * pk_ia["gg"] * pk_ia["gE"]
                   + cx["x0"] * pk_ia["gE"]) / (cgg["N"] * cgp["M"])
        cov_x = self._cov(Q, self.jb0, self.jb2, sign=-1.0)

        n = len(self.a_bin)
        comb = np.zeros((2 * n, 2 * n))
        comb[:n, :n] = cov_gg
        comb[n:, n:] = cov_gp
        comb[:n, n:] = cov_x
        comb[n:, :n] = cov_x.T
        return cov_gg, cov_gp, cov_x, comb
