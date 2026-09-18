"""ia2pt.gausscov: kernel identities, the shot-noise-only analytic limit, and a
synthetic characterisation fixture of both covariance classes.

The refactor out of scripts/compute_gaussian_cov*.py was validated bit-for-bit
against the stored LRG z1 covariances of the UNIONS x DESI analysis
(2026-09-18); the fixture here pins the same code on synthetic inputs so the
package can be tested without the catalogues (tests/make_reference_gausscov.py).
"""
from pathlib import Path

import numpy as np
import pytest

from ia2pt import gausscov as gc

REF = Path(__file__).parent / "data" / "ia2pt_reference_gausscov.npz"


def _uniform_profile(z_edges, nbar, area_sr, chi_of_z):
    """density_profile of a uniform sample with unit weights: nbar_w = nbar_eff = nbar."""
    chi = chi_of_z(z_edges)
    v = area_sr * np.diff(chi**3) / 3.0
    return np.full_like(v, nbar), np.full_like(v, nbar), v


def chi_flat(z):
    """Comoving distance of a flat Om=0.31 cosmology in h^-1 Mpc (astropy-free)."""
    from scipy.integrate import quad
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([2997.92458 * quad(lambda x: 1 / np.sqrt(0.31 * (1 + x) ** 3 + 0.69), 0, zz)[0]
                     for zz in z])


# ------------------------------------------------------------- kernel closures
def test_jbar_closure():
    """(1/2 pi^2) Int k^2 jbar_l(k; i) jbar_l(k; j) dk = delta_ij / V_bin."""
    s_edges = np.geomspace(20.0, 80.0, 7)
    k = np.concatenate([np.geomspace(1e-4, 0.1, 400, endpoint=False), np.arange(0.1, 60.0, 2e-3)])
    v_bin = 4 * np.pi * np.diff(s_edges**3) / 3
    for ell in (0, 2):
        jb = gc.jbar(ell, k, s_edges[:-1], s_edges[1:])
        w = k**2 * np.gradient(k) / (2 * np.pi**2)
        closure = (jb * w[:, None]).T @ jb * v_bin[:, None]
        np.testing.assert_allclose(closure, np.eye(len(v_bin)), atol=2e-2)


def test_Jbar_closure():
    """Int kp dkp Jbar_m(kp; i) Jbar_m(kp; j) / 2pi = delta_ij / A_bin."""
    rp_edges = np.geomspace(5.0, 60.0, 8)
    kp = np.concatenate([np.geomspace(1e-4, 0.1, 300, endpoint=False), np.arange(0.1, 80.0, 2.5e-3)])
    a_bin = np.pi * np.diff(rp_edges**2)
    for m in (0, 2):
        jb = gc.Jbar(m, kp, rp_edges[:-1], rp_edges[1:])
        w = kp * np.gradient(kp) / (2 * np.pi)
        closure = (jb * w[:, None]).T @ jb * a_bin[:, None]
        np.testing.assert_allclose(closure, np.eye(len(a_bin)), atol=2e-2)


# ------------------------------------------------- shot-noise-only analytic limit
def test_shot_noise_limit():
    """With P = 0 the covariances are the analytic white terms: for a uniform
    density nbar over a volume V_s, cov(xi0)_ii = 2 / (nbar^2 V_s V_bin),
    cov(xi~22)_ii = (5/24) sigma^2 / (nbar_g nbar_s V_s V_bin), and the
    projected analogues with V_bin -> A_bin / (2 Pi_max)."""
    z_edges = np.arange(0.4, 0.7501, 0.01)
    area_sr = 0.3
    ng, ns, sig2 = 5e-4, 2e-4, 0.05
    ng_w, ng_eff, v = _uniform_profile(z_edges, ng, area_sr, chi_flat)
    ns_w, ns_eff, _ = _uniform_profile(z_edges, ns, area_sr, chi_flat)
    V = v.sum()
    cgg = gc.gg_coeffs(ng_w, ng_eff, v)
    cgp = gc.gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sig2)
    cx = gc.cross_coeffs(ng_w, ng_eff, ns_w, v)

    s_edges = np.geomspace(10.0, 100.0, 6)
    G = gc.GaussianCov(s_edges, area_sr, kmax=2.0, n_mu=20)
    zero = {"gg": np.zeros((len(G.k), len(G.mu))), "gE": np.zeros((len(G.k), len(G.mu))),
            "EE": np.zeros((len(G.k), len(G.mu)))}
    c00, c22, c02, comb = G.blocks(zero, zero, cgg, cgp, cx)
    np.testing.assert_allclose(np.diag(c00), 2.0 / (ng**2 * V * G.v_bin), rtol=1e-12)
    np.testing.assert_allclose(np.diag(c22), (5.0 / 24.0) * sig2 / (ng * ns * V * G.v_bin), rtol=1e-12)
    assert np.all(c02 == 0)
    np.testing.assert_allclose(comb[len(G.v_bin):, len(G.v_bin):], c22)

    rp_edges = np.geomspace(5.0, 60.0, 6)
    P = gc.GaussianCovProjected(rp_edges, area_sr, kmax=2.0, pimax=100.0)
    zero = {"gg": np.zeros_like(P.k2d), "gE": np.zeros_like(P.k2d), "EE": np.zeros_like(P.k2d)}
    cgg_, cgp_, cx_, comb = P.blocks(zero, zero, cgg, cgp, cx)
    np.testing.assert_allclose(np.diag(cgg_), 2.0 / (ng**2 * V) * 2 * P.pimax / P.a_bin, rtol=1e-12)
    np.testing.assert_allclose(np.diag(cgp_), sig2 / (ng * ns * V) * 2 * P.pimax / P.a_bin, rtol=1e-12)
    assert np.all(cx_ == 0)


# --------------------------------------------------------- characterisation
@pytest.fixture(scope="module")
def synthetic():
    """LRG-like synthetic sample: Gaussian n(z), unit-ish weights, sigma_gamma = 0.21."""
    rng = np.random.default_rng(1)
    z_edges = np.arange(0.4, 0.7501, 0.01)
    area_sr = 0.3667
    zg = rng.normal(0.58, 0.08, 200_000); zg = zg[(zg > 0.4) & (zg < 0.75)]
    wg = rng.lognormal(0.0, 0.1, zg.size)
    zs = rng.normal(0.56, 0.08, 40_000); zs = zs[(zs > 0.4) & (zs < 0.75)]
    ws = rng.lognormal(0.0, 0.2, zs.size)
    return dict(z_edges=z_edges, area_sr=area_sr, zg=zg, wg=wg, zs=zs, ws=ws, sig2=0.0456)


def _cosmo():
    import pyccl as ccl
    ref = dict(np.load(Path(__file__).parent / "data" / "ia2pt_reference_lrg_z1.npz"))
    Omc, Omb, mnu, As, ns, h, _ = ref["cosmology"]
    return ccl.Cosmology(Omega_c=Omc, Omega_b=Omb, h=h, A_s=As, n_s=ns,
                         transfer_function="boltzmann_camb", matter_power_spectrum="camb",
                         extra_parameters={"camb": {"halofit_version": "mead2020"}})


def compute_synthetic(syn):
    ng_w, ng_eff, v = gc.density_profile(syn["zg"], syn["wg"], syn["z_edges"], syn["area_sr"], chi_flat)
    ns_w, ns_eff, _ = gc.density_profile(syn["zs"], syn["ws"], syn["z_edges"], syn["area_sr"], chi_flat)
    cgg = gc.gg_coeffs(ng_w, ng_eff, v)
    cgp = gc.gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, syn["sig2"])
    cx = gc.cross_coeffs(ng_w, ng_eff, ns_w, v)
    cosmo = _cosmo()
    out = {}
    G = gc.GaussianCov(np.geomspace(6.0, 100.0, 11), syn["area_sr"], kmax=5.0)
    pk_gg, pk_ia = G.pk_model(cosmo, 0.58, 1.9, 1.0), G.pk_model(cosmo, 0.56, 1.9, 1.0)
    out["m_xi0"], out["m_xi2"], out["m_x"], out["m_comb"] = G.blocks(pk_gg, pk_ia, cgg, cgp, cx)
    tpl, f = G.xi0_template(cosmo, 0.58)
    out["template"], out["f"] = tpl, f
    P = gc.GaussianCovProjected(np.geomspace(6.0, 100.0, 11), syn["area_sr"], kmax=5.0, pimax=100.0)
    pk_gg, pk_ia = P.pk_model(cosmo, 0.58, 1.9, 1.0), P.pk_model(cosmo, 0.56, 1.9, 1.0)
    out["p_wp"], out["p_wgp"], out["p_x"], out["p_comb"] = P.blocks(pk_gg, pk_ia, cgg, cgp, cx)
    return out


def test_characterisation(synthetic):
    ref = dict(np.load(REF))
    out = compute_synthetic(synthetic)
    for k, v in out.items():
        np.testing.assert_allclose(v, ref[k], rtol=1e-8, err_msg=k)
    # the cross blocks are non-zero with a1 != 0 and negative-signed like the JK
    assert np.any(out["m_x"] != 0) and np.any(out["p_x"] != 0)


def test_fit_b1_roundtrip():
    """fit_b1 recovers b1 from a Kaiser-amplitude-scaled template."""
    s_mid = np.geomspace(8, 90, 10)
    tpl = 1.0 / s_mid**1.8
    f, b1 = 0.79, 1.9
    K = b1**2 + 2 * b1 * f / 3 + f**2 / 5
    assert abs(gc.fit_b1(K * tpl, np.ones_like(tpl), tpl, f, s_mid, (5, 100)) - b1) < 1e-12
