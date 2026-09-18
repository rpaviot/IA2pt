# Conventions

The short list of things that are easy to get wrong when combining this
package with a measurement or another code.

## Units

$h^{-1}$ Mpc for every separation ($s$, $r_p$, $\Pi$), $h$ Mpc$^{-1}$ for
wavenumbers, $(h^{-1}$ Mpc$)^3$ for power spectra. The cosmology is passed as
`[Omega_c, Omega_b, sum m_nu (eV), A_s, n_s, h, z_eff]`; pyccl's $k$ in
Mpc$^{-1}$ and $P$ in Mpc$^3$ are converted internally.

## Sign of the IA amplitude

$e_+$ is positive for **radial** alignment (a galaxy pointing at its
neighbour), i.e. $e_+ = -e_t$ of the lensing convention. A positive `a1`
therefore gives a positive $\tilde\xi_{2,2}$ and a positive $w_{g+}$, as in
Singh et al. 2015, 2023. Estimators built on the lensing tangential shear
(e.g. treecorr `NG` correlations) return $w_{g+}$ with the opposite sign:
flip such a measurement before fitting, or negate the fitted `a1`.

## Normalisation of `a1`, `a2`, `bTA`

`a1` is the pyccl `translate_IA_norm` amplitude: for NLA it is $A_{\rm IA}$
with $C_1 \rho_{\rm crit} = 0.0134$, so $P_{gI} = -a_1 C_1 \rho_{\rm crit}
\Omega_m / D(z)\, b_1 P_{\delta\delta}$ at $\mu = 0$. `a2` is the TATT
tidal-torquing amplitude in the same normalisation (`Om_m2_for_c2=False`,
i.e. $C_2 \propto \Omega_m$, not $\Omega_m^2$) and `bTA` the density-weighting
bias, $a_{1\delta} = a_1 b_{\rm TA}$. In TATT the tidal and third-order
biases of the density tracer are tied to $b_1$: $b_s = -\tfrac{4}{7}(b_1-1)$,
$b_{3nl} = b_1 - 1$; both are zero in NLA.

## Redshift-space distortions

Only the density side carries a Kaiser factor $(1 + \beta\mu^2)$,
$\beta = f/b_1$; the shape field has no velocity term at this order, so
galaxy–shape statistics get one factor (Singh et al. 2023 eqs. 16–17) and
shape–shape statistics none. `do_rsd=False` sets $\beta = 0$ everywhere.

## Line-of-sight dependence of the IA spectra

The sky projection of the tidal field gives $P_{gE}(k,\mu) \propto
(1-\mu^2)$ and, for the linear-alignment term, $P_{EE}(k,\mu) \propto
(1-\mu^2)^2$. For $P_{gE}$ this is exact for any tensor field at the
two-point level (there is a single isotropic rank-2 structure), and the
package puts it in exactly through the associated-Legendre expansion. For
$P_{EE}$ and $P_{BB}$ it is exact only at linear order: the one-loop TATT
terms have several helicity components with different $\mu$ dependence,
and FAST-PT computes them at $\mu = 0$ (the Limber configuration). Following
Maion et al. 2024 ([arXiv:2307.13754](https://arxiv.org/abs/2307.13754),
sect. 3.2.3) the package multiplies the full $P_{EE}$ and $P_{BB}$ by
$(1-\mu^2)^2$ — the helicity-0 approximation. It affects only the
shape–shape statistics in TATT; galaxy–shape statistics are unaffected.

## Wedges

The multipoles are computed with the $r_p \ge$ `rp_min_wedge` cut of Singh
et al. 2023 eq. 18, i.e. from $\xi(s,\mu)$ with the small-$r_p$ (Fingers-of-God
dominated) region removed. This must match the measurement: a wedged
measurement fitted with an unwedged model (or vice versa) is biased at small
$s$. Set `rp_min_wedge=0` for a plain multipole.

## Bin averages

With `bin_avg=True` every statistic takes bin **edges** and returns the
average of the model over each bin: annular ($\propto r\,dr$) for the
projected statistics, spherical ($\propto r^2 dr$) for the multipoles. Pair
counts weight bins this way, so bin averages are the right comparison for
wide logarithmic bins; point values at the bin centres are adequate for
narrow ones.

## Covariance handling in the likelihood

`jack=True` applies the $(n - n_{\rm data} - 2)/(n-1)$ factor of Hartlap et
al. 2007 to the inverse covariance. It is appropriate for a covariance
estimated from $n$ independent realisations; for a delete-one jackknife over
$n$ patches the realisations are strongly correlated and the factor
over-corrects, so the UNIONS × DESI fits use `jack=False`. `zero_cross_cov`
discards the covariance between different statistics; `taper_scale` applies
the tapering of Paz & Sánchez 2015. Whatever the choice, keep it fixed across
fits that are to be compared.
