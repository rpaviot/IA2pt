"""Regenerate tests/data/ia2pt_reference_pp_lrg_z1.npz, the characterisation
fixture of the shape-shape statistics (xi~_{0,0}^{++}, xi~_{4,4}^{++}, w_++,
w_xx) on the LRG z1 grids of ia2pt_reference_lrg_z1.npz.

Unlike the gg / g+ fixture this one is NOT an independent reference: it pins
the ia2pt implementation of 2026-09-18 (validated against the Limber-limit
identity, see test_pp_limber_limit). Rerun only after an intended change.
"""
from pathlib import Path

import numpy as np

from ia2pt import TwoPointModel

HERE = Path(__file__).parent / "data"
ref = dict(np.load(HERE / "ia2pt_reference_lrg_z1.npz"))
out = {}
for cfg in ("NLA", "TATT"):
    p = tuple(ref["pars_tatt" if cfg == "TATT" else "pars_nla"])
    for bin_avg in (False, True):
        tag = "ba" if bin_avg else "pt"
        r_s = ref["s_edges"] if bin_avg else ref["s_mid"]
        r_p = ref["rp_edges"] if bin_avg else ref["rp_mid"]
        for nz in (False, True):
            m = TwoPointModel(list(ref["cosmology"]), cfg, do_rsd=True, pimax=100,
                              rp_min_wedge=5.0, n_mu_wedge=101, bin_avg=bin_avg)
            if nz:
                m.set_nz(ref["z_bins"], ref["z_nz"], ref["nz_clustering"], ref["nz_shape"])
            k = f"{cfg}_{tag}_{'nz' if nz else 'z'}"
            out[f"{k}_xi0pp"] = m.compute_xi_pp_wedge_monopole(r_s, *p)
            out[f"{k}_xi4pp"] = m.compute_xi_pp_wedge_hexadecapole(r_s, *p)
            out[f"{k}_wpp"] = m.compute_wpp(r_p, *p)
            out[f"{k}_wxx"] = m.compute_wxx(r_p, *p)
np.savez(HERE / "ia2pt_reference_pp_lrg_z1.npz", **out)
print("wrote", len(out), "arrays")
