"""ia2pt -- NLA / TATT two-point intrinsic-alignment model and likelihood.

The modelling layer of the UNIONS x DESI intrinsic-alignment analysis
(Paviot et al.): wedged multipoles xi~_{0,0} / xi~_{2,2} / xi~^{++}, projected
w_gg / w_g+ / w_++ / w_xx from pyccl + FAST-PT, a chi^2 likelihood with
jackknife-covariance handling, iminuit / nautilus / emcee drivers, and the
Gaussian covariance of both estimators (``ia2pt.gausscov``).

>>> from ia2pt import TwoPointModel, IALikelihood, run_minuit
>>> from ia2pt.gausscov import GaussianCov, GaussianCovProjected
"""

from .model import TwoPointModel, L00, L20, L40, L22, L42, L44
from .likelihood import IALikelihood, PARAM_NAMES
from .samplers import run_minuit, run_nautilus, run_emcee, weighted_quantile
from . import gausscov

__version__ = "0.2.0"
__all__ = ["TwoPointModel", "IALikelihood", "PARAM_NAMES",
           "run_minuit", "run_nautilus", "run_emcee", "weighted_quantile",
           "gausscov", "L00", "L20", "L40", "L22", "L42", "L44", "__version__"]
