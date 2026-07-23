from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed

from beetroots.modelling.component_distribution import ComponentDistribution


class TargetDistribution(ABC):
    __slots__ = (
        "D",
        "L",
        "N",
        "dict_sites",
        "distribution_components",
        "var_name",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        distribution_components: dict[str:ComponentDistribution],
        var_name: str = "",
        separable: bool = True,
    ):
        self.D = D
        """int: number of distinct physical parameters"""

        self.L = L
        """int: number of observables per pixel"""

        self.N = N
        """int: number of pixels"""

        self.distribution_components = distribution_components
        """dict: dict of all distributions that compose the target distribution"""

        self.var_name = var_name
        """str: name of the variable of the target distribution"""

        if separable is True:  # all terms are independent (separable)
            self.dict_sites = {0: xp.arange(self.N)}
        else:
            self.dict_sites = {n: xp.array([n]) for n in range(self.N)}

        return

    @abstractmethod
    def neglog_pdf(
        self,
        current: dict[str, Union[dict, float, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        pixelwise: bool = False,
        full: bool = False,
        update_nlpdf_utils: bool = True,
        chromatic_gibbs: bool = False,
    ) -> float:
        """
        Should output something
        """
        pass  # NOTE: when calling update_nlpdf_utils (when it is true), it should (in general) deactivate the the computation of the utils for the derivatives to avoid unecessary computations.

    @abstractmethod
    def grad_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        pass

    @abstractmethod
    def hess_diag_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        pass

    @abstractmethod
    def compute_all_for_saver(
        self,
        current: dict[str, dict],
        **kwargs,
    ) -> Tuple[dict[str, Union[float, xp.ndarray]], xp.ndarray]:
        """computes negative log pdf of each component distribution and posterior (detailed values to be saved, not to be used in sampling)

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : [str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_nll_utils()`` method

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            values to be saved
        """
        pass

    @abstractmethod
    def compute_all(
        self,
        current_sampler: dict,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        update_nlpdf_utils: bool = True,
        **kwargs,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the target distribution

        Parameters
        ----------
        nlpdf_utils : dict[str, np.ndarray]
            output of the union of the outputs of the method ``evaluate_all_nll_utils()`` of each component distribution
        compute_derivatives : bool, optional
            whether to compute 1st order derivatives, by default True
        compute_derivatives_second_order : bool, optional
            whether to compute second order derivatives, by default True

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            negative log pdf and derivatives of the posterior distribution
        """
        pass

    def update_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
        **kwargs,
    ) -> None:
        """Update all utilities for the negative log-pdf and its eventual derivatives

        Parameters
        ----------
        current : dict[str, dict]
            current iterate
        idx_pix : np.ndarray, optional
            indices of the pixels, by default None
        compute_derivatives : bool, optional
            whether to compute 1st order derivatives, by default True
        compute_derivatives_2nd_order : bool, optional
            whether to compute 2nd order derivatives, by default True
        mtm : bool, optional
            whether to use the MTM, by default False
        """
        self.mtm = mtm

        for cd in self.distribution_components.values():
            cd.evaluate_all_nlpdf_utils(
                current,
                idx_pix,
                compute_derivatives,
                compute_derivatives_2nd_order,
                mtm,
                **kwargs,
            )
