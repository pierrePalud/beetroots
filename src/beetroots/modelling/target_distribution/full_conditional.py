from typing import Dict, Optional, Tuple, Union

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed


class FullConditional(TargetDistribution):
    __slots__ = (
        "D",
        "L",
        "N",
        "distribution_components",
        "var_name",
        "dict_sites",
        "var_shape",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        distribution_components: dict[str, ComponentDistribution],
        var_name: str,
        var_shape: Tuple[int],
        separable: bool = True,
        dict_sites: Optional[Dict[int, xp.ndarray]] = None,
    ):
        super().__init__(D, L, N, distribution_components, var_name=var_name)

        self.var_shape = var_shape
        """Tuple[int]: shape of the variable. Necessary for the computation of the gradient of the negative log pdf of the target distribution."""

        self.dict_sites = {}
        """dict[int, np.ndarray]: sites for pixels to be sampled in parallel in the MTM-chromoatic Gibbs kernel"""
        if dict_sites is not None:
            self.dict_sites = dict_sites
        elif separable is True:
            self.dict_sites = {0: xp.arange(self.N)}
        else:
            self.dict_sites = {n: xp.array([n]) for n in range(self.N)}

        return

    def neglog_pdf(
        self,
        current: dict[str, Union[dict, float, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        pixelwise: bool = False,
        full: bool = False,
        update_nlpdf_utils: bool = True,
        chromatic_gibbs: bool = False,
    ) -> float:
        assert (
            xp.sum([pixelwise, full]) < 2
        ), "pixelwise and full cannot be True at the same time"
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(
                current,
                idx_pix=idx_pix,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
            )

        random_component = list(self.distribution_components.values())[
            0
        ]  # just to extract information that should be present in every component distribution
        is_mtm = random_component.nlpdf_utils["mtm"]
        k_mtm = random_component.nlpdf_utils["k_mtm"]
        n_pix = random_component.nlpdf_utils["n_pix"]

        if full:
            nlpdf_ = (
                xp.zeros((n_pix, *self.var_shape[1:]))
                if not is_mtm
                else xp.zeros((n_pix, k_mtm, *self.var_shape[1:]))
            )
        elif pixelwise:
            nlpdf_ = (
                xp.zeros(n_pix)
                if not is_mtm
                else xp.zeros(
                    (
                        n_pix,
                        k_mtm,
                    )
                )
            )
        else:
            nlpdf_ = 0.0 if not is_mtm else xp.zeros(k_mtm)

        for component in self.distribution_components.values():
            # if is_mtm and isinstance(component, PriorProbaDistribution):
            #     pass
            # else:
            # is_mtm = random_component.nlpdf_utils['mtm']
            # if not(is_mtm and isinstance(component, PriorProbaDistribution)):
            comp_nlpdf = component.neglog_pdf(
                pixelwise=pixelwise, full=full, chromatic_gibbs=chromatic_gibbs
            )
            nlpdf_ += comp_nlpdf

        return nlpdf_

    def grad_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(
                current,
                idx_pix=idx_pix,
                compute_derivatives=True,
                compute_derivatives_2nd_order=False,
            )

        size_ = self.N if idx_pix is None else idx_pix.size
        grad_ = xp.zeros((size_, *self.var_shape[1:]))

        for component in self.distribution_components.values():
            comp_grad = component.gradient_neglog_pdf(deriv_var_name=self.var_name)
            assert (
                comp_grad.shape == grad_.shape
            ), f"comp_grad.shape is {comp_grad.shape}, grad_.shape is {grad_.shape}"
            assert xp.sum(~xp.isfinite(comp_grad)) == 0.0
            grad_ += comp_grad

        return grad_

    def hess_diag_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(
                current,
                idx_pix=idx_pix,
                compute_derivatives=True,
                compute_derivatives_2nd_order=True,
            )

        size_ = self.N if idx_pix is None else idx_pix.size
        hess_diag_ = xp.zeros((size_, *self.var_shape[1:]))

        for component in self.distribution_components.values():
            hess_diag_ += component.hess_diag_neglog_pdf(deriv_var_name=self.var_name)

        return hess_diag_

    def compute_all_for_saver(
        self,
        current: dict[str, dict],
        model_checking_component_name: str,
        save_nll_full: bool = False,
        **kwargs,
    ) -> Tuple[dict[str, Union[float, xp.ndarray]], xp.ndarray]:
        assert xp.sum(xp.isnan(current[self.var_name]["var"])) == 0, xp.sum(
            xp.isnan(current[self.var_name]["var"])
        )

        dict_objective = dict()
        posterior_nlpdf = 0.0
        nll_full = xp.zeros((self.N, self.L))
        for component_name, component in self.distribution_components.items():
            component.evaluate_all_nlpdf_utils(
                current,
                idx_pix=None,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
                mtm=False,
            )
            if isinstance(component, PriorProbaDistribution):
                nll_comp = component.neglog_pdf(paramwise=True, chromatic_gibbs=False)
                nll_comp_float = nll_comp.sum()
                dict_objective["nlpdf_" + component_name] = nll_comp  # (D,)
                posterior_nlpdf += nll_comp_float
            elif component_name == model_checking_component_name:
                nlpdf_comp = component.neglog_pdf(full=True, chromatic_gibbs=False)
                assert isinstance(
                    nlpdf_comp, xp.ndarray
                ), "nlpdf_comp shoud be an array, check the component distribution 'neglog_pdf' method"
                assert nlpdf_comp.shape == (
                    self.N,
                    self.L,
                ), f"nlpdf_comp with wrong shape. is {nlpdf_comp.shape}, should be {(self.N, self.L)}"
                nll_full += nlpdf_comp
                nlpdf_comp_float = nlpdf_comp.sum()
                dict_objective["nll"] = (
                    nlpdf_comp_float  # We assume here the likelihood is the model checking component
                )
                if save_nll_full:
                    dict_objective["nll_full"] = nll_full
                posterior_nlpdf += nlpdf_comp_float
            else:
                nll_comp = component.neglog_pdf(chromatic_gibbs=False)
                assert isinstance(nll_comp, float)
                dict_objective["nlpdf_" + component_name] = nll_comp
                posterior_nlpdf += nll_comp

        dict_objective["objective"] = posterior_nlpdf

        return dict_objective, nll_full

    def compute_all(
        self,
        current: dict[str, Union[dict, float, xp.ndarray]] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        update_nlpdf_utils: bool = True,
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
        assert xp.sum(xp.isnan(current[self.var_name]["var"])) == 0, xp.sum(
            xp.isnan(current[self.var_name]["var"])
        )

        if update_nlpdf_utils:
            self.update_nlpdf_utils(
                current,
                compute_derivatives=compute_derivatives,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
            )

        nlpdf_utils, iterate = dict(), dict()
        posterior_nlpdf_pix_global = xp.zeros(self.N)
        posterior_nlpdf_pix_chromatic = xp.zeros(self.N)
        for component_name, component in self.distribution_components.items():
            if not isinstance(
                component, SpatialPrior
            ):  # TODO: change to make it more generic, (not only spatial priors can use the chromatic_gibbs parameter)
                nlpdf_comp_pixelwise = component.neglog_pdf(
                    pixelwise=True, chromatic_gibbs=False
                )
                posterior_nlpdf_pix_global += nlpdf_comp_pixelwise
                posterior_nlpdf_pix_chromatic += nlpdf_comp_pixelwise
            else:
                nlpdf_comp_pixelwise_chromatic, nlpdf_comp_pixelwise = (
                    component.neglog_pdf(pixelwise=True, chromatic_gibbs="both")
                )
                posterior_nlpdf_pix_global += nlpdf_comp_pixelwise
                posterior_nlpdf_pix_chromatic += nlpdf_comp_pixelwise_chromatic

            nlpdf_utils["nlpdf_" + component_name] = nlpdf_comp_pixelwise.sum()
            if hasattr(component, "forward_map"):
                iterate["forward_map_evals_" + component_name] = (
                    component.forward_map_evals
                )

        iterate["var"] = current[self.var_name]["var"] * 1
        # iterate["nlpdf_utils"] = nlpdf_utils
        iterate["objective_pix_chromatic"] = posterior_nlpdf_pix_chromatic
        iterate["objective_chromatic"] = posterior_nlpdf_pix_chromatic.sum()
        iterate["objective_pix_global"] = posterior_nlpdf_pix_global
        iterate["objective_global"] = posterior_nlpdf_pix_global.sum()

        if compute_derivatives:
            iterate["grad"] = self.grad_neglog_pdf(update_nlpdf_utils=False)
            if compute_derivatives_2nd_order:
                iterate["hess_diag"] = self.hess_diag_neglog_pdf(
                    update_nlpdf_utils=False
                )
            else:
                iterate["hess_diag"] = None

        return iterate

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
        for cd in self.distribution_components.values():
            cd.evaluate_all_nlpdf_utils(
                current=current,
                idx_pix=idx_pix,
                compute_derivatives=compute_derivatives,
                compute_derivatives_2nd_order=compute_derivatives_2nd_order,
                mtm=mtm,
                deriv_var_name=self.var_name,
                **kwargs,
            )
