from typing import Dict, Optional, Tuple, Union

from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp


class Posterior(TargetDistribution):  # TODO: generalize for any number of priors.
    __slots__ = (
        "D",
        "L",
        "N",
        "likelihood",
        "prior_spatial",
        "prior_indicator",
        "dict_sites",
        "var_shape",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        likelihood: Likelihood,
        var_name: str,
        prior_spatial: Optional[Union[None, SpatialPrior]] = None,
        prior_indicator: Optional[Union[None, SmoothIndicatorPrior]] = None,
        separable: bool = True,
        dict_sites: Optional[Dict[int, xp.ndarray]] = None,
    ):
        distribution_components = {
            "likelihood": likelihood,
            "prior_spatial": prior_spatial,
            "prior_indicator": prior_indicator,
        }  # not filtered for None
        super().__init__(
            D,
            L,
            N,
            {
                key: val
                for key, val in distribution_components.items()
                if val is not None
            },
            var_name,
            separable,
        )

        self.likelihood = likelihood
        """Likelihood: data-fidelity term"""

        self.prior_spatial = prior_spatial
        """SpatialPrior: spatial prior term"""

        self.prior_indicator = prior_indicator
        """SmoothIndicatorPrior: prior term encoding validity intervals"""

        self.dict_sites = {}
        """dict[int, xp.ndarray]: sites for pixels to be sampled in parallel in the MTM-chromoatic Gibbs kernel"""
        if dict_sites is not None:
            self.dict_sites = dict_sites
        elif self.prior_spatial is not None:
            self.dict_sites = self.prior_spatial.dict_sites
        elif separable is True:  # all terms are independent (separable)
            self.dict_sites = {0: xp.arange(self.N)}
        else:
            self.dict_sites = {n: xp.array([n]) for n in range(self.N)}

        self.var_shape = (self.N, self.D)

        return

    def neglog_pdf_priors(
        self,
        pixelwise: bool = False,
        chromatic_gibbs: bool = False,
    ) -> Union[float, xp.ndarray]:
        is_mtm = self.likelihood.nlpdf_utils["mtm"]
        k_mtm = self.likelihood.nlpdf_utils["k_mtm"]
        n_pix = self.likelihood.nlpdf_utils["n_pix"]
        if pixelwise:
            nlpdf_priors = (
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
            nlpdf_priors = 0.0 if not is_mtm else xp.zeros(k_mtm)

        if self.prior_spatial is not None:
            nlpdf_priors += self.prior_spatial.neglog_pdf(
                pixelwise=pixelwise, chromatic_gibbs=chromatic_gibbs
            )

        if self.prior_indicator is not None:
            nlpdf_priors += self.prior_indicator.neglog_pdf(pixelwise=pixelwise)

        return nlpdf_priors

    def neglog_pdf(
        self,
        current: dict[str, Union[dict, float, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        pixelwise: bool = False,
        full: bool = False,
        update_nlpdf_utils: bool = True,
        chromatic_gibbs: bool = False,
    ) -> float:
        assert xp.sum([pixelwise, full]) < 2, (
            "pixelwise and full cannot be True at the same time"
        )
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(
                current,
                idx_pix=idx_pix,
                compute_derivatives=False,
                compute_derivatives_2nd_order=False,
            )

        random_component = list(
            self.distribution_components.values()
        )[
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

        nll = self.likelihood.neglog_pdf(
            pixelwise=pixelwise,
        )  # NOTE: idx_pix not required because it is only required in the evaluate_all_nlpdf_utils method. It computes everything while taking care of nlpdf_utils.
        nlpdf_ += nll

        nlpdf_priors = self.neglog_pdf_priors(
            pixelwise=pixelwise, chromatic_gibbs=chromatic_gibbs
        )
        nlpdf_ += nlpdf_priors

        assert xp.sum(xp.isnan(nll)) == 0, xp.sum(xp.isnan(nll))
        # assert xp.sum(xp.isnan(nlpdf_priors)) == 0, xp.sum(xp.isnan(nlpdf_priors))
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
        grad_ = xp.zeros((size_, self.D))

        nll = self.likelihood.gradient_neglog_pdf()  # (N, D)
        grad_ += nll * 1
        # assert grad_.shape == (self.N, self.D), grad_nll.shape

        if self.prior_spatial is not None:
            grad_nl_prior_spatial = self.prior_spatial.gradient_neglog_pdf()
            assert grad_nl_prior_spatial.shape == grad_.shape
            assert xp.sum(xp.isnan(grad_nl_prior_spatial)) == 0, (
                f"nan grad prior spatial {xp.sum(xp.isnan(grad_nl_prior_spatial))}"
            )
            grad_ += grad_nl_prior_spatial * 1

        if self.prior_indicator is not None:
            grad_nl_prior_indicator = self.prior_indicator.gradient_neglog_pdf()
            assert grad_nl_prior_indicator.shape == grad_.shape
            assert xp.sum(xp.isnan(grad_nl_prior_indicator)) == 0, (
                f"nan grad prior indicator {xp.sum(xp.isnan(grad_nl_prior_indicator))}"
            )
            grad_ += grad_nl_prior_indicator * 1

        grad_ = xp.nan_to_num(grad_)
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

        hess_diag_ += self.likelihood.hess_diag_neglog_pdf()  # (N, D)

        if self.prior_spatial is not None:
            hess_diag_nl_prior_spatial = self.prior_spatial.hess_diag_neglog_pdf()
            # assert xp.sum(xp.isnan(hess_diag_nl_prior_spatial)) == 0
            # assert hess_diag_nl_prior_spatial.shape == (self.N, self.D)
            hess_diag_ += hess_diag_nl_prior_spatial

        if self.prior_indicator is not None:
            hess_diag_nl_prior_indicator = self.prior_indicator.hess_diag_neglog_pdf()
            # assert xp.sum(xp.isnan(hess_diag_nl_prior_indicator)) == 0
            # assert hess_diag_nl_prior_indicator.shape == (self.N, self.D)
            hess_diag_ += hess_diag_nl_prior_indicator

        hess_diag_ = xp.nan_to_num(hess_diag_)
        return hess_diag_

    def compute_all_for_saver(
        self,
        current: dict[str, dict],
        **kwargs,
    ) -> Tuple[dict[str, Union[float, xp.ndarray]], xp.ndarray]:
        # NOTE: the model checking component is assumed to be the likelihood in the posterior case.
        """computes negative log pdf of likelihood, priors and posterior (detailed values to be saved, not to be used in sampling)

        Parameters
        ----------
        Theta : xp.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, Union[float, xp.ndarray]]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : [str, Union[float, xp.ndarray]]
            output of the ``likelihood.evaluate_all_nll_utils()`` method

        Returns
        -------
        dict[str, Union[float, xp.ndarray]]
            values to be saved
        """
        assert xp.sum(xp.isnan(current[self.var_name]["var"])) == 0, xp.sum(
            xp.isnan(current[self.var_name]["var"])
        )
        dict_objective = dict()
        posterior_nlpdf = 0.0

        # Likelihood
        nll_full = self.likelihood.neglog_pdf(
            full=True,
        )  # (N, L)
        nll_float = xp.sum(nll_full)
        dict_objective["nll"] = nll_float
        posterior_nlpdf += nll_float

        # Priors
        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(paramwise=True)
            dict_objective["nl_prior_spatial"] = nl_prior_spatial
            posterior_nlpdf += xp.sum(nl_prior_spatial)
        else:
            dict_objective["nl_prior_spatial"] = xp.zeros((self.D,))

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf(paramwise=True)
            dict_objective["nl_prior_indicator"] = nl_prior_indicator
            posterior_nlpdf += xp.sum(nl_prior_indicator)
        else:
            dict_objective["nl_prior_indicator"] = xp.zeros((self.D,))

        dict_objective["objective"] = posterior_nlpdf

        assert isinstance(nll_full, xp.ndarray), (
            "nll_full shoud be an array, check likelihood.neglog_pdf method"
        )
        assert nll_full.shape == (
            self.N,
            self.L,
        ), (
            f"nll_full with wrong shape. is {nll_full.shape}, should be {(self.N, self.L)}"
        )

        return dict_objective, nll_full

    def compute_all(
        self,
        current: dict[str, dict],
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        update_nlpdf_utils: bool = True,
        **kwargs,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the posterior distribution

        Parameters
        ----------
        Theta : xp.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, xp.ndarray], optional
            output of the ``likelihood.evaluate_all_forward_map()`` method, by default {}
        nll_utils : dict[str, xp.ndarray], optional
            output of the ``likelihood.evaluate_all_nll_utils()`` method, by default {}
        compute_derivatives : bool, optional
            wether to compte derivatives, by default True

        Returns
        -------
        dict[str, Union[float, xp.ndarray]]
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

        nlpdf_utils, iterate = {}, {}
        posterior_nlpdf_pix_global = xp.zeros(self.N)
        posterior_nlpdf_pix_chromatic = xp.zeros(self.N)

        # * Compute the negative log-pdf of the components.
        # * We do it manually for each component to keep track of their individual contributions if needed.
        # Compute the negative log-pdf of the likelihood
        nll = self.likelihood.neglog_pdf(pixelwise=True, chromatic_gibbs=False)
        posterior_nlpdf_pix_global += nll
        posterior_nlpdf_pix_chromatic += nll
        nlpdf_utils["nll"] = nll

        # Compute the negative log-pdf of the priors
        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf(pixelwise=True)
            posterior_nlpdf_pix_global += nl_prior_indicator
            posterior_nlpdf_pix_chromatic += nl_prior_indicator
        else:
            nl_prior_indicator = xp.zeros((self.N,))
        nlpdf_utils["nlpdf_prior_indicator"] = nl_prior_indicator

        if self.prior_spatial is not None:
            nl_prior_spatial_chromatic, nl_prior_spatial_global = (
                self.prior_spatial.neglog_pdf(pixelwise=True, chromatic_gibbs="both")
            )
            posterior_nlpdf_pix_global += nl_prior_spatial_global
            posterior_nlpdf_pix_chromatic += nl_prior_spatial_chromatic
        else:
            nl_prior_spatial_global = xp.zeros((self.N,))
        nlpdf_utils["nlpdf_prior_spatial"] = nl_prior_spatial_global

        iterate = {
            "var": current[self.var_name]["var"] * 1,
            "nll": nlpdf_utils,
            "objective_pix_global": posterior_nlpdf_pix_global,
            "objective_pix_chromatic": posterior_nlpdf_pix_chromatic,
            "objective_global": xp.sum(posterior_nlpdf_pix_global),
            "objective_chromatic": xp.sum(posterior_nlpdf_pix_chromatic),
        }

        # Compute the gradient and the Hessian of the negative log-pdf if needed
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
                **kwargs,
            )
