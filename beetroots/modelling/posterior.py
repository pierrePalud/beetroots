from typing import Dict, Optional, Tuple, Union

import numpy as np


class Posterior:
    __slots__ = (
        "D",
        "L",
        "N",
        "likelihood",
        "prior_spatial",
        "prior_indicator",
        "dict_sites",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        likelihood,
        prior_spatial=None,
        prior_indicator=None,
        separable: bool = True,
        dict_sites: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.D = D
        """int: number of distinct physical parameters"""

        self.L = L
        """int: number of observables per pixel"""

        self.N = N
        """int: number of pixels"""

        self.likelihood = likelihood
        """Likelihood: data-fidelity term"""

        self.prior_spatial = prior_spatial
        """SpatialPrior: spatial prior term"""

        self.prior_indicator = prior_indicator
        """SmoothIndicatorPrior: prior term encoding validity intervals"""

        self.dict_sites = {}
        """dict[int, np.ndarray]: sites for pixels to be sampled in parallel in the MTM-chromoatic Gibbs kernel"""
        if dict_sites is not None:
            self.dict_sites = dict_sites
        elif self.prior_spatial is not None:
            self.dict_sites = self.prior_spatial.dict_sites
        elif separable is True:
            self.dict_sites = {0: np.arange(self.N)}
        else:
            self.dict_sites = {n: np.array([n]) for n in range(self.N)}

        return

    def mtm_neglog_pdf_priors(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        with_weights: Optional[bool] = True,
        use_indicator_prior: Optional[bool] = True,
        use_spatial_prior: Optional[bool] = True,
        chromatic_gibbs: Optional[bool] = True,
        # compute_indicator: bool = False,
    ) -> np.ndarray:
        r"""computes the neg log-prior when only one pixel is modified

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        idx_pix : int
            the index of the pixel to consider (0 <= idx_pix <= N - 1)
        list_pixel_candidates : np.ndarray of shape (N_candidates, D)
            the list of all candidates for pixel idx_pi
        spatial_weights : Optional[np.ndarray], optional
            vector of shape (D,) containing the weights of the spatial prior, by default None
        use_indicator_prior : bool, optional
            wether to use the indicator prior term, by default True
        use_spatial_prior : bool, optional
            wether to use the spatial prior term, by default True

        Returns
        -------
        np.ndarray of shape (N_candidates,)
            the negative log-prior of the candidates
        """
        N, k_mtm, D = Theta.shape
        n_pix = idx_pix.size
        assert D == Theta.shape[-1]

        nl_priors = np.zeros((n_pix, k_mtm))
        if self.prior_spatial is not None and use_spatial_prior:
            nl_priors_spatial = self.prior_spatial.neglog_pdf(
                Theta, idx_pix, with_weights, chromatic_gibbs=chromatic_gibbs, full=True
            ).sum(
                axis=tuple(range(2, Theta.ndim))
            )  # (n_pix, k_mtm)
            assert nl_priors_spatial.shape == (n_pix, k_mtm)
            nl_priors += nl_priors_spatial

        if self.prior_indicator is not None and use_indicator_prior:
            Theta_reshaped = Theta[idx_pix].reshape((n_pix * k_mtm, D))
            nl_priors_indicator = self.prior_indicator.neglog_pdf(
                Theta_reshaped, pixelwise=True
            )
            nl_priors += nl_priors_indicator.reshape((n_pix, k_mtm))
        return nl_priors

    def neglog_pdf_priors(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        pixelwise: bool = False,
        chromatic_gibbs: bool = False,
    ) -> Union[float, np.ndarray]:
        """evaluates the negative log-pdf of the priors

        Parameters
        ----------
        Theta : np.ndarray
            vector to evaluate
        pixelwise : bool, optional
            whether to return the prior neg log pdf per pixel, by default False

        Returns
        -------
        Union[float, np.ndarray]
            returns a float if pixelwise is False, otherwise an array of shape (N,)
        """
        if pixelwise:
            nl_priors = np.zeros((self.N,))
        else:
            nl_priors = 0.0

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(
                Theta, idx_pix, pixelwise=pixelwise, chromatic_gibbs=chromatic_gibbs
            )
            if pixelwise:
                nl_priors += nl_prior_spatial  # (n_pix, D) -> (n_pix,)
            else:
                nl_priors += np.sum(nl_prior_spatial)

        if self.prior_indicator is not None:
            nl_prior_ind = self.prior_indicator.neglog_pdf(
                Theta[idx_pix], pixelwise=pixelwise
            )  # (n_pix,) if pixelwise, (D,) otherwise
            if pixelwise:
                nl_priors += nl_prior_ind  # (n_pix,)
            else:
                nl_priors += np.sum(nl_prior_ind)  # (D,) -> float

        return nl_priors

    def neglog_pdf(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
        pixelwise: bool = False,
        chromatic_gibbs: bool = False,
    ) -> Union[float, np.ndarray]:
        """evaluates the negative log pdf of the posterior at Theta

        Parameters
        ----------
        Theta : np.ndarray
            point at which the posterior negative log pdf is to be evaluated
        forward_map_evals : dict[str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : [str, Union[float, np.ndarray]]
            output of the ``likelihood.evaluate_all_nll_utils()`` method
        pixelwise : bool, optional
            whether to return the prior neg log pdf per pixel, by default False
        chromatic_gibbs : bool, optional
            by default False

        Returns
        -------
        Union[float, np.ndarray]
            returns a float if pixelwise is False, otherwise an array of shape (N,)
        """
        nl_llh = self.likelihood.neglog_pdf(
            forward_map_evals,
            nll_utils,
            pixelwise=pixelwise,
        )
        nl_prior = self.neglog_pdf_priors(
            Theta,
            idx_pix,
            pixelwise=pixelwise,
            chromatic_gibbs=chromatic_gibbs,
        )

        if pixelwise:
            assert nl_llh.shape == (self.N,)
            assert isinstance(nl_prior, np.ndarray)
            assert nl_prior.shape == (self.N,)
        else:
            assert isinstance(nl_llh, float)
            assert isinstance(nl_prior, float)

        # assert np.sum(np.isnan(nll)) == 0, np.sum(np.isnan(nll))
        # assert np.sum(np.isnan(nl_priors)) == 0, np.sum(np.isnan(nl_priors))
        return nl_llh + nl_prior

    def grad_neglog_pdf(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        forward_map_evals: dict[str, Union[float, np.ndarray]],
        nll_utils: dict[str, Union[float, np.ndarray]],
    ) -> np.ndarray:
        grad_ = self.likelihood.gradient_neglog_pdf(
            forward_map_evals, nll_utils
        )  # (N, D)
        # assert grad_.shape == (self.N, self.D), grad_nll.shape

        if self.prior_spatial is not None:
            # grad_nl_prior_spatial = self.prior_spatial.gradient_neglog_pdf(Theta)
            # assert grad_nl_prior_spatial.shape == (self.N, self.D)
            # assert (
            #     np.sum(np.isnan(grad_nl_prior_spatial)) == 0
            # ), f"nan grad prior spatial {np.sum(np.isnan(grad_nl_prior_spatial))}"
            grad_ += self.prior_spatial.gradient_neglog_pdf(Theta, idx_pix)

        if self.prior_indicator is not None:
            # grad_nl_prior_indicator = self.prior_indicator.gradient_neglog_pdf(Theta)
            # assert grad_nl_prior_indicator.shape == (self.N, self.D)
            # assert (
            #     np.sum(np.isnan(grad_nl_prior_indicator)) == 0
            # ), f"nan grad prior indicator {np.sum(np.isnan(grad_nl_prior_indicator))}"
            grad_ += self.prior_indicator.gradient_neglog_pdf(Theta[idx_pix])

        grad_ = np.nan_to_num(grad_)
        return grad_

    def hess_diag_neglog_pdf(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
    ) -> np.ndarray:
        hess_diag = self.likelihood.hess_diag_neglog_pdf(
            forward_map_evals, nll_utils
        )  # (N, D)
        # assert hess_diag.shape == (self.N, self.D)

        if self.prior_spatial is not None:
            # hess_diag_nl_prior_spatial = self.prior_spatial.hess_diag_neglog_pdf(Theta)
            # assert np.sum(np.isnan(hess_diag_nl_prior_spatial)) == 0
            # assert hess_diag_nl_prior_spatial.shape == (self.N, self.D)
            hess_diag += self.prior_spatial.hess_diag_neglog_pdf(Theta)

        if self.prior_indicator is not None:
            # hess_diag_nl_prior_indicator = self.prior_indicator.hess_diag_neglog_pdf(Theta)
            # assert np.sum(np.isnan(hess_diag_nl_prior_indicator)) == 0
            # assert hess_diag_nl_prior_indicator.shape == (self.N, self.D)
            hess_diag += self.prior_indicator.hess_diag_neglog_pdf(Theta)

        hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def compute_all_for_saver(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict[str, Union[float, np.ndarray]],
        nll_utils: dict,
    ) -> Tuple[dict[str, Union[float, np.ndarray]], np.ndarray]:
        """computes negative log pdf of likelihood, priors and posterior (detailed values to be saved, not to be used in sampling)

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
        assert Theta.shape == (self.N, self.D)
        dict_objective = dict()

        nll_full = self.likelihood.neglog_pdf(
            forward_map_evals,
            nll_utils,
            full=True,
        )  # (N, L)

        assert isinstance(
            nll_full, np.ndarray
        ), "nll_full should be an array, check likelihood.neglog_pdf method"
        assert nll_full.shape == (
            self.N,
            self.L,
        ), f"nll_full with wrong shape. is {nll_full.shape}, should be {(self.N, self.L)}"

        dict_objective["nll"] = np.sum(nll_full)  # float

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(
                Theta, idx_pix=np.arange(self.N), chromatic_gibbs=False
            )
            dict_objective["nl_prior_spatial"] = nl_prior_spatial  # (D,)
        else:
            nl_prior_spatial = np.zeros((self.D,))

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf(Theta)
            dict_objective["nl_prior_indicator"] = nl_prior_indicator  # (D,)
        else:
            nl_prior_indicator = np.zeros((self.D,))

        nl_posterior = (
            np.sum(nll_full) + np.sum(nl_prior_spatial) + np.sum(nl_prior_indicator)
        )
        dict_objective["objective"] = nl_posterior

        return dict_objective, nll_full

    def compute_all(
        self,
        Theta: np.ndarray,
        idx_pix: Union[np.ndarray, None] = None,
        forward_map_evals: dict = {},
        nll_utils: dict = {},
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        # chromatic_gibbs: bool = True,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the posterior distribution

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, np.ndarray], optional
            output of the ``likelihood.evaluate_all_forward_map()`` method, by default {}
        nll_utils : dict[str, np.ndarray], optional
            output of the ``likelihood.evaluate_all_nll_utils()`` method, by default {}
        compute_derivatives : bool, optional
            wether to compte derivatives, by default True

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            negative log pdf and derivatives of the posterior distribution
        """
        assert np.sum(np.isnan(Theta)) == 0, np.sum(np.isnan(Theta))

        if idx_pix is None:
            idx_pix = np.arange(self.N)

        if forward_map_evals == {}:
            forward_map_evals = self.likelihood.evaluate_all_forward_map(
                Theta[idx_pix], compute_derivatives, compute_derivatives_2nd_order
            )

        if nll_utils == {}:
            nll_utils = self.likelihood.evaluate_all_nll_utils(
                forward_map_evals,
                idx_pix,
                compute_derivatives,
                compute_derivatives_2nd_order,
            )

        nll_pixelwise = self.likelihood.neglog_pdf(
            forward_map_evals, nll_utils, pixelwise=True
        )
        if self.prior_indicator is not None:
            nl_prior_indicator_pixelwise = self.prior_indicator.neglog_pdf(
                Theta, pixelwise=True
            )
        else:
            nl_prior_indicator_pixelwise = np.zeros((self.N,))

        nll_utils["nll"] = nll_pixelwise.sum()
        nll_utils["nl_prior_indicator"] = nl_prior_indicator_pixelwise.sum()

        if self.prior_spatial is not None:
            (
                nl_prior_spatial_pixelwise_chromatic,
                nl_prior_spatial_pixelwise_global,
            ) = self.prior_spatial.neglog_pdf(
                Theta, idx_pix, pixelwise=True, chromatic_gibbs="both"
            )
        else:
            nl_prior_spatial_pixelwise_chromatic = np.zeros((self.N,))
            nl_prior_spatial_pixelwise_global = nl_prior_spatial_pixelwise_chromatic * 1

        nlpdf_pixelwise_chromatic = (
            nll_pixelwise
            + nl_prior_spatial_pixelwise_chromatic
            + nl_prior_indicator_pixelwise
        )
        nlpdf_pixelwise_global = (
            nll_pixelwise
            + nl_prior_spatial_pixelwise_global
            + nl_prior_indicator_pixelwise
        )
        # nlpdf_pixelwise = self.neglog_pdf(
        #     Theta,
        #     idx_pix,
        #     forward_map_evals,
        #     nll_utils,
        #     pixelwise=True,
        #     chromatic_gibbs=chromatic_gibbs,
        # )  # (N,)

        iterate = {
            "Theta": Theta,
            "forward_map_evals": forward_map_evals,
            "nll_utils": nll_utils,
            "objective_pix_chromatic": nlpdf_pixelwise_chromatic,
            "objective_pix_global": nlpdf_pixelwise_global,
            "objective_chromatic": np.sum(nlpdf_pixelwise_chromatic),
            "objective_global": np.sum(nlpdf_pixelwise_global),
        }

        if compute_derivatives:
            iterate["grad"] = self.grad_neglog_pdf(
                Theta,
                idx_pix,
                forward_map_evals,
                nll_utils,
            )
            if compute_derivatives_2nd_order:
                iterate["hess_diag"] = self.hess_diag_neglog_pdf(
                    Theta,
                    idx_pix,
                    forward_map_evals,
                    nll_utils,
                )

        return iterate
