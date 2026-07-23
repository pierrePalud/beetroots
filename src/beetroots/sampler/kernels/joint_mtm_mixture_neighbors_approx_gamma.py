try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from typing import Dict, Tuple, Union

from beetroots.modelling.hierarchical.aux_var_u import (
    AuxiliaryGivenTarget,
)
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_ind_mtm import KernelIndependentMTM
from beetroots.sampler.kernels.gamma_approx_aux import (
    KernelGammaApproximationCensoredIMTM,
)
from beetroots.sampler.kernels.mixture_neighbors import KernelMixtureNeighbors
from beetroots.sampler.utils import utils


class MTMMixtureNeighborsApprox(KernelUtils):
    __slots__ = (
        "is_stochastic",
        "k_mtm",
        "list_edges",
        "weights",
        "chromatic_gibbs",
        "objective_type",
        "var_shape_aux",
        "var_shape_param",
        "plot_name",
        "acceptance_wise_type",
        "omega",
    )

    def __init__(
        self,
        is_stochastic: str,
        target_distributions: TargetDistribution,
        k_mtm: int,
        acceptance_wise_type: str,
        do_indicator: bool = False,
        plot_name: str = "MTM_mixture_neighbors_approx",
    ):
        self.is_stochastic = is_stochastic
        if not isinstance(plot_name, str):
            raise TypeError("plot_name should be a string")
        self.plot_name = plot_name

        if not isinstance(k_mtm, int):
            raise TypeError("k_mtm should be an integer")
        self.k_mtm = k_mtm

        self.do_indicator = do_indicator
        if not isinstance(self.do_indicator, bool):
            raise TypeError("do_indicator should be a boolean")

        for target_dist in target_distributions.values():
            for comp in target_dist.distribution_components.values():
                list_edges = getattr(comp, "list_edges", False)
                if list_edges is not False:
                    self.list_edges = list_edges
                omega = getattr(comp, "omega", False)
                if omega is not False:
                    self.omega = omega
                sigma_a = getattr(comp, "sigma_a", False)
                if sigma_a is not False:
                    self.sigma_a_imtm = xp.repeat(
                        sigma_a[:, None], self.k_mtm + 1, axis=1
                    )
                    self.sigma_a2_imtm = getattr(comp, "sigma_a2", False)
                    self.sigma_a2_imtm = xp.repeat(
                        self.sigma_a2_imtm[:, None], self.k_mtm + 1, axis=1
                    )
                sigma_m = getattr(comp, "sigma_m", False)
                if sigma_m is not False:
                    self.sigma_m_imtm = xp.repeat(
                        sigma_m[:, None], self.k_mtm + 1, axis=1
                    )
                    self.sigma_m2_imtm = getattr(comp, "sigma_m2", False)
                    self.sigma_m2_imtm = xp.repeat(
                        self.sigma_m2_imtm[:, None], self.k_mtm + 1, axis=1
                    )

        if self.list_edges is False:
            raise ValueError("No list_edges found in the distribution components")
        if self.omega is False:
            raise ValueError("No omega found in the distribution components")
        if self.sigma_a_imtm is False or self.sigma_a2_imtm is False:
            raise ValueError("No sigma_a found in the distribution components")
        if self.sigma_m_imtm is False or self.sigma_m2_imtm is False:
            raise ValueError("No sigma_m found in the distribution components")

        self.omega_imtm = xp.repeat(self.omega[:, None, :], self.k_mtm + 1, axis=1)
        assert acceptance_wise_type in ["pixelwise", "entire_map"]
        self.acceptance_wise_type = acceptance_wise_type


class KernelMTMMixtureNeighborsApproxCensored(
    KernelIndependentMTM, MTMMixtureNeighborsApprox
):
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    def __init__(
        self,
        target_distributions: Dict[str, TargetDistribution],
        k_mtm: int,
        param_interest_name: str,
        do_indicator: bool = False,
        scaling_std_approx: float = 1.0,
        scaling_std_censored: float = 1.0,
        is_stochastic: bool = False,
        acceptance_wise_type: str = "pixelwise",
        **kwargs: dict,
    ) -> None:
        r"""
        Parameters
        ----------
        key : str
        list_edges : List[tuple[int, int]]
            list of edges
        """
        super().__init__(
            is_stochastic=is_stochastic,
            target_distributions=target_distributions,
            k_mtm=k_mtm,
            do_indicator=do_indicator,
            acceptance_wise_type=acceptance_wise_type,
            plot_name="MTM_mixture_neighbors_indicator_approx_censored",
        )

        self.scaling_std_approx = scaling_std_approx
        self.scaling_std_censored = scaling_std_censored

        self.param_name = param_interest_name
        self.auxiliary_var_name = tuple(
            set(target_distributions.keys()) - {param_interest_name}
        )[0]  # There should be two variables names, so this operation should be safe.
        self.approx_kernel = KernelGammaApproximationCensoredIMTM(
            key=self.auxiliary_var_name,
            target_distribution=target_distributions[self.auxiliary_var_name],
            is_stochastic=is_stochastic,
            k_mtm=k_mtm,
            key_forward_var=param_interest_name,
            scaling_factor_sigma_m=self.scaling_std_censored,
        )
        self.approx_kernel.summing_axes = (-1,)
        self.mixture_kernel = KernelMixtureNeighbors(
            key=self.param_name,
            target_distribution=target_distributions[self.param_name],
            is_stochastic=is_stochastic,
            k_mtm=k_mtm,
            do_indicator=do_indicator,
        )

        self.chromatic_gibbs = None
        self.objective_type = None
        self.var_shape = None
        self.weights = False

    def sample(
        self,
        t: int,
        current_Var: xp.ndarray,
        target_distributions: Dict[str, TargetDistribution],
        idx_pix: Union[dict, xp.ndarray],
        k_mtm,
        rng: xp.random.Generator,
    ) -> xp.ndarray:
        r"""draws a random vectors for components :math:`n` (e.g., a pixel :math:`\theta_n`). The distribution used to draw these vectors is:

        * the smooth indicator prior
        * a combination of the smooth indicator prior and of a Gaussian mixture defined with the set of all combinations of neighbors of component :math:`n`

        Parameters
        ----------
        Theta : xp.ndarray
            current iterate
        idx_pix : xp.ndarray
            indices of the pixels

        Returns
        -------
        xp.array of shape (n_pix, self.k_mtm, D)
            random element of the hypercube defined by the lower and upper bounds with uniform distribution

        Raises
        ------
        ValueError : if ``target_distribution.prior_indicator`` is None
        """
        param_Var = current_Var[self.param_name]["var"]
        auxiliary_Var = current_Var[self.auxiliary_var_name]["var"]

        seed = rng.integers(0, 1_000_000_000)

        if self.weights is False:
            weights = xp.ones(self.var_shape_param[-1])
        else:
            weights = self.weights
        if self.do_indicator:
            samples_param = utils.sample_conditional_spatial_and_indicator_prior(
                param_Var,
                self.list_edges,
                spatial_weights=weights,
                indicator_lower_bounds=self.mixture_kernel.lower_bounds,
                indicator_upper_bounds=self.mixture_kernel.upper_bounds,
                indicator_margin_scale=self.mixture_kernel.indicator_margin_scale,
                idx_pix=idx_pix,
                k_mtm=k_mtm,
                seed=seed,
            )  # (n_pix, self.k_mtm, D)
        else:
            samples_param = utils.sample_conditional_spatial_prior(
                param_Var,
                self.list_edges,
                spatial_weights=weights,
                idx_pix=idx_pix,
                k_mtm=k_mtm,
                seed=seed,
            )  # (n_pix, self.k_mtm, D)

        samples_param = xp.concatenate(
            [samples_param, param_Var[idx_pix, None]], axis=1
        )

        current_temp = {
            self.param_name: {"var": samples_param},
            self.auxiliary_var_name: {"var": auxiliary_Var[idx_pix]},
        }
        (
            mean_approx,
            log_mean_approx,
            alpha,
            theta,
        ) = self.approx_kernel.get_means_proposal(
            current_temp,
            idx_pix,
            t,
            target_distributions,
        )

        samples_aux = self.approx_kernel.sample(
            mean_candidates=mean_approx,
            alpha=alpha,
            theta=theta,
            rng=rng,
            idx_pix=idx_pix,
        )

        return (
            samples_param[:, :-1],
            samples_aux[:, :-1],
            mean_approx,
            log_mean_approx,
            alpha,
            theta,
        )

    def neglog_proposal_candidates_current(
        self, current, idx_pix, mean_approx, log_mean_approx, alpha, theta
    ) -> xp.ndarray:
        n_pix = idx_pix.size
        candidates_param = current[self.param_name]["var"]
        candidates_aux = current[self.auxiliary_var_name]["var"]
        k_mtm = candidates_param.shape[1]

        if self.weights is False:
            weights = xp.ones(candidates_param.shape[-1])
        else:
            weights = self.weights

        if self.do_indicator:
            _pdf_theta = utils.compute_nlpdf_spatial_indicator_proposal(
                candidates_param,
                self.list_edges,
                weights,
                idx_pix,
                indicator_lowerbounds=self.mixture_kernel.lower_bounds,
                indicator_upperbounds=self.mixture_kernel.upper_bounds,
            )
        else:
            _pdf_theta = utils.compute_nlpdf_spatial_proposal(
                candidates_param,
                self.list_edges,
                weights,
                idx_pix,
            )

        # _pdf_cond_prior = self.approx_kernel.neglog_proposal_candidates_current(
        #     candidates_aux[idx_pix],
        #     mean_approx,
        #     log_mean_approx,
        #     weight_lower,
        #     idx_pix,
        # )

        selection = xp.zeros(candidates_aux.shape, dtype=bool)
        selection[idx_pix] = True
        selection = xp.logical_and(selection, self.approx_kernel.censored_imtm)

        _pdf_cond_prior = (
            self.approx_kernel.neglog_proposal_candidates_current_selection_bias(
                candidates_aux[idx_pix],
                mean_approx,
                log_mean_approx,
                alpha,
                theta,
                selection,
                idx_pix,
            )
        )

        return (_pdf_theta + _pdf_cond_prior).reshape((n_pix, k_mtm))
        # return _pdf_theta.reshape((n_pix, k_mtm))

    def selection_bias(
        self,
        current_candidates,
        forward_map_evals,
        log_forward_map_evals,
        idx_pix,
    ):
        candidates_aux = current_candidates[self.auxiliary_var_name]["var"]

        bias = xp.zeros_like(candidates_aux[idx_pix])

        bias[forward_map_evals < 1e-12] += 1e100  # TODO check if needed

        return bias.sum(axis=-1)

    def get_new_sample_sampling(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        new_var_param = current[self.param_name]["var"] * 1
        new_var_aux = current[self.auxiliary_var_name]["var"] * 1
        self.var_shape_param = current[self.param_name]["var"].shape
        self.var_shape_aux = current[self.auxiliary_var_name]["var"].shape

        list_components = []
        for target_distribution in target_distributions.values():
            list_components.extend(target_distribution.distribution_components.values())
        list_components = set(list_components)
        for comp in list_components:
            if isinstance(comp, SpatialPrior):
                self.weights = xp.minimum(comp.weights, comp.initial_weights)
                break

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, len(self.var_shape_param))
            accept_total = xp.zeros((self.var_shape_param[0],))
            log_rg_total = xp.zeros((self.var_shape_param[0],))
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(len(self.var_shape))
            accept_total = xp.zeros((1,))
            log_rg_total = xp.zeros((1,))

        list_idx = False
        for target_distribution in target_distributions.values():
            if hasattr(target_distribution, "dict_sites"):
                dict_sites = target_distribution.dict_sites
                list_idx = xp.array(list(dict_sites.keys()))
                break
        if list_idx is False:
            raise ValueError(
                "No dict_sites found in the target distributions. Please check the target distributions."
            )
        self.chromatic_gibbs = not (len(list_idx) == 1)
        self.mixture_kernel.chromatic_gibbs = self.chromatic_gibbs
        self.objective_type = (
            "objective_pix_chromatic"
            if self.chromatic_gibbs
            else "objective_pix_global"
        )

        for idx_site in list_idx:
            idx_pix = dict_sites[idx_site]
            n_pix = idx_pix.size

            current_var_param = new_var_param * 1
            current_var_aux = new_var_aux * 1

            current_var = {
                self.param_name: {"var": current_var_param * 1.0},
                self.auxiliary_var_name: {"var": current_var_aux * 1.0},
            }

            samples_param, samples_aux, mean_approx, log_mean_approx, alpha, theta = (
                self.sample(
                    t,
                    current_var,
                    target_distributions,
                    idx_pix,
                    self.k_mtm,
                    rng,
                )
            )  # (n_pix, self.k_mtm, ...)

            current_candidates = {
                self.param_name: {
                    "var": xp.repeat(current_var_param[:, None], self.k_mtm + 1, axis=1)
                },
                self.auxiliary_var_name: {
                    "var": xp.repeat(current_var_aux[:, None], self.k_mtm + 1, axis=1)
                },
            }
            current_candidates[self.param_name]["var"][idx_pix, :-1] = samples_param

            current_candidates[self.auxiliary_var_name]["var"][idx_pix, :-1] = (
                samples_aux
            )

            neglogpdf_candidates_current = (
                xp.zeros((n_pix, self.k_mtm + 1))
                if self.acceptance_wise_type == "pixelwise"
                else xp.zeros((self.k_mtm + 1,))
            )

            selection = xp.zeros(
                current_candidates[self.auxiliary_var_name]["var"].shape, dtype=bool
            )
            selection[idx_pix] = True
            selection = xp.logical_and(selection, self.approx_kernel.censored_imtm)

            for comp in list_components:
                comp.evaluate_all_nlpdf_utils(
                    current_candidates,
                    idx_pix,
                    compute_derivatives=False,
                    compute_derivatives_2nd_order=False,
                    mtm=True,
                )

                if isinstance(
                    comp, (AuxiliaryGivenTarget,)
                ):  # ObservationsGivenAuxiliary)):
                    comp_nlpdf = xp.zeros(selection[idx_pix].shape)
                    comp_nlpdf[~selection[idx_pix]] += comp.neglog_pdf(
                        full=True,
                        update_nlpdf_utils=False,
                        chromatic_gibbs=self.chromatic_gibbs,
                    )[~selection[idx_pix]]
                    neglogpdf_candidates_current += comp_nlpdf.sum(axis=-1)
                else:
                    comp_nlpdf = comp.neglog_pdf(
                        pixelwise=True,
                        update_nlpdf_utils=False,
                        chromatic_gibbs=self.chromatic_gibbs,
                    )
                    neglogpdf_candidates_current += comp_nlpdf

            assert neglogpdf_candidates_current.shape == (
                n_pix,
                self.k_mtm + 1,
            )
            assert xp.isnan(neglogpdf_candidates_current).sum() == 0

            neglog_proposal_candidates_current = (
                self.neglog_proposal_candidates_current(
                    current_candidates,
                    idx_pix,
                    mean_approx,
                    log_mean_approx,
                    alpha=alpha,
                    theta=theta,
                )
            )

            # neglog_proposal_candidates_current -= self.selection_bias(
            #     current_candidates,
            #     target_distributions[self.auxiliary_var_name]
            #     .distribution_components["aux_given_theta"]
            #     .forward_map_evals["f_Var"]
            #     .reshape(selection[idx_pix].shape),
            #     target_distributions[self.auxiliary_var_name]
            #     .distribution_components["aux_given_theta"]
            #     .forward_map_evals["log_f_Var"]
            #     .reshape(selection[idx_pix].shape),
            #     idx_pix,
            # )

            shape_ = neglog_proposal_candidates_current.shape
            assert shape_ == (n_pix, self.k_mtm + 1)

            current_candidates = {
                self.param_name: current_candidates[self.param_name]["var"][
                    idx_pix, :-1
                ],
                self.auxiliary_var_name: current_candidates[self.auxiliary_var_name][
                    "var"
                ][idx_pix, :-1],
            }

            current_var = {
                self.param_name: current_var_param[idx_pix],
                self.auxiliary_var_name: current_var_aux[idx_pix],
            }
            candidate_accepted, accept_arr, log_rg = self.accept_reject(
                current_candidates,
                current_var,
                neglogpdf_candidates_current,
                neglog_proposal_candidates_current,
                self.acceptance_wise_type,
                rng,
                joint=True,
            )
            new_var_param[idx_pix] = candidate_accepted[self.param_name] * 1
            new_var_aux[idx_pix] = candidate_accepted[self.auxiliary_var_name] * 1

            accept_total[idx_pix] = accept_arr * 1
            log_rg_total[idx_pix] = log_rg * 1

        if accept_total.max() > 0:
            current[self.param_name]["var"] = new_var_param * 1
            current[self.auxiliary_var_name]["var"] = new_var_aux * 1

            for k, target_dist in target_distributions.items():
                current[k] = target_distributions[k].compute_all(
                    current,
                    compute_derivatives_2nd_order=compute_derivatives_2nd_order,
                )

        return current, accept_total, log_rg_total

    def get_new_sample_optim(
        self,
        current: dict,
        t: int,
        target_distributions: Dict[str, TargetDistribution],
        rng: xp.random.Generator,
        compute_derivatives_2nd_order: bool,
    ) -> Tuple:
        raise NotImplementedError(
            "get_new_sample_optim is not implemented for MTMMixtureNeighborsApprox"
        )
