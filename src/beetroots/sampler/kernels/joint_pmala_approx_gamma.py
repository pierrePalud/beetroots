try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from beetroots.modelling.component_distribution import ComponentDistribution
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.abstract_kernel import KernelUtils
from beetroots.sampler.kernels.abstract_kernel_metropolis_hastings import (
    KernelMetropolisHastings,
)
from beetroots.sampler.kernels.gamma_approx_aux import KernelGammaApproximationCensored
from beetroots.sampler.kernels.pmala import KernelPMALA


class PMALAApprox(KernelUtils):
    __slots__ = (
        "is_stochastic",
        "chromatic_gibbs",
        "objective_type",
        "var_shape_aux",
        "var_shape_param",
        "plot_name",
        "acceptance_wise_type",
    )

    def __init__(
        self,
        is_stochastic: str,
        target_distributions: TargetDistribution,
        acceptance_wise_type: str,
        plot_name: str = "PMALA_approx",
    ):
        self.is_stochastic = is_stochastic
        if not isinstance(plot_name, str):
            raise TypeError("plot_name should be a string")
        self.plot_name = plot_name

        for target_dist in target_distributions.values():
            for comp in target_dist.distribution_components.values():
                self.list_edges = getattr(comp, "list_edges", False)
                if self.list_edges is not False:
                    break
            if self.list_edges is not False:
                break
        if self.list_edges is False:
            raise ValueError("No list_edges found in the distribution components")

        assert acceptance_wise_type in ["pixelwise", "entire_map"]
        self.acceptance_wise_type = acceptance_wise_type


class KernelPMALAApproxCensored(KernelMetropolisHastings, PMALAApprox):
    r"""Dataclass that implement the proposal distribution based on the smooth indicator prior and the spatial prior"""

    def __init__(
        self,
        target_distributions: Dict[str, TargetDistribution],
        param_interest_name: str,
        alpha: float,
        eps0: float,
        lambda_: float,
        compute_correction_term: bool,
        add_perturbation: bool,
        scaling_std_approx: float = 1.0,
        scaling_std_censored: float = 1.0,
        is_stochastic: bool = False,
        acceptance_wise_type: str = "pixelwise",
        lower_bound: float = 1e-100,
        upper_bound: Union[float, int] = 1e100,
        **kwargs: dict,
    ) -> None:
        r"""
        Parameters
        ----------
        key : str
        list_edges : List[tuple[int, int]]
            list of edges
        weights : xp.ndarray of shape (n_edges,)
        """
        super().__init__(
            is_stochastic=is_stochastic,
            target_distributions=target_distributions,
            acceptance_wise_type=acceptance_wise_type,
            plot_name="PMALA_approx_censored",
        )

        self.scaling_std_approx = scaling_std_approx
        self.scaling_std_censored = scaling_std_censored

        self.param_name = param_interest_name
        self.auxiliary_var_name = tuple(
            set(target_distributions.keys()) - {param_interest_name}
        )[0]  # There should be two variables names, so this operation should be safe.
        self.approx_kernel = KernelGammaApproximationCensored(
            key=self.auxiliary_var_name,
            target_distribution=target_distributions[self.auxiliary_var_name],
            is_stochastic=is_stochastic,
            param_interest_name=param_interest_name,
            scaling_factor_sigma_a=self.scaling_std_approx,
            scaling_factor_sigma_m=self.scaling_std_censored,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self.approx_kernel.summing_axes = (1,)
        self.pmala_kernel = KernelPMALA(
            key=self.param_name,
            alpha=alpha,
            eps0=eps0,
            lambda_=lambda_,
            compute_correction_term=compute_correction_term,
            add_perturbation=add_perturbation,
            target_distribution=target_distributions[self.param_name],
            is_stochastic=is_stochastic,
            acceptance_wise_type="pixelwise",
        )
        self.sub_kernels = {"pmala": self.pmala_kernel, "approx": self.approx_kernel}
        self.chromatic_gibbs = None
        self.objective_type = None
        self.var_shape = None

    def initialize_hyperparameters(self, current: dict) -> None:
        self.pmala_kernel.initialize_hyperparameters(current)

    def external_update_hyperparameters(
        self, t: int, current: dict, acceptance_stats: dict
    ) -> None:
        self.pmala_kernel.external_update_hyperparameters(
            t=t,
            current=current,
            acceptance_stats=acceptance_stats,
        )

    def _dict_parameters_saving(self, **kwargs):
        return self.pmala_kernel._dict_parameters_saving()

    def sample(
        self,
        current_Var: Dict[str, xp.ndarray],
        forward_map_evals_current: Optional[Dict[str, xp.ndarray]],
        target_distributions: Dict[str, TargetDistribution],
        list_components: List[ComponentDistribution],
        idx_pix: Union[dict, xp.ndarray],
        rng: xp.random.Generator,
        compute_derivatives_components: Optional[List[bool]] = None,
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
        auxiliary_Var = current_Var[self.auxiliary_var_name]["var"]
        param_Var = current_Var[self.param_name]["var"]

        current_pmala = {
            "var": param_Var,
            "grad": xp.zeros(param_Var.shape),
        }
        if self.pmala_kernel.compute_correction_term:
            current_pmala["hess_diag"] = xp.zeros(param_Var.shape)
        else:
            current_pmala["hess_diag"] = None

        for comp, compute_deriv in zip(list_components, compute_derivatives_components):
            if compute_deriv:
                current_pmala["grad"][idx_pix] += comp.gradient_neglog_pdf(
                    deriv_var_name=self.param_name,
                )
                if self.pmala_kernel.compute_correction_term:
                    current_pmala["hess_diag"][idx_pix] += comp.hess_diag_neglog_pdf(
                        deriv_var_name=self.param_name,
                    )

        sample_param, diag_G_t, mu_current_pmala = self.pmala_kernel.sample(
            current_pmala,
            idx_pix,
            rng,
        )

        current_temp = {
            self.param_name: {"var": sample_param},
            self.auxiliary_var_name: {"var": auxiliary_Var[idx_pix]},
        }  # New samples for physical parameters / current values for auxiliary variables.
        (
            mean_approx_current_given_candidate,
            log_mean_approx_current_given_candidate,
            alpha_current_given_candidate,
            theta_current_given_candidate,
        ) = self.approx_kernel.get_mean_proposal(
            current_temp,
            0,
            target_distributions,
            idx_pix=idx_pix,
        )
        (
            mean_approx_candidate_given_current,
            log_mean_approx_candidate_given_current,
            alpha_candidate_given_current,
            theta_candidate_given_current,
        ) = self.approx_kernel.get_mean_proposal(
            current_temp,  # Not used
            0,
            target_distributions,
            forward_map_evals_current=forward_map_evals_current,
            idx_pix=idx_pix,
        )
        sample_aux = self.approx_kernel.sample(
            mean_candidate_given_current=mean_approx_candidate_given_current,
            alpha=alpha_candidate_given_current,
            theta=theta_candidate_given_current,
            rng=rng,
            idx_pix=idx_pix,
        )

        return (
            sample_param,
            diag_G_t,
            mu_current_pmala,
            sample_aux,
            mean_approx_candidate_given_current,
            log_mean_approx_candidate_given_current,
            alpha_candidate_given_current,
            theta_candidate_given_current,
            mean_approx_current_given_candidate,
            log_mean_approx_current_given_candidate,
            alpha_current_given_candidate,
            theta_current_given_candidate,
        )

    def neglog_proposal_candidate_given_current(
        self,
        candidate,
        idx_pix,
        mu_current_pmala,
        diag_G_t,
        mean_approx_candidate_given_current,
        log_mean_approx_candidate_given_current,
        alpha_candidate_given_current,
        theta_candidate_given_current,
    ) -> xp.ndarray:
        n_pix = idx_pix.size
        candidate_param = candidate[self.param_name]["var"][idx_pix]
        candidate_aux = candidate[self.auxiliary_var_name]["var"][idx_pix]

        _nlpdf_theta = self.pmala_kernel.neglog_proposal_candidate_given_current(
            candidate=candidate_param,
            mu_current=mu_current_pmala,
            diag_G_t=diag_G_t,
        )

        _nlpdf_cond_prior = self.approx_kernel.neglog_proposal_candidate_given_current(
            candidate_aux,
            log_mean_approx_candidate_given_current,
            alpha_candidate_given_current,
            theta_candidate_given_current,
            idx_pix,
        )

        assert _nlpdf_theta.shape == (n_pix,)
        assert _nlpdf_cond_prior.shape == (n_pix,)

        return _nlpdf_theta + _nlpdf_cond_prior

    def neglog_proposal_current_given_candidate(
        self,
        candidate,
        current,
        idx_pix,
        mean_approx_current_given_candidate,
        log_mean_approx_current_given_candidate,
        alpha_current_given_candidate,
        theta_current_given_candidate,
        list_components,
        compute_derivatives_components: Optional[List[bool]] = None,
    ):
        n_pix = idx_pix.size
        candidate_aux = candidate[self.auxiliary_var_name]["var"][idx_pix]

        _nlpdf_theta, _nlpdf_candidate, v_cand = (
            self.pmala_kernel.neglog_proposal_current_given_candidate(
                current=current,
                candidate=candidate,
                list_components=list_components,
                compute_derivatives_components=compute_derivatives_components,
                idx_pix=idx_pix,
            )
        )

        _nlpdf_cond_prior = self.approx_kernel.neglog_proposal_current_given_candidate(
            candidate_aux,
            log_mean_approx_current_given_candidate,
            alpha_current_given_candidate,
            theta_current_given_candidate,
            idx_pix,
        )

        nlpdf_sum = _nlpdf_theta + _nlpdf_cond_prior
        assert _nlpdf_theta.shape == _nlpdf_cond_prior.shape
        assert nlpdf_sum.shape == (n_pix,)

        return (
            nlpdf_sum,
            _nlpdf_candidate,
            v_cand,
        )

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

        # Get the list of components and the ones to compute the derivatives from.
        components_auxiliary = set(
            target_distributions[
                self.auxiliary_var_name
            ].distribution_components.values()
        )
        components_param = set(
            target_distributions[self.param_name].distribution_components.values()
        )
        components_auxiliary = list(
            set(components_auxiliary).difference(components_param)
        )
        list_components = list(components_param)
        list_components.extend(components_auxiliary)
        compute_derivatives_components = [True] * len(components_param) + [False] * len(
            components_auxiliary
        )

        if self.acceptance_wise_type == "pixelwise":
            summing_axes = xp.arange(1, len(self.var_shape_param))
            accept_total = xp.zeros((self.var_shape_param[0],))
            log_proba_accept_total = xp.zeros((self.var_shape_param[0],))
        elif self.acceptance_wise_type == "entire_map":
            summing_axes = xp.arange(len(self.var_shape))
            accept_total = xp.zeros((1,))
            log_proba_accept_total = xp.zeros((1,))

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
        self.pmala_kernel.chromatic_gibbs = self.chromatic_gibbs
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
                self.param_name: {"var": current_var_param},
                self.auxiliary_var_name: {"var": current_var_aux},
            }

            neglogpdf_current = xp.zeros((n_pix,))
            for comp in list_components:
                comp.evaluate_all_nlpdf_utils(
                    current_var,
                    idx_pix,
                    compute_derivatives=True,
                    compute_derivatives_2nd_order=self.pmala_kernel.compute_correction_term,
                    mtm=False,
                    deriv_var_name=self.param_name,
                )
                neglogpdf_current += comp.neglog_pdf(
                    pixelwise=True,
                    update_nlpdf_utils=False,
                    chromatic_gibbs=self.chromatic_gibbs,
                )
                if (
                    hasattr(comp, "forward_map_evals")
                    and comp.forward_map_evals is not None
                ):
                    forward_map_evals_current = getattr(comp, "forward_map_evals")

            (
                sample_param,  # (n_pix, ...)
                diag_G_t,
                mu_current_pmala,
                sample_aux,  # (n_pix, ...)
                mean_approx_candidate_given_current,
                log_mean_approx_candidate_given_current,
                alpha_candidate_given_current,
                theta_candidate_given_current,
                mean_approx_current_given_candidate,
                log_mean_approx_current_given_candidate,
                alpha_current_given_candidate,
                theta_current_given_candidate,
            ) = self.sample(
                current_var,
                forward_map_evals_current,
                target_distributions,
                list_components,
                idx_pix,
                rng,
                compute_derivatives_components=compute_derivatives_components,
            )

            candidate = deepcopy(current_var)
            candidate[self.param_name]["var"][idx_pix] = sample_param * 1  # (N,)
            candidate[self.auxiliary_var_name]["var"][idx_pix] = sample_aux * 1  # (N,)

            neglog_proposal_candidate_given_current = (
                self.neglog_proposal_candidate_given_current(
                    candidate,
                    idx_pix,
                    mu_current_pmala,
                    diag_G_t,
                    mean_approx_candidate_given_current,
                    log_mean_approx_candidate_given_current,
                    alpha_candidate_given_current,
                    theta_candidate_given_current,
                )
            )

            neglog_proposal_current_given_candidate, neglogpdf_candidate, v_cand = (
                self.neglog_proposal_current_given_candidate(
                    current=current,
                    candidate=candidate,
                    list_components=list_components,
                    compute_derivatives_components=compute_derivatives_components,
                    idx_pix=idx_pix,
                    mean_approx_current_given_candidate=mean_approx_current_given_candidate,
                    log_mean_approx_current_given_candidate=log_mean_approx_current_given_candidate,
                    alpha_current_given_candidate=alpha_current_given_candidate,
                    theta_current_given_candidate=theta_current_given_candidate,
                )
            )

            assert xp.isnan(neglogpdf_current).sum() == 0
            assert xp.isnan(neglogpdf_candidate).sum() == 0
            assert xp.isnan(neglog_proposal_candidate_given_current).sum() == 0
            assert xp.isnan(neglog_proposal_current_given_candidate).sum() == 0
            assert neglog_proposal_candidate_given_current.shape == (n_pix,)
            assert neglog_proposal_current_given_candidate.shape == (n_pix,)
            assert neglogpdf_candidate.shape == (n_pix,)
            assert neglogpdf_current.shape == (n_pix,)

            current_var = {
                self.param_name: current_var_param[idx_pix],
                self.auxiliary_var_name: current_var_aux[idx_pix],
            }
            candidate = {
                self.param_name: candidate[self.param_name]["var"][idx_pix],
                self.auxiliary_var_name: candidate[self.auxiliary_var_name]["var"][
                    idx_pix
                ],
            }
            candidate_accepted, accept_arr, log_proba_accept = self.accept_reject(
                candidates=candidate,
                current_var=current_var,
                neglogpdf_candidates=neglogpdf_candidate,
                neglogpdf_current=neglogpdf_current,
                neglog_q_current_given_candidate=neglog_proposal_current_given_candidate,
                neglog_q_candidate_given_current=neglog_proposal_candidate_given_current,
                acceptance_wise_type=self.acceptance_wise_type,
                rng=rng,
                joint=True,
            )

            new_var_param[idx_pix] = candidate_accepted[self.param_name] * 1
            new_var_aux[idx_pix] = candidate_accepted[self.auxiliary_var_name] * 1

            accept_total[idx_pix] = accept_arr * 1
            log_proba_accept_total[idx_pix] = log_proba_accept * 1

            self.pmala_kernel.internal_update_hyperparameters(
                t=t, v_cand=v_cand, accept_arr=accept_arr, idx_pix=idx_pix
            )

        if accept_total.max() > 0:
            current[self.param_name]["var"] = new_var_param * 1
            current[self.auxiliary_var_name]["var"] = new_var_aux * 1

            for k, target_dist in target_distributions.items():
                current[k] = target_distributions[k].compute_all(
                    current,
                    compute_derivatives_2nd_order=compute_derivatives_2nd_order,
                )

        return current, accept_total, log_proba_accept_total

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
