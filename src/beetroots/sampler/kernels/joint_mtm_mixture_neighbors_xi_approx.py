from typing import Dict

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.joint_mtm_mixture_neighbors_approx import (
    KernelMTMMixtureNeighborsApproxCensored as BaseKernelMTMMixtureNeighborsApproxCensored,
)
from beetroots.sampler.kernels.lognormal_xi_approx_aux import (
    KernelLogNormalXiApproximationCensoredIMTM,
)


class KernelMTMMixtureNeighborsApproxCensored(
    BaseKernelMTMMixtureNeighborsApproxCensored
):
    r"""Joint MTM wrapper using the non-centered xi approximation proposal."""

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
        super().__init__(
            target_distributions=target_distributions,
            k_mtm=k_mtm,
            param_interest_name=param_interest_name,
            do_indicator=do_indicator,
            scaling_std_approx=scaling_std_approx,
            scaling_std_censored=scaling_std_censored,
            is_stochastic=is_stochastic,
            acceptance_wise_type=acceptance_wise_type,
            **kwargs,
        )

        self.approx_kernel = KernelLogNormalXiApproximationCensoredIMTM(
            key=self.auxiliary_var_name,
            target_distribution=target_distributions[self.auxiliary_var_name],
            is_stochastic=is_stochastic,
            k_mtm=k_mtm,
            key_forward_var=param_interest_name,
            joint=False,
            acceptance_wise_type="full",
        )
        self.approx_kernel.summing_axes = (-1,)
