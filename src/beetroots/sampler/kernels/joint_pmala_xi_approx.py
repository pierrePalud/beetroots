from typing import Dict, Union

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.kernels.joint_pmala_approx import (
    KernelPMALAApproxCensored as BaseKernelPMALAApproxCensored,
)
from beetroots.sampler.kernels.lognormal_xi_approx_aux import (
    KernelLogNormalXiApproximationCensored,
)


class KernelPMALAApproxCensored(BaseKernelPMALAApproxCensored):
    r"""Joint PMALA wrapper using the non-centered xi approximation proposal."""

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
        super().__init__(
            target_distributions=target_distributions,
            param_interest_name=param_interest_name,
            alpha=alpha,
            eps0=eps0,
            lambda_=lambda_,
            compute_correction_term=compute_correction_term,
            add_perturbation=add_perturbation,
            scaling_std_approx=scaling_std_approx,
            scaling_std_censored=scaling_std_censored,
            is_stochastic=is_stochastic,
            acceptance_wise_type=acceptance_wise_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            **kwargs,
        )

        self.approx_kernel = KernelLogNormalXiApproximationCensored(
            key=self.auxiliary_var_name,
            target_distribution=target_distributions[self.auxiliary_var_name],
            is_stochastic=is_stochastic,
            key_forward_var=param_interest_name,
            joint=False,
            acceptance_wise_type="full",
        )
        self.approx_kernel.summing_axes = (1,)
        self.sub_kernels["approx"] = self.approx_kernel
