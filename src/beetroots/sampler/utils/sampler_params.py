"""Defines a Dataclass that stores the parameters of the augmented PSGLD sampler"""

from typing import List, Union

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

import importlib

# import beetroots.sampler.kernels as kernels
from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)


class MySamplerParams(object):
    r"""Dataclass that stores the parameters of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023`"""

    def __init__(
        self,
        initial_step_size: float,
        extreme_grad: float,
        history_weight: float,
        selection_probas: Union[xp.ndarray, List[float]],
        k_mtm: int,
        is_stochastic: bool = True,
        compute_correction_term: bool = True,
    ) -> None:
        r"""

        Parameters
        ----------
        initial_step_size : float
            step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon` in the article
        extreme_grad : float
            limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta` in the article
        history_weight : float
            weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha` in the article
        selection_probas : xp.ndarray of shape (2,)
            vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`
        k_mtm : int
            number of candidates in the MTM kernel, denoted :math:`K` in the article
        is_stochastic : bool
            if True, the algorithm performs sampling, and optimization otherwise, by default True
        compute_correction_term : bool
            wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`), by default True
        """
        assert initial_step_size > 0
        assert extreme_grad > 0
        assert 0 < history_weight < 1

        assert isinstance(k_mtm, int) and k_mtm >= 1

        if isinstance(selection_probas, list):
            selection_probas = xp.array(selection_probas)

        assert xp.all(0 <= selection_probas)
        assert selection_probas.sum() == 1

        assert isinstance(is_stochastic, bool)
        assert isinstance(compute_correction_term, bool)

        self.initial_step_size = initial_step_size
        self.extreme_grad = extreme_grad
        self.history_weight = history_weight

        self.selection_probas = selection_probas
        self.k_mtm = k_mtm

        self.is_stochastic = is_stochastic
        self.compute_correction_term = compute_correction_term


class MyGibbsSamplerParams(object):
    r"""Dataclass that stores the parameters of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023`"""

    def __init__(
        self,
        target_distributions: dict[str, TargetDistribution],
        is_stochastic: bool,
        param_interest_name: str,
        kernels_dic: dict,
        sample_variables: dict[str, bool],
        model_checking_component_name: str,
        **kwargs,
    ) -> None:
        r"""

        Parameters
        ----------
        target_distributions : dict[str, TargetDistribution]
            dictionary of target distributions
        is_stochastic : bool
            whether the sampler is used for optimization or not
        param_interest_name : str
            name of the parameter of interest
        kernels_dic : dict
            dictionary of kernels and parameters
        param_interest_name : str
            name of the parameter of interest
        sample_variables : dict[str, bool]
            dictionary of boolean whether variables aare sampled or not
        start_from_type : dict[str, str]
            dictionary stating the type of the starting point for each variable
        model_checking_component_name : str
            name of the model checking component distribution
        """
        # CHECK CONDITIONS

        if not (
            isinstance(sample_variables, dict)
            and xp.all([isinstance(v, bool) for v in sample_variables.values()])
        ):
            raise TypeError("sample_variables should be a dictionary of boolean values")
        self.sample_variables = sample_variables
        r"""Whether or not to sample the variables."""

        if not isinstance(param_interest_name, str):
            raise TypeError("param_interest_name should be a string")
        self.param_interest_name = param_interest_name
        r"""Name of the parameter of interest."""

        self.selection_probas = self.check_selection_probas(kernels_dic)
        r"""Selection probabilities for the kernels."""

        if not isinstance(is_stochastic, bool):
            raise TypeError("is_stochastic should be a boolean")
        self.is_stochastic = is_stochastic
        r"""Whether the algorithm is in stochastic or optimization mode."""

        if not isinstance(model_checking_component_name, str):
            raise TypeError("model_checking_component_name should be a string")
        self.model_checking_component_name = model_checking_component_name
        r"""Name of the component that forms the likelihood for the model checking process."""

        self.compute_derivatives_2nd_order = False

        kernels_joint = {}
        for key, val in kernels_dic["joint"].items():  # Get the joint kernels
            if isinstance(key, int):
                module_name = val["module_name"]
                class_name = val["class_name"]
                params = val["external_parameters"]
                params.update(self.__dict__)
                params.update(
                    {
                        "target_distributions": target_distributions,
                        "is_stochastic": is_stochastic,
                    }
                )

                module = importlib.import_module(
                    "." + module_name, package="beetroots.sampler.kernels"
                )
                class_ = getattr(module, class_name)
                kernels_joint[key - 1] = class_(
                    **params
                )  # We remove 1 to have indices starting to 0

                self.compute_derivatives_2nd_order = (
                    self.compute_derivatives_2nd_order
                    or getattr(
                        kernels_joint[key - 1],
                        "compute_derivatives_2nd_order",
                        False,
                    )
                )
        kernels_disjoint = {}
        for target_distribution_name in self.sample_variables.keys():
            kernels_disjoint[target_distribution_name] = {}
            for key, val in kernels_dic["disjoint"][target_distribution_name].items():
                if isinstance(key, int):
                    module_name = val["module_name"]
                    class_name = val["class_name"]
                    params = val["external_parameters"]
                    params.update(self.__dict__)
                    params.update(
                        {
                            "target_distribution": target_distributions[
                                target_distribution_name
                            ],
                            "key": target_distribution_name,
                            "is_stochastic": is_stochastic,
                        }
                    )

                    module = importlib.import_module(
                        "." + module_name, package="beetroots.sampler.kernels"
                    )
                    class_ = getattr(module, class_name)
                    kernels_disjoint[target_distribution_name][key - 1] = class_(
                        **params
                    )  # We remove 1 to have indices starting to 0

                    self.compute_derivatives_2nd_order = (
                        self.compute_derivatives_2nd_order
                        or getattr(
                            kernels_disjoint[target_distribution_name][key - 1],
                            "compute_derivatives_2nd_order",
                            False,
                        )
                    )
        self.kernels_dic = {"joint": kernels_joint, "disjoint": kernels_disjoint}
        r"""Dictionary of joint and disjoint kernels"""

    def check_selection_probas(self, kernels_dic):
        r"""Check the selection probabilities"""
        assert isinstance(kernels_dic, dict)
        assert (
            all([key in ["joint", "disjoint"] for key in kernels_dic.keys()])
            and len(kernels_dic.keys()) > 0
        ), (
            f'keys should be either "joint" or "disjoint" or both, got {list(kernels_dic.keys())}'
        )

        if "joint" not in kernels_dic.keys():
            kernels_dic["joint"] = {"global_proba": 0.0, "sub_proba": {}}
        if "disjoint" not in kernels_dic.keys():
            kernels_dic["disjoint"] = {"global_proba": 0.0, "sub_proba": {}}

        # * Check global_proba
        global_proba_joint = kernels_dic["joint"]["global_proba"]
        global_proba_disjoint = kernels_dic["disjoint"]["global_proba"]
        if (
            not (0 <= global_proba_joint <= 1)
            or not (0 <= global_proba_disjoint <= 1)
            or not (global_proba_joint + global_proba_disjoint == 1)
        ):
            raise ValueError("global_proba should both be in [0, 1] and sum to 1.")

        #  Check sub_proba
        sub_proba_joint = xp.array(list(kernels_dic["joint"]["sub_proba"].values()))
        if sub_proba_joint.size > 0:
            is_correct_sub_proba_joint = (
                xp.all(xp.logical_and(0 <= sub_proba_joint, sub_proba_joint <= 1))
                and sub_proba_joint.sum() == 1
            )
        else:
            assert global_proba_joint == 0.0, (
                "if no kernels provided for joint, global_proba should be 0"
            )
            is_correct_sub_proba_joint = True

        target_distribution_keys = list(kernels_dic["disjoint"].keys())
        target_distribution_keys.remove("global_proba")
        for key in target_distribution_keys:
            sub_proba = xp.array(
                list(kernels_dic["disjoint"][key]["sub_proba"].values())
            )
            if sub_proba.size == 0:
                assert global_proba_disjoint == 0.0 or not self.sample_variables[key], (
                    "if no kernels provided for disjoint, global_proba should be 0 or the variable should not be sampled"
                )
            elif not (
                xp.all(xp.logical_and(0 <= sub_proba, sub_proba <= 1))
                and sub_proba.sum() == 1
                and is_correct_sub_proba_joint
            ):
                raise ValueError("sub_probas should all be in [0, 1] and sum to 1.")

        selection_probas_disjoint = {
            key: xp.array(list(kernels_dic["disjoint"][key]["sub_proba"].values()))
            for key in target_distribution_keys
        }
        selection_probas = {
            "joint": {
                "proba": global_proba_joint,
                "sub_proba": sub_proba_joint,
            },
            "disjoint": {
                "proba": global_proba_disjoint,
                "sub_proba": selection_probas_disjoint,
            },
        }
        return selection_probas
