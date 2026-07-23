import numpy as np
import pytest

from beetroots.modelling.target_distribution.abstract_target_distribution import (
    TargetDistribution,
)
from beetroots.sampler.utils.sampler_params import MyGibbsSamplerParams


def test_check_selection_probas_valid():
    kernels = {
        "joint": {"global_proba": 0.5, "sub_proba": {0: 0.5, 1: 0.5}},
        "disjoint": {
            "global_proba": 0.5,
            "dist1": {"sub_proba": {0: 0.5, 1: 0.5}},
            "dist2": {"sub_proba": {0: 0.5, 1: 0.5}},
        },
    }
    target_distributions = {
        "dist1": TargetDistribution(),
        "dist2": TargetDistribution(),
    }
    sample_variables = {"var1": True, "var2": False}
    params = MyGibbsSamplerParams(
        target_distributions, kernels, sample_variables, "model_check", True
    )
    selection_probas = params.check_selection_probas(kernels)
    assert selection_probas["joint"]["proba"] == 0.5
    assert np.all(selection_probas["joint"]["sub_proba"] == np.array([0.5, 0.5]))
    assert selection_probas["disjoint"]["proba"] == 0.5
    assert np.all(
        selection_probas["disjoint"]["sub_proba"]["dist1"] == np.array([0.5, 0.5])
    )
    assert np.all(
        selection_probas["disjoint"]["sub_proba"]["dist2"] == np.array([0.5, 0.5])
    )


def test_check_selection_probas_invalid_global_proba():
    kernels = {
        "joint": {"global_proba": 0.7, "sub_proba": {0: 0.5, 1: 0.5}},
        "disjoint": {"global_proba": 0.4, "dist1": {"sub_proba": {0: 0.5, 1: 0.5}}},
    }
    target_distributions = {"dist1": TargetDistribution()}
    sample_variables = {"var1": True}
    params = MyGibbsSamplerParams(
        target_distributions, kernels, sample_variables, "model_check", True
    )
    with pytest.raises(ValueError):
        params.check_selection_probas(kernels)


def test_check_selection_probas_invalid_sub_proba():
    kernels = {
        "joint": {"global_proba": 0.5, "sub_proba": {0: 0.5, 1: 0.6}},
        "disjoint": {"global_proba": 0.5, "dist1": {"sub_proba": {0: 0.5, 1: 0.5}}},
    }
    target_distributions = {"dist1": TargetDistribution()}
    sample_variables = {"var1": True}
    params = MyGibbsSamplerParams(
        target_distributions, kernels, sample_variables, "model_check", True
    )
    with pytest.raises(ValueError):
        params.check_selection_probas(kernels)
