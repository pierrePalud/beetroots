import numpy as np

import beetroots.modelling.likelihoods.utils as utils


def test_norm_pdf_cdf_ratio():
    # values
    assert np.isclose(utils.norm_pdf_cdf_ratio(0.0), 2 / np.sqrt(2 * np.pi))
    assert utils.norm_pdf_cdf_ratio(10.0) < 1e-2

    # decreasing function
    x = np.linspace(-10, 10, 100)
    f_Theta = utils.norm_pdf_cdf_ratio(x)
    grad_f_Theta = np.gradient(f_Theta)
    assert np.all(grad_f_Theta < 0)
