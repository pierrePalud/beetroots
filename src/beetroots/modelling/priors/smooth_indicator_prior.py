from typing import List, Optional

import numba as nb

# try:
#     import cupy as xp # type: ignore
#     decorator_nb = nb.cuda.jit
# except:
import numpy as xp

from beetroots.modelling.priors.abstract_prior import PriorProbaDistribution

decorator_nb = nb.njit


@decorator_nb("float64[:](float64[:,:], float64[:], float64[:], float64)")
def penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    neglog_p = xp.zeros((D,))
    for d in range(D):
        neglog_p[d] = xp.sum(
            xp.where(
                Var[:, d] > upper_bounds[d],
                (Var[:, d] - upper_bounds[d]) / indicator_margin_scale,
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (lower_bounds[d] - Var[:, d]) / indicator_margin_scale,
                    0,
                ),
            )
            ** 4
        )
    return neglog_p  # (D,)


@decorator_nb()
def penalty_one_pix(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    neglog_p_full = (
        xp.where(
            Var > xp.expand_dims(upper_bounds, 0),
            (Var - xp.expand_dims(upper_bounds, 0)) / indicator_margin_scale,
            xp.where(
                Var < xp.expand_dims(lower_bounds, 0),
                (xp.expand_dims(lower_bounds, 0) - Var) / indicator_margin_scale,
                xp.zeros_like(Var),
            ),
        )
        ** 4
    )  # (N_candidates, D)
    return neglog_p_full


@decorator_nb("float64[:,:](float64[:,:], float64[:], float64[:], float64)")
def gradient_penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    g = xp.zeros_like(Var)
    for d in range(D):
        g[..., d] = (
            4
            / indicator_margin_scale**4
            * xp.where(
                Var[..., d] > upper_bounds[d],
                (Var[..., d] - upper_bounds[d]),
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Var[..., d]),
                    0,
                ),
            )
            ** 3
        )
    return g


@decorator_nb("float64[:,:](float64[:,:], float64[:], float64[:], float64)")
def hess_diag_penalty(
    Var: xp.ndarray,
    lower_bounds: xp.ndarray,
    upper_bounds: xp.ndarray,
    indicator_margin_scale: float,
) -> xp.ndarray:
    D = lower_bounds.size

    hess_diag = xp.zeros_like(Var)
    for d in range(D):
        hess_diag[:, d] = (
            12
            / indicator_margin_scale**4
            * xp.where(
                Var[:, d] > upper_bounds[d],
                (Var[:, d] - upper_bounds[d]),
                xp.where(
                    Var[:, d] < lower_bounds[d],
                    (-lower_bounds[d] + Var[:, d]),
                    0,
                ),
            )
            ** 2
        )
    return hess_diag  # (N, D)


class SmoothIndicatorPrior(PriorProbaDistribution):
    r"""This prior encodes validity intervals :math:`[\underline{\theta}_{d}, \overline{\theta}_{d}]]` the physical parameters :math:`\theta_{nd}` (for :math:`d \in [\![1, D]\!]`).
    The negative log of this prior is

    .. math::

        \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
        0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
        \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
        \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
        \end{cases}

    with :math:`\Delta > 0` a margin scaling parameter.
    """

    __slots__ = (
        "D",
        "N",
        "indicator_margin_scale",
        "lower_bounds",
        "upper_bounds",
    )

    def __init__(
        self,
        D: int,
        N: int,
        indicator_margin_scale: float,
        lower_bounds: xp.ndarray,
        upper_bounds: xp.ndarray,
        list_idx_sampling: List[int],
        **kwargs,
    ) -> None:
        super().__init__(D, N, **kwargs)
        self.indicator_margin_scale = indicator_margin_scale
        r"""float: scaling parameter :math:`\Delta`"""

        self.lower_bounds_full = lower_bounds
        r"""xp.ndarray: validity interval lower bounds of the full set of D physical parameters"""
        self.upper_bounds_full = upper_bounds
        r"""xp.ndarray: validity interval upper bounds of the full set of D physical parameters"""

        self.list_idx_sampling = list_idx_sampling
        r"""list: list of indices of the physical parameters that are sampled"""

        self.lower_bounds = lower_bounds[list_idx_sampling]
        r"""xp.ndarray: validity interval lower bounds of the set of sampled physical parameters"""
        self.upper_bounds = upper_bounds[list_idx_sampling]
        r"""xp.ndarray: validity interval upper bounds of the set of sampled physical parameters"""

        assert self.lower_bounds.size == self.D, (
            f"should be {self.D}, is {self.lower_bounds.size}"
        )
        assert self.upper_bounds.size == self.D, (
            f"should be {self.D}, is {self.upper_bounds.size}"
        )

        return

    def neglog_pdf(
        self,
        full: bool = False,
        pixelwise: bool = False,
        paramwise: bool = False,
        **kwargs,
    ) -> xp.ndarray:
        r"""compute the negative log of the prior that approximates the indicator function

        .. math::

            \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
            0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
            \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
            \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
            \end{cases}

        Parameters
        ----------
        Var : xp.array of shape (N, D)
            current iterate
        pixelwise : bool, optional
            wether to return an aggregated result per pixel (if True) or per map (if False), by default False

        Returns
        -------
        neglog_p : xp.ndarray of shape (D,) or (N,)
            negative log of the smooth indicator prior pdf
        """
        k_mtm = self.nlpdf_utils["k_mtm"]
        n_pix = self.nlpdf_utils["n_pix"]

        neglog_p = self.nlpdf_utils["nlpdf_full"]

        if full:
            return neglog_p  # (n_pix, D) or (n_pix, k_mtm, D)
        elif pixelwise:
            neglog_p = (
                neglog_p.sum(axis=tuple(range(1, neglog_p.ndim)))
                if k_mtm == 0
                else neglog_p.sum(axis=tuple(range(2, neglog_p.ndim)))
            )
        elif paramwise:
            neglog_p = neglog_p.sum(axis=0) if k_mtm == 0 else neglog_p.sum(axis=(0, 1))
        else:
            neglog_p = (
                neglog_p.sum()
                if k_mtm == 0
                else neglog_p.swapaxes(0, 1).sum(axis=tuple(range(1, neglog_p.ndim)))
            )

        # neglog_p /= self.N * self.D
        return neglog_p  # (n_pix, k_mtm, D,) or (n_pix, D) or (D,) or (k_mtm, D) depending on the values of pixelwise and k_mtm

    # def neglog_pdf_one_pix(self, Var: xp.ndarray) -> xp.ndarray:
    #     r"""compute the negative log of the prior that approximates the indicator function

    #     .. math::

    #         \forall n, d, \quad \iota^{\Delta}_{[\underline{\theta}_{d}, \overline{\theta}_{d}]}(\theta_{n,d}) = \begin{cases}
    #         0 \quad \text{ if } \theta_{n,d} \in [\underline{\theta}_{d}, \overline{\theta}_{d}]\\
    #         \left(\frac{\theta_{n,d} - \underline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} < \underline{\theta}_{d}\\
    #         \left(\frac{\theta_{n,d} - \overline{\theta}_{d}}{\Delta} \right)^4 \quad \text{ if } \theta_{n,d} > \overline{\theta}_{d}
    #         \end{cases}

    #     Parameters
    #     ----------
    #     Var : xp.array of shape (N_candidates, D)
    #         current iterate

    #     Returns
    #     -------
    #     neglog_p : numpy array of shape (N_candidates,)
    #         negative log of the smooth indicator prior per map
    #     """
    #     assert len(Var.shape) == 2 and Var.shape[1] == self.D
    #     neglog_p = penalty_one_pix(
    #         Var,
    #         self.lower_bounds,
    #         self.upper_bounds,
    #         self.indicator_margin_scale,
    #     )  # (N_candidates,)
    #     return neglog_p

    def gradient_neglog_pdf(self, **kwargs) -> xp.ndarray:
        r"""gradient of the negative log pdf of the smooth indicator prior

        Parameters
        ----------

        Returns
        -------
        g : xp.array of shape (N, D)
            gradient
        """
        return self.nlpdf_utils["grad"]  # / (self.N * self.D)

    def hess_diag_neglog_pdf(self, **kwargs) -> xp.ndarray:
        r"""diagonal of the Hessian of the negative log pdf of the smooth indicator prior

        Parameters
        ----------
        Returns
        -------
        hess_diag : xp.array of shape (N, D)
            [description]
        """
        return self.nlpdf_utils["hess_diag"]  # / (self.N * self.D)

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        mtm: bool = False,
        **kwargs,
    ) -> None:
        Var = current[self.var_name]["var"] * 1
        original_var_shape = Var.shape

        assert Var.shape[0] == self.N
        assert Var.shape[-1] == self.D

        self.nlpdf_utils["mtm"] = mtm
        self.nlpdf_utils["k_mtm"] = original_var_shape[1] if mtm else 0
        self.nlpdf_utils["n_pix"] = idx_pix.size if idx_pix is not None else self.N

        if idx_pix is not None:
            Var = Var[idx_pix]
        if mtm:
            Var = Var.reshape(-1, *original_var_shape[2:])

        self.nlpdf_utils["laplacian_local"] = penalty_one_pix(
            Var,
            self.lower_bounds,
            self.upper_bounds,
            self.indicator_margin_scale,
        )  # (N_candidates, D)

        if mtm:
            self.nlpdf_utils["nlpdf_full"] = self.nlpdf_utils[
                "laplacian_local"
            ].reshape(
                self.nlpdf_utils["n_pix"],
                self.nlpdf_utils["k_mtm"],
                *original_var_shape[2:],
            )
        else:
            self.nlpdf_utils["nlpdf_full"] = self.nlpdf_utils[
                "laplacian_local"
            ].reshape(self.nlpdf_utils["n_pix"], *original_var_shape[1:])

        if compute_derivatives:
            grad_ = gradient_penalty(
                Var,
                self.lower_bounds,
                self.upper_bounds,
                self.indicator_margin_scale,
            )
            if mtm:
                self.nlpdf_utils["grad"] = grad_.reshape(
                    self.nlpdf_utils["n_pix"],
                    self.nlpdf_utils["k_mtm"],
                    *original_var_shape[2:],
                )
            else:
                self.nlpdf_utils["grad"] = grad_.reshape(
                    self.nlpdf_utils["n_pix"], *original_var_shape[1:]
                )
            if compute_derivatives_2nd_order:
                hess_diag_ = hess_diag_penalty(
                    Var,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.indicator_margin_scale,
                )
                if mtm:
                    self.nlpdf_utils["hess_diag"] = hess_diag_.reshape(
                        self.nlpdf_utils["n_pix"],
                        self.nlpdf_utils["k_mtm"],
                        *original_var_shape[2:],
                    )
                else:
                    self.nlpdf_utils["hess_diag"] = hess_diag_.reshape(
                        self.nlpdf_utils["n_pix"], *original_var_shape[1:]
                    )
