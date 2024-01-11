"""Marginal maximum likelihood update for scalar regularization parameters.
"""
from abc import ABC, abstractmethod

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def pgd_rate(theta, stepsize, vmin, vmax, dim, gX, homogeneity):
    r"""Update parameter :math:`\theta` from iteration :math:`k` to
    :math:`k+1` using projected gradient ascent, assuming
    :math:`p(x \mid \theta)` is of the form

    .. math::
        p(x \mid \theta) \propto \exp(- \theta g(x)),

    with :math:`g \in \Gamma_0(\mathbb{R^N})`.

    Parameters
    ----------
    theta : float
        Current value of the rate parameter :math:`\theta^{(k)}` to be
        optimized.
    stepsize : float
        Stepsize.
    vmin : float
        Lower limit of the admissible interval.
    vmax : float
        Upper limit of the admissible interval.
    dim : float
        Dimension of the space over which the distribution
            :math:`p(x \mid \theta)` is defined.
    gX : float
        Value of the potential function :math:`g` evaluated using the
        current iterate.
    homogeneity : float
        Positive homogeneity factor of the function
        :math:`- \log p(x \mid \theta)` with respect to :math:`x`. By
        default 1.

    Returns
    -------
    float
        Updated value :math:`\theta^{(k+1)}`.
    """
    theta = np.maximum(
        np.minimum(
            theta + stepsize * (dim / (homogeneity * theta) - gX),
            vmax,
        ),
        vmin,
    )
    return theta


@jit(nopython=True, cache=True)
def pgd_scale(theta, stepsize, vmin, vmax, dim, gX, homogeneity):
    r"""Update parameter :math:`\theta` from iteration :math:`k` to
    :math:`k+1` using projected gradient ascent, assuming
    :math:`p(x \mid \theta)` is of the form

    .. math::
        p(x \mid \theta) \propto \exp(- g(x) / \theta),

    with :math:`g \in \Gamma_0(\mathbb{R^N})`.

    Parameters
    ----------
    theta : float
        Current value of the scale parameter :math:`\theta^{(k)}` to be
        optimized.
    stepsize : float
        Stepsize.
    vmin : float
        Lower limit of the admissible interval.
    vmax : float
        Upper limit of the admissible interval.
    dim : float
        Dimension of the space over which the distribution
            :math:`p(x \mid \theta)` is defined.
    gX : float
        Value of the potential function :math:`g` evaluated using the
        current iterate.
    homogeneity : float
        Positive homogeneity factor of the function
        :math:`- \log p(x \mid \theta)` with respect to :math:`x`. By
        default 1.

    Returns
    -------
    float
        Updated value :math:`\theta^{(k+1)}`.
    """
    theta = np.maximum(
        np.minimum(
            theta + (stepsize / theta) * (-dim / homogeneity + gX / theta),
            vmax,
        ),
        vmin,
    )
    return theta


@jit(nopython=True, cache=True)
def pgd_rate_log(theta, stepsize, vmin, vmax, dim, gX, homogeneity):
    r"""Update parameter :math:`\log \theta^{(k)}` from iteration :math:`k` to
    :math:`k+1` using projected gradient ascent (in log scale), assuming
    :math:`p(x \mid \theta)` is of the form

    .. math::
        p(x \mid \theta) \propto \exp(- \theta g(x)),

    with :math:`g \in \Gamma_0(\mathbb{R^N})`.

    Parameters
    ----------
    theta : float
        Current value of the rate parameter :math:`\theta^{(k)}` to be
        optimized.
    stepsize : float
        Stepsize.
    vmin : float
        Lower limit of the admissible interval for :math:`\log \theta`.
    vmax : float
        Upper limit of the admissible interval for :math:`\log \theta`.
    dim : float
        Dimension of the space over which the distribution
        :math:`p(x \mid \theta)` is defined.
    gX : float
        Value of the potential function :math:`g` evaluated using the
        current iterate.
    homogeneity : float
        Positive homogeneity factor of the function
        :math:`- \log p(x \mid \theta)` with respect to :math:`x`. By
        default 1.

    Returns
    -------
    float
        Updated value :math:`\theta^{(k+1)}` (in linear scale).
    """
    log_theta = np.log(theta)
    theta = np.exp(
        np.maximum(
            np.minimum(
                log_theta + stepsize * (dim / homogeneity - gX * theta),
                vmax,
            ),
            vmin,
        )
    )
    return theta


@jit(nopython=True, cache=True)
def pgd_scale_log(theta, stepsize, vmin, vmax, dim, gX, homogeneity):
    r"""Update parameter :math:`\theta` from iteration :math:`k` to
    :math:`k+1` using projected gradient ascent (in log scale), assuming
    :math:`p(x \mid \theta)` is of the form

    .. math::
        p(x \mid \theta) \propto \exp(- g(x) / \theta),

    with :math:`g \in \Gamma_0(\mathbb{R^N})`.

    Parameters
    ----------
    theta : float
        Current value of the scale parameter :math:`\theta^{(k)}` to be
        optimized.
    stepsize : float
        Stepsize.
    vmin : float
        Lower limit of the admissible interval.
    vmax : float
        Upper limit of the admissible interval.
    dim : float
        Dimension of the space over which the distribution
            :math:`p(x \mid \theta)` is defined.
    gX : float
        Value of the potential function :math:`g` evaluated using the
        current iterate.
    homogeneity : float
        Positive homogeneity factor of the function
        :math:`- \log p(x \mid \theta)` with respect to :math:`x`. By
        default 1.

    Returns
    -------
    float
        Updated value :math:`\theta^{(k+1)}` (in linear scale).
    """
    log_theta = np.log(theta)
    theta = np.exp(
        np.maximum(
            np.minimum(
                log_theta + stepsize * (-dim / homogeneity + gX / theta),
                vmax,
            ),
            vmin,
        )
    )
    return theta


class EBayesMMLE(ABC):
    r"""Abstract class implementing an empirical Bayes maximum marginal
    likelihood approach to estimate a scalar regularization parameter.

    Empirical Bayes maximum marginal likelihood estimation of a scalar
    regularization parameter :math:`\theta` from within an MCMC algorithm using
    a projected gradient ascent. The implementation follows the description
    provided in :cite:t:`vidalMaximumLikelihoodEstimation2020`.
    """
    # :cite:p:`Vidal2020`

    __slots__ = (
        "scale",
        "N0",
        "N1",
        "dim",
        "vmin",
        "vmax",
        "homogeneity",
        "exponent",
        "_c",
        "mean_theta",
        "mean_theta_old",
        "sum_weights",
    )

    def __init__(
        self,
        scale,
        N0,
        N1,
        dim,
        vmin,
        vmax,
        homogeneity=1.0,
        exponent=0.8,
    ):
        r"""BayesMMLE constructor.

        Parameters
        ----------
        scale : float
            Scale parameter involved in the definition of the projected gradient
            stepsize.
        N0 : int
            Number of iterations defining the initial update phase.
        N1 : int
            Number of iterations for the stabilization phase.
        dim : int
            Dimension of the space over which the distribution
            :math:`p(x \mid \theta)` is defined.
        vmin : float
            Lower limit of the admissible interval.
        vmax : float
            Upper limit of the admissible interval.
        homogeneity : float, optional
            Positive homogeneity factor of the function
            :math:`- \log p(x \mid \theta)` with respect to :math:`x`. By
            default 1.
        exponent : float, optional
            Exponent involved in the evolution of the stepsize of the
            projected gradient. By default 0.8
        """
        self.scale = scale
        self.N0 = N0
        self.N1 = N1
        self.dim = dim
        self.vmin = vmin
        self.vmax = vmax
        self.homogeneity = homogeneity
        self.exponent = exponent
        self._c = scale / dim
        self.mean_theta = 0.0
        self.mean_theta_old = 0.0
        self.sum_weights = 0.0

    def _compute_stepsize(self, iteration):
        r"""Compute current value for the stepsize of the projected gradient.

        Parameters
        ----------
        iteration : int
            Current iteration of the sampler.

        Returns
        -------
        float
            Current value for the stepsize.
        """
        stepsize = self._c / ((iteration - self.N0 + 1) ** self.exponent)
        return stepsize

    def _update_mean_theta(self, theta, iteration, stepsize):
        """Compute average value for the parameter.

        Parameters
        ----------
        theta : float
            Current state of the parameter.
        iteration : int
            Current iteration index of the sampler.
        stepsize : float
            Stepsize for the update of the parameter investigated.
        """
        self.mean_theta_old = self.mean_theta
        if iteration > self.N1:
            step = stepsize  # phase 3: decreasing stepsize (weights)
        else:
            step = 1  # phase 2: uniform weights
        self.mean_theta = self.sum_weights * self.mean_theta + step * theta
        self.sum_weights += step
        self.mean_theta /= self.sum_weights

    @staticmethod
    @abstractmethod
    def _update_step(theta, stepsize, vmin, vmax, dim, gX, homogeneity):
        return NotImplemented

    def update(self, theta, iteration, gX):
        r"""Update parameter :math:`\theta` from iteration :math:`k` to
        :math:`k+1`.

        Parameters
        ----------
        theta : float
            Current value of the parameter to be optimized.
        iteration : int
            Current iteration of the sampler.
        gX : float
            Value of the potential function :math:`g` evaluated using the
            current iterate.

        Returns
        -------
        float
            Updated value \theta^{(k+1)}.
        """
        if iteration + 1 > self.N0:  # no update if not in phase > 1
            stepsize = self._compute_stepsize(iteration)
            theta = self._update_step(
                theta,
                stepsize,
                self.vmin,
                self.vmax,
                self.dim,
                gX,
                self.homogeneity,
            )
            self._update_mean_theta(theta, iteration, stepsize)
        return theta


class EBayesMMLERate(EBayesMMLE):
    r"""Empirical Bayes maximum marginal likelihood approach to estimate a
    parameter :math:`\theta` in linear scale when

        .. math::
            p(x, y \mid \theta) \propto p(y \mid x) p(x \mid \theta)
            p(x \mid \theta) = Z^{-1}(\theta) \exp(-\theta g(x))

    with :math:`g` a proper, closed, convex and :math:`\alpha`-positively
    homogeneous function.
    """
    _update_step = staticmethod(pgd_rate)


class EBayesMMLELogRate(EBayesMMLE):
    r"""Empirical Bayes maximum marginal likelihood approach to estimate a
    parameter :math:`\theta` in log scale when

        .. math::
            p(x, y \mid \theta) \propto p(y \mid x) p(x \mid \theta)
            p(x \mid \theta) = Z^{-1}(\theta) \exp(-\theta g(x))

    with :math:`g` a proper, closed, convex and :math:`\alpha`-positively
    homogeneous function.
    """

    def __init__(
        self,
        scale,
        N0,
        N1,
        dim,
        vmin=1e-8,
        vmax=1e8,
        homogeneity=1.0,
        exponent=0.8,
    ):
        super(EBayesMMLELogRate, self).__init__(
            scale,
            N0,
            N1,
            dim,
            np.log(vmin),
            np.log(vmax),
            homogeneity=homogeneity,
            exponent=exponent,
        )

    _update_step = staticmethod(pgd_rate_log)


class EBayesMMLEScale(EBayesMMLE):
    r"""Empirical Bayes maximum marginal likelihood approach to estimate a
    parameter :math:`\theta` in linear scale when

        .. math::
            p(x, y \mid \theta) \propto p(y \mid x) p(x \mid \theta)
            p(x \mid \theta) = Z^{-1}(\theta) \exp(-\theta^{-1} g(x))

    with :math:`g` a proper, closed, convex and :math:`\alpha`-positively
    homogeneous function.
    """
    _update_step = staticmethod(pgd_scale)


class EBayesMMLELogScale(EBayesMMLE):
    r"""Empirical Bayes maximum marginal likelihood approach to estimate a
    parameter :math:`\theta` in log scale when

        .. math::
            p(x, y \mid \theta) \propto p(y \mid x) p(x \mid \theta)
            p(x \mid \theta) = Z^{-1}(\theta) \exp(-\theta^{-1} g(x))

    with :math:`g` a proper, closed, convex and :math:`\alpha`-positively
    homogeneous function.
    """

    def __init__(
        self,
        scale,
        N0,
        N1,
        dim,
        vmin,
        vmax,
        homogeneity=1.0,
        exponent=0.8,
    ):
        super(EBayesMMLELogScale, self).__init__(
            scale,
            N0,
            N1,
            dim,
            np.log(vmin),
            np.log(vmax),
            homogeneity=homogeneity,
            exponent=exponent,
        )

    _update_step = staticmethod(pgd_scale_log)
