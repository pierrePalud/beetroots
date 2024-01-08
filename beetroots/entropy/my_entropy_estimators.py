import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


# CONTINUOUS ESTIMATORS
def entropy(x: np.ndarray, k: int = 3) -> float:
    r"""
    Kozachenko-Leonenko k-nearest neighbor continuous entropy estimator for the entropy of X
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 2, x.shape
    n_elements, n_features = x.shape

    assert k <= n_elements - 1, "Set k smaller than num. samples - 1"

    # build max-norm distances tree
    tree = KDTree(x, metric="chebyshev")

    # get k nearest neighbors distance
    nn = tree.query(x, k=k + 1)[0][:, k]

    # compute constant (for max-norm: c_d = 1 and log c_d = 0)
    const = digamma(n_elements) - digamma(k)

    #
    return const + n_features * np.log(2 * nn).mean()


def centropy(x, y, k=3):
    """The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k)
    entropy_y = entropy(y, k=k)
    return entropy_union_xy - entropy_y


def cross_entropy(xq, xp, k=3):
    r"""
    Kozachenko-Leonenko k-nearest neighbor continuous entropy estimator for the cross entropy of X
    ie cost of using q (surrogate) instead of p (true distribution)
    """
    assert isinstance(xq, np.ndarray)
    assert len(xq.shape) == 2, xq.shape

    assert isinstance(xp, np.ndarray)
    assert len(xp.shape) == 2, xp.shape

    n_elements_q, n_features = xq.shape
    n_elements_p, _ = xp.shape

    assert k < min(n_elements_q, n_elements_p), "Set k smaller than num. samples - 1"
    assert xq.shape[1] == xp.shape[1], "Two distributions must have same dim."

    const = digamma(n_elements_q) - digamma(k)

    tree_q = KDTree(xq, metric="chebyshev")

    nn = tree_q.query(xp, k=k)[0][:, k - 1]

    return const + n_features * np.log(2 * nn).mean()


def kldiv(x, xp, k=3):
    """KL Divergence between p and q for x~p(x), xp~q(x)
    x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    n, d = x.shape
    m, _ = xp.shape
    const = np.log(m) - np.log(n - 1)
    tree = KDTree(x, metric="chebyshev")
    treep = KDTree(xp, metric="chebyshev")
    nn = query_neighbors(tree, x, k)
    nnp = query_neighbors(treep, x, k - 1)
    return const + d * (np.log(nnp).mean() - np.log(nn).mean())

    # """
    # .. math::
    #     :label: eq:my-label
    #     h(p, q) = - \int p(x) \log q(x) dx

    # estimator from n samples of p and q
    # .. math::
    #     :label: eq:my-label
    #     \hat(p, q) = \psi (n) - \psi (k) + \frac{d}{n} \sum_{i=1}^n \log (\epsilon^{(i)})
    # """


# UTILITY FUNCTIONS


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]
