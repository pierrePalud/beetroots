import abc


class ApproxOptimForwardMap(abc.ABC):
    r"""abstract class that handles the generation of a dataset of :math:`\log_{10} f_{\ell}(\theta)` values"""

    @abc.abstractmethod
    def compute_log10_f_Theta(self):
        r"""Evaluates :math:`\log_{10} f` on a list of ``self.N_samples_theta`` :math:`\theta` values, each randomly generated in the hypercube defined by its lower and upper bounds.

        Parameters
        ----------
        forward_model_name : str
            name of the forward model to load (i.e., of the corresponding folder)
        angle : float
            angle at which the cloud is observed
        lower_bounds_lin : np.ndarray of shape (D,)
            array of lower bounds in linear scale
        upper_bounds_lin : np.ndarray of shape (D,)
            array of upper bounds in linear scale

        Returns
        -------
        np.ndarray of shape (self.N_samples_theta, L)
            evaluations of the ``self.N_samples_theta`` :math:`\theta` values for the L considered lines.
        """
        pass
