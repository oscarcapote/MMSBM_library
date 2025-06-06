import numpy as np
from .metadata_layer import metadata_layer

class inclusive_metadata(metadata_layer):
    """
    Class for handling inclusive metadata in a nodes layer.

    This class inherits from metadata_layer and adds functionality for handling
    inclusive metadata, where each node can have multiple attributes from a set
    of possible attributes.

    Parameters
    ----------
    lambda_val : float
        Metadata visibility parameter
    meta_name : str
        Name of the metadata column in the node_layer class
    Tau : int
        Number of membership groups of this metadata

    Attributes
    ----------
    q_k_tau : np.array
        Probability matrix between groups, membership groups and attributes
    """
    def __init__(self, lambda_val, meta_name, Tau):
        """
        Initialization of the inclusive_metadata class

        Parameters
        ----------
        lambda_val: float
            Metadata visibility
        meta_name: str
            Name of the metadata column in the node_layer class
        Tau: int
            Number of membership groups of this metadata
        """
        super().__init__(lambda_val, meta_name)
        self.Tau = Tau

    @property
    def q_k_tau(self):
        """
        Probability matrix between groups, membership groups and attributes.

        Returns
        -------
        np.array
            Matrix of probabilities between groups, membership groups and attributes
        """
        return self._q_k_tau

    @q_k_tau.setter
    def q_k_tau(self, q_k_tau):
        """
        Setter for the q_k_tau matrix.

        Parameters
        ----------
        q_k_tau : np.array
            Matrix of probabilities between groups, membership groups and attributes

        Returns
        -------
        None
        """
        self._q_k_tau = q_k_tau

    def init_q_k_tau(self, K, Tau):
        """
        Initialization of the q_k_tau matrix.

        Parameters
        ----------
        K : int
            Number of groups
        Tau : int
            Number of membership groups

        Returns
        -------
        np.array
            Initialized q_k_tau matrix of shape (K, Tau, N_att)

        Raises
        ------
        ValueError
            If K or Tau are not positive
        """
        if K <= 0: raise ValueError("Value of K must be positive!")
        if Tau <= 0: raise ValueError("Value of Tau must be positive!")
        self._q_k_tau = np.random.rand(K, self.Tau, self.N_att)
        return self._q_k_tau 