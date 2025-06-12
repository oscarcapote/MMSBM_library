import numpy as np
from .metadata_layer import metadata_layer

class inclusive_metadata(metadata_layer):
    """
    Class for handling inclusive metadata in a nodes layer.

    This class inherits from metadata_layer and adds functionality for handling
    inclusive metadata, where each node can have multiple attributes from a set
    of possible attributes.

    Attributes
    ----------
    lambda_val : float
        Metadata visibility parameter
    meta_name : str
        Name of the metadata column in the node_layer class
    N_att : int
        Number of different categorical attributes of the metadata.
    dict_codes : dict
        A dictionary to store codes related to the metadata.
        Codes are integers ranged from 0 to N_att-1.
    links : 2D NumPy array
        Array representing links between nodes and metadata using its codes.
    Tau : int
        Number of membership groups of this metadata
    q_k_tau : np.array
        Probability matrix between groups, membership groups and attributes
    neighbours_meta : list
        List where the index is the attribute and the element is an array of the nodes that are connected to the same attribute.
    masks_att_list : list
        List of arrays of ints where the array number att has all the index positions of links that connects the attribute att
    labels : np.array
        Array of labels of the links: 0 if not connected 1 if connected
    masks_label_list : list
        masks list to know wich links have label r (that is the index of the list).

    Methods
    -------
    init_q_k_tau(K, Tau)
        Initialization of the q_k_tau matrix.
    code_inclusive_metadata(meta_row)
        Code the inclusive metadata that is a string of metadata separated by inclusive_metadata._separator.
    __len__()
        Returns the number of different categorical attributes.
    __str__()
        Returns the name of the metadata.
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
    
    def code_inclusive_metadata(self, meta_row):
       """
       Code the inclusive metadata that is a string of metadata separated by inclusive_metadata._separator.

       Parameters
       ----------
       meta_row : str
           String of metadata separated by inclusive_metadata._separator

       Returns
       -------
       str
           String of metadata ids separated by inclusive_metadata._separator
       """
       meta_list = meta_row.split(self._separator)
       #print(meta_list)
       meta_list_ids = [str(self.dict_codes[meta]) for meta in meta_list]
       #print(meta_list_ids)
       return self._separator.join(meta_list_ids)