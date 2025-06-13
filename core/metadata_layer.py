import numpy as np

class metadata_layer:
    """
    Principal class of nodes_layer metadata. It contains extra information about the nodes.

    It has two subclasses:
        - exclusive_metadata
        - inclusive_metadata
    Parameters
    ----------
    lambda_val : float
        A parameter representing a lambda value.
    meta_name : str
        Name of the metadata.

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

    Methods
    -------
    __len__()
        Returns the number of different categorical attributes.
    __str__()
        Returns the name of the metadata.

    Notes
    -----
    This class provides a structure to manage metadata associated with nodes.
    """
    def __init__(self, lambda_val, meta_name):
        """
        Initialize the MetadataLayer instance.

        Parameters
        ----------
        lambda_val : float
            Parameter that represent the importance of the metadata when the model is inferred.
        meta_name : str
            Name of the metadata.
        """
        self.meta_name = meta_name
        self.lambda_val = lambda_val

    @property
    def dict_codes(self):
        """
        A dictionary property to store codes related to the metadata.

        Returns
        -------
        dict
            Dictionary containing codes related to the metadata.
        """
        return self._dict_codes

    @dict_codes.setter
    def dict_codes(self,dc):
        self._dict_codes = dc
        return dc

    @property
    def N_att(self):
        """
        Number of different categorical attributes of the metadata.

        Returns
        -------
        int
            Number of different categorical attributes
        """
        return self._N_att

    @N_att.setter
    def N_att(self, N_att):
        """
        Number of different categorical attributes of the metadata

        Parameters
        -----------
        N_att : Int
            Number of different categorical attributes of the metadata
        """
        self._N_att = N_att
        return self.N_att

    @property
    def links(self):
        """
        Array representing links between nodes and metadata using its codes.

        Returns
        -------
        np.array
            2D array containing the links between nodes and metadata
        """
        return self._links

    @links.setter
    def links(self, links):
        """
        Setter for the links property.

        Parameters
        ----------
        links : np.array
            2D array containing the links between nodes and metadata

        Returns
        -------
        None
        """
        self._links = links
        self.N_links = len(links)

    def __len__(self):
        """
        Returns the number of different categorical attributes.

        Returns
        -------
        int
            Number of different categorical attributes
        """
        return self.N_att

    def __str__(self):
        """
        Returns the name of the metadata.

        Returns
        -------
        str
            Name of the metadata
        """
        return self.meta_name 