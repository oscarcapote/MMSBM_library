from .metadata_layer import metadata_layer

class exclusive_metadata(metadata_layer):
    """
    Class for handling exclusive metadata in a nodes layer.

    This class inherits from metadata_layer and adds functionality for handling
    exclusive metadata, where each node can only have one attribute from a set
    of possible attributes.

    Parameters
    ----------
    lambda_val : float
        Metadata visibility parameter
    meta_name : str
        Name of the metadata column in the node_layer class

    Attributes
    ----------
    qka : np.array
        Probability matrix between groups and attributes
    """
    def __init__(self, lambda_val, meta_name):
        """
        Initialization of the exclusive_metadata class

        Parameters
        ----------
        lambda_val: float
            Metadata visibility
        meta_name: str
            Name of the metadata column in the node_layer class
        """
        super().__init__(lambda_val, meta_name)

    @property
    def qka(self):
        """
        Probability matrix between groups and attributes.

        Returns
        -------
        np.array
            Matrix of probabilities between groups and attributes
        """
        return self._qka

    @qka.setter
    def qka(self, qka):
        """
        Setter for the qka property.

        Parameters
        ----------
        qka : np.array
            Matrix of probabilities between groups and attributes

        Returns
        -------
        None
        """
        self._qka = qka 