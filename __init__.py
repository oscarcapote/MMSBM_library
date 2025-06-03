# coding: utf-8


import pandas as pd
import numpy as np
from MMSBM_library.functions import *

import functions.utils


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
        return self._links

    @links.setter
    def links(self, links):

        self._links = links
        self.N_links = len(links)

    def __len__(self):
        return self.N_att

    def __str__(self):
        return self.meta_name



class exclusive_metadata(metadata_layer):

    def __init__(self, lambda_val, meta_name):
        """
        Initialization of the exclusive_metadata class

        Parameters
        ----------
        lambda_val: float
            Metadata visibility
        meta_name: str
            Name of the metadata column in the node_layer class
        K: int
            Number of membership groups of this metadata
        """
        super().__init__(lambda_val, meta_name)

    @property
    def qka(self):
        return self._qka

    @qka.setter
    def qka(self, qka):
        self._qka = qka


class inclusive_metadata(metadata_layer):

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
        return self._q_k_tau

    @q_k_tau.setter
    def q_k_tau(self, q_k_tau):
        """
        Setter of the q_k_tau matrix
        """
        self._q_k_tau =  q_k_tau


    def init_q_k_tau(self, K, Tau):
        """
        Initialization of the q_k_tau matrix
        """
        if K <= 0: raise ValueError("Value of K must be positive!")
        if Tau <= 0: raise ValueError("Value of Tau must be positive!")
        self._q_k_tau = np.random.rand(K, self.Tau, self.N_att)
        return self._q_k_tau


class nodes_layer:
    """
    Base class of a layer that contains nodes

    Is initialized using a dataframe and can be modify it  using the df attribute

    The rest of the columns of the dataframe can contain information (metadata) from the nodes.
    This metadata can be added as a metadata_layer object considering the network as multipartite network.
    This metadata can be classified it as exclusive_metadata (if a node only accepts one attribute) and inclusive_metadata (if the node accepts more than one attribute)

    See for more information of metadata: metadata_layer, exclusive_metadata and inclusive_metadata.

    These objects can be added into a BiNet (bipartite network) where connections between nodes_layer are considered to infer links and their labels  (see BiNet)

    ...

    Attributes
    ----------
    K : int
        Number of memberships groups for the layer.

    node_type : str
        Name of the layer. It corresponds with the column where the nodes' name are contained.

    df : pandas DataFrame
        DataFrame that contains information of the nodes. It contains one column
        with the nodes' name and the rest are its metadata.

    dict_codes : dict
        Dictionary with the integer id of the nodes. The key is the nodes' name and the value its id.
    meta_exclusives : list of metadata_layer
        List with the metadata exclusives objects that contains the metadata that will be used in the inference.
    meta_inclusives : list of metadata_layer
        List with the metadata inclusives object that contains the metadata that will be used in the inference.
    meta_neighbours_exclusives :
        Dictionaries of lists that contains, for each node its exclusives metadata neighbours.
    meta_neighbours_inclusives :
        Dictionaries of lists that contains, for each node its inclusives metadata neighbours.
    nodes_observed_inclusive :
        List of arrays for each metadata with the nodes that has assigned an attribute of the metadata
    """


    def __init__(self, K, nodes_name, nodes_info, *, separator="\t", dict_codes = None, **kwargs):
        """
        Initialization of the nodes_layer class

        Parameters
        ----------
        K : int
            Number of memberships groups for the layer.
        nodes_name : str
            Name of the nodes column in the nodes_layer class
        nodes_info : str or pandas DataFrame
            If it is a string, it is the directory of the file with the nodes information.
            If it is a DataFrame, it is the DataFrame with the nodes information.
        separator : str, default \t
            Separator if the columns of the file that contains the nodes information
        dict_codes : dict, default None
            Dictionary with the integer id of the nodes. The key is the nodes' name and the value its id.
        """

        self.K = K
        self.node_type = nodes_name

        if type(nodes_info) == type("d"):
            self.df = self.read_file(nodes_info, separator)
        elif type(nodes_info) == type(pd.DataFrame()):
            self.df = nodes_info


        if dict_codes != None:
            if self.df.dtypes[self.node_type] == np.dtype("int64") or self.df.dtypes[self.node_type] == np.dtype("int32") or self.df.dtypes[self.node_type] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            else:
                for k in dict_codes:
                    dict_codes[k] = int(dict_codes[k])

            self._dict_codes = dict_codes
            self.df.replace({nodes_name+"_id":dict_codes}, inplace=True)
        else:
            self._dict_codes = add_codes(self, nodes_name)


        self.nodes_list = self.df[nodes_name].unique()


        self.meta_exclusives = {}
        self.meta_inclusives = {}
        self.meta_neighbours_exclusives = {}
        self.meta_neighbours_inclusives = {}  # all neighbours (connected and not) of inclusive metadata

        self.nodes_observed_inclusive = {}
        self.nodes_observed_exclusive = {}
        self._has_metas = False  #Boolean that tells you if you have metadata initialized with non 0 values of lambda_val

        self.N_nodes = len(self.nodes_list)
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0


    @property
    def dict_codes(self):
        return self._dict_codes

    @dict_codes.setter
    def dict_codes(self,dc):
        self._dict_codes = dc
        return dc

    def __getitem__(self, metadata_name):
        """
        Returns the metadata object with the name metadata_name

        Parameters
        ----------
        metadata_name : str
            Name of the metadata

        Returns
        -------
        metadata_layer
            metadata_layer object with the name metadata_name
        """
        if metadata_name in self.meta_exclusives:
            return self.meta_exclusives[metadata_name]
        elif metadata_name in self.meta_inclusives:
            return self.meta_inclusives[metadata_name]
        else:
            raise ValueError(f"Metadata {metadata_name} name not found")

    def __setitem__(self, metadata_name, metadata):
        """
        Sets the metadata object with the name metadata_name

        Parameters
        ----------
        metadata_name : str
            Name of the metadata

        metadata : metadata_layer
            metadata_layer object with the name metadata_name
        """
        if isinstance(metadata, exclusive_metadata):
            self.meta_exclusives[metadata_name] = metadata
        elif isinstance(metadata, inclusive_metadata):
            self.meta_inclusives[metadata_name] = metadata
        else:
            raise ValueError(f"Metadata {metadata_name} is not a metadata_layer object")

    def __delitem__(self, metadata_name):
        """
        Deletes the metadata object with the name metadata_name

        Parameters
        ----------
        metadata_name : str
            Name of the metadata
        """
        if metadata_name in self.meta_exclusives:
            del self.meta_exclusives[metadata_name]
            self.N_meta_exclusive -= 1
            self.N_meta -= 1
        elif metadata_name in self.meta_inclusives:
            del self.meta_inclusives[metadata_name]
            self.N_meta_inclusive -= 1
            self.N_meta -= 1
        else:
            raise ValueError(f"Metadata {metadata_name} name not found")

    def read_file(self, filename, separator="\t"):
        """
        Reads the nodes information from a file and returns it as a pandas DataFrame.

        Parameters
        ----------
        filename : str
            The filename or path to the file containing nodes information.

        separator : str, default: "\t"
            Separator of the nodes DataFrame. Default is "\t".

        Returns
        -------
        DataFrame
            A pandas DataFrame containing nodes information.
        """
        return pd.read_csv(filename, sep=separator, engine='python')

    @classmethod
    def create_simple_layer(cls, K, nodes_list, nodes_name, dict_codes=None):
        '''
        Create a nodes_layer object from a list or DataSeries only with the known nodes and without metadata

        Parameters
        ----------
        K: Int
            Number of membership groups of nodes_layer
        nodes_list: array-like, DataFrame or DataSeries
            array-like, DataFrame or DataSeries with all the nodes
        nodes_name: str
            Name of the nodes type (users, movies, metabolites...) that are or will be in DataFrame
        dict_codes: dict, None, default: None
            Dictionary where the keys are the names of nodes, and the values are their ids. If None, the program will generate the ids.
        '''
        if isinstance(nodes_list, list) or isinstance(nodes_list, np.ndarray):
            new_df = pd.DataFrame({nodes_name: nodes_list})
        elif isinstance(nodes_list, pd.Series):
            new_df = pd.DataFrame(nodes_list)
        elif isinstance(nodes_list, pd.DataFrame):
            new_df = nodes_list

        return cls(K, nodes_name, new_df, dict_codes=dict_codes)

    def __str__(self):
        return self.node_type

    def update_N(self, N_nodes):
        '''
        Update the number of nodes and reinitialize the membership matrix

        Parameters
        -----------
        N_nodes: Int
            Number of nodes
        '''
        self.N_nodes = N_nodes
        self.theta = init_P_matrix(N_nodes, self.K)

    def update_K(self, K):
        '''
        Update the number of membership groups of nodes_layer and reinitialize the membership matrix

        Parameters
        -----------
        K: Int
            Number of membership groups of nodes_layer
        '''
        self.K = K
        self.theta = init_P_matrix(self.N_nodes, K)

    def __len__(self):
        return self.N_nodes

    def add_exclusive_metadata(self, lambda_val, meta_name,*,dict_codes=None,**kwargs):
        '''
        Add exclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_val: Float
            Value of the metadata visibility

        dict_codes: dict, None, default: None
            Dictionary where the keys are the names of metadata's type, and the values are the ids. If None, the program will generate the ids.
        '''


        if lambda_val>1.e-16:self._has_metas = True



        # create metadata object
        em = exclusive_metadata(lambda_val, meta_name)


        # print(self.df.columns,df_dropna.columns)

        if dict_codes != None:
            if df_dropna.dtypes[meta_name] == np.dtype("int64") or df_dropna.dtypes[meta_name] == np.dtype("int32") or df_dropna.dtypes[meta_name] == np.dtype("int16"):
                dict_codes = {int(k):int(v) for k,v in dict_codes.items()}

            em.dict_codes = dict_codes
            df_dropna.replace({meta_name +"_id":dict_codes}, inplace=True)
        else:
            em.dict_codes = add_codes(self, meta_name)

        df_dropna = self.df.dropna(subset=[meta_name])
        observed = df_dropna[str(self)+"_id"].values

        em._meta_code = self.N_meta_exclusive

        em.links = df_dropna[[self.node_type + "_id", meta_name + "_id"]].values
        em.N_att = len(em.dict_codes)

        #list of arrays of ints where the array number att has all the index positions of links that connects the attribute att
        em.masks_att_list = []
        for r in range(em.N_att):
            mask = np.argwhere(em.links[:,1]==r)[:,0]
            em.masks_att_list.append(mask)


        # update meta related nodes attributes
        self.meta_exclusives[meta_name] = em
        self.nodes_observed_exclusive[str(em)] = observed
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes, dtype=np.int32)

        for n in observed:
            meta_neighbours[n] = self.df[self.df[self.node_type + "_id" ]== n][meta_name + "_id"]#.values

        self.meta_neighbours_exclusives[meta_name] = meta_neighbours

        return em






    def add_inclusive_metadata(self, lambda_val, meta_name, Tau,*,dict_codes=None, separator="|",**kwargs):
        '''
        Add inclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_val: float
            Value of the metadata visibility

        Tau: Int
            Number of membership groups of metadata


        separator: str, default: "|"
            Separator that is used to differentiate the different metadata assigned for each node

        dict_codes: dict, None, default: None Dictionary where the keys are the names of metadata's type,
        and the values are the ids. If None, the program will generate the ids.
        '''

        # create metadata object
        im = inclusive_metadata(lambda_val, meta_name, Tau)
        im._separator = separator
        im._meta_code = self.N_meta_inclusive
        # im.q_k_tau(self.K, Tau, 2)lambda_val, meta_name, Tau

        # links and neighbours
        df_dropna = self.df.dropna(subset=[meta_name])
        meta_list = self.df[meta_name].values

        observed = df_dropna[self.node_type].values  # Nodes with known metadata
        observed_id = df_dropna[self.node_type + "_id"].values  # Nodes with known metadata


        if lambda_val>1.e-16:self._has_metas = True


        # encode metadata
        meta_neighbours = []#meta connected with 1

        for arg in np.argsort(observed_id):
            i = meta_list[arg]
            if i == None or i == np.NaN or i == pd.NaT:
                meta_neighbours.append(None)
            else:
                meta_neighbours.append([j for j in i.split(separator)])


        if dict_codes is not None:
            if self.df.dtypes[meta_name] == np.dtype("int64") or self.df.dtypes[meta_name] == np.dtype("int32") or self.df.dtypes[meta_name] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            else:
                for k in dict_codes:
                    dict_codes[k] = int(dict_codes[k])
            codes = dict_codes
            im.dict_codes = dict_codes
        else:
            codes = {}

            for l in meta_neighbours:
                if l == None: continue
                for m in l:
                    codes[m] = codes.get(m, len(codes))
            im.dict_codes = codes


        decodes = {codes[i]:i for i in codes}
        im.decodes = decodes
        im.N_att = len(set(codes))

        meta_neighbours = [[im.dict_codes[i] for i in L] for L in meta_neighbours]

        # Links between node and metadata type
        links = np.ones((len(observed) * im.N_att, 2),dtype=np.int64)
        # Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed) * im.N_att,dtype=np.int64)

        index = 0
        for i, o in enumerate(observed_id):
            for a in range(im.N_att):
                links[index, 0] = o
                links[index, 1] = a
                if a in meta_neighbours[i]:
                    labels[index] = 1

                index += 1

        #list where the index is the attribute and the element is an array of the nodes that are connected to the same attribute
        im.neighbours_meta = []
        for att in range(im.N_att):
            im.neighbours_meta.append(links[links[:,1]==att][:,0])

        im.masks_att_list = [np.argwhere(links[:,1]==att)[:,0] for att in range(len(im))]

        im.links = links
        im.labels = labels
        im.N_labels = 2#connected or disconnected
        #nodes neigbours


        #masks list to know wich links have label r (that is the index of the list)
        im.masks_label_list = []
        for r in range(2):
            mask = np.argwhere(im.labels==r)[:,0]
            im.masks_label_list.append(mask)

        # codes = pd.Categorical(self.df[meta_name]).codes
        # self.df = self.df.join(pd.DataFrame(codes, columns=[meta_name+"_id"]))
        # self.inclusive_linked.append([[int(j) for j in i.split(separator)] for i in df_dropna [meta_name+"_id"].values])

        self.nodes_observed_inclusive[str(im)] = observed_id

        # update meta related nodes attributes
        self.meta_inclusives[str(im)] = im
        self.N_meta_inclusive += 1
        self.N_meta += 1

        self.meta_neighbours_inclusives[str(im)] = meta_neighbours

        return im


    def update_exclusives_id(self, em, dict_codes):
        '''
        Changes the ids (the integer assigned to each metadata attribute) given the dict_codes.

        Parameters
        -----------

        dict_codes: dict
            Dictionary where the keys are the names of metadata's type, and the values are the ids.
        '''
        replacer = {}
        for att in dict_codes:
            replacer[em.dict_codes[att]]= dict_codes[att]

        em.dict_codes = dict_codes
        self.df.replace({em.meta_name +"_id":replacer}, inplace=True)

        em.links = self.df[[self.node_type + "_id", em.meta_name + "_id"]].values

        #list of arrays of ints where the array number att has all the index positions of links that connects the attribute att
        em.masks_att_list = []
        for r in range(em.N_att):
            mask = np.argwhere(em.links[:,1]==r)[:,0]
            em.masks_att_list.append(mask)



        meta_neighbours = np.ones(self.N_nodes, dtype=np.int32)

        for n in range(self.N_nodes):
            meta_neighbours[n] = self.df[self.df[self.node_type + "_id" ]== n][em.meta_name + "_id"]#.values

        self.meta_neighbours_exclusives[str(em)] = meta_neighbours


    def update_inclusives_id(self, im, dict_codes):
        '''
        Changes the ids (the integer assigned to each metadata attribute) given the dict_codes.

        Parameters
        -----------

        dict_codes: dict Dictionary where the keys are the names of metadata's type, and the values are the ids.
                    If None, ids will be generated automatically.
        '''
        observed = self.nodes_observed_inclusive[str(im)]
        meta_neighbours = []


        #Replacer to change the ids
        replacer = {}
        for att in dict_codes:
            replacer[im.dict_codes[att]]= dict_codes[att]


        #New ids into the neigbours list
        for neig in self.meta_neighbours_inclusives[str(im)]:
            N = []
            for ids in neig:
                N.append(replacer[ids])
            meta_neighbours.append(N)



        #changing codes dicts
        codes = dict_codes
        im.dict_codes = dict_codes


        decodes = {codes[i]:i for i in codes}
        im.decodes = decodes
        im.N_att = len(set(codes))

        # Links between node and metadata type
        links = np.ones((len(observed) * im.N_att, 2),dtype=np.int64)
        # Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed) * im.N_att,dtype=np.int64)

        index = 0
        for i, o in enumerate(observed_id):
            for a in range(im.N_att):
                links[index, 0] = o
                links[index, 1] = a
                if a in meta_neighbours[i]:
                    labels[index] = 1

                index += 1

        #list where the index is the attribute and the element is an array of the nodes that are connected to the same attribute
        im.neighbours_meta = []
        for att in range(im.N_att):
            im.neighbours_meta.append(links[links[:,1]==att][:,0])

        im.masks_att_list = [np.argwhere(links[:,1]==att)[:,0] for att in range(len(im))]

        im.links = links
        im.labels = labels
        im.N_labels = 2#connected or disconnected
        #nodes neigbours


        #masks list to know wich links have label r (that is the index of the list)
        im.masks_label_list = []
        for r in range(2):
            mask = np.argwhere(im.labels==r)[:,0]
            im.masks_label_list.append(mask)



        # update meta related nodes attributes
        self.meta_neighbours_inclusives[str(im)] = [[m for m in L] for L in meta_neighbours]

    def save_nodes_layer(self, dir="."):
        '''
        It saves the nodes_layer object

        Parameters
        -----------
        dir: str
            Directory where the json with the nodes_layer information will be saved
        '''

        functions.utils.save_nodes_layer_dict(self, dir)

    @classmethod
    def load_nodes_layer_from_file(cls, df, json_dir="."):
        '''
        It loads the nodes_layer object from a JSON file

        Parameters
        -----------
        dir: str
            Directory where the json with the nodes_layer information is saved
        '''

        with open(json_dir, "r") as f:
            data = json.load(f)
        layer = cls(nodes_info=df,**data)

        for m in data["metadata_exclusives"]:
            layer.add_exclusive_metadata(**m)

        for m in data["metadata_inclusives"]:
            layer.add_inclusive_metadata(**m)

        return layer



class BiNet:
    """
    Class of a Bipartite Network, where two layers of different types of nodes are connected (users->items,
    politicians->bills, patient->microbiome...) and these links can be labeled with information of the
    interaction (ratings, votes...).


    Attributes
    ----------
    labels_name: str
        Name of the labels column
    N_labels: int
        Number of different types labels.
    labels_array: ndarray
        Array with all the ids of the labels.
    labels_name:
        List of the names of the diferents labels.
    labels_training:
        Array with all the ids of the labels used to train the MMSBM
    df:
        Dataframe with the links information, who connected to who and with which label are connected
    dict_codes: dict
        Dictionary with the integer ids of the labels. Keys are label names, and values are corresponding ids.
    nodes_a,nodes_b: nodes_layer
        nodes_layer objects of the nodes that are part from the bipartite network.
    links: 2D-array
        2D-array with the links of the nodes that are connected
    links_training: 2D-array
        2D-array with the links of the nodes that are connected used to train the MMSBM

    Methods
    -------
    __len__()
        Returns the number of different labels.
    __str__()
        Returns the name of the labels column.

    init_EM(tol=0.001, training=None, seed=None)
        Initializes the EM algorithm to find the most plausible membership parameters of the MMSBM.

    init_EM_from_directory(training=None, dir=".")
        Initializes the EM algorithm using parameters saved in files in a specified directory.

    EM_step(N_steps=1)
        Performs N_steps steps of the EM algorithm.

    get_loglikelihood()
        Returns the loglikelihood of the current state of the MMSBM.

    get_links_probabilities(links=None)
        Returns the probability of each link in links.

    get_predicted_links(links=None)
        Returns the predicted label of each link in links.

    get_accuracy(predicted_labels = None, test_labels = None, Pij = None,links = None, estimator = "max_probability")
        Returns the accuracy of the predicted labels.

    deep_copying()
        Returns a deep copy of the BiNet instance.

    converges()
        Returns True if the EM algorithm has converged, False otherwise.

    Notes
    -----
    This class provides a structure to manage bipartite networks.


    """
    def __init__(self, links, links_label,*, nodes_a = None, nodes_b = None, Ka=1, nodes_a_name="nodes_a", Kb=1,
                 nodes_b_name="nodes_b", separator="\t", dict_codes = None, dict_codes_a = None, dict_codes_b = None):
        """
         Initialization of a BiNet class

         Parameters
         -----------
         links: str, pandas DataFrame
            DataFrame or directory containing the links between nodes_a and nodes_b and their labels.

         links_label: str
             Name of the column where the labels are stored in the links DataFrame.

         nodes_a: nodes_layer, str, DataFrame, None, default: None
             One of the nodes layer that forms the bipartite network
             - If nodes_layer: Existing instance of the nodes_layer class representing the first layer.
             - If str or pd.DataFrame: If str, a directory containing the file information about nodes_a.
             - If pd.DataFrame, DataFrame with nodes_a information.
             - If None: A simple nodes_layer will be created from the information in links.

         nodes_b: nodes_layer, str, DataFrame, None, default: None
             One of the nodes layer that forms the bipartite network
             - If nodes_layer: Existing instance of the nodes_layer class representing the first layer.
             - If str or pd.DataFrame: If str, a directory containing the file information about nodes_b.
             - If pd.DataFrame, DataFrame with nodes_b information.
             - If None: A simple nodes_layer will be created from the information in links.

         Ka: int, default: 1
            Number of membership groups for nodes_a layer

         Kb: int, default: 1
            Number of membership groups for nodes_b layer

         nodes_a_name: str, default: nodes_a
            Name of the column where the names of nodes_a are in the links DataFrame and nodes_a DataFrame

         nodes_b_name: str, default: nodes_b
            Name of the column where the names of nodes_b are in the links DataFrame and nodes_b DataFrame

         dict_codes: dict, None, default: None
            Dictionary where the keys are the names of the labels, and the values are the ids. If None, new ids will be provided.

         dict_codes_a: dict, None, default: None
            Dictionary where the keys are the names of the nodes from nodes_a and the values are the ids. If None, new ids will be provided.

         dict_codes_b: dict, None, default: None
            Dictionary where the keys are the names of the nodes from nodes_b and the values are the ids. If None, new ids will be provided.

         separator: str, default: \t
            Separator used to read links DataFrame file. Default is \t
        """
        #Checking type of links
        self._separator = separator

        if isinstance(links, pd.DataFrame):
            self.df = links
        elif isinstance(links, str):
            self.df = pd.read_csv(links, sep=separator, engine='python')


        # creating first layer class
        if isinstance(nodes_a, nodes_layer):
            self.nodes_a = nodes_a
            nodes_a_name = str(self.nodes_a)
            Ka = nodes_a.K
        elif isinstance(nodes_a, str):
            self.nodes_a = nodes_layer(Ka, nodes_a_name, nodes_a, dict_codes = dict_codes_a)
        elif isinstance(nodes_a, pd.DataFrame):
            self.nodes_a = nodes_layer(Ka, nodes_a_name, nodes_a, dict_codes = dict_codes_a)
        elif  nodes_a == None:
            self.nodes_a = nodes_layer.create_simple_layer(Ka, self.df[nodes_a_name].unique(), nodes_a_name, dict_codes = dict_codes_a)

        # creating second layer class
        if isinstance(nodes_b, nodes_layer):
            self.nodes_b = nodes_b
            nodes_b_name = str(self.nodes_b)
            Kb = nodes_b.K
        elif isinstance(nodes_b, str):
            self.nodes_b = nodes_layer(Kb, nodes_b_name, nodes_b, dict_codes = dict_codes_b)
        elif isinstance(nodes_b, pd.DataFrame):
            self.nodes_b = nodes_layer(Kb, nodes_b_name, nodes_b, dict_codes = dict_codes_b)
        elif nodes_b == None:
            self.nodes_b = nodes_layer.create_simple_layer(Kb, self.df[nodes_b_name].unique(), nodes_b_name, dict_codes = dict_codes_b)


        ## Coding labels
        self.labels_name = links_label
        self._dict_codes = add_codes(self,links_label)
        if dict_codes is not None:
            #Checking type of labels
            if self.df.dtypes[self.labels_name] == np.dtype("int64") or self.df.dtypes[self.labels_name] == np.dtype("int32") or self.df.dtypes[self.labels_name] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            else:
                new_dict = {}
                for k in dict_codes:
                    dict_codes[k] = int(dict_codes[k])
                dict_codes = new_dict

            replacer = {}
            for att in dict_codes:
                replacer[self._dict_codes[att]]= dict_codes[att]

            self._dict_codes = dict_codes
            self.df.replace({self.labels_name+"_id":replacer}, inplace=True)

        #Links
        self.df[nodes_a_name + "_id"] = self.df[[nodes_a_name]].replace({nodes_a_name:self.nodes_a.dict_codes})
        self.df[nodes_b_name + "_id"] = self.df[[nodes_b_name]].replace({nodes_b_name:self.nodes_b.dict_codes})


        self.df = self.df.drop_duplicates()
        self.labels_array = self.df[links_label + "_id"].values
        self.links = self.df[[nodes_a_name + "_id", nodes_b_name + "_id"]].values




        self.N_labels = max(self.labels_array)+1


    @property
    def dict_codes(self):
        """
        Dictionary with the integer ids of the labels. Keys are label names, and values are corresponding ids.
        The ids go from 0 to N_labels-1.
        """
        return self._dict_codes

    @dict_codes.setter
    def dict_codes(self,dc):
        """
        Changes the ids (the integer assigned to each label) given the dict_codes.
        """
        self._dict_codes = dc
        return dc

    @classmethod
    def load_BiNet_from_json(cls, json_file, links, links_label,*, nodes_a = None, nodes_b = None, nodes_a_dir = None, nodes_b_dir = None, separator="\t"):
        """
        Load a BiNet instance from a JSON file containing MMSBM parameters and link information.

        Parameters
        ----------
        json_file: str
            Path to the JSON files containing MMSBM parameters.

        links: str, pandas DataFrame
            DataFrame or directory containing the links between nodes_a and nodes_b and their labels.

        links_label: array-like
            Array-like object representing the labels corresponding to the links.

        nodes_a: nodes_layer, str, pd.DataFrame, None, default: None
            - If nodes_layer: Existing instance of the nodes_layer class representing the first layer.
            - If str or pd.DataFrame: If str, a name for the first layer. If pd.DataFrame, DataFrame with nodes and attributes.
            - If None: The first layer will be created later.

        nodes_b: nodes_layer, str, pd.DataFrame, None, default: None
            - If nodes_layer: Existing instance of the nodes_layer class representing the second layer.
            - If str or pd.DataFrame: If str, a name for the second layer. If pd.DataFrame, DataFrame with nodes and attributes.
            - If None: The second layer will be created later as a simple layer (no metadata)

        separator: str, default: "\t"
            Separator used in the provided JSON file.

        Returns
        -------
        BN: BiNet
            Instance of the BiNet class loaded from the JSON file.

        Notes
        -----
        This class method allows loading a BiNet instance from a JSON file, along with links and labels. It constructs both
        nodes layers' objects with metadata initialized based on the provided information.

        """
        #open json
        with open(json_file, 'r') as f:
            data = json.load(f)


        #construct both nodes layers objects with metadata initialized
        # creating first layer class
        if isinstance(nodes_a, nodes_layer):
            na = nodes_a
            nodes_a_name = str(na)
            Ka = nodes_a.K
        elif isinstance(nodes_a, str) or isinstance(nodes_a, pd.DataFrame):
            if nodes_a_dir is None:
                raise ValueError("If nodes_a is a string or a DataFrame, nodes_a_dir must be provided (the same for nodes_b).")
            na = load_nodes_layer_from_file(nodes_a, nodes_a_dir)
            Ka = nodes_a.K
        elif  nodes_a == None:
            #later it will be created
            na = None

        # creating second layer class
        if isinstance(nodes_b, nodes_layer):
            nb = nodes_b
            nodes_b_name = str(nb)
            Kb = nodes_b.K
        elif isinstance(nodes_b, str) or isinstance(nodes_b, pd.DataFrame):
            if nodes_b_dir is None:
                raise ValueError(
                    "If nodes_b is a string or a DataFrame, nodes_b_dir must be provided (the same for nodes_a).")
            nb = load_nodes_layer_from_file(nodes_b, nodes_b_dir)
            Kb = nodes_b.K
        elif  nodes_b == None:
            #later it will be created
            nb = None



        #creating BiNet
        if na == None and nb == None:
            BN = cls(links,links_label,
                     nodes_a = na, Ka=data["Ka"], nodes_a_name=data["nodes_a_name"],
                               dict_codes_a=data["dict_codes_a"],
                     nodes_b = nb, Kb=data["Kb"], nodes_b_name=data["nodes_b_name"],
                               dict_codes_b=data["dict_codes_b"],
                     separator=separator,dict_codes = data["dict_codes"])



        elif na == None:
            BN = cls(links,links_label,
                     nodes_a = na, Ka=data["Ka"], nodes_a_name=data["nodes_a_name"],
                     nodes_b = nb, separator=data["separator"],dict_codes = data["dict_codes"])

        elif nb == None:
            BN = cls(links,links_label,
                     nodes_a = na,
                     nodes_b = nb, Kb=data["K"], nodes_b_name=data["nodes_b_name"],
                     separator=data["separator"],dict_codes = data["dict_codes"])
        else:
            BN = cls(links,links_label,
                     nodes_a = na,
                     nodes_b = nb,
                     separator=data["separator"],dict_codes = data["dict_codes"])

        return BN

    def __getitem__(self, nodes_type):
        """
        Returns the nodes_layer object of the specified type.

        Parameters
        ----------
        nodes_type: str
            Name of the nodes_layer object to return.

        Returns
        -------
        nodes_layer
            The nodes_layer object of the specified type.
        """
        if nodes_type == str(self.nodes_a):
            return self.nodes_a
        elif nodes_type == str(self.nodes_b):
            return self.nodes_b
        else:
            raise ValueError(f"Nodes type {nodes_type} not found")

    def __setitem__(self, nodes_type, nodes_layer):
        """
        Sets the nodes_layer object of the specified type.

        Parameters
        ----------
        nodes_type: str
            Name of the nodes_layer object to set.

        nodes_layer: nodes_layer
            The nodes_layer object to set.
        """
        if nodes_type == str(self.nodes_a):
            self.nodes_a = nodes_layer
        elif nodes_type == str(self.nodes_b):
            self.nodes_b = nodes_layer
        else:
            raise ValueError("Nodes type not found")


    #EM algorithm
    def init_EM(self,tol=0.001, training = None, seed=None):
        '''
        Initialize the EM algorithm to get the most plausible membership parameters of the MMSBM

        Parameters
        -----------
        tol : float, default: 0.001
            Tolerance of the algorithm when finding the parameters.

        seed : int, None, default: None
            Seed to generate the matrices. Is initialized using the np.random.RandomState(seed) method.

        training : DataFrame, list, default: None
            - If DataFrame: DataFrame with the links used to train the MMSBM.
            - If list or ndarray: List or array containing the indexes of the links list used for training.
            - If None: Uses self.links and self.labels_array.

        Notes
        -----
        This method initializes the EM algorithm by setting up probability matrices (BiNet.pkl), memberships (BiNet.nodes_a.theta and BiNet.nodes_b.theta), and managing
        links to train. The tolerance, seed, and training data can be specified to customize the initialization process.
        '''
        # Probability matrices

        self.tol = tol

        #BiNet matrices
        self.pkl = init_P_matrix(self.nodes_a.K, self.nodes_b.K, self.N_labels)

        #memberships (thetas)
        self.nodes_a.theta = init_P_matrix(len(self.nodes_a),self.nodes_a.K)
        self.nodes_b.theta = init_P_matrix(len(self.nodes_b),self.nodes_b.K)

        # Links to train management
        if isinstance(training,pd.DataFrame):
            self.links_training = training[[str(self.nodes_a)+"_id",str(self.nodes_b)+"_id"]].values
            self.labels_training = training[self.labels_name+"_id"].values
        elif isinstance(training,list) or isinstance(training,np.ndarray):
            self.links_training = self.links[training]
            self.labels_training = self.labels_array[training]
        elif training is None:
            self.links_training = self.links
            self.labels_training = self.labels_array

        #masks list to know wich links have label r (that is the index of the list)
        self.masks_label_list = []
        for r in range(self.N_labels):
            mask = np.argwhere(self.labels_training==r)[:,0]
            self.masks_label_list.append(mask)

        #observed nodes in each layer
        self.observed_nodes_a = np.unique(self.links_training[:,0])
        self.observed_nodes_b = np.unique(self.links_training[:,1])

        #non_observed nodes in each layer
        self.non_observed_nodes_a = np.array([i for i in range(len(self.nodes_a)) if i not in self.observed_nodes_a])
        self.non_observed_nodes_b = np.array([i for i in range(len(self.nodes_b)) if i not in self.observed_nodes_b])




        self.omega = omega_comp_arrays(len(self.nodes_a),len(self.nodes_b),self.pkl,self.nodes_a.theta,self.nodes_b.theta,self.nodes_a.K,self.nodes_b.K,self.links_training,self.labels_training)



        #Metadata
        ## qka and omegas
        for meta_name,meta in self.nodes_a.meta_exclusives.items():
            meta.qka = init_P_matrix(self.nodes_a.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_a.theta,len(self.nodes_a),self.nodes_a.K,meta.links)

        for meta_name,meta in self.nodes_b.meta_exclusives.items():
            meta.qka = init_P_matrix(self.nodes_b.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_b.theta,len(self.nodes_b),self.nodes_b.K,meta.links)

        ## q_k_tau, zetes and omegas omega_comp_arrays(omega,p_kl,theta,eta,K,L,links_array,links_ratings):
        for meta_name,meta in self.nodes_a.meta_inclusives.items():
            meta.q_k_tau = init_P_matrix(self.nodes_a.K, meta.Tau, 2)
            meta.zeta = init_P_matrix(len(meta), meta.Tau)
            meta.omega = omega_comp_arrays(len(self.nodes_a),len(meta),meta.q_k_tau,self.nodes_a.theta,meta.zeta,self.nodes_a.K,meta.Tau,meta.links,meta.labels)

            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta[m])
            meta.denominators = meta.denominators[:,np.newaxis]



        for meta_name,meta in self.nodes_b.meta_inclusives.items():
            meta.q_k_tau = init_P_matrix(self.nodes_b.K, meta.Tau, 2)
            meta.zeta = init_P_matrix(len(meta), meta.Tau)
            meta.omega = omega_comp_arrays(len(self.nodes_b),len(meta),meta.q_k_tau,self.nodes_b.theta,meta.zeta,self.nodes_b.K,meta.Tau,meta.links,meta.labels)

            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta)
            meta.denominators = meta.denominators[:,np.newaxis]


        #creating arrays with the denominator (that are constants) of each node in both layers and em layers

        ##nodes a
        self.nodes_a.denominators = np.zeros(len(self.nodes_a))

        self.neighbours_nodes_a = [] #list of list of neighbours
        for node in range(len(self.nodes_a)):
            #neighbours in BiNet
            self.neighbours_nodes_a.append(self.links_training[self.links_training[:,0]==node,1])
            self.nodes_a.denominators[node] += len(self.neighbours_nodes_a[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(self.nodes_a.meta_exclusives.values()):
            for node in meta.links[:,0]:
                self.nodes_a.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(self.nodes_a)):
            for i, meta in enumerate(self.nodes_a.meta_inclusives.values()):
                self.nodes_a.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])


            #for i, meta in enumerate(self.nodes_a.meta_exclusives):
                #self.node_a.denominators[node] += meta.lambda_vals*

        self.nodes_a.denominators = self.nodes_a.denominators[:,np.newaxis]

        ##nodes b
        self.nodes_b.denominators = np.zeros(len(self.nodes_b))

        self.neighbours_nodes_b = [] #list of list of neighbours
        for node in range(len(self.nodes_b)):
            #neighbours in BiNet
            self.neighbours_nodes_b.append(self.links_training[self.links_training[:,1]==node,0])
            self.nodes_b.denominators[node] += len(self.neighbours_nodes_b[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(self.nodes_b.meta_exclusives.values()):
            for node in meta.links[:,0]:
                self.nodes_b.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(self.nodes_b)):
            for i, meta in enumerate(self.nodes_b.meta_inclusives.values()):
                self.nodes_b.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])

            #neighbours in meta inclusives

        self.nodes_b.denominators = self.nodes_b.denominators[:,np.newaxis]
            # for meta in self.nodes_b.meta_exclusives:

    def save_BiNet(self, dir=".",layers=True):
        '''
        It saves the BiNet data into a JSON file in dir. If layers==True,
        it saves the nodes_layer objects in JSONs files in the same directory.

        Parameters
        -----------
        dir: str
            Directory where the JSON with the BiNet information will be saved
        layers: bool, default: True
            If True, it saves the nodes_layer objects in JSONs files in the same directory.
        '''
        functions.utils.save_BiNet_dict(self, dir)

        if layers:
            self.nodes_a.save_nodes_layer(dir)
            self.nodes_b.save_nodes_layer(dir)

    @classmethod
    def load_BiNet_from_file(cls, df_links,json_dir, layers=True,*, nodes_a = None, nodes_b = None):
        '''
        It loads the BiNet data from a JSON file in dir

        Parameters
        -----------
        dir: str
            Directory where the JSON with the BiNet information is saved
        layers: bool, default: True
            If True, it loads the nodes_layer objects from the JSON file in the same directory.
        '''
        with open(json_dir, "r") as f:
            data = json.load(f)
        BN = cls(links=df_links, **data)

        if layers:
            BN.nodes_a = nodes_layer.load_nodes_layer_from_file(json_dir)
            BN.nodes_b = nodes_layer.load_nodes_layer_from_file(json_dir)
        else:
            if nodes_a is not None:
                BN.nodes_a = nodes_a
            if nodes_b is not None:
                BN.nodes_b = nodes_b
            


        
        return BN
    
    
    def init_EM_from_directory(self,training=None,dir="."):
        '''
        Initialize the Expectation Maximization (EM) algorithm to obtain the most plausible membership parameters of the
        Mixed-Membership Stochastic Block Model (MMSBM) using parameters saved in files located in a specified directory.

            Parameters
            ----------
            dir: str, default: "."
                Directory where the files with the MMSBM parameters will be loaded.

            training: pd.DataFrame, list, ndarray, default: None
                - If pd.DataFrame: DataFrame containing the training links and labels.
                - If list or ndarray: List or array containing the positions of the links list from self.df attribute.
                - If None: Uses self.links_training and self.labels_training.
        '''
        na = self.nodes_a

        nb = self.nodes_b


        load_EM_parameters(self,dir)

        # Links to train management
        if isinstance(training,pd.DataFrame):
            if str(self.nodes_a)+"_id" not in training.columns:
                training[str(self.nodes_a)+"_id"] = training[[str(self.nodes_a)]].replace({str(self.nodes_a):self.nodes_a.dict_codes})
            if str(self.nodes_b)+"_id" not in training.columns:
                training[str(self.nodes_b)+"_id"] = training[[str(self.nodes_b)]].replace({str(self.nodes_b):self.nodes_b.dict_codes})

            self.links_training = training[[str(self.nodes_a)+"_id",str(self.nodes_b)+"_id"]].values
            self.labels_training = training[self.labels_name+"_id"].values
        elif isinstance(training,list) or isinstance(training,np.ndarray):
            self.links_training = self.links[training]
            self.labels_training = self.labels_array[training]
        elif training is None:
            self.links_training = self.links
            self.labels_training = self.labels_array

        #Omegas and denominators
        self.omega = omega_comp_arrays(len(na),len(nb),self.pkl,na.theta,nb.theta,na.K,nb.K,self.links_training,self.labels_training)


        #masks list to know wich links have label r (that is the index of the list)
        self.masks_label_list = []
        for r in range(self.N_labels):
            mask = np.argwhere(self.labels_training==r)[:,0]
            self.masks_label_list.append(mask)

        #Metadata
        ## qka and omegas
        for i,meta in enumerate(na.meta_exclusives.values()):
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,na.theta,len(na),na.K,meta.links)

        for i,meta in enumerate(nb.meta_exclusives.values()):
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,nb.theta,len(nb),nb.K,meta.links)


        for i,meta in enumerate(na.meta_inclusives.values()):
            meta.omega = omega_comp_arrays(len(na),len(meta),meta.q_k_tau,na.theta,meta.zeta,na.K,meta.Tau,meta.links,meta.labels)
            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta[m])
            meta.denominators = meta.denominators[:,np.newaxis]



        for i,meta in enumerate(nb.meta_inclusives.values()):
            meta.omega = omega_comp_arrays(len(nb),len(meta),meta.q_k_tau,nb.theta,meta.zeta,nb.K,meta.Tau,meta.links,meta.labels)

            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta)
            meta.denominators = meta.denominators[:,np.newaxis]


        #creating arrays with the denominator (that are constants) of each node in both layers and em layers

        ##nodes a
        na.denominators = np.zeros(len(na))

        #observed and nonobserved nodes
        self.observed_nodes_a = np.unique(self.links_training[:,0])
        self.non_observed_nodes_a = np.array([i for i in range(len(self.nodes_a)) if i not in self.observed_nodes_a])


        self.neighbours_nodes_a = [] #list of list of neighbours
        for node in self.observed_nodes_a:
            #neighbours in BiNet
            self.neighbours_nodes_a.append(self.links_training[self.links_training[:,0]==node][:,1])
            na.denominators[node] += len(self.neighbours_nodes_a[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(na.meta_exclusives.values()):
            for node in meta.links[:,0]:
                na.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(na)):
            for i, meta in enumerate(na.meta_inclusives.values()):
                na.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])


        na.denominators = na.denominators[:,np.newaxis]



        ##nodes b
        #observed and nonobserved nodes
        self.observed_nodes_b = np.unique(self.links_training[:,1])
        self.non_observed_nodes_b = np.array([i for i in range(len(self.nodes_b)) if i not in self.observed_nodes_b])


        nb.denominators = np.zeros(len(nb))

        self.neighbours_nodes_b = [] #list of list of neighbours
        for node in self.observed_nodes_b:
            #neighbours in BiNet
            self.neighbours_nodes_b.append(self.links_training[self.links_training[:,1]==node][:,0])
            nb.denominators[node] += len(self.neighbours_nodes_b[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(nb.meta_exclusives.values()):
            for node in meta.links[:,0]:
                nb.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(nb)):
            for i, meta in enumerate(nb.meta_inclusives.values()):
                nb.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])

            #neighbours in meta inclusives

        nb.denominators = nb.denominators[:,np.newaxis]


    def EM_step(self,N_steps=1):
        """
        Performs the N_steps number of steps to update the model parameters.

        Parameters
        ----------
        N_steps: int, default: 1
            Number of EM steps to be performed. Default is 1.

        Notes
        -----
        This method updates the model parameters using the Expectation Maximization (EM) estimation.
        The Maximum a Posteriori algorithm is employed for iterative updates.

        During each step, the following updates are performed:
        - Update of nodes_a parameters (BiNet.nodes_a.theta).
        - Update of exclusive_meta and inclusive_meta for nodes_a (BiNet.nodes_a.meta.theta).
        - Update of nodes_b parameters ((BiNet.nodes_b.theta)).
        - Update of exclusive_meta and inclusive_meta for nodes_b (BiNet.nodes_b.meta.theta)..
        - Update of link probabilities (BiNet.pkl) and omega (BiNet.omega).

        After each step, a deep copy of the current model parameters is stored for convergence tracking.

        It is recommended to perform multiple EM steps to refine the model parameters.

        """

        #getting copies from the actual parameters
        self.deep_copying()

        na = self.nodes_a

        nb = self.nodes_b

        for step in range(N_steps):

            for layer,layer_str in [(na,"a"),(nb,"b")]:
                #layer update
                # print(f"layer {layer} ({layer_str})")
                # print(f"\t\t theta")

                layer.theta = theta_comp_arrays_multilayer(self,layer_str)

                ##nodes_a exclusive_meta update
                for i, meta in enumerate(layer.meta_exclusives.values()):
#                     print(f"\t\tmeta {meta}")
                    meta.qka = q_ka_comp_arrays(layer.K,meta.N_att,meta.omega,meta.links,meta.masks_att_list)
                    meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,layer.theta,len(layer),layer.K,meta.links)


                ##nodes_a inclusive_meta update
                for i, meta in enumerate(layer.meta_inclusives.values()):
                    #print(f"\t\tmeta: {meta}")
                    meta.zeta = theta_comp_array(meta.N_att,meta.Tau,meta.omega,meta.denominators,meta.links,meta.masks_att_list)
                    meta.q_k_tau = p_kl_comp_arrays(layer.K,meta.Tau,2,meta.links,meta.omega,meta.masks_label_list)
                    meta.omega = omega_comp_arrays(len(layer),len(meta),meta.q_k_tau,layer.theta,meta.zeta,layer.K,meta.Tau,meta.links,meta.labels)

            self.pkl = p_kl_comp_arrays(na.K,nb.K,self.N_labels, self.links_training, self.omega, self.masks_label_list)
            self.omega = omega_comp_arrays(len(self.nodes_a),len(self.nodes_b),self.pkl,self.nodes_a.theta,self.nodes_b.theta,self.nodes_a.K,self.nodes_b.K,self.links_training,self.labels_training)


    def get_log_likelihoods(self):
        """
        It computes the log_likelihoods from every bipartite network of the multipartite network
        """
        na = self.nodes_a

        nb = self.nodes_b

        #log-like from the labels network
        self.log_likelihood = log_like_comp(na.theta,nb.theta,self.pkl,self.links_training,self.labels_training)

        #log-like from the metadata networks
        for layer in [na,nb]:
            #log-like inclusives meta
            for i, meta in enumerate(layer.meta_inclusives.values()):
                meta.log_likelihood = log_like_comp(layer.theta,meta.zeta,meta.q_k_tau,meta.links,meta.labels)
            #log-like exclusives meta
            for i, meta in enumerate(layer.meta_exclusives.values()):
                meta.log_likelihood = log_like_comp_exclusive(layer.theta,meta.qka,meta.links)

    def get_links_probabilities(self, links = None):
        """
        Computes the label probabilities for links in the trained BiNet.

        Parameters
        ----------
        links : ndarray or DataFrame, optional, default: None
            Array or DataFrame with links for which probabilities are computed.
            - If 2D array, the first column should contain node IDs from nodes_a layer, and the second column from nodes_b layer.
            - If 1D array, it should contain positions of links in self.df attribute.
            - If DataFrame, it should have at least two columns with names of the nodes layers.
            - If None, self.links_training will be used.

        Returns
        -------
        Pij_r : ndarray, shape (len(links), self.N_labels)
            Pij_r[l, r] is the probability that link l has label r.
        """
        if links is None:
            Pij = total_p_comp_test(self.nodes_a.theta,self.nodes_b.theta,self.pkl,self.links_training)
        elif isinstance(links,pd.DataFrame):
            N = links.columns.isin([str(self.nodes_a)+"_id",str(self.nodes_b)+"_id"]).sum()

            if N==2:
                links = links[[str(self.nodes_a)+"_id",str(self.nodes_b)+"_id"]].values
            elif N==0:
                links = links[[str(self.nodes_a),str(self.nodes_b)]].replace(to_replace={str(self.nodes_a):self.nodes_a.dict_codes,str(self.nodes_b):self.nodes_b.dict_codes}).values
            elif N==1:
                if links.columns.isin([str(self.nodes_a)+"_id"]).any():
                    links = links[[str(self.nodes_a)+"_id",str(self.nodes_b)]].replace(to_replace={str(self.nodes_b):self.nodes_b.dict_codes}).values
                else:
                    links = links[[str(self.nodes_a),str(self.nodes_b)+"_id"]].replace(to_replace={str(self.nodes_a):self.nodes_a.dict_codes}).values

            Pij = total_p_comp_test(self.nodes_a.theta,self.nodes_b.theta,
                                            self.pkl,links)

        elif len(links.shape) == 1:
            Pij = total_p_comp_test(self.nodes_a.theta,self.nodes_b.theta,
                                            self.pkl,self.links[links])
        elif len(links.shape) == 2:
            Pij = total_p_comp_test(self.nodes_a.theta,self.nodes_b.theta,
                                            self.pkl,links)

        return Pij


    def get_predicted_labels(self, links = None, Pij = None,  estimator = "max_probability", to_return = "df"):
        """

        Computes the predicted labels of the model based on the MMSBM parameters, using different estimators. They can be measured by different estimators:
            - max_probability: The predicted label will be the most plausible label
            - mean: The predicted label will be the mean

        Parameters
        ----------
        links: ndarray of 1 or 2 dimensions, pandas DataFrame, default: None
            Array with links for which label probabilities are computed.
            -If a 2d-array, the first column must contain the ids from nodes_a layer and the second
             column must contain the ids from nodes_b layers.
            -If a 1d-array, it must contain the positions of the links list from self.df attribute
            -If a pandas DataFrame, it must contain at least two columns with the name of the nodes' layers
             and a column with the same name as the labels column from BiNet.df.
            -If None, self.links_training will be used.

        Pij: ndarray, default: None
            Array with the probabilities of the links to have each label. If None, it will compute the probabilities using self.get_links_probabilities(links).

        estimator: {"max_probability","average"}, default: max_probability
            Estimator used to get predicted labels:
            - "max_probability": Selects the most plausible label.
            - "average": Selects the average label (sum [Pij(l) * l]).

        to_return: {"df","ids", "both"}, default: df
            Option to choose how the predicted labels will be returned.
             -"df": Returns a DataFrame with columns for nodes from both layers and an additional column called "Predicted + self.label_name".
             -"ids": Returns a ndarray of ints with the ids of the predicted labels.
             -"both": Returns both the DataFrame and the ndarray with the ids in this order.


        Returns
        -------
        labels_id: ndarray
            Predicted labels id.

        labels_df: pandas DataFrame
            DataFrame whose columns are nodes_a, nodes_b and the predicted labels


        Notes
        -----
        If Pij is provided, it will use the given probabilities; otherwise, it will compute probabilities using self.get_links_probabilities(links).
        """
        if isinstance(Pij,np.ndarray):
            if estimator=="max_probability":
                labels_id =  Pij.argmax(axis=1)
            elif estimator=="mean":
                labels_id = np.rint(Pij@np.arange(0,self.N_labels)[:,np.newaxis])[:,0]
        else:
            Pij = self.get_links_probabilities(links)
            if estimator=="max_probability":
                labels_id = Pij.argmax(axis=1)
            elif estimator=="mean":
                labels_id = np.rint(Pij@np.arange(0,self.N_labels)[:,np.newaxis])[:,0]

        if to_return == "df" or to_return == "both":
            if links is None:
                to_link = self.links_training
            elif isinstance(links,pd.DataFrame):
                decoder = {self._dict_codes[n]:n for n in self._dict_codes}
                labels = [decoder[n] for n in labels_id]
                result_df = links.copy()
                result_df.loc[:, "Predicted "+self.labels_name] = labels

                if to_return == "df":
                    return result_df
                elif to_return == "both":
                    return result_df, labels_id
            elif len(links.shape)==1:
                to_link = self.links[links]
            elif len(links.shape)==2:
                to_link = links


            na = self.nodes_a
            nb = self.nodes_b

            decoder = {na.dict_codes[n]:n for n in na.dict_codes}
            A = [decoder[n] for n in to_link[:,0]]

            decoder = {nb.dict_codes[n]:n for n in nb.dict_codes}
            B = [decoder[n] for n in to_link[:,1]]

            decoder = {self._dict_codes[n]:n for n in self._dict_codes}
            labels = [decoder[n] for n in labels_id]

            if to_return == "df":
                return pd.DataFrame({str(na):A,str(nb):B,"Predicted "+self.labels_name:labels})
            elif to_return == "both":
                return pd.DataFrame({str(na):A,str(nb):B,"Predicted "+self.labels_name:labels}), labels_id

        elif to_return == "ids":
            return labels_id

    def get_accuracy(self, predicted_labels = None, test_labels = None, Pij = None,links = None, estimator = "max_probability"):
        """
        Computes the ratio of correctly predicted labels of the model given the MMSBM parameters. They can be measured by different estimators:
            -max_probability: The predicted label will be the most plausible label
            -mean: The predicted label will be the mean

        Parameters
        ----------
        predicted_labels: array-like, default:None.
            Array-like with the predicted labels ids given by the MMSBM. If None, predictions will be generated using
        the specified links and estimator.

        test_labels: array-like, default:None.
            List or array with the observed labels. If None, labels from self.labels_array are taken given pos_test_labels

        links: ndarray of 1 or 2 dimensions, pandas DataFrame, default: None
            Array with links for which label probabilities are computed.
            -If a 2d-array, the first column must contain the ids from nodes_a layer and the second
             column must contain the ids from nodes_b layers.
            -If a 1d-array, it must contain the positions of the links list from self.df attribute
            -If a pandas DataFrame, it must contain at least two columns with the name of the nodes' layers
             and a column with the same name as the labels column from BiNet.df.
            -If None, self.links_training will be used.


        estimator: {"max_probability","mean"}, default: max_probability
            Estimator used to get the predicted labels:
            -max_probability: Selects the most plausible label
            -mean: Selects the mean label (sum [Pij(l)*l])

        Returns
        -------
        accuracy: float
            Ratio of correctly predicted labels to the total number of predicted labels.
        """
        #If predicted labels are not provided, it will compute them
        if predicted_labels is None:
            predicted_labels = self.get_predicted_labels(links=links, Pij=Pij, estimator=estimator, to_return="ids")

        #If test labels are not provided, we get them from links
        if test_labels is None:
            if isinstance(links,pd.DataFrame):
                if links.columns.isin([str(self.labels_name)+"_id"]).any():
                    test_labels = links[str(self.labels_name)+"_id"].values
                elif links.columns.isin([str(self.labels_name)]).any():
                    test_labels = links.replace({self.labels_name:self._dict_codes}).values[:,2]

            elif isinstance(links,np.ndarray):
                if len(links.shape)==1:
                    test_labels = self.labels_array[links]
                else:
                    raise TypeError("""Missing test label information to compare:
                                     -Array in test_labels with labels ids.
                                     -Pandas DataFrame with links and their labels in links parameter.
                                     -Links position of the BiNet.links_array.""")
            elif links is None:
                test_labels = self.labels_training
        return (predicted_labels==test_labels).sum()/len(predicted_labels)

    def deep_copying(self):
        """
        Performs a deep copy of all parameters in the EM algorithm.

        Notes
        -----
        This method creates deep copies of various parameters to store their current states for future reference and convergence checking.

        - Link Parameters:
            - pkl_old: Deep copy of the link probabilities (self.pkl).
            - omega_old: Deep copy of omega (self.omega).

        - Metadata parameters (for each layer):
            - theta_old: Deep copy of the layer's theta parameters (self.theta).
            - Inclusive metadata:
                - zeta_old: Deep copy of zeta (meta.zeta).
                - q_k_tau_old: Deep copy of q_k_tau (meta.q_k_tau).
                - omega_old: Deep copy of omega (meta.omega).
            - Exclusive metadata:
                - qka_old: Deep copy of qka (meta.qka).
                - omega_old: Deep copy of omega (meta.omega).

        """

        na = self.nodes_a

        nb = self.nodes_b


        self.pkl_old = self.pkl.copy()
        self.omega_old = self.omega.copy()


        ##Metas copies
        for layer in [na,nb]:
            layer.theta_old = layer.theta.copy()
            ##inclusive_meta copies
            for i, meta in enumerate(layer.meta_inclusives.values()):
                meta.zeta_old = meta.zeta.copy()
                meta.q_k_tau_old = meta.q_k_tau.copy()
                meta.omega_old = meta.omega.copy()

            ##exclusive_meta copies
            for i, meta in enumerate(layer.meta_exclusives.values()):
                meta.qka_old = meta.qka.copy()
                meta.omega_old = meta.omega.copy()

    def converges(self):
        """
        Checks if the parameters have converged during the EM procedure.

        Returns
        -------
        bool
            True if the parameters have converged, False otherwise.

        Notes
        -----
        Convergence is determined based on the tolerance (self.tol) set for the model.

        - Meta Convergence:
            - Checks convergence for each layer's theta and metadata parameters.
            - Metadata parameters include zeta, q_k_tau, and omega for both inclusive and exclusive metadata.

        - Links Convergence:
            - Checks convergence for pkl (link probabilities) and omega parameters.
        """
        na = self.nodes_a

        nb = self.nodes_b

        tol = self.tol


        ##Metas convergence
        for layer in [na,nb]:
            if not finished(layer.theta_old,layer.theta,tol): return False
            ##inclusive_meta convergence
            for i, meta in enumerate(layer.meta_inclusives.values()):
                if not finished(meta.zeta_old,meta.zeta,tol): return False
                if not finished(meta.q_k_tau_old,meta.q_k_tau,tol): return False
                if not finished(meta.omega_old,meta.omega,tol): return False

            ##exclusive_meta convergence
            for i, meta in enumerate(layer.meta_exclusives.values()):
                if not finished(meta.qka_old,meta.qka,tol): return False
                if not finished(meta.omega_old,meta.omega,tol): return False

        #links convergence
        if not finished(self.pkl_old,self.pkl,tol):return False
        if not finished(self.omega_old,self.omega,tol):return False

        return True
