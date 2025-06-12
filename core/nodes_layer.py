import pandas as pd
import numpy as np
import json
from .metadata_layer import metadata_layer
from .exclusive_metadata import exclusive_metadata
from .inclusive_metadata import inclusive_metadata
from ..functions import *


class nodes_layer:
    """
    Base class of a layer that contains nodes

    Is initialized using a dataframe and can be modify it using the df attribute

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
    dict_decodes : dict
        Dictionary with the integer id of the nodes. The key is the nodes' id and the value its name.
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

        self.dict_decodes = {v: k for k, v in self.dict_codes.items()}


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
    
    @property
    def dict_decodes(self):
        return self._dict_decodes

    @dict_decodes.setter
    def dict_decodes(self,dd):
        self._dict_decodes = dd
        return dd

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
        """
        Returns the number of nodes in the layer.

        Returns
        -------
        int
            Number of nodes
        """
        return len(self.df)

    def __iter__(self):
        """
        Returns an iterator over the nodes.

        Returns
        -------
        iterator
            Iterator over node IDs
        """
        return iter(self.df[self.node_type + "_id"])

    def __contains__(self, node_id):
        """
        Check if a node exists in the layer.

        Parameters
        ----------
        node_id : int
            Node identifier

        Returns
        -------
        bool
            True if the node exists, False otherwise
        """
        return node_id in self.df[self.node_type + "_id"]
    
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
        
        if lambda_val > 1.e-16:
            self._has_metas = True



        # create metadata object
        em = exclusive_metadata(lambda_val, meta_name)


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
        ----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_val: float
            Value of the metadata visibility

        Tau: Int
            Number of membership groups of metadata

        separator: str, default: "|"
            Separator that is used to differentiate the different metadata assigned for each node

        dict_codes: dict, None, default: None 
            Dictionary where the keys are the names of metadata's type,
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


        if lambda_val > 1.e-16:
            self._has_metas = True


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

        # Add the new column with metadata IDs
        self.df[str(im) + '_id'] = self.df[str(im)].apply(
            lambda x: im.code_inclusive_metadata(x)
        )

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
        im.N_labels = 2  #connected or disconnected
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
        ----------

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
        ----------
        im: inclusive_metadata
            inclusive_metadata object to update the ids

        dict_codes: dict
            Dictionary where the keys are the names of metadata's type, and the values are the ids.
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

        utils.save_nodes_layer_dict(self, dir)

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
            replacer[im.dict_codes[att]] = dict_codes[att]

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
        links = np.ones((len(observed) * im.N_att, 2), dtype=np.int64)
        # Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed) * im.N_att, dtype=np.int64)

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
        im.N_labels = 2  #connected or disconnected

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
        utils.save_nodes_layer_dict(self, dir)

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
        layer = cls(nodes_info=df, **data)

        for m in data["metadata_exclusives"]:
            layer.add_exclusive_metadata(**m)

        for m in data["metadata_inclusives"]:
            layer.add_inclusive_metadata(**m)

        return layer 