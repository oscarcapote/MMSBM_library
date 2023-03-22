# coding: utf-8


import pandas as pd
import numpy as np
from MMSBM_library.functions import *


class metadata_layer:
    """
    Principal class of nodes_layer metadata. It contains extra information about the nodes.

    It has two subclasses:
        - exclusive_metadata
        - inclusive_metadata
    """
    def __init__(self, lambda_val, meta_name):
        self.meta_name = meta_name
        self.lambda_val = lambda_val

    # @property
    # def N_meta(self):
    #     return self._N_meta
    #
    # @N_meta.setter
    # def N_meta(self, N_meta):
    #     self._N_meta = N_meta
    #     return self._N_meta

    @property
    def dict_codes(self):
        return self.dict_codes

    @dict_codes.setter
    def dict_codes(self,dict_codes):
        self.dict_codes = dict_codes
        return dict_codes

    @property
    def N_att(self):
        return self._N_att

    @N_att.setter
    def N_att(self, N_att):
        """
        Number of different categorical attributes of the metadata

        Parameters
        -----------
        N_att: Int
            Number of different categorical attributes of the metadata

        """
        self._N_att = N_att
        return self.N_att

    @property
    def links(self):
        return self._links

    @links.setter
    def links(self, links):
        """
        Adds the links between nodes and metadata and update number of links

        Parameters
        -----------
        links: 2D NumPy array
            Array with (N_meta, N_links)

        Returns
        -------
        links: 2D NumPy array with the links between nodes and metadata
        """

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
        # self.qka = K
        # print("---",self.qka.shape)

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
        # self.zeta = self.zeta(Tau)


    @property
    def q_k_tau(self):
        return self._q_k_tau

    @q_k_tau.setter
    def q_k_tau(self, q_k_tau):
        self._q_k_tau =  q_k_tau



    # @q_k_tau.setter
    def init_q_k_tau(self, K, Tau):
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
    This metadata can be classify it as exclusive_metadata (if a node only accepts one attribute) and inclusive_metadata (if the node accepts more than one attribute)

    See for more information of metadata: metadata_layer, exclusive_metadata and inclusive_metadata.

    These objects can be added into a BiNet (bipartite network) where connection between nodes_layer are considered to infer links and their labels  (see BiNet)

    ...

    Attributes
    ----------
    K: int
        Number of groups.
    node_type: str
        Name of the layer. Also is the name of the column where the nodes' name are contained.
    df: pandas dataFrame
        DataFrame that contains the nodes information. It contains one column
        with the nodes' name and the rest are attributes.
    dict_codes: dict
        Dictionary with the integer id of the nodes. The key is the nodes' name and the value its id.
    meta_exclusives: list of metadata_layer
        List with the metadata exclusives object that contains the metadata that will be used in the MAP algorithm.
    meta_inclusives: list of metadata_layer
        List with the metadata inclusives object that contains the metadata that will be used in the MAP algorithm.
    meta_neighbours_exclusives:
        List of lists that contains, for each node its exclusives metadata neighbours
    meta_neighbours_inclusives:
        List of lists that contains, for each node its inclusives metadata neighbours
    nodes_observed_inclusive:
        List of arrays for each metadata with the nodes that has assigned an attribute of the metadata

    """


    def __init__(self, K, nodes_name, nodes_info, *, separator="\t", dict_codes = None, **kwargs):
        """
        Initialization of the nodes_layer class

        Parameters
        ----------
        K: int
            Number of groups
        nodes_name: str
            Name of the nodes column in the nodes_layer class
        nodes_info: str or pandas DataFrame
            Is it is a string, it is the directory of the file with the nodes informations
        separator: str, default \t
            Separator if the columns of the file that contains the nodes information
        dict_codes: dict, default None
            Dictionary with the integer id of the nodes. The key is the nodes' name and the value its id.
        """

        self.K = K
        self.node_type = nodes_name

        if type(nodes_info) == type("d"):
            self.df = self.read_file(nodes_info, separator)
        elif type(nodes_info) == type(pd.DataFrame()):
            self.df = nodes_info

        # codes = pd.Categorical(self.df[nodes_name]).codes
        # self.codes = codes
        self.dict_codes = add_codes(self,nodes_name)

        if dict_codes != None:
            if self.df.dtypes[self.node_type] == np.dtype("int64") or self.df.dtypes[self.node_type] == np.dtype("int32") or self.df.dtypes[self.node_type] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            else:
                for k in dict_codes:
                    dict_codes[k] = int(dict_codes[k])

            replacer = {}
            for att in dict_codes:
                replacer[self.dict_codes[att]]= dict_codes[att]
            self.dict_codes = dict_codes
            self.df.replace({nodes_name+"_id":replacer}, inplace=True)


        # self.df = self.df.join(pd.DataFrame(codes, columns=[nodes_name+"_id"]))
        # print(self.df)
        # self.df = pd.concat([self.df, pd.DataFrame({nodes_name + "_id": codes})], axis=1, ignore_index=True)
        # self.df[nodes_name + "_id"] = codes
        # print(self.df)
        self.nodes_list = self.df[nodes_name].unique()


        self.meta_exclusives = []
        self.meta_inclusives = []
        self.meta_neighbours_exclusives = []
        self.meta_neighbours_inclusives = []  # all neighbours (connected and not) of inclusive metadata
        # self.inclusive_linked = []  # metadata inclusive for each node
        self.nodes_observed_inclusive = []
        self._has_metas = False  #Boolean that tells you if you have metadata initialized with non 0 values of lambda_val

        self.N_nodes = len(self.nodes_list)
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0



    def read_file(self, filename, separator="\t"):
        return pd.read_csv(filename, sep=separator, engine='python')

    @classmethod
    def create_simple_layer(cls, K, nodes_list, nodes_name, dict_codes=None):
        '''
        Create a nodes_layer object from a list or DataSeries without only with the known nodes

        Parameters
        -----------
        K: Int
            Number of membership groups of nodes_layer

        nodes_list: array-like, DataFrame or DataSeries
            array-like, DataFrame or DataSeries with all the nodes

        nodes_name: str
            Name of the nodes type (users, movies, metobolites...) that are or will be in DataFrame

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

    def add_exclusive_metadata(self, lambda_val, meta_name,*,dict_codes=None):
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

        df_dropna = self.df.dropna(subset=[meta_name])
        observed = df_dropna[str(self)+"_id"].values

        if lambda_val>1.e-16:self._has_metas = True

        # encode metadata
        # codes = pd.Categorical(self.df[meta_name]).codes
        # self.df = self.df.join(pd.DataFrame(codes, columns=[meta_name + "_id"]))


        # create metadata object
        em = exclusive_metadata(lambda_val, meta_name)
        em.dict_codes = add_codes(self,meta_name)
        em._meta_code = self.N_meta_exclusive

        if dict_codes != None:
            if self.df.dtypes[meta_name] == np.dtype("int64") or self.df.dtypes[meta_name] == np.dtype("int32") or self.df.dtypes[meta_name] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            replacer = {}
            for att in dict_codes:
                replacer[em.dict_codes[att]]= dict_codes[att]

            em.dict_codes = dict_codes
            self.df.replace({meta_name +"_id":replacer}, inplace=True)

        em.links = self.df[[self.node_type + "_id", meta_name + "_id"]].values
        em.N_att = len(em.dict_codes)
        # em.qka = em.init_qka(self.K)

        #list of arrays of ints where the array number att has all the index positions of links that connects the attribute att
        em.masks_att_list = []
        for r in range(em.N_att):
            mask = np.argwhere(em.links[:,1]==r)[:,0]
            em.masks_att_list.append(mask)


        # update meta related nodes attributes
        self.meta_exclusives.append(em)
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes, dtype=np.int32)

        for n in range(self.N_nodes):
            meta_neighbours[n] = self.df[self.df[self.node_type + "_id" ]== n][meta_name + "_id"]#.values

        self.meta_neighbours_exclusives.append(meta_neighbours)



    def add_inclusive_metadata(self, lambda_val, meta_name, Tau,*,dict_codes=None, separator="|"):
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
            Separator that is used to differenciate the differents metadata assigned for each node

        dict_codes: dict, None, default: None
            Dictionary where the keys are the names of metadata's type, and the values are the ids. If None, the program will generate the ids.
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
        meta_neighbours = []#[[int(j) for j in i.split(separator)] for i in df_dropna[meta_name].values]#meta connected with 1

        for arg in np.argsort(observed_id):
            i = meta_list[arg]
            if i == None or i == np.NaN or i == pd.NaT:
                meta_neighbours.append(None)
            else:
                meta_neighbours.append([j for j in i.split(separator)])


        if dict_codes != None:
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

        self.nodes_observed_inclusive.append(observed_id)

        # update meta related nodes attributes
        self.meta_inclusives.append(im)
        self.N_meta_inclusive += 1
        self.N_meta += 1

        self.meta_neighbours_inclusives.append(meta_neighbours)


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

        for i,meta in enumerate(self.meta_exclusives):
            if meta==em:
                self.meta_neighbours_exclusives[i] = meta_neighbours

    def update_inclusives_id(self, im, dict_codes):
        '''
        Changes the ids (the integer assigned to each metadata attribute) given the dict_codes.

        Parameters
        -----------

        dict_codes: dict
            Dictionary where the keys are the names of metadata's type, and the values are the ids. If None, the program will generate the ids.
        '''
        observed = self.nodes_observed_inclusive[im._meta_code]
        meta_neighbours = []


        #Replacer to change the ids
        replacer = {}
        for att in dict_codes:
            replacer[im.dict_codes[att]]= dict_codes[att]


        #New ids into the neigbours list
        for neig in self.meta_neighbours_inclusives[im._meta_code]:
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
        self.meta_neighbours_inclusives[im._meta_code] = [[m for m in L] for L in meta_neighbours]

class BiNet:
    """
    Class of a Bipartite Network, where two layers of different types of nodes are connected (users->items,
    politicians->bills, patient->microbiome...) and these links can be labeled with information of the
    interaction (ratings, votes...).

    ...

    Attributes
    ----------
    labels_name:
        Name of the labels column
    N_labels:
        Number of diferent types labels.
    labels_array:
        Array with all the ids of the labels.
    labels_name:
        List of the names of the diferents labels.
    labels_training:
        Array with all the ids of the labels used to train the MMSBM
    df:
        Dataframe with the links information, who connected to who and with which label are connected
    dict_codes:
        Dictionary with the integer id of the labels. The key is the labels' name and the value its id.
    nodes_a,nodes_b:
        nodes_layer objects of the nodes that are part from the bipartite network
    links:
        2D-array with the links of the nodes that are connected
    links_training:
        2D-array with the links of the nodes that are connected used to train the MMSBM


    labels_array


    """
    def __init__(self, links, links_label,*, nodes_a = None, nodes_b = None, Ka=1, nodes_a_name="nodes_a", Kb=1,
                 nodes_b_name="nodes_b", separator="\t", dict_codes = None, dict_codes_a = None, dict_codes_b = None):
        """
         Initialization of a BiNet class

         Parameters
         -----------
         links: str, DataFrame
            DataFrame or directory where the DataFrame is. It should contains the links list between nodes_a and nodes_b and their labels.

         links_label: str
             Name of the links column where the labels are

         nodes_a: nodes_layer, str, DataFrame, None, default: None
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type a are.
             If it is a pandas DatFrame, it has to contain the information of the nodes of type a.
             If None, it a simple nodes_layer will be created from the information from links.

         nodes_b: nodes_layer, str, DataFrame, None, default: None
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type b are.
             If it is a pandas DatFrame, it has to contain the information of the nodes of type b.
             If None, it a simple nodes_layer will be created from the information from links.

         Ka: int, default: 1
            Number of membership groups from nodes_a layer

         Kb: int, default: 1
            Number of membership groups from nodes_b layer

         nodes_a_name: str, default: nodes_a
            Name of the column where the names of nodes_a are in the links DataFrame and nodes_a DataFrame

         nodes_b_name: str, default: nodes_b
            Name of the column where the names of nodes_b are in the links DataFrame and nodes_b DataFrame

         dict_codes: dict, None, default: None
            Dictionary where the keys are the names of the labels, and the values are the ids. If None, the program will generate the ids.

         dict_codes_a: dict, None, default: None
            Dictionary where the keys are the names of the nodes from nodes_a and the values are the ids. If None, the program will generate the ids.

         dict_codes_b: dict, None, default: None
            Dictionary where the keys are the names of the nodes from nodes_b and the values are the ids. If None, the program will generate the ids.

         separator: str, default: \t
            Separator of the links DataFrame. Default is \t
        """
        if type(links) == type(pd.DataFrame()):
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
        self.dict_codes = add_codes(self,links_label)
        if dict_codes != None:
            if self.df.dtypes[self.labels_name] == np.dtype("int64") or self.df.dtypes[self.labels_name] == np.dtype("int32") or self.df.dtypes[self.labels_name] == np.dtype("int16"):
                new_dict = {}
                for k in dict_codes:
                    new_dict[int(k)] = int(dict_codes[k])
                dict_codes = new_dict
            else:
                for k in new_dict:
                    dict_codes[k] = int(dict_codes[k])

            replacer = {}
            for att in dict_codes:
                replacer[self.dict_codes[att]]= dict_codes[att]

            self.dict_codes = dict_codes
            self.df.replace({self.labels_name+"_id":replacer}, inplace=True)

        # codes = pd.Categorical(self.df[links_label]).codes
        # self.df = self.df.join(pd.DataFrame(codes, columns=[links_label + "_id"]))




        #Links
        self.df[nodes_a_name + "_id"] = self.df[[nodes_a_name]].replace({nodes_a_name:self.nodes_a.dict_codes})
        self.df[nodes_b_name + "_id"] = self.df[[nodes_b_name]].replace({nodes_b_name:self.nodes_b.dict_codes})

        #self.df = self.df.join(self.nodes_a.df[[nodes_a_name,nodes_a_name + "_id"]].set_index(nodes_a_name),on=nodes_a_name)
        #self.df = self.df.join(self.nodes_b.df[[nodes_b_name,nodes_b_name + "_id"]].set_index(nodes_b_name),on=nodes_b_name)

        self.df = self.df.drop_duplicates()
        self.labels_array = self.df[links_label + "_id"].values
        self.links = self.df[[nodes_a_name + "_id", nodes_b_name + "_id"]].values


        # #observed nodes in each layer
        # self.observed_nodes_a = np.unique(self.links[:,0])
        # self.observed_nodes_b = np.unique(self.links[:,1])
        #
        # #non_observed nodes in each layer
        # self.non_observed_nodes_a = np.array([i for i in range(len(self.nodes_a)) if i not in self.observed_nodes_a])
        # self.non_observed_nodes_b = np.array([i for i in range(len(self.nodes_b)) if i not in self.observed_nodes_b])



        self.N_labels = max(self.labels_array)+1

        #masks list to know wich links have label r (that is the index of the list)
        # self.masks_label_list = []
        # for r in range(self.N_labels):
        #     mask = np.argwhere(self.labels_array==r)[:,0]
        #     self.masks_label_list.append(mask)


    @classmethod
    def load_BiNet_from_json(cls, json_file, links, links_label,*, nodes_a = None, nodes_b = None, separator="\t"):

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
            na = nodes_layer(data["layer a"]["K"],data["layer a"]["name"],nodes_a,dict_codes=data["layer a"]["dict_codes"])
        elif  nodes_a == None:
            #later it will be created
            na = None

        # creating second layer class
        if isinstance(nodes_b, nodes_layer):
            nb = nodes_b
            nodes_b_name = str(nb)
            Kb = nodes_b.K
        elif isinstance(nodes_b, str) or isinstance(nodes_b, pd.DataFrame):
            nb = nodes_layer(data["layer b"]["K"],data["layer b"]["name"],nodes_b,dict_codes=data["layer b"]["dict_codes"])
        elif  nodes_b == None:
            #later it will be created
            nb = None



        #creating BiNet
        if na == None and nb == None:
            BN = cls(links,links_label,
                     nodes_a = None, Ka=data["layer a"]["K"], nodes_a_name=data["layer a"]["name"],
                               dict_codes_a=data["layer a"]["dict_codes"],
                     nodes_b = None, Kb=data["layer b"]["K"], nodes_b_name=data["layer b"]["name"],
                               dict_codes_b=data["layer b"]["dict_codes"],
                     separator=separator,dict_codes = data["dict_codes"])


            #layer a
            na = BN.nodes_a
            #layer b
            nb = BN.nodes_b

        elif na == None:
            BN = cls(links,links_label,
                     nodes_a = None, Ka=data["layer a"]["K"], nodes_a_name=data["layer a"]["name"],dict_codes_a=data["layer a"]["dict_codes"],
                     nodes_b = nb,
                     separator=separator,dict_codes = data["dict_codes"])
            #layer a
            na = BN.nodes_a
            #layer b
            nb = BN.nodes_b


        elif nb == None:
            BN = cls(links,links_label,
                     nodes_a = na,
                     nodes_b = None, Kb=data["layer b"]["K"], nodes_b_name=data["layer b"]["name"],dict_codes_b=data["layer b"]["dict_codes"],
                     separator=separator,dict_codes = data["dict_codes"])


            #layer a
            na = BN.nodes_a
            #layer b
            nb = BN.nodes_b

        else:
            BN = cls(links,links_label,
                     nodes_a = na,
                     nodes_b = nb,
                     separator=separator,dict_codes = data["dict_codes"])

        #metadatas
        for l,layer in [("a",na),("b",nb)]:

            #inclusives metadata
            for meta in data["layer {}".format(l)]["metadata_inclusives"]:
                layer.add_inclusive_metadata(meta["lambda"],
                                  meta["Meta_name"],
                                  meta["Tau"],
                                  dict_codes = meta["dict_codes"])

            #inclusives metadata
            for meta in data["layer {}".format(l)]["metadata_exclusives"]:
                layer.add_exclusive_metadata(meta["lambda"],
                                  meta["Meta_name"],
                                  dict_codes = meta["dict_codes"])


        return BN



    #MAP ALGORITHM
    def init_MAP(self,tol=0.001, training = None, seed=None):
        '''
        Initialize the MAP algorithm to get the most plausible memberhip parameters of the MMSBM

        Parameters
        -----------
        tol: float, default: 0.001
            Tolerance of the algorithm when finding the parameters.

        seed: int, None, default: None
            Seed to generate the matrices. Is initialized using the np.random.RandomState(seed) method.

        training: DataFrame, list, default: None
            DataFrame with the links that you want to use to train your MMSBM.
            If it is a list, it must contain the indexes of the links list that will be used to train the MMSBM.

        '''
        # Probability matrices
        # np.random.RandomState(seed)

        self.tol = tol

        #BiNet matrices
        self.pkl = init_P_matrix(self.nodes_a.K, self.nodes_b.K, self.N_labels)

        #memberships (thetas)
        self.nodes_a.theta = init_P_matrix(len(self.nodes_a),self.nodes_a.K)
        self.nodes_b.theta = init_P_matrix(len(self.nodes_b),self.nodes_b.K)

        # Links to train management
        if isinstance(training,pd.DataFrame):
            self.links_training = training[str(self.nodes_a)+"_id",str(self.nodes_b)+"_id"].values
            self.labels_training = training[self.labels_name+"_id"].values
        elif isinstance(training,list) or isinstance(training,np.ndarray):
            self.links_training = self.links[training]
            self.labels_training = self.labels_array[training]
        elif training == None:
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
        for meta in self.nodes_a.meta_exclusives:
            meta.qka = init_P_matrix(self.nodes_a.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_a.theta,len(self.nodes_a),self.nodes_a.K,meta.links)

        for meta in self.nodes_b.meta_exclusives:
            meta.qka = init_P_matrix(self.nodes_b.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_b.theta,len(self.nodes_b),self.nodes_b.K,meta.links)

        # print("aqui2")
        ## ql_tau, zetes and omegas omega_comp_arrays(omega,p_kl,theta,eta,K,L,links_array,links_ratings):
        for meta in self.nodes_a.meta_inclusives:
            meta.q_k_tau = init_P_matrix(self.nodes_a.K, meta.Tau, 2)
            meta.zeta = init_P_matrix(len(meta), meta.Tau)
            meta.omega = omega_comp_arrays(len(self.nodes_a),len(meta),meta.q_k_tau,self.nodes_a.theta,meta.zeta,self.nodes_a.K,meta.Tau,meta.links,meta.labels)
            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta[m])
            meta.denominators = meta.denominators[:,np.newaxis]



        for meta in self.nodes_b.meta_inclusives:
            meta.q_k_tau = init_P_matrix(self.nodes_b.K, meta.Tau, 2)
            meta.zeta = init_P_matrix(len(meta), meta.Tau)
            meta.omega = omega_comp_arrays(len(self.nodes_b),len(meta),meta.q_k_tau,self.nodes_b.theta,meta.zeta,self.nodes_b.K,meta.Tau,meta.links,meta.labels)

            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta)


        #creating arrays with the denominator (that are constants) of each node in both layers and em layers

        ##nodes a
        self.nodes_a.denominators = np.zeros(len(self.nodes_a))

        self.neighbours_nodes_a = [] #list of list of neighbours
        for node in range(len(self.nodes_a)):
            #neighbours in BiNet
            self.neighbours_nodes_a.append(self.links_training[self.links_training[:,0]==node,1])
            self.nodes_a.denominators[node] += len(self.neighbours_nodes_a[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(self.nodes_a.meta_exclusives):
            for node in meta.links[:,0]:
                self.nodes_a.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(self.nodes_a)):
            for i, meta in enumerate(self.nodes_a.meta_inclusives):
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
        for i, meta in enumerate(self.nodes_b.meta_exclusives):
            for node in meta.links[:,0]:
                self.nodes_b.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(self.nodes_b)):
            for i, meta in enumerate(self.nodes_b.meta_inclusives):
                self.nodes_b.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])

            #neighbours in meta inclusives

        self.nodes_b.denominators = self.nodes_b.denominators[:,np.newaxis]
            # for meta in self.nodes_b.meta_exclusives:


    def init_MAP_from_directory(self,dir="."):
        '''
        Initialize the MAP algorithm to get the most plausible membership parameters of the MMSBM using parameters saved in files that are in a directory

        Parameters
        -----------

        dir: str, default: "."
            Directory where the files with the MMSBM parameters will be loaded

        '''
        na = self.nodes_a

        nb = self.nodes_b


        load_MAP_parameters(self,dir)



        #Omegas and denominators
        self.omega = omega_comp_arrays(len(na),len(nb),self.pkl,na.theta,nb.theta,na.K,nb.K,self.links_training,self.labels_training)



        #Metadata
        ## qka and omegas
        for meta in na.meta_exclusives:
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,na.theta,len(na),na.K,meta.links)

        for meta in nb.meta_exclusives:
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,nb.theta,len(nb),nb.K,meta.links)


        for meta in na.meta_inclusives:
            meta.omega = omega_comp_arrays(len(na),len(meta),meta.q_k_tau,na.theta,meta.zeta,na.K,meta.Tau,meta.links,meta.labels)
            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta[m])
            meta.denominators = meta.denominators[:,np.newaxis]



        for meta in nb.meta_inclusives:
            meta.omega = omega_comp_arrays(len(nb),len(meta),meta.q_k_tau,nb.theta,meta.zeta,nb.K,meta.Tau,meta.links,meta.labels)

            #neighbours and denominators from meta
            meta.denominators = np.zeros(len(meta))

            for m in range(len(meta)):
                meta.denominators[m] += len(meta.neighbours_meta)


        #creating arrays with the denominator (that are constants) of each node in both layers and em layers

        ##nodes a
        na.denominators = np.zeros(len(na))

        self.neighbours_nodes_a = [] #list of list of neighbours
        for node in self.observed_nodes_a:
            #neighbours in BiNet
            self.neighbours_nodes_a.append(self.links_training[self.links_training[:,0]==node][:,1])
            na.denominators[node] += len(self.neighbours_nodes_a[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(na.meta_exclusives):
            for node in meta.links[:,0]:
                na.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(na)):
            for i, meta in enumerate(na.meta_inclusives):
                na.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])


        na.denominators = na.denominators[:,np.newaxis]



        ##nodes b
        nb.denominators = np.zeros(len(nb))

        self.neighbours_nodes_b = [] #list of list of neighbours
        for node in self.observed_nodes_b:
            #neighbours in BiNet
            self.neighbours_nodes_b.append(self.links_training[self.links_training[:,1]==node][:,0])
            nb.denominators[node] += len(self.neighbours_nodes_b[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(nb.meta_exclusives):
            for node in meta.links[:,0]:
                nb.denominators[node] += meta.lambda_val

        #neighbours in meta inclusives
        for node in range(len(nb)):
            for i, meta in enumerate(nb.meta_inclusives):
                nb.denominators[node] += meta.lambda_val*len(meta.links[meta.links[:,0]==node,:])

            #neighbours in meta inclusives

        nb.denominators = nb.denominators[:,np.newaxis]


    def MAP_step(self,N_steps=1):
        """

        Parameters
        ----------
        N_steps: int, default: 1
            Number of MAP steps that will be performed. N_steps = 1 as default.
        """
        na = self.nodes_a

        nb = self.nodes_b

        for step in range(N_steps):
            #nodes_a update
            na.theta = theta_comp_arrays_multilayer(self)

            ##nodes_a exclusive_meta update
            for i, meta in enumerate(na.meta_exclusives):
                meta.qka = q_ka_comp_arrays(na.K,meta.N_att,meta.omega,meta.links,meta.masks_att_list)
                meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_a.theta,len(self.nodes_a),self.nodes_a.K,meta.links)


            ##nodes_a inclusive_meta update
            for i, meta in enumerate(na.meta_inclusives):
                meta.zeta = theta_comp_array(meta.N_att,meta.Tau,meta.omega,meta.denominators,meta.links,meta.masks_att_list)#(meta.N_att,meta.Tau,meta.omega,meta.links,meta.masks_att_list)
                meta.q_k_tau = p_kl_comp_arrays(na.K,meta.Tau,2,meta.links,meta.omega,meta.masks_label_list)
                meta.omega = omega_comp_arrays(len(self.nodes_a),len(meta),meta.q_k_tau,self.nodes_a.theta,meta.zeta,self.nodes_a.K,meta.Tau,meta.links,meta.labels)



            #nodes_b update
            nb.theta = theta_comp_arrays_multilayer(self,"b")

            ##nodes_b exclusive_meta update
            for i, meta in enumerate(nb.meta_exclusives):
                meta.qka = q_ka_comp_arrays(meta.omega,nb.K,meta.links,meta.N_att)
                meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,nb.theta,len(nb),nb.K,meta.links)

            ##nodes_b inclusive_meta update
            for i, meta in enumerate(nb.meta_inclusives):
                meta.zeta = theta_comp_array(meta.N_att,meta.Tau,meta.omega,meta.denominators,meta.links,meta.masks_att_list)#(meta.N_att,meta.Tau,meta.omega,meta.links,meta.masks_att_list)
                meta.q_k_tau = p_kl_comp_arrays(nb.K,meta.Tau,2,meta.links,meta.omega,meta.masks_label_list)
                meta.omega = omega_comp_arrays(len(self.nodes_b),len(meta),meta.q_k_tau,self.nodes_b.theta,meta.zeta,self.nodes_b.K,meta.Tau,meta.links,meta.labels)

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
            for i, meta in enumerate(layer.meta_inclusives):
                meta.log_likelihood = log_like_comp(layer.theta,meta.zeta,meta.q_k_tau,meta.links,meta.labels)
            #log-like exclusives meta
            for i, meta in enumerate(layer.meta_exclusives):
                meta.log_likelihood = log_like_comp_exclusive(layer.theta,meta.qka,meta.links)

    def get_links_probabilities(self, links = None):
        """
        Computes the links labels probabilities trained BiNet for a set of links.

        Parameters
        ----------
        links: ndarray of 1 or 2 dimensions, pandas DataFrame, default:None
            Array with the links that you want to get probabilites that are connected for each label
            -If is a 2d-array, the first column must contain the ids from nodes_a layer and the second
            column must contains the ids from nodes_b layer.
            -If it is a 1d-array, it must contain the positions of the links list from self.df attribute
            -If it is a pandas DataFrame, it must contains, at least, two columns with the names of the nodes layers.
            -If it is None, self.links_training will be used.

        Returns
        -------
        Pij_r: ndarray of shape (len(links),self.N_labels)
            Pij_r[l,r] is the probability that the link l has a label r
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


    def get_predicted_labels(self, to_return = "df", Pij = None, links = None, estimator = "max_probability"):
        """
        Computes the predicted labels of the model given the MMSBM parameters. They can be measured by different estimators:
            max_probability: The predicted label will be the most plausible label
            mean: The predicted label will be the mean

        Parameters
        ----------
        to_return: {"df","ids", "both"}, default: df
            Option to choose how the predicted labels will be returned.
             -df: A dataframe with the columns being the nodes from both layers and an extra column called predicted_+self.label_name
             -ids: A ndarray of ints with the ids of the predicted labels
             -both: It will return the df and the ndarray in this order.

        links: ndarray of 1 or 2 dimensions, pandas DataFrame, default: None
            Array with the links that you want to get probabilites that are connected for each label.
            -If it is a 2d-array, the first column must contain the ids from nodes_a layer and the second
            column must contains the ids from nodes_b layer.
            -If it is a 1d-array, it must contains the positions of the links list from self.df attribute
            -If it is a pandas dataFrame, it must contains at less two columns with the name of the nodes layer.
            -If it is None, self.links_training will be used.

        estimator: {"max_probability","average"}, default: max_probability
            Estimator used to get the predicted labels:
            -max_probability: The selected label is the most plausible label
            -mean: The selected label is mean label (sum [Pij(l)*l])

        Returns
        -------
        labels_id: ndarray
            Predicted labels id

        labels_df: pandas DataFrame
            Dataframe whose columns are nodes_a, nodes_b and the prediced labels
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
                decoder = {self.dict_codes[n]:n for n in self.dict_codes}
                labels = [decoder[n] for n in labels_id]
                links["Predicted "+self.labels_name] = labels

                if to_return == "df":
                    return links
                elif to_return == "both":
                    return links, labels_id
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

            decoder = {self.dict_codes[n]:n for n in self.dict_codes}
            labels = [decoder[n] for n in labels_id]

            if to_return == "df":
                return pd.DataFrame({str(na):A,str(nb):B,"Predicted "+self.labels_name:labels})
            elif to_return == "both":
                return pd.DataFrame({str(na):A,str(nb):B,"Predicted "+self.labels_name:labels}), labels_id

        elif to_return == "ids":
            return labels_id

    def get_accuracy(self, predicted_labels = None, test_labels = None, Pij = None,links = None, estimator = "max_probability"):
        """
        Computes the predicted labels of the model given the MMSBM parameters. They can be measured by different estimators:
            -max_probability: The predicted label will be the most plausible label
            -mean: The predicted label will be the mean

        Parameters
        ----------
        predicted_labels: array-like, default:None.
            Array-like with the predicted labels ids given by the MMSBM

        test_labels: array-like, default:None.
            List or array with the observed labels
            If it is None, labels from self.labels_array are taken given pos_test_labels

        links: ndarray of 1 or 2 dimensions, pandas DataFrame, default: None
            Array with the links that you want to get probabilites that are connected for each label.
            -If it is a 2d-array, the first column must contain the ids from nodes_a layer and the second
             column must contains the ids from nodes_b layers.
            -If it is a 1d-array, it must contains the positions of the links list from self.df attribute
            -If it is a pandas dataFrame, it must contains at less two columns with the name of the nodes layer
             and a column with the same name as the labels column from BiNet.df.
            -If it is None, self.links_training will be used.


        estimator: {"max_probability","mean"}, default: max_probability
            Estimator used to get the predicted labels:
            -max_probability: The selected label is the most plausible label
            -mean: The selected label is mean label (sum [Pij(l)*l])

        Returns
        -------
        accuracy: float
            Ratio of well predicted labels
        """

        if predicted_labels is None:
            predicted_labels = self.get_predicted_labels(to_return = "ids", links = links, Pij = Pij, estimator = estimator)

        if test_labels is None:
            if isinstance(links,pd.DataFrame):
                if links.columns.isin([str(self.labels_name)+"_id"]).any():
                    test_labels = links[str(self.labels_name)+"_id"].values
                elif links.columns.isin([str(self.labels_name)]).any():
                    test_labels = links.replace({self.labels_name:self.dict_codes}).values[:,2]

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
        # print(predicted_labels.shape,test_labels.shape,predicted_labels==test_labels,predicted_labels,test_labels)
        return (predicted_labels==test_labels).sum()/len(predicted_labels)

    def deep_copying(self):
        """
        It makes a deep copy of all the parameters from the MAP algorithm
        """
        na = self.nodes_a

        nb = self.nodes_b


        self.pkl_old = self.pkl.copy()
        self.omega_old = self.omega.copy()


        ##Metas copies
        for layer in [na,nb]:
            layer.theta_old = layer.theta.copy()
            ##inclusive_meta copies
            for i, meta in enumerate(layer.meta_inclusives):
                meta.zeta_old = meta.zeta.copy()
                meta.q_k_tau_old = meta.q_k_tau.copy()
                meta.omega_old = meta.omega.copy()

            ##exclusive_meta copies
            for i, meta in enumerate(layer.meta_exclusives):
                meta.qka_old = meta.qka.copy()
                meta.omega_old = meta.omega.copy()

    def converges(self):
        """
        Returns True if the parameters have converged or False if they haven't converged
        """
        na = self.nodes_a

        nb = self.nodes_b

        tol = self.tol


        ##Metas convergence
        for layer in [na,nb]:
            if not finished(layer.theta_old,layer.theta,tol): return False
            ##inclusive_meta convergence
            for i, meta in enumerate(layer.meta_inclusives):
                if not finished(meta.zeta_old,meta.zeta,tol): return False
                if not finished(meta.q_k_tau_old,meta.q_k_tau,tol): return False
                if not finished(meta.omega_old,meta.omega,tol): return False

            ##exclusive_meta convergence
            for i, meta in enumerate(layer.meta_exclusives):
                if not finished(meta.qka_old,meta.qka,tol): return False
                if not finished(meta.omega_old,meta.omega,tol): return False

        #links convergence
        if not finished(self.pkl_old,self.pkl,tol):return False
        if not finished(self.omega_old,self.omega,tol):return False

        return True
