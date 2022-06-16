# coding: utf-8


import pandas as pd
import numpy as np
from string import ascii_lowercase
from copy import deepcopy
# from numba import jit,prange,int64,double,vectorize,float64
from time import time
import os, sys
from MMSBM_library.functions import *


class metadata_layer:
    """
    Principal class of nodes_layer metadata. It contains extra information about the nodes.

    It has two subclasses:
        - exclusive_metadata
        - inclusive_metadata
    """
    def __init__(self, lambda_meta, meta_name):
        self.meta_name = meta_name
        self.lambda_meta = lambda_meta

    # @property
    # def N_meta(self):
    #     return self._N_meta
    #
    # @N_meta.setter
    # def N_meta(self, N_meta):
    #     self._N_meta = N_meta
    #     return self._N_meta

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
    #


class exclusive_metadata(metadata_layer):

    def __init__(self, lambda_meta, meta_name):
        """
        Initialization of the exclusive_metadata class

        Parameters
        ----------
        lambda_meta: float
            Metadata visibility
        meta_name: str
            Name of the metadata column in the node_layer class
        K: int
            Number of membership groups of this metadata
        """
        super().__init__(lambda_meta, meta_name)
        # self.qka = K
        # print("---",self.qka.shape)

    @property
    def qka(self):
        return self._qka

    @qka.setter
    def qka(self, qka):
        self._qka = qka

    def init_qka(self, K):
        # print("Hola!!!!", K)
        if K <= 0: raise ValueError("Value of K must be positive!")
        self.qka = np.random.rand(K, self.N_att)


class inclusive_metadata(metadata_layer):

    def __init__(self, lambda_meta, meta_name, Tau):
        """
        Initialization of the inclusive_metadata class

        Parameters
        ----------
        lambda_meta: float
            Metadata visibility
        meta_name: str
            Name of the metadata column in the node_layer class
        Tau: int
            Number of membership groups of this metadata
        """
        super().__init__(lambda_meta, meta_name)
        self.Tau = Tau
        # self.zeta = self.zeta(Tau)

    @property
    def zeta(self):
        return self.zeta

    @zeta.setter
    def zeta(self, Tau):
        self.zeta = zeta
        return self.zeta

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

    Is initialized using a dataframe and can be modify it  using the df_nodes attribute

    The rest of the columns of the dataframe can contain information (metadata) from the nodes.
    This metadata can be added as a metadata_layer object considering the network as multipartite network.
    This metadata can be classify it as exclusive_metadata (if a node only accepts one attribute) and inclusive_metadata (if the node accepts more than one attribute)

    See for more information of metadata: metadata_layer, exclusive_metadata and inclusive_metadata.

    These objects can be added into a BiNet (bipartite network) where connection between nodes_layer are considered to infer links and their labels  (see BiNet)
    """


    def __init__(self, K, nodes_name, nodes_info, *, separator="\t", **kwargs):
        self.K = K
        self.node_type = nodes_name

        if type(nodes_info) == type("d"):
            self.df_nodes = self.read_file(nodes_info, separator)
        elif type(nodes_info) == type(pd.DataFrame()):
            self.df_nodes = nodes_info

        codes = pd.Categorical(self.df_nodes[nodes_name]).codes
        # self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[nodes_name+"_id"]))
        # print(self.df_nodes)
        # self.df_nodes = pd.concat([self.df_nodes, pd.DataFrame({nodes_name + "_id": codes})], axis=1, ignore_index=True)
        self.df_nodes[nodes_name + "_id"] = codes
        # print(self.df_nodes)
        self.nodes_list = self.df_nodes[nodes_name].unique()


        self.meta_exclusives = []
        self.meta_inclusives = []
        self.meta_neighbours_exclusives = []
        self.meta_neighbours_inclusives = []  # all neighbours (connected and not) of inclusive metadata
        self.inclusive_linked = []  # metadata inclusive for each node
        self.nodes_observed_inclusive = []
        self.has_meta = False  #Boolean that tells you if you have metadata initialized with non 0 values of lambda

        self.N_nodes = len(codes)
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0

        self.theta = init_P_matrix(self.N_nodes,self.K)

    def read_file(self, filename, separator="\t"):
        return pd.read_csv(filename, sep=separator, engine='python')

    @classmethod
    def create_simple_layer(cls, K, nodes_list, nodes_name):
        '''
        Create a nodes_layer object from a list or DataSeries without only with the known nodes

        Parameters
        -----------
        K: Int
            Number of membership groups of nodes_layer

        nodes_list: list or DataSeries
            List or DataSeries with all the nodes

        nodes_name: str
            Name of the nodes type (users, movies, metobolites...) that are or will be in DataFrame

        '''
        if isinstance(nodes_list, list):
            new_df = pd.DataFrame({nodes_name: nodes_list})
        elif isinstance(nodes_list, pd.Series):
            new_df = pd.DataFrame(nodes_list)
        elif isinstance(nodes_list, pd.DataFrame):
            new_df = nodes_list

        return cls(K, nodes_name, new_df)

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

    def add_exclusive_metadata(self, lambda_meta, meta_name):
        '''
        Add exclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_meta: Float
            Value of the metadata visibility
        '''

        df_dropna = self.df_nodes.dropna(subset=[meta_name])
        observed = df_dropna[str(self)+"_id"].values

        if lambda_meta>1.e-16:self.has_meta = True

        # encode metadata
        codes = pd.Categorical(self.df_nodes[meta_name]).codes
        self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name + "_id"]))

        # create metadata object
        em = exclusive_metadata(lambda_meta, meta_name)
        em.links = self.df_nodes[[self.node_type + "_id", meta_name + "_id"]].values
        em.N_att = len(set(codes))
        # em.qka = em.init_qka(self.K)

        # update meta related nodes attributes
        self.meta_exclusives.append(em)
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes, dtype=np.int32)

        for n in range(self.N_nodes):
            meta_neighbours[n] = self.df_nodes[self.df_nodes[self.node_type + "_id" ]== n][meta_name + "_id"]#.values

        self.meta_neighbours_exclusives.append(meta_neighbours)

    def add_inclusive_metadata(self, lambda_meta, meta_name, Tau, separator="|"):
        '''
        Add inclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_meta: float
            Value of the metadata visibility

        Tau: Int
            Number of membership groups of metadata


        Separator: str
            Separator that is used to differenciate the differents metadata assigned for each node
        '''

        # create metadata object
        im = inclusive_metadata(lambda_meta, meta_name, Tau)
        # im.q_k_tau(self.K, Tau, 2)lambda_meta, meta_name, Tau

        # links and neighbours
        df_dropna = self.df_nodes.dropna(subset=[meta_name])
        meta_list = self.df_nodes[meta_name].values

        observed = df_dropna[self.node_type].values  # Nodes with known metadata
        observed_id = df_dropna[self.node_type + "_id"].values  # Nodes with known metadata


        if lambda_meta>1.e-16:self.has_meta = True
        # encode metadata
        meta_neighbours = []#[[int(j) for j in i.split(separator)] for i in df_dropna[meta_name].values]#meta connected with 1

        for i in meta_list:
            if i == None or i == np.NaN or i == pd.NaT:
                meta_neighbours.append(None)
            else:
                meta_neighbours.append([j for j in i.split(separator)])

        codes = {}

        for l in meta_neighbours:
            if l == None: continue
            for m in l:
                codes[m] = codes.get(m, len(codes))

        decodes = {codes[i]:i for i in codes}

        im.codes = codes
        im.decodes = decodes
        im.N_att = len(set(codes))

        # Links between node and metadata type
        links = np.ones((len(observed) * im.N_att, 2))
        # Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed) * im.N_att)

        index = 0
        for i, o in enumerate(observed):
            for a in range(im.N_att):
                links[index, 0] = o
                links[index, 1] = a
                if decodes[a] in meta_neighbours[i]:
                    labels[index] = 1

                index += 1

        im.links = links
        im.labels = labels

        # codes = pd.Categorical(self.df_nodes[meta_name]).codes
        # self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name+"_id"]))
        # self.inclusive_linked.append([[int(j) for j in i.split(separator)] for i in df_dropna [meta_name+"_id"].values])

        self.nodes_observed_inclusive.append(observed)

        # update meta related nodes attributes
        self.meta_inclusives.append(im)
        self.N_meta_inclusive += 1
        self.N_meta += 1

        self.meta_neighbours_inclusives.append([[codes[m] for m in L] for L in meta_neighbours])


class BiNet:
    """
    Class of a Bipartite Network, where two layers of different types of nodes are connected (users->items, politicians->bills, patient->microbiome...) and these links can be labeled with informations of the interaction (ratings, votes...)


    """
    def __init__(self, links, links_label,*, nodes_a = None, nodes_b = None, Ka=1, nodes_a_name="nodes_a", Kb=1,
                 nodes_b_name="nodes_b", separator="\t"):
        """
         Initialization of a BiNet class

         Parameters
         -----------
         links: str, DataFrame
            DataFrame or directory where the dataframe is. It should contains the links list between nodes_a and nodes_b and their labels.

         links_label: str
            Name of the links column where the labels are

         nodes_a: nodes_layer, str, None
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type a are.
             If None, it a simple nodes_layer will be created from the information from links.

         nodes_b: nodes_layer, str, None
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type b are.
             If None, it a simple nodes_layer will be created from the information from links.

         Ka: int
            Number of membership groups from nodes_a layer

         Kb: int
            Number of membership groups from nodes_b layer

         nodes_a_name: str
            Name of the column where the names of nodes_a are in the links DataFrame and nodes_a DataFrame

         nodes_b_name: str
            Name of the column where the names of nodes_b are in the links DataFrame and nodes_b DataFrame

         separator: str
            Separator of the links DataFrame. Default is \t
        """
        if type(links) == type(pd.DataFrame()):
            self.links_df = links
        elif isinstance(links, str):
            self.links_df = pd.read_csv(links, sep=separator, engine='python')


        # creating first layer class
        if isinstance(nodes_a, nodes_layer):
            self.nodes_a = nodes_a
            nodes_a_name = str(self.nodes_a)
            Ka = nodes_a.K
        elif isinstance(nodes_a, str):
            self.nodes_a = nodes_layer(Ka, nodes_a_name, nodes_a)
        elif  nodes_a == None:
            self.nodes_a = nodes_layer.create_simple_layer(Ka, self.links_df[nodes_a_name], nodes_a_name)

        # creating second layer class
        if isinstance(nodes_b, nodes_layer):
            self.nodes_b = nodes_b
            nodes_b_name = str(self.nodes_b)
            Kb = nodes_b.K
        elif isinstance(nodes_b, str):
            self.nodes_b = nodes_layer(Kb, nodes_b_name, nodes_b)
        elif nodes_b == None:
            self.nodes_b = nodes_layer.create_simple_layer(Kb, self.links_df[nodes_b_name], nodes_b_name)


        ## Coding labels
        codes = pd.Categorical(self.links_df[links_label]).codes
        self.links_df = self.links_df.join(pd.DataFrame(codes, columns=[links_label + "_id"]))
        self.labels_array = self.links_df[links_label + "_id"].values

        print("labels y arrays:",len(self.labels_array),len(self.links_df),len(codes))



        #Links
        self.links_df = self.links_df.join(self.nodes_a.df_nodes[[nodes_a_name,nodes_a_name + "_id"]].set_index(nodes_a_name),on=nodes_a_name)
        self.links_df = self.links_df.join(self.nodes_b.df_nodes[[nodes_b_name,nodes_b_name + "_id"]].set_index(nodes_b_name),on=nodes_b_name)

        self.links_df = self.links_df.drop_duplicates()
        self.links = self.links_df[[nodes_a_name + "_id", nodes_b_name + "_id"]].values

        print("labels y arrays2:",len(self.labels_array),len(self.links_df),len(codes))

        #observed nodes in each layer
        self.observed_nodes_a = np.unique(self.links[:,0])
        self.observed_nodes_b = np.unique(self.links[:,1])

        self.N_ratings = max(self.labels_array)+1

    def init_MAP(self, seed=None):
        '''
        Initialize the MAP algorithm to get the most plausible memberhip parameters of the MMSBM

        Parameters
        -----------
        seed: int
            Seed to generate the matrices. Is initialized using the np.random.RandomState(seed) method.

        '''
        # Probability matrices
        # np.random.RandomState(seed)

        #BiNet matrices
        print((self.nodes_a.K, self.nodes_b.K, self.N_ratings))
        self.pkl = init_P_matrix(self.nodes_a.K, self.nodes_b.K, self.N_ratings)
        # print("aqui1",self.pkl.shape)
        self.omega = omega_comp_arrays(len(self.nodes_a),len(self.nodes_b),self.pkl,self.nodes_a.theta,self.nodes_b.theta,self.nodes_a.K,self.nodes_b.K,self.links,self.labels_array)

        # print("aqui1")
        #Metadata
        ## qka and omegas
        for meta in self.nodes_a.meta_exclusives:
            meta.qka = init_P_matrix(self.nodes_a.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_a.theta,len(self.nodes_a),self.nodes_a.K,meta.links)

        for meta in self.nodes_b.meta_exclusives:
            meta.qka = init_P_matrix(self.nodes_b.K, meta.N_att)
            meta.omega = omega_comp_arrays_exclusive(meta.qka,meta.N_att,self.nodes_b.theta,len(self.nodes_b),self.nodes_b.K,meta.links)

        # print("aqui2")
        ## ql_tau and omegas omega_comp_arrays(omega,p_kl,theta,eta,K,L,links_array,links_ratings):
        for meta in self.nodes_a.meta_inclusives:
            meta.q_k_tau = init_P_matrix(self.nodes_a.K, meta.Tau, 2)
            meta.omega = omega_comp_arrays(len(self.nodes_a),len(meta),meta.q_k_tau,self.nodes_a.theta,meta.zeta,self.nodes_a.K,meta.Tau,meta.links,meta.labels_array)

        for meta in self.nodes_b.meta_inclusives:
            meta.q_k_tau = init_P_matrix(self.nodes_b.K, meta.Tau, 2)
            meta.omega = omega_comp_arrays(len(self.nodes_b),len(meta),meta.q_k_tau,self.nodes_b.theta,meta.zeta,self.nodes_b.K,meta.Tau,meta.links,meta.labels_array)

        #omega amd equivalents from inclusive metadata
        #self.omega = np.array((len(self.nodes_a), len(self.nodes_b), self.nodes_a.Ka, self.nodes_b.Kb))


#         for meta in self.nodes_a.meta_inclusives:
#             meta.omega = np.array((len(meta.nodes_a), len(self.nodes_a), meta.Tau, self.nodes_a.Ka))

#         for meta in self.nodes_b.meta_inclusives:
#             meta.omega = np.array((len(meta.nodes_b), len(self.nodes_b), meta.Tau, self.nodes_b.Kb))

        #creating arrays with the denominator (that are constants) of each node in both layers and em layers

        ##nodes a
        self.nodes_a.denominators = np.zeros(len(self.nodes_a))

        self.neighbours_nodes_a = [] #list of list of neighbours
        for node in range(len(self.nodes_a)):
            #neighbours in BiNet
            self.neighbours_nodes_a.append(self.links_df[self.links_df[str(self.nodes_a)+"_id"] == node][str(self.nodes_b)+"_id"].values)
            self.nodes_a.denominators[node] += len(self.neighbours_nodes_a[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(self.nodes_a.meta_exclusives):
            for node in meta.links[:,0]:
                self.nodes_a.denominators[node] += meta.lambda_meta

        #neighbours in meta inclusives
        for node in range(len(self.nodes_a)):
            for i, meta in enumerate(self.nodes_a.meta_inclusives):
                self.nodes_a.denominators[node] += meta.lambda_meta*len(self.links[self.links[:,0]==node,:])
            #for i, meta in enumerate(self.nodes_a.meta_exclusives):
                #self.node_a.denominators[node] += meta.lambda_metas*

        ##nodes b
        self.nodes_b.denominators = np.zeros(len(self.nodes_b))

        self.neighbours_nodes_b = [] #list of list of neighbours
        for node in range(len(self.nodes_b)):
            #neighbours in BiNet
            self.neighbours_nodes_b.append(self.links_df[self.links_df[str(self.nodes_b)+"_id"] == node][str(self.nodes_a)+"_id"].values)
            self.nodes_b.denominators[node] += len(self.neighbours_nodes_b[-1])

        #neighbours in meta exclusives
        for i, meta in enumerate(self.nodes_b.meta_exclusives):
            for node in meta.links[:,0]:
                self.nodes_b.denominators[node] += meta.lambda_meta

        #neighbours in meta inclusives
        for node in range(len(self.nodes_b)):
            for i, meta in enumerate(self.nodes_b.meta_inclusives):
                self.nodes_b.denominators[node] += meta.lambda_meta*len(self.links[self.links[:,0]==node,:])

            #neighbours in meta inclusives
            # for meta in self.nodes_b.meta_exclusives:


        def MAP_step(N_steps=1):
            """

            Parameters
            ----------
            N_steps: int
                Number of MAP steps that will be performed
            """
            for step in range(N_steps):
                print(step)
