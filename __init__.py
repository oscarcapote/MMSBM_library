# coding: utf-8


import pandas as pd
import numpy as np
from string import ascii_lowercase
from copy import deepcopy
# from numba import jit,prange,int64,double,vectorize,float64
from time import time
import os, sys
import time as time_lib
import argparse
import yaml


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
    # def add_links(self, links):
    #     self.links(links)
    #     self.N_links = len(self.links)


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
        return self._zeta

    @zeta.setter
    def zeta(self, Tau):
        if Tau <= 0: raise ValueError("Value of Tau must be positive!")
        self.zeta = np.random.rand(self.N_att, Tau)
        return self._zeta

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

        self.N_nodes = len(codes)
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0

        self.theta = np.random.rand(self.N_nodes, K)

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
        elif isinstance(nodes_list, pd.DataFrame):
            new_df = nodes_list

        return cls(K, nodes_name, new_df)

    def update_N(self, N_nodes):
        '''
        Update the number of nodes and reinitialize the membership matrix

        Parameters
        -----------
        N_nodes: Int
            Number of nodes
        '''
        self.N_nodes = N_nodes
        self.theta = np.random.rand(N_nodes, self.K)

    def update_K(self, K):
        '''
        Update the number of membership groups of nodes_layer and reinitialize the membership matrix

        Parameters
        -----------
        K: Int
            Number of membership groups of nodes_layer
        '''
        self.K = K
        self.theta = np.random.rand(self.N_nodes, K)

    def __len__(self):
        return self.N_nodes

    def add_exclusive_metadata(self, meta_name, lambda_meta):
        '''
        Add exclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: str
            Name of the metadata that should be in the node dataframe

        lambda_meta: Float
            Value of the metadata visibility
        '''
        # encode metadata
        codes = pd.Categorical(self.df_nodes[meta_name]).codes
        self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name + "_id"]))

        # create metadata object
        em = exclusive_metadata(meta_name, lambda_meta)
        em.links = self.df_nodes[[self.node_type, meta_name]].values
        em.N_att = len(set(codes))
        em.qka = em.init_qka(self.K)

        # update meta related nodes attributes
        self.meta_exclusives.append(em)
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes, dtype=np.int32)

        for n in range(self.N_nodes):
            meta_neighbours[n] = self.df_nodes[self.df_nodes[self.node_type + "_id" ]== n][meta_name + "_id"]

        self.meta_neighbours_exclusives.append(meta_neighbours)

    def add_inclusive_metadata(self, meta_name, lambda_meta, Tau, separator="|"):
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
        im = inclusive_metadata(meta_name, lambda_meta, Tau)
        # im.q_k_tau(self.K, Tau, 2)

        # links and neighbours
        df_dropna = self.df_nodes.dropna(subset=meta_name)
        observed = df_dropna[self.node_type + "_id"].values  # Nodes with known metadata

        # encode metadata
        meta_neighbours = [[int(j) for j in i.split(separator)] for i in df_dropna[meta_name].values]
        codes = {}

        for l in range(len(meta_neighbours)):
            for m in range(len(l)):
                codes[m] = codes[m].get(len(codes), m)

        im.N_att(len(set(codes)))

        # Links between node and metadata type
        links = np.ones((len(observed) * im.N_att, 2))
        # Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed) * im.N_att)

        index = 0
        for i, o in enumerate(observed):
            for a in range(im.N_att):
                links[index, 0] = o
                links[index, 1] = a

                if a in meta_neighbours[i]:
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

        self.meta_neighbours_inclusives.append(meta_neighbours)


class BiNet:
    """
    Class of a Bipartite Network, where two layers of different types of nodes are connected (users->items, politicians->bills, patient->microbiome...) and these links can be labeled with informations of the interaction (ratings, votes...)


    """
    def __init__(self, nodes_a, nodes_b, links, links_label,*, Ka=1, nodes_a_name="nodes_a", Kb=1,
                 nodes_b_name="nodes_b", separator="\t"):
         """
         Initialization of a BiNet class

         Parameters
         -----------
         nodes_a: nodes_layer, str
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type a are.

         nodes_b:
             One of the nodes layer that forms the bipartite network
             If it is a string, it should contain the directory where the information of the nodes of type b are.

         links: str, DataFrame
            DataFrame or directory where the dataframe is. It should contains the links list between nodes_a and nodes_b and their labels.

         links_label: str
            Name of the links column where the labels are

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
            self.links = links
        elif isinstance(links, str):
            self.links = pd.read_csv(links, sep=separator, engine='python')


        # creating first layer class
        if isinstance(nodes_a, nodes_layer):
            self.nodes_a = nodes_a
        elif isinstance(nodes_a, str):
            self.nodes_a = nodes_layer.create_simple_layer(Ka, links[nodes_a], nodes_a)

        # creating second layer class
        if isinstance(nodes_b, nodes_layer):
            self.nodes_b = nodes_b
        elif isinstance(nodes_a, str):
            self.nodes_b = nodes_layer.create_simple_layer(Kb, links[nodes_b], nodes_b)


        ## Coding labels
        self.ratings_array = self.links[links_label].values
        codes = pd.Categorical(self.links[links_label]).codes
        self.links = self.links.join(pd.DataFrame(codes, columns=[links_label + "_id"]))

        #Links
        self.links = self.links.join(self.nodes_a[[nodes_a_name,nodes_a_name + "_id"]].set_index(nodes_a_name),on=nodes_a_name)
        self.links = self.links.join(self.nodes_b[[nodes_b_name,nodes_b_name + "_id"]].set_index(nodes_b_name),on=nodes_b_name)
        self.links_array = self.links[[nodes_a_name + "_id", nodes_b_name + "_id"]].values



        self.N_ratings = max(self.ratings_array)

    def init_MAP(self, seed=None):
        '''
        Initialize the MAP algorithm to get the most plausible memberhip parameters of the MMSBM

        Parameters
        -----------
        seed: int
            Seed to generate the matrices. Is initialized using the np.random.RandomState(seed) method.

        '''
        # Probability matrices
        np.random.RandomState(seed)
        self.pkl = np.array((self.nodes_a.Ka, self.nodes_b.Kb, self.N_ratings))

        ## qka
        for meta in self.nodes_a.meta_exclusives:
            meta.qka = np.array((self.nodes_a.Ka, meta.N_att))

        for meta in self.nodes_b.meta_exclusives:
            meta.qka = np.array((self.nodes_b.Kb, meta.N_att))

        ## ql_tau
        for meta in self.nodes_a.meta_inclusives:
            meta.q_k_tau = np.array((self.nodes_a.Ka, meta.Tau, 2))

        for meta in self.nodes_a.meta_inclusives:
            meta.q_k_tau = np.array((self.nodes_b.Kb, meta.Tau, 2))

        #omega amd equivalents from inclusive metadata
        self.omega = np.array((len(self.nodes_a), len(self.nodes_b), self.nodes_a.Ka, self.nodes_b.Kb))


        for meta in self.nodes_a.meta_inclusives:
            meta.omega = np.array((len(meta.nodes_a), len(self.nodes_a), meta.Tau, self.nodes_a.Ka))

        for meta in self.nodes_b.meta_inclusives:
            meta.omega = np.array((len(meta.nodes_b), len(self.nodes_b), meta.Tau, self.nodes_b.Kb))

        #creating arrays with the denominator (that are constants) of each node in both layers and em layers
        self.node_a.denominators = np.zeros(len(self.nodes_a))



        def MAP_step(N_steps=1):
            """

            Parameters
            ----------
            N_steps: int
                Number of MAP steps that will be performed
            """
            for step in range(N_steps):
                print(step)
