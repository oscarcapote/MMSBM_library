# coding: utf-8


import pandas as pd
import numpy as np
from string import ascii_lowercase
from copy import deepcopy
#from numba import jit,prange,int64,double,vectorize,float64
from time import time
import os,sys
import time as time_lib
import argparse
import yaml

class metadata_layer:
    def __init__(self, lambda_meta, meta_name):
        self.meta_name = meta_name
        self.lambda_meta = lambda_meta

    @property
    def N_meta(self):
        return self._N_meta

    @N_meta.setter
    def N_meta(self, N_meta):
        self._N_meta = N_meta
        return self._N_meta

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
    #
    # def add_links(self, links):
    #     self.links(links)
    #     self.N_links = len(self.links)
class exclusive_metadata(metadata_layer):

    def __init__(self,lambda_meta,meta_name,K):
        super().__init__(lambda_meta,meta_name)
        #self.qka = K
        #print("---",self.qka.shape)

    @property
    def qka(self):
        return self._qka

    @qka.setter
    def qka(self, K):
        print("Hola!!!!",K)
        if K<=0:raise ValueError("Value of K must be positive!")
        self._qka = np.random.rand(K,self.N_meta)


class inclusive_metadata(metadata_layer):


    def __init__(self, Tau):
        super().__init__(lambda_meta,meta_name)
        self.Tau = Tau
        # self.zeta = self.zeta(Tau)

    @property
    def zeta(self):
        return self._zeta
    @zeta.setter
    def zeta(self, Tau):
        if Tau<=0:raise ValueError("Value of Tau must be positive!")
        self.zeta = np.random.rand(self.N_att, Tau)
        return self._zeta

    @property
    def q_k_tau(self):
        return self._q_k_tau

    @q_k_tau.setter
    def q_k_tau(self, K, Tau):
        if K<=0:raise ValueError("Value of K must be positive!")
        if Tau<=0:raise ValueError("Value of Tau must be positive!")
        self._q_k_tau = np.random.rand(K,self.Tau,self.N_att)
        return self._q_k_tau





class nodes_layer:
    def __init__(self, K,  nodes_name, nodes_info, *, separator="\t", **kwargs):
        self.K = K
        self.node_type = nodes_name

        if type(nodes_info)==type("d"):
            self.df_nodes = self.read_file(nodes_info, separator)
        elif type(nodes_info)==type(pd.DataFrame()):
            self.df_nodes = nodes_info

        codes = pd.Categorical(self.df_nodes[nodes_name]).codes
        # self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[nodes_name+"_id"]))
        print(self.df_nodes)
        self.df_nodes = pd.concat([self.df_nodes,pd.DataFrame({nodes_name+"_id":codes})], axis=1, ignore_index=True)
        self.nodes_list = self.df_nodes[nodes_name].unique()

        self.meta_exclusives = []
        self.meta_inclusives = []
        self.meta_neighbours_exclusives = []
        self.meta_neighbours_inclusives = [] #all neighbours (connected and not) of inclusive metadata
        self.inclusive_linked = [] #metadata inclusive for each node
        self.nodes_observed_inclusive = []


        self.N_nodes = len(codes)
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0

        self.theta = np.random.rand(self.N_nodes,K)

    def read_file(self, filename, separator="\t"):
        return pd.read_csv(filename,sep=separator, engine='python')


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
        if isinstance(nodes_list, List):
            new_df = pd.DataFrame({nodes_name:nodes_list})
        elif isinstance(nodes_list, pd.DataFrame):
            new_df = nodes_list

        return cls(K,  nodes_name, new_df)




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
        #encode metadata
        codes = pd.Categorical(self.df_nodes[meta_name]).codes
        self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name+"_id"]))

        #create metadata object
        em = exclusive_metadata(meta_name, lambda_meta, self.K)
        print("---",self.node_type)
        em.links = self.df_nodes[[self.node_type,meta_name]].values
        em.N_meta = len(codes)
        em.qka = self.K
        em.N_att = len(set(codes))

        #update meta related nodes attributes
        self.meta_exclusives.append(em)
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes,dtype=np.int32)

        for n in range(self.N_nodes):
            print(n)
            meta_neighbours[n] = self.df_nodes[[self.node_type+"_id" == n]][meta_name+"_id"]

        self.meta_neighbours_exclusives.append(meta_neighbours)


    def add_inclusive_metadata(self, meta_name, lambda_meta, Tau, separator="|"):
        '''
        Add inclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: Str
            Name of the metadata that should be in the node dataframe

        lambda_meta: Float
            Value of the metadata visibility

        Tau: Int
            Number of membership groups of metadata


        Separator: str
            Separator that is used to differenciate the differents metadata assigned for each node
        '''

        #create metadata object
        im = inclusive_metadata(meta_name, lambda_meta, Tau)
        im.q_k_tau(self.K,Tau,2)

        #links and neighbours
        df_dropna = df.dropna(subset=meta_name)
        observed = df_dropna[self.node_type+"_id"].values #Nodes with known metadata

        #encode metadata
        meta_neighbours = [[int(j) for j in i.split(separator)] for i in df_dropna [meta_name].values]
        codes = {}

        for l in range(len(meta_neighbours)):
            for m in range(len(l)):
                codes[m] = codes[m].get(len(codes),m)

        im.N_att(len(set(codes)))

        #Links between node and metadata type
        links = np.ones((len(observed)*im.N_att,2))
        #Label of the link: 0 if not connected 1 if connected
        labels = np.zeros(len(observed)*im.N_att)

        index = 0
        for i,o in enumerated(observed):
            for a in range(N_att):
                links[index,0] = o
                links[index,1] = a

                if a in meta_neighbours[i]:
                    labels[index] = 1

                index += 1

        im.links = links
        im.labels = labels

        #codes = pd.Categorical(self.df_nodes[meta_name]).codes
        #self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name+"_id"]))
        # self.inclusive_linked.append([[int(j) for j in i.split(separator)] for i in df_dropna [meta_name+"_id"].values])

        self.nodes_observed_inclusive.append(observed)


        #update meta related nodes attributes
        self.meta_inclusives.append(im)
        self.N_meta_inclusive += 1
        self.N_meta += 1


        self.meta_neighbours_inclusives.append(meta_neighbours)




class BiNet:
    def __init__(self,nodes_a,nodes_b,links, links_name,*,Ka=1, nodes_a_name="nodes_a",Kb=1, nodes_b_name="nodes_b"):
        if type(links)==type(pd.DataFrame()):
            self.links = links
        elif isinstance(links, str):
            self.links = pd.read_csv(filename,sep=separator, engine='python')

        self.links_list = self.links[links_name].values

        #creating first layer class
        if isinstance(nodes_a, nodes_layer):
            self.nodes_a = nodes_a
        elif isinstance(nodes_a, str):
            self.nodes_a = create_simple_layer(Ka, links[nodes_a], nodes_a)

        #creating second layer class
        if isinstance(nodes_b, nodes_layer):
            self.nodes_b = nodes_b
        elif isinstance(nodes_a, str):
            self.nodes_b = create_simple_layer(Kb, links[nodes_b], nodes_b)
