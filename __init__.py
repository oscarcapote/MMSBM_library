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
    def N_meta(self, N_meta):
        self.N_meta = N_meta
        return self.N_meta

    @property
    def N_att(self, N_att):
        """
        Number of different categorical attributes of the metadata

        Parameters
        -----------
        N_att: Int
            Number of different categorical attributes of the metadata

        """
        self.N_att = N_att
        return self.N_att

    #@property
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

        self.links = links
        self.N_links = len(links)
    #
    # def add_links(self, links):
    #     self.links(links)
    #     self.N_links = len(self.links)
class exclusive_metadata(metadata_layer):


    @property
    def qka(self, K):
        self.qka = np.random.rand(K,self.N_meta)


class inclusive_metadata(metadata_layer):


    def __init__(self, Tau):
        self.Tau = Tau


    @property
    def zeta(self, K):
        self.zeta = np.random.rand(self.N_att, Tau)

    @property
    def q_k_tau(self, K, Tau):
        self.q_k_tau = np.random.rand(K,self.Tau,self.N_att)





class nodes_layer:
    def __init__(self, L,  node_name, nodes_info, *, separator="\t", **kwargs):
        self.L = L
        self.node_type = node_name

        if type(nodes_info)==type("d"):
            self.df_nodes = self.read_file(nodes_info, separator)
        elif type(nodes_info)==type(pd.DataFrame()):
            self.df_nodes = nodes_info
        codes = pd.Categorical(self.df_nodes[node_name]).codes
        self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[node_name+"_id"]))
        self.nodes_list = self.df_nodes[node_name].unique()

        self.meta_exclusives = []
        self.meta_inclusives = []
        self.meta_neighbours_exclusives = []
        self.meta_neighbours_inclusives = [] #all neighbours (connected and not) of inclusive metadata
        self.inclusive_linked = [] #metadata inclusive for each node
        self.nodes_observed_inclusive = []


        self.N_nodes = np.max(self.df_nodes[node_name+"_id"].max()) + 1
        self.N_meta_exclusive = 0
        self.N_meta_inclusive = 0
        self.N_meta = 0

        self.theta = np.random.rand(self.N_nodes,L)

    def read_file(self, filename, separator="\t"):
        return pd.read_csv(filename,sep=separator, engine='python')

    def update_N(self, N_nodes):
        '''
        Update the number of nodes and reinitialize the membership matrix

        Parameters
        -----------
        N_nodes: Int
            Number of nodes
        '''
        self.N_nodes = N_nodes
        self.theta = np.random.rand(N_nodes, self.L)

    def add_exclusive_metadata(self, meta_name, lambda_meta):
        '''
        Add exclusive_metadata object to node_layer object

        Parameters
        -----------
        meta_name: Str
            Name of the metadata that should be in the node dataframe

        lambda_meta: Float
            Value of the metadata visibility
        '''
        #encode metadata
        codes = pd.Categorical(self.df_nodes[meta_name]).codes
        self.df_nodes = self.df_nodes.join(pd.DataFrame(codes, columns=[meta_name+"_id"]))

        #create metadata object
        em = exclusive_metadata(meta_name, lambda_meta)
        em.links(self.df_nodes[[self.node_type,meta_name]].values)
        em.qka(self.K)
        em.N_att(len(set(codes)))

        #update meta related nodes attributes
        self.meta_exclusives.append(em)
        self.N_meta_exclusive += 1
        self.N_meta += 1

        meta_neighbours = np.ones(self.N_nodes,dtype=np.int32)

        for n in range(self.N_nodes):
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
    def __init__(self,nodes_a,Ka,nodes_b,Kb,link):
        self.nodes_a = nodes_layer(Ka,nodes_a)
        self.nodes_b = nodes_layer(Kb,nodes_b)
        self.links = links
