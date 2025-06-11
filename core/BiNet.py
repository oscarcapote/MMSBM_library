import pandas as pd
import numpy as np
import json
from .metadata_layer import metadata_layer
from .exclusive_metadata import exclusive_metadata
from .inclusive_metadata import inclusive_metadata
from .nodes_layer import nodes_layer
from ..functions import *


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

    get_log_likelihoods()
        Returns the loglikelihood of the current state of the MMSBM.

    get_links_probabilities(links=None)
        Returns the probability of each link in links.

    get_predicted_labels(links=None)
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
        utils.save_BiNet_dict(self, dir)

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
        It computes the log_likelihoods from every bipartite network of the multipartite network, that means the log_likelihoods of the BiNet network and the log_likelihoods of the metadata networks.
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

    def add_node(self, node_id, node_name=None):
        """
        Add a node to the layer.

        Parameters
        ----------
        node_id : int
            Unique identifier for the node
        node_name : str, optional
            Name of the node, defaults to None

        Returns
        -------
        dict
            The created node dictionary

        Raises
        ------
        ValueError
            If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists!")
        self.nodes[node_id] = {"name": node_name}
        return self.nodes[node_id]

    def add_nodes_from_list(self, node_list):
        """
        Add multiple nodes from a list.

        Parameters
        ----------
        node_list : list
            List of node identifiers

        Returns
        -------
        None
        """
        for node_id in node_list:
            self.add_node(node_id)

    def add_metadata_to_node(self, node_id, meta_name, meta_value):
        """
        Add metadata to a specific node.

        Parameters
        ----------
        node_id : int
            Node identifier
        meta_name : str
            Name of the metadata column
        meta_value : any
            Value of the metadata

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If node_id or meta_name doesn't exist
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} doesn't exist!")
        if meta_name not in self.metadata:
            raise KeyError(f"Metadata {meta_name} doesn't exist!")
        self.nodes[node_id][meta_name] = meta_value

    def get_node_metadata(self, node_id, meta_name):
        """
        Get metadata value for a specific node.

        Parameters
        ----------
        node_id : int
            Node identifier
        meta_name : str
            Name of the metadata column

        Returns
        -------
        any
            The metadata value

        Raises
        ------
        KeyError
            If node_id or meta_name doesn't exist
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} doesn't exist!")
        if meta_name not in self.metadata:
            raise KeyError(f"Metadata {meta_name} doesn't exist!")
        return self.nodes[node_id].get(meta_name)

    def get_all_nodes_metadata(self, meta_name):
        """
        Get metadata values for all nodes.

        Parameters
        ----------
        meta_name : str
            Name of the metadata column

        Returns
        -------
        dict
            Dictionary mapping node IDs to their metadata values

        Raises
        ------
        KeyError
            If meta_name doesn't exist
        """
        if meta_name not in self.metadata:
            raise KeyError(f"Metadata {meta_name} doesn't exist!")
        return {node_id: node_data.get(meta_name) for node_id, node_data in self.nodes.items()}

    def remove_node(self, node_id):
        """
        Remove a node from the layer.

        Parameters
        ----------
        node_id : int
            Node identifier

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If node_id doesn't exist
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} doesn't exist!")
        del self.nodes[node_id]

    def remove_metadata(self, meta_name):
        """
        Remove a metadata column from all nodes.

        Parameters
        ----------
        meta_name : str
            Name of the metadata column

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If meta_name doesn't exist
        """
        if meta_name not in self.metadata:
            raise KeyError(f"Metadata {meta_name} doesn't exist!")
        del self.metadata[meta_name]
        for node_data in self.nodes.values():
            if meta_name in node_data:
                del node_data[meta_name]
        self.N_meta -= 1

    def clear(self):
        """
        Remove all nodes and metadata from the layer.
        """
        self.nodes.clear()
        self.metadata.clear()
        self.N_meta = 0





