import numpy as np
import pandas as pd
import json
import os,sys
# from numba import jit

def init_P_matrix(*shape):
    """
    Generates a normalized probability matrix

    Parameters:
    -----------
    shape: int or tuple of ints
        Dimension of the probability matrix

    Returns:
    -------
    A normalized probability matrix that is normalized along the last axis
    """
    P = np.random.rand(*shape)
    S = P.sum(axis = len(shape)-1)
    return P/S.reshape((*shape[:-1],-1))




def add_codes(layer,col_name):
    """
    Adds to the df property a column with an id of the value of the col_name column

    Parameters
    -----------
    layer: node_layer, BiNet
        Objects with a df attribute

    col_name: str
        Name of the column in the df attribute that an integer id will be assigned to every different value of the column.
        This id is an integer that goes from 0 to N-1, where N is the number of different values in the col_name column.

    Return
    -----------
    dict_codes: dict
        Dictionary where each key is a different value of the col_name column and the value is the id integer that was assigned.

    """
    dict_codes = {}
    c = pd.Categorical(layer.df[col_name])#.codes
    c2 = pd.Categorical(layer.df[col_name]).codes
    for i,att in enumerate(c):
    #     print(i)
        dict_codes[layer.df[col_name].iloc[i]] = c2[i]
    # layer.df = layer.df.join(pd.DataFrame(c2, columns=[col_name + "_id"]))
    layer.df[col_name + "_id"] = c2
    return dict_codes



def finished(A,A_old,tol):
    """
    Returns True if A and A_old are simmilar by looking if the mean absolute difference between A and A_old is lower than a
    tolerance tol
    """
    finished = False
    if(np.mean(np.abs(A-A_old))<tol):
        finished = True
    return finished


def save_MMSBM_parameters(BiNet,dir=".",matrix_format="npy",BiNet_json=False):
    """
    It saves the parameters into matrices in the dir directory

    Parameters:
    -----------
    BiNet: BiNet object
        Bipartite network with the MMSBM initialized

    dir: str, default: "."
        Directory where the files with the MMSBM parameters will be saved

    matrix_format: str, default: npy
        Format that the matrices parameters will be saved. It can be, npy or npz.

    BiNet_json: boolean, default: False
        If it is True, the information of the BiNet class will be saved into a json.

    """
    na = BiNet.nodes_a

    nb = BiNet.nodes_b

    if matrix_format == "npy":
        save_func = np.save
    if matrix_format == "npz":
        save_func = np.savez
    # else:
    #     save_func = np.savetxt


    save_func(dir+"/pkl."+matrix_format,BiNet.pkl)
    # save_func(dir+"/omega."+matrix_format,BiNet.omega)


    ##Metas saves
    for layer,str_layer in [(na,"a"),(nb,"b")]:
        save_func(dir+"/theta_{}.".format(str_layer)+matrix_format,layer.theta)
        ##inclusive_meta saves
        for i, meta in enumerate(layer.meta_inclusives.values()):
            save_func(dir+"/zeta_{}.".format(str(meta))+matrix_format,meta.zeta)
            save_func(dir+"/q_k_tau_{}.".format(str(meta))+matrix_format,meta.q_k_tau)
            # save_func(dir+"/omega_{}_in_{}.".format(str_layer,str(meta))+matrix_format,meta.omega)

        ##exclusive_meta saves
        for i, meta in enumerate(layer.meta_exclusives.values()):
            save_func(dir+"/qka_{}.".format(str(meta))+matrix_format,meta.qka)
            # save_func(dir+"/omega_{}_ex_{}.".format(str_layer,str(meta))+matrix_format,meta.omega)

    #BiNet json
    if BiNet_json:
        save_BiNet_dict(BiNet,dir=dir)

def save_nodes_layer_dict(layer,dir="."):
    """
    It saves the some information, including the dict_codes from each layer, into a json called layer_data.json

    Parameters:
    -----------
    layer: node_layer
        Bipartite network object

    dir: str
        Directory where the files with the MMSBM parameters will be saved

    """
    dict_info = {}
    dict_info["dict_codes"] ={str(k):str(v) for k,v in layer.dict_codes.items()}
    dict_info["nodes_name"] = str(layer)
    dict_info["N_nodes"] = len(layer)
    dict_info["N_metas"] = layer.N_meta
    dict_info["K"] = layer.K
    dict_info["N_metas_exclusives"] = str(layer.N_meta_exclusive)
    dict_info["N_metas_inclusives"] = str(layer.N_meta_inclusive)
    dict_info["metadata_exclusives"] = []
    for i, meta in enumerate(layer.meta_exclusives.values()):
        dict_info["metadata_exclusives"].append({
                                        "meta_name":str(meta),
                                        "lambda_val":meta.lambda_val,
                                        "N_atts":len(meta),
                                        # "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
                                        "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
                                        })
    dict_info["metadata_inclusives"] = []
    for i, meta in enumerate(layer.meta_inclusives.values()):
        dict_info["metadata_inclusives"].append({
                                        "meta_name":str(meta),
                                        "lambda_val":meta.lambda_val,
                                        "Tau":meta.Tau,
                                        "N_atts":len(meta),
                                        "separator":meta._separator,
                                        # "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
                                        "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
                                        })
    with open(dir+f'/layer_{str(layer)}_data.json', 'w') as outfile:
        json.dump(dict_info, outfile)

def save_BiNet_dict(BiNet,dir="."):
    """
    It saves the some information, including the dict_codes from each layer, into a json called BiNet_data.json

    Parameters:
    -----------
    BiNet: BiNet object
        Bipartite network object

    dir: str
        Directory where the files with the MMSBM parameters will be saved

    """
    na = BiNet.nodes_a

    nb = BiNet.nodes_b
    #other values from EM and MMSBM saves
    dict_info = {}
    dict_info["dict_codes"] ={str(k):str(v) for k,v in BiNet.dict_codes.items()}
    dict_info["links_label"] = BiNet.labels_name
    dict_info["nodes_a_name"] = str(na)
    dict_info["nodes_b_name"] = str(nb)
    dict_info["Ka"] = na.K
    dict_info["dict_codes_a"] = {str(k): str(v) for k, v in na.dict_codes.items()}
    dict_info["Kb"] = nb.K
    dict_info["separator"] = BiNet._separator
    dict_info["dict_codes_b"] = {str(k): str(v) for k, v in nb.dict_codes.items()}


    # for layer,str_layer in [(na,"a"),(nb,"b")]:
    #     layer_label = "layer "+str_layer
    #     dict_info[layer_label] = {"name":str(layer),
    #                              "N_nodes":len(layer),
    #                              "N_metas":layer.N_meta,
    #                              "K":layer.K,
    #                              "N_metas_exclusives":str(layer.N_meta_exclusive),
    #                              "N_metas_inclusives":str(layer.N_meta_inclusive),
    #                              "dict_codes":{str(k):str(v) for k,v in layer.dict_codes.items()}
    #                               }
    #     ##exclusive_meta saves
    #     dict_info[layer_label]["metadata_exclusives"] = []
    #     for i, meta in enumerate(layer.meta_exclusives.values()):
    #         dict_info[layer_label]["metadata_exclusives"].append({
    #                                                         "Meta_name":str(meta),
    #                                                         "lambda":meta.lambda_val,
    #                                                         "N_atts":len(meta),
    #                                                         # "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
    #                                                         "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
    #                                                         })
    #     ##inclusive_meta saves
    #     dict_info[layer_label]["metadata_inclusives"] = []
    #     for i, meta in enumerate(layer.meta_inclusives.values()):
    #         dict_info[layer_label]["metadata_inclusives"].append({
    #                                                         "Meta_name":str(meta),
    #                                                         "lambda":meta.lambda_val,
    #                                                         "Tau":meta.Tau,
    #                                                         "N_atts":len(meta),
    #                                                         # "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
    #                                                         "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
    #                                                         })
    BiNet.info = dict_info
    with open(dir+'/BiNet_data.json', 'w') as outfile:
        json.dump(dict_info, outfile)


def load_EM_parameters(BiNet,directory="."):
    """
    It loads the parameters from matrices in the directory

    Parameters:
    -----------
    BiNet: BiNet object
        Bipartite network with the MMSBM initialized

    directory: str, default: "."
        Directory where the files with the MMSBM parameters will be loaded

    """
    na = BiNet.nodes_a

    nb = BiNet.nodes_b

    if directory[-1] != "/":
        directory += "/"

    #format
    for f in ["npy","npz","txt","dat"]:
        if os.path.isfile(directory+"pkl."+f) or os.path.isfile(directory+"pkl."+f):
            matrix_format = f
            break

    BiNet.pkl = np.load(directory+"pkl." + matrix_format)
    # BiNet.omega = np.load(directory+"omega." + matrix_format)



    ##Metas saves
    for layer,str_layer in [(na,"a"),(nb,"b")]:
        layer.theta = np.load(directory+"theta_{}.".format(str_layer)+matrix_format)
        ##inclusive_meta saves
        for i, meta in enumerate(layer.meta_inclusives.values()):
            meta.zeta = np.load(directory+"zeta_{}.".format(str(meta))+matrix_format)
            meta.q_k_tau = np.load(directory+"q_k_tau_{}.".format(str(meta))+matrix_format)
            # meta.omega = np.load(directory+"omega_{}_in_{}.".format(str_layer,str(meta))+matrix_format)

        ##exclusive_meta saves
        for i, meta in enumerate(layer.meta_exclusives.values()):
            meta.qka = np.load(directory+"qka_{}.".format(str(meta))+matrix_format)
            # meta.omega = np.load(directory+"omega_{}_ex_{}.".format(str_layer,str(meta))+matrix_format)
