import numpy as np
import pandas as pd
import json
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
    print(shape)
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
    Returns True if A and A_old are simmilar by looking if the mean absolute difference between A and A_old is lower than a tolerance tol
    """
    finished = False
    if(np.mean(np.abs(A-A_old))<tol):
        finished = True
    return finished


def save_MMSBM_parameters(BiNet,dir=".",matrix_format="npy"):
    """
    It saves the parameters into matrices in the dir directory

    Parameters:
    -----------
    BiNet: BiNet object
        Bipartite network with the MMSBM initialized

    dir: str
        Directory where the files with the MMSBM parameters will be saved

    matrix_format: str
        Format that the matrices parameters will be saved. It can be, npy, npz, txt or dat
        If you choose a format that is not npy or npz, they will be saved in a text file using the numpy.savetxt function

    """
    na = BiNet.nodes_a

    nb = BiNet.nodes_b

    if matrix_format == "npy" or matrix_format == "npz":
        save_func = np.save
    else:
        save_func = np.savetxt


    save_func(dir+"/pkl."+matrix_format,BiNet.pkl)
    save_func(dir+"/omega."+matrix_format,BiNet.omega)


    ##Metas saves
    for layer,str_layer in [(na,"a"),(nb,"b")]:
        print(str_layer)
        save_func(dir+"/theta_{}.".format(str_layer)+matrix_format,layer.theta)
        ##inclusive_meta saves
        for i, meta in enumerate(layer.meta_inclusives):
            save_func(dir+"/zeta_{}.".format(str(meta))+matrix_format,meta.zeta)
            save_func(dir+"/q_k_tau_{}.".format(str(meta))+matrix_format,meta.q_k_tau)
            save_func(dir+"/omega_{}_in_{}.".format(str_layer,str(meta))+matrix_format,meta.omega)

        ##exclusive_meta saves
        for i, meta in enumerate(layer.meta_exclusives):
            save_func(dir+"/qka_{}.".format(str(meta))+matrix_format,meta.qka)
            save_func(dir+"/omega_{}_ex_{}.".format(str_layer,str(meta))+matrix_format,meta.omega)

    #other values from MAP and MMSBM saves
    # f = open("BiNet_info.dat")
    dic_info = {}
    dic_info["dict_codes"] ={str(k):str(v) for k,v in BiNet.dict_codes.items()}
    dic_info["N_links"] = len(BiNet.links)
    

    for layer,str_layer in [(na,"a"),(nb,"b")]:
        layer_label = "layer "+str_layer
        dic_info[layer_label] = {"name":str(layer),
                                 "N_nodes":len(layer),
                                 "N_metas":layer.N_meta,
                                 "K":layer.K,
                                 "N_metas_exclusives":str(layer.N_meta_exclusive),
                                 "N_metas_inclusives":str(layer.N_meta_inclusive),
                                 "dict_codes":{str(k):str(v) for k,v in layer.dict_codes.items()}
                                    }
        ##exclusive_meta saves
        for i, meta in enumerate(layer.meta_exclusives):
            dic_info[layer_label]["metadata_exclusives"] = {
                                                            "Meta_name":str(meta),
                                                            "lambda":meta.lambda_val,
                                                            "N_atts":len(meta),
                                                            "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
                                                            "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
                                                            }
        ##inclusive_meta saves
        for i, meta in enumerate(layer.meta_inclusives):
            dic_info[layer_label]["metadata_inclusives"] = {
                                                            "Meta_name":str(meta),
                                                            "lambda":meta.lambda_val,
                                                            "Tau":meta.Tau,
                                                            "N_atts":len(meta),
                                                            "Attributes":[str(i) for i in np.unique(layer.df[str(meta)])],
                                                            "dict_codes":{str(k):str(v) for k,v in meta.dict_codes.items()}
                                                            }
    BiNet.info = dic_info
    with open(dir+'/BiNet_data.json', 'w') as outfile:
    #     json_string = json.dumps(politics.dict_codes)
        json.dump(dic_info, outfile)
