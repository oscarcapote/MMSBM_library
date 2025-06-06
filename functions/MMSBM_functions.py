import numpy as np


try:
    from MMSBM_library.functions.numba_functions import *
    numba_imported = True
except ImportError:
    numba_imported = False

# @jit(nopython=True,locals=dict(i=int64,j=int64,L=int64,K=int64,Na=int64,Nb=int64,suma=double))
def omega_comp_arrays(Na,Nb,p_kl,theta,eta,K,L,links_array,links_ratings):
    """
    Computes the omega matrix for a bipartite network.

    Parameters
    ----------
    Na : int
        Number of nodes in layer a
    Nb : int
        Number of nodes in layer b
    p_kl : np.array
        Probability matrix between groups k and l
    theta : np.array
        Membership matrix for layer a
    eta : np.array
        Membership matrix for layer b
    K : int
        Number of groups in layer a
    L : int
        Number of groups in layer b
    links_array : np.array
        Array containing the links between nodes
    links_ratings : np.array
        Array containing the ratings/labels for each link

    Returns
    -------
    np.array
        The computed omega matrix of shape (Na, Nb, K, L)
    """
    #new_omega = np.array(omega)
    omega = np.zeros((Na,Nb,K,L))
    for link  in range(len(links_ratings)):
        i = links_array[link][0]
        j = links_array[link][1]
        rating = links_ratings[link]
        omega[i,j,:,:] = p_kl[:,:,rating]*(np.expand_dims(theta[i,:], axis=1)@np.expand_dims(eta[j,:],axis=0))
        suma = omega[i,j,:,:].sum()
        omega[i,j,:,:] /= suma+1e-16
    return omega

# @jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double),parallel=True)
def omega_comp_arrays_exclusive(q_ka,N_att,theta,N_nodes,K,metas_links_arrays_nodes):
    """
    Computes the omega matrix for exclusive metadata.

    Parameters
    ----------
    q_ka : np.array
        Probability matrix between groups k and attributes a
    N_att : int
        Number of attributes
    theta : np.array
        Membership matrix for nodes
    N_nodes : int
        Number of nodes
    K : int
        Number of groups
    metas_links_arrays_nodes : np.array
        Array containing the links between nodes and metadata

    Returns
    -------
    np.array
        The computed omega matrix of shape (N_nodes, N_att, K)
    """
    omega = np.zeros((N_nodes,N_att,K))
    for j in range(len(metas_links_arrays_nodes)):
        i = metas_links_arrays_nodes[j,0]
        a = metas_links_arrays_nodes[j,1]
        s = 0
        for k in range(K):
            omega[i,a,k] = theta[i,k]*q_ka[k,a]
            s +=omega[i,a,k]
        omega[i,a,:] /= s
    return omega


def theta_comp_arrays_multilayer(BiNet,layer = "a"):
    """
    Computes the membership matrix for a layer in a bipartite network.

    Parameters
    ----------
    BiNet : BiNet
        The bipartite network object
    layer : str, default="a"
        The layer to compute the membership matrix for ("a" or "b")

    Returns
    -------
    np.array
        The computed membership matrix for the specified layer

    Raises
    ------
    TypeError
        If layer is not "a" or "b"
    """
    #new_theta = np.zeros((N_nodes,K))

    if layer == "a":
        na = BiNet.nodes_a
        observed = BiNet.observed_nodes_a
        non_observed = BiNet.non_observed_nodes_a
    elif layer == "b":
        na = BiNet.nodes_b
        observed = BiNet.observed_nodes_b
        non_observed = BiNet.non_observed_nodes_b
    else: raise TypeError("Layer must be 'a' or 'b'")

    #Nodes b
#     nb = BiNet.nodes_b

    ##Binet
    new_theta = np.zeros((len(na),na.K))
    if layer == "a":
        new_theta[BiNet.observed_nodes_a,:] = sum_omega_13(BiNet.omega[BiNet.observed_nodes_a[:,np.newaxis],BiNet.observed_nodes_b,:,:])#BiNet.omega.sum(axis=1).sum(axis=2)
    else:
        new_theta[BiNet.observed_nodes_b,:] = sum_omega_02(BiNet.omega[BiNet.observed_nodes_a[:,np.newaxis],BiNet.observed_nodes_b,:,:])#BiNet.omega.sum(axis=1).sum(axis=2)

    ##Meta exclusive
    N_metas = na.N_meta_exclusive

    for k, meta in enumerate(na.meta_exclusives.values()):
        new_theta += sum_omega_ax_lambda(meta.omega,1,meta.lambda_val)
#         new_theta += meta.omega.sum(axis=1)*meta.lambda_val


    ##Meta inclusive
    N_metas = na.N_meta_inclusive

    for k, meta in enumerate(na.meta_inclusives.values()):
        new_theta += sum_omega_13_lambda(meta.omega,meta.lambda_val)#meta.omega.sum(axis=1).sum(axis=2)*meta.lambda_val
        # new_theta += meta.omega.sum(axis=1).sum(axis=2)*meta.lambda_val

    ##Denominator
    new_theta /= na.denominators

    ##if no metadata we put means in cold starts
    if not na._has_metas and len(non_observed)!=0:
        means = np.mean(new_theta[observed,:],axis=0)
        new_theta[non_observed] = means

    return new_theta


def theta_comp_array(N_nodes,K,omega,denominators,links,masks_list):
    """
    Computes the membership matrix of a nodes layer with no metadata.

    Parameters
    ----------
    N_nodes : int
        Number of nodes
    K : int
        Number of groups
    omega : np.array
        The omega matrix
    denominators : np.array
        Array of denominators for normalization
    links : np.array
        Array containing the links
    masks_list : list
        List of masks for each attribute

    Returns
    -------
    np.array
        The computed membership matrix of shape (N_nodes, K)
    """
    theta = np.empty((N_nodes,K))
    theta_unfold = omega[links[:,0],links[:,1],:,:].sum(1)
    for att,mask in enumerate(masks_list):
        theta[att,:] = theta_unfold[mask].sum(0)
    theta /= denominators#[:,np.newaxis]
    return theta

def q_ka_comp_arrays(K,N_att,omega,links,masks_att_list):
    """
    Computes the probability matrix between nodes in group k and attribute a.

    Parameters
    ----------
    K : int
        Number of groups
    N_att : int
        Number of attributes
    omega : np.array
        The omega matrix of the BiNet
    links : np.array
        The links of the BiNet
    masks_att_list : list
        A list of masks for each attribute

    Returns
    -------
    np.array
        The probability matrix of shape (K, N_att) between nodes in group k and attribute a
    """
    qka2 = np.zeros((K,N_att))
    unfolded_q = omega[links[:,0],links[:,1],:]
    for att,mask in enumerate(masks_att_list):
        qka2[:,att] += unfolded_q[mask,:].sum(axis=0)
    suma = np.sum(qka2,axis =1)
    qka2 /=suma[:,np.newaxis]
    return qka2

def p_kl_comp_arrays(Ka,Kb,N_labels, links, omega, mask_list):
    """
    Computes the probability matrix between groups in nodes_a layer and group nodes_b layer in a BiNet network.

    Parameters
    ----------
    Ka : int
        Number of groups in layer a
    Kb : int
        Number of groups in layer b
    N_labels : int
        Number of possible labels
    links : np.array
        Array containing the links
    omega : np.array
        The omega matrix
    mask_list : list
        List of masks for each label

    Returns
    -------
    np.array
        The probability matrix of shape (Ka, Kb, N_labels) between nodes_a and nodes_b
    """
    p_kl = np.zeros((Ka,Kb,N_labels))
    sum_list = omega[links[:,0],links[:,1],:,:]
    for l,mask in enumerate(mask_list):
        p_kl[:,:,l] = sum_list[mask].sum(0)
    suma = p_kl.sum(axis=2)
    p_kl[:,:,:] /= (suma[:,:,np.newaxis]+1e-16)
    return p_kl



def theta_meta(theta, meta):
    """
    Computes the membership matrix for metadata.

    Parameters
    ----------
    theta : np.array
        The membership matrix
    meta : metadata_layer
        The metadata layer object

    Returns
    -------
    np.array
        The computed membership matrix for the metadata
    """
    for j,veins in enumerate(veins_items_array):
        for l in range(L):
            new_eta[j,l] = np.sum(omega[veins,j,:,l])
            new_eta[j,l] /= N_veins_items[j]
    return new_eta


def log_like_comp(theta, eta, pkl, links, labels):
    """
    Computes the log_likelihood given the membership matrices of two nodes layer and its matrix probabilities.

    Parameters
    ----------
    theta : np.array
        Membership matrix for layer a
    eta : np.array
        Membership matrix for layer b
    pkl : np.array
        Probability matrix between groups
    links : np.array
        Array containing the indexes of the links
    labels : np.array
        Array containing the indexes of the labels for each link

    Returns
    -------
    float
        The computed log likelihood
    """
    T = theta[links[:,0]][:,:,np.newaxis]
    E = eta[links[:,1]][:,np.newaxis,:]
    P = np.moveaxis(pkl[:,:,labels], -1,0)

    return np.log(((T@E)*P).sum(axis=(1,2))).sum()


def total_p_comp_test(theta, eta, pkl, links):
    """
    Computes the probabilities that two nodes are connected with each link label.

    Parameters
    ----------
    theta : np.array
        Membership matrix for layer a
    eta : np.array
        Membership matrix for layer b
    pkl : np.array
        Probability matrix between groups
    links : np.array
        Array containing the links

    Returns
    -------
    np.array
        Array of probabilities for each link label
    """
    TE = theta[links[:,0]][:,:,np.newaxis]@eta[links[:,1]][:,np.newaxis,:]
    return (TE[:,:,:,np.newaxis]*pkl[:,:,:]).sum(axis=(1,2))


def log_like_comp_exclusive(theta, qka, links):
    """
    Computes the log_likelihood of a bipartite network of exclusive metadata.

    Parameters
    ----------
    theta : np.array
        Membership matrix for nodes
    qka : np.array
        Probability matrix between groups and attributes
    links : np.array
        Array containing the indexes of the links

    Returns
    -------
    float
        The computed log likelihood for exclusive metadata
    """
    I = links[:,0]
    A = links[:,1]

    T = theta[I]
    Q = qka[:,A]

    return np.log((T.T*Q).sum(0)).sum()

if not numba_imported:
    def A_lambda(A, lambda_val):
        """
        Multiplies a matrix A by a scalar lambda_val.

        Parameters
        ----------
        A : np.array
            Input matrix
        lambda_val : float
            Scalar multiplier

        Returns
        -------
        np.array
            The matrix A multiplied by lambda_val
        """
        return A*lambda_val



    def sum_omega_02(omega):
        """
        Sum over axis 0 and 2 the omega matrix. Is equivalent to np.sum(omega, axis=(0,2)).

        Parameters
        ----------
        omega : np.array
            Input omega matrix

        Returns
        -------
        np.array
            The sum over axis 0 and 2 of omega
        """
        return omega.sum(axis=(0,2))

    def sum_omega_13(omega):
        """
        Sum over axis 1 and 3 the omega matrix. Is equivalent to np.sum(omega, axis=(1,3)).

        Parameters
        ----------
        omega : np.array
            Input omega matrix

        Returns
        -------
        np.array
            The sum over axis 1 and 3 of omega
        """
        return omega.sum(axis=(1,3))



    def sum_omega_ax(omega, ax):
        """
        Sum over axis ax the omega matrix.

        Parameters
        ----------
        omega : np.array
            Input omega matrix
        ax : int
            Axis to sum over

        Returns
        -------
        np.array
            The sum over axis ax of omega
        """
        return omega.sum(axis=ax)




    def sum_omega_02_lambda(omega, lambda_val):
        """
        Sum over axis 0 and 2 the omega matrix and multiplies the result by a scalar lambda_val.
        Is equivalent to np.sum(omega, axis=(0,2))*lambda_val.

        Parameters
        ----------
        omega : np.array
            Input omega matrix
        lambda_val : float
            Scalar multiplier

        Returns
        -------
        np.array
            The sum over axis 0 and 2 of omega multiplied by lambda_val
        """
        return A_lambda(sum_omega_02(omega),lambda_val)

    def sum_omega_13_lambda(omega, lambda_val):
        """
        Sum over axis 1 and 3 the omega matrix and multiplies the result by a scalar lambda_val.
        Is equivalent to np.sum(omega, axis=(1,3))*lambda_val.

        Parameters
        ----------
        omega : np.array
            Input omega matrix
        lambda_val : float
            Scalar multiplier

        Returns
        -------
        np.array
            The sum over axis 1 and 3 of omega multiplied by lambda_val
        """
        return A_lambda(sum_omega_13(omega),lambda_val)


    def sum_omega_ax_lambda(omega, ax, lambda_val):
        """
        Sum over axis ax the omega matrix and multiplies the result by a scalar lambda_val.
        Is equivalent to np.sum(omega, axis=(ax))*lambda_val.

        Parameters
        ----------
        omega : np.array
            Input omega matrix
        ax : int
            Axis to sum over
        lambda_val : float
            Scalar multiplier

        Returns
        -------
        np.array
            The sum over axis ax of omega multiplied by lambda_val
        """
        return A_lambda(sum_omega_ax(omega,ax),lambda_val)
