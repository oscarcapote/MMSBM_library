import numpy as np
try:
    from numba_functions import *
    numba_imported = True
except ImportError:
    numba_imported = False

# @jit(nopython=True,locals=dict(i=int64,j=int64,L=int64,K=int64,Na=int64,Nb=int64,suma=double))
def omega_comp_arrays(Na,Nb,p_kl,theta,eta,K,L,links_array,links_ratings):
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
    It computes the membership matrix of a nodes layer with no metadata
    """
    theta = np.empty((N_nodes,K))
    theta_unfold = omega[links[:,0],links[:,1],:,:].sum(1)
    for att,mask in enumerate(masks_list):
        theta[att,:] = theta_unfold[mask].sum(0)
    theta /= denominators#[:,np.newaxis]
    return theta

def q_ka_comp_arrays(K,N_att,omega,links,masks_att_list):
    """
    It computes the probability matrix between nodes in group k and attribute a

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
    It computes the probability matrix between nodes_a and nodes_b in a BiNet network
    """
    p_kl = np.zeros((Ka,Kb,N_labels))
    sum_list = omega[links[:,0],links[:,1],:,:]
    for l,mask in enumerate(mask_list):
        p_kl[:,:,l] = sum_list[mask].sum(0)
    suma = p_kl.sum(axis=2)
    p_kl[:,:,:] /= (suma[:,:,np.newaxis]+1e-16)
    return p_kl



def theta_meta(theta,meta):
    for j,veins in enumerate(veins_items_array):
        for l in range(L):
            #eta_jl = eta[j,l]
            new_eta[j,l] = np.sum(omega[veins,j,:,l])
            new_eta[j,l] /= N_veins_items[j]
    return new_eta


def log_like_comp(theta,eta,pkl,links,labels):
    """
    Computes the log_likelihood given the membership matrices of two nodes layer (theta and eta), the matrix probabilities (pkl), and the links (links) with the labels (labels)
    """

    T = theta[links[:,0]][:,:,np.newaxis]
    E = eta[links[:,1]][:,np.newaxis,:]
    P = np.moveaxis(pkl[:,:,labels], -1,0)

    return np.log(((T@E)*P).sum(axis=(1,2))).sum()


def total_p_comp_test(theta,eta,pkl,links):
    """
    Computes the probabilities that two nodes are connected with each link label.
    """
    TE = theta[links[:,0]][:,:,np.newaxis]@eta[links[:,1]][:,np.newaxis,:]
    return (TE[:,:,:,np.newaxis]*pkl[:,:,:]).sum(axis=(1,2))


def log_like_comp_exclusive(theta,qka,links):
    """
    Computes the log_likelihood of a bipartite network of exclusive metadata.
    given the membership matrix of a nodes layer (theta), the matrix probabilities (pkl), and the links (links) with the labels (labels)
    """
    I = links[:,0]
    A = links[:,1]

    T = theta[I]#[:,:].T
    Q = qka[:,A]

    return np.log((T.T*Q).sum(0)).sum()

if not numba_imported:
    def A_lambda(A,lambda_val):
        """
        Multiplies a matrix A, by a scalar lambda_val
        """
        return A*lambda_val



    def sum_omega_02(omega):
        """
        Sum over axis 0 and 2 the omega matrix. Is equivalent to np.sum(omega, axis=(0,2))
        """
        return omega.sum(axis=(0,2))#.sum(axis=1)

    def sum_omega_13(omega):
        """
        Sum over axis 1 and 3 the omega matrix. Is equivalent to np.sum(omega, axis=(1,3))
        """
        return omega.sum(axis=(1,3))#.sum(axis=2)



    def sum_omega_ax(omega,ax):
        """
        Sum over axis ax the omega matrix.
        """
        return omega.sum(axis=ax)




    def sum_omega_02_lambda(omega,lambda_val):
        """
        Sum over axis 0 and 2 the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(0,2))*lambda_val
        """
        return A_lambda(sum_omega_02(omega),lambda_val)

    def sum_omega_13_lambda(omega,lambda_val):
        """
        Sum over axis 1 and 3 the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(1,3))*lambda_val
        """
        return A_lambda(sum_omega_13(omega),lambda_val)


    def sum_omega_ax_lambda(omega,ax,lambda_val):
        """
        Sum over axis ax the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(ax))*lambda_val
        """
        return A_lambda(sum_omega_ax(omega,ax),lambda_val)
