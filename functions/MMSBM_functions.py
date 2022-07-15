import numpy as np
try:
    from numba_functions import *
    numba_imported = True
except ImportError:
    numba_imported = False

# @jit(nopython=True,locals=dict(i=int64,j=int64,L=int64,K=int64,Na=int64,Nb=int64,suma=double))
def omega_comp_arrays(Na,Nb,p_kl,theta,eta,K,L,links_array,links_ratings):
    #new_omega = np.array(omega)
    print((Na,Nb,K,L))
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
    else: raise TypeError("layer must be 'a' or 'b'")

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

    for meta in na.meta_exclusives:
        new_theta += sum_omega_ax_lambda(meta.omega,1,meta.lambda_meta)
#         new_theta += meta.omega.sum(axis=1)*meta.lambda_meta


    ##Meta inclusive
    N_metas = na.N_meta_inclusive

    for meta in na.meta_inclusives:
        new_theta += sum_omega_13_lambda(meta.omega,meta.lambda_meta)#meta.omega.sum(axis=1).sum(axis=2)*meta.lambda_meta
        # new_theta += meta.omega.sum(axis=1).sum(axis=2)*meta.lambda_meta

    ##Denominator
    new_theta /= na.denominators

    ##if no metadata we put means in cold starts
    if not na.has_metas and len(non_observed)!=0:
        means = np.sum(new_theta[observed,:],axis=0)/float(len(observed))
        new_theta[non_observed] = means

    return new_theta




def q_ka_comp_arrays(omega,K,links_array,N_att):
    """
    It computes the probability matrix between nodes in group k and attribute a
    """
    q_ka2 = np.zeros((K,N_att))

    for link  in range(len(links_array)):
        i = links_array[link][0]
        a = links_array[link][1]
        for k in range(K):
            #print(i,k,a)
            q_ka2[k,a] = omega[i,a,k]#/att_elements[a]

    suma = np.sum(q_ka2,axis =1)
    q_ka2  /=suma[:,np.newaxis]
    return q_ka2

def theta_meta(theta,meta):
    for j,veins in enumerate(veins_items_array):
        for l in range(L):
            #eta_jl = eta[j,l]
            new_eta[j,l] = np.sum(omega[veins,j,:,l])
            new_eta[j,l] /= N_veins_items[j]
    return new_eta


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
