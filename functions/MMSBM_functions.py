import numpy as np
from numba import jit,int64,double

# @jit(nopython=True,locals=dict(i=int64,j=int64,L=int64,K=int64,Na=int64,Nb=int64,suma=double))
def omega_comp_arrays(Na,Nb,p_kl,theta,eta,K,L,links_array,links_ratings):
    print("HOLA!!")
    #new_omega = np.array(omega)
    print((Na,Nb,K,L))
    omega = np.ones((Na,Nb,K,L))
    for link  in range(len(links_ratings)):
        i = links_array[link][0]
        j = links_array[link][1]
        rating = links_ratings[link]
        omega[i,j,:,:] = p_kl[:,:,rating]*(np.expand_dims(theta[i,:], axis=1)@np.expand_dims(eta[j,:],axis=0))
        suma = omega[i,j,:,:].sum()
        omega[i,j,:,:] /= suma+1e-16
    return

# @jit(nopython=True,locals=dict(i=int64,a=int64,k=int64,link=int64,suma=double),parallel=True)
def omega_comp_arrays_exclusive(omega,q_ka,theta,N_nodes,metas_links_arrays_nodes):
    for j in prange(len(metas_links_arrays_nodes)):
        i = metas_links_arrays_nodes[j,0]
        a = metas_links_arrays_nodes[j,1]
        s = 0
        for k in range(K):
            omega[i,a,k] = theta[i,k]*q_ka[k,a]
            s +=omega[i,a,k]
        omega[i,a,:] /= s
    return
