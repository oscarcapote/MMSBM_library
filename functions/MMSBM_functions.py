import numpy as np
from numba import jit,int64,float64,double,prange,vectorize

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
    for j in prange(len(metas_links_arrays_nodes)):
        i = metas_links_arrays_nodes[j,0]
        a = metas_links_arrays_nodes[j,1]
        s = 0
        for k in range(K):
            omega[i,a,k] = theta[i,k]*q_ka[k,a]
            s +=omega[i,a,k]
        omega[i,a,:] /= s
    return omega



@vectorize([float64(float64,float64)])
def A_lambda(A,lambda_val):
    """
    Multiplies a matrix A, by a scalar lambda_val
    """
    return A*lambda_val



@jit([float64[:,:](float64[:,:,:,:])])
def sum_omega_02(omega):
    """
    Sum over axis 0 and 2 the omega matrix. Is equivalent to np.sum(omega, axis=(0,2))
    """
    return omega.sum(axis=0).sum(axis=1)
    
@jit([float64[:,:](float64[:,:,:,:])])
def sum_omega_13(omega):
    """
    Sum over axis 1 and 3 the omega matrix. Is equivalent to np.sum(omega, axis=(1,3))
    """
    return omega.sum(axis=1).sum(axis=2)


    
@jit([float64[:,:](float64[:,:,:],int64)])
def sum_omega_ax(omega,ax):
    """
    Sum over axis ax the omega matrix. 
    """
    return omega.sum(axis=ax)




@jit([float64[:,:](float64[:,:,:,:],float64)])
def sum_omega_02_lambda(omega,lambda_val):
    """
    Sum over axis 0 and 2 the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(0,2))*lambda_val
    """
    return A_lambda(sum_omega_02(omega),lambda_val)
    
@jit([float64[:,:](float64[:,:,:,:],float64)])
def sum_omega_13_lambda(omega,lambda_val):
    """
    Sum over axis 1 and 3 the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(1,3))*lambda_val
    """
    return A_lambda(sum_omega_13(omega),lambda_val)

    
@jit([float64[:,:](float64[:,:,:],int64,float64)])
def sum_omega_ax_lambda(omega,ax,lambda_val):
    """
    Sum over axis ax the omega matrix and multiplies the result by a scalar labda_val. Is equivalent to np.sum(omega, axis=(ax))*lambda_val
    """
    return A_lambda(sum_omega_ax(omega,ax),lambda_val)


