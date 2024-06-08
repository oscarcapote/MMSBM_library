from numba import jit,int64,float64,double,prange,vectorize

@vectorize([float64(float64,float64)])
def A_lambda(A,lambda_val):
    """
    Multiplies a matrix A, by a scalar lambda_val
    """
    return A*lambda_val



@jit([float64[:,:](float64[:,:,:,:])])
def sum_omega_02(omega):
    """
    Sum over axis 0 and 2 the omega matrix. Is equivalent to np.sum(omega, axis=(0,2)) prueba
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
