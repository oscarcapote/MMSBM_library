import numpy as np
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
