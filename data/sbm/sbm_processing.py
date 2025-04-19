import numpy as np
import pyemb as eb


def get_sbm_data():
    r"""
    Generate a Stochastic Block Model (SBM) dataset as described in 
    `https://arxiv.org/abs/2405.19230 <https://arxiv.org/abs/2405.19230>`_.
    
    This dataset represents a three-community Dynamic Stochastic Block Model (DSBM) with an 
    inter-community edge probability matrix:
    
    .. math::
    
        B(t) =
        \begin{bmatrix}
        s_1 & 0.02 & 0.02 \\
        0.02 & s_2 & 0.02 \\
        0.02 & 0.02 & s_3
        \end{bmatrix}
    
    where :math:`s_1`, :math:`s_2`, and :math:`s_3` represent within-community connection states. 
    Each :math:`s` can take one of two values: 0.08 or 0.16.
    
    We simulate a dynamic network over :math:`T = 8` time points, corresponding to the 
    :math:`2^3 = 8` possible combinations of :math:`s_1`, :math:`s_2`, and :math:`s_3`. 
    For each time point, the adjacency matrix :math:`A(t)` is drawn from the corresponding 
    probability matrix :math:`B(t)`. The ordering of these time points is random. 
    
    The task is to predict the community label of each node.

    Returns:
        tuple: A tuple containing:
            - As (list of np.ndarray): List of adjacency matrices for each time point.
            - node_labels (np.ndarray): Array of node labels for each time point.
    """
    K = 3
    n = 100 * K
    T = 8
    pi = np.repeat(1 / K, K)

    a = [0.08, 0.16]
    Bs = 0.02 * np.ones((T, K, K))

    T_list = [t for t in range(T)]
    np.random.shuffle(T_list)

    for t in range(T):
        for k in range(K):
            Bs[t, k, k] = a[(T_list[t] & (1 << k)) >> k]

    As, spatial_node_labels = eb.simulation.SBM(n, Bs, pi)
    node_labels = np.tile(spatial_node_labels, T)

    return As, node_labels
