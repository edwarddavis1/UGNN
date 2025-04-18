import numpy as np
import pyemb as eb


def get_sbm_data():

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
