import os
import numpy as np


def get_school_data(return_all_labels=False):
    """
    A dynamic social network between pupils at a primary school in Lyon, France
    (Stehlé et al., 2011).

    Each of the 232 pupils wore a radio identification device such that each interaction,
    with its timestamp, could be recorded, forming a dynamic network. An interaction was
    defined by close proximity for 20 seconds.

    The task is to predict the classroom allocation of each pupil. This dataset has a
    temporal structure that particularly distinguishes:

    - **Class time**: Pupils cluster together based on their class (easier).
    - **Lunchtime**: The cluster structure breaks down (harder).

    The data covers two full school days, making it roughly repeating.

    Returns:
        tuple: A tuple containing:
            - As (np.ndarray): Adjacency matrices for each time window.
            - node_labels (np.ndarray): Labels for each node at each time window.
    """

    window = 60 * 60

    day_1_start = (8 * 60 + 30) * 60
    day_1_end = (17 * 60 + 30) * 60
    day_2_start = ((24 + 8) * 60 + 30) * 60
    day_2_end = ((24 + 17) * 60 + 30) * 60

    T1 = int((day_1_end - day_1_start) // window)
    T2 = int((day_2_end - day_2_start) // window)
    T = T1 + T2
    print(f"Number of time windows: {T}")

    base_dir = os.path.dirname(__file__)
    fname = base_dir + "/ia-primary-school-proximity-attr.edges"
    file = open(fname)

    label_dict = {
        "1A": 0,
        "1B": 1,
        "2A": 2,
        "2B": 3,
        "3A": 4,
        "3B": 5,
        "4A": 6,
        "4B": 7,
        "5A": 8,
        "5B": 9,
        "Teachers": 10,
    }
    nodes = []
    spatial_node_labels = []
    edge_tuples = []

    for line in file:
        node_i, node_j, time, id_i, id_j = line.strip("\n").split(",")

        if day_1_start <= int(time) < day_1_end:
            t = (int(time) - day_1_start) // window
        elif day_2_start <= int(time) < day_2_end:
            t = T1 + (int(time) - day_2_start) // window
        else:
            continue

        if node_i not in nodes:
            nodes.append(node_i)
            spatial_node_labels.append(label_dict[id_i])

        if node_j not in nodes:
            nodes.append(node_j)
            spatial_node_labels.append(label_dict[id_j])

        edge_tuples.append([t, node_i, node_j])

    edge_tuples = np.unique(edge_tuples, axis=0)
    nodes = np.array(nodes)

    n = len(nodes)
    print(f"Number of nodes: {n}")

    node_dict = dict(zip(nodes[np.argsort(spatial_node_labels)], range(n)))
    spatial_node_labels = np.sort(spatial_node_labels)

    As = np.zeros((T, n, n))

    for m in range(len(edge_tuples)):
        t, i, j = edge_tuples[m]
        As[int(t), node_dict[i], node_dict[j]] = 1
        As[int(t), node_dict[j], node_dict[i]] = 1

    node_labels = np.tile(spatial_node_labels, T)

    if return_all_labels:
        all_labels = np.array(list(label_dict.keys()))
        return As, node_labels, all_labels
    else:
        return As, node_labels
