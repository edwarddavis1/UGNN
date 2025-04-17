import numpy as np


def non_zero_degree_mask(As, n, T):
    """
    Create a data mask which removes nodes with zero connections at each time step.

    Args:
        As (list of np.ndarray): List of adjacency matrices for each time step.
        n (int): Number of nodes.
        T (int): Number of time steps.

    Returns:
        np.ndarray: A boolean mask indicating usable node/time pairs.
    """
    data_mask = np.array([[True] * T for _ in range(n)])
    for t in range(T):
        data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False
    return data_mask


def mask_split(mask, split_props, seed=0, regime="transductive"):
    """
    Split a mask into train/valid/calib/test based on the specified regime.

    Args:
        mask (np.ndarray): A boolean mask of shape (n, T), where n is the number of nodes
            and T is the number of time steps.
        split_props (list of float): Proportions for splitting the data into
            train/valid/calib/test. The proportions should sum to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        regime (str, optional): Splitting regime, either "transductive" or "semi-inductive".
            Defaults to "transductive".

    Returns:
        list of np.ndarray: A list of boolean masks for each split (train, valid, calib, test).
            Each mask has the same shape as the input mask.
    """
    np.random.seed(seed)

    n, T = mask.shape

    if regime == "transductive":
        flat_mask = mask.T.reshape(-1)
        n_masks = np.sum(flat_mask)

        flat_mask_idx = np.where(flat_mask)[0]
        np.random.shuffle(flat_mask_idx)
        split_ns = np.cumsum([round(n_masks * prop) for prop in split_props[:-1]])
        split_idx = np.split(flat_mask_idx, split_ns)

    elif regime == "semi-inductive":
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)

        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-1] / np.sum(split_props[:-1])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)
        split_idx.append(n * T_trunc + np.where(flat_mask_end)[0])

    elif regime == "temporal transductive":
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)
        n_masks_end = np.sum(flat_mask_end)

        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-2] / np.sum(split_props[:-2])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        flat_mask_end_idx = np.where(flat_mask_end)[0]
        np.random.shuffle(flat_mask_end_idx)
        split_props_end = split_props[-2:] / np.sum(split_props[-2:])
        split_ns = np.cumsum(
            [round(n_masks_end * prop) for prop in split_props_end[:-1]]
        )
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[0])
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[1])

    split_masks = np.array([[False] * n * T for _ in range(len(split_props))])
    for i in range(len(split_props)):
        split_masks[i, split_idx[i]] = True

    return split_masks


def mask_mix(mask_1, mask_2, seed=0):
    np.random.seed(seed)

    n = len(mask_1)
    n1 = np.sum(mask_1)

    mask_idx = np.where(mask_1 + mask_2)[0]
    np.random.shuffle(mask_idx)
    split_idx = np.split(mask_idx, [n1])

    split_masks = np.array([[False] * n for _ in range(2)])
    for i in range(2):
        split_masks[i, split_idx[i]] = True

    return split_masks


def pad_unfolded_mask(mask, n):
    """
    Padding required due to the n extra anchor nodes introduced in the unfolded representation.

    These nodes are not included in training.

    Args:
        mask (np.ndarray): The original mask.
        n (int): The number of nodes.

    Returns:
        np.ndarray: The padded mask.
    """
    return np.concatenate((np.array([False] * n), mask))
