import numpy as np
from ugnn.utils.masks import mask_split, mask_mix


def test_mask_split():
    mask = np.array([[True, True], [True, False]])
    split_props = [0.5, 0.5]
    split_masks = mask_split(mask, split_props, seed=42, mode="transductive")
    assert split_masks.shape == (2, 4)
    assert np.sum(split_masks[0]) + np.sum(split_masks[1]) == np.sum(mask)


def test_mask_mix():
    mask_1 = np.array([True, False, True, False])
    mask_2 = np.array([False, True, False, True])
    mixed_masks = mask_mix(mask_1, mask_2, seed=42)
    assert mixed_masks.shape == (2, 4)
    assert np.sum(mixed_masks[0]) + np.sum(mixed_masks[1]) == np.sum(mask_1 + mask_2)
