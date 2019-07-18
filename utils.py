import numpy as np

def consecutive_integer(mask):
    """
    Convert input mask into consecutive integer values, starting from 0. 
    If the background class is missing to start with, we manually inject a background pixel at [0, 0]
    so that the loss function will run properly. We realize that this is suboptimal and will explore 
    better solutions in the future. 
    """

    mask_buffer = np.zeros(mask.shape)
    if (0 not in np.unique(mask)):
        mask[0, 0] = 0
    mask_values = np.unique(mask)
    change_log = np.zeros(shape=(len(mask_values)))
    counter = 0
    for value in mask_values:
        mask_buffer[mask == value] = counter
        change_log[counter] = value
        counter += 1
    mask = mask_buffer.astype(int)
    return mask, change_log


def totuple(a):
    """
    Convert a numpy array to a tuple of tuples in the format of [(), (), ...]
    """
    try:
        return [tuple(i) for i in a]
    except TypeError:
        return a