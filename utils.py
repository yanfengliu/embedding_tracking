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


def prep_for_model(image_info):
    x = image_info['image']
    x = x * 2 - 1
    x = np.expand_dims(x, axis = 0)

    class_mask = image_info['class_mask']
    class_mask = np.expand_dims(class_mask, axis = 0)
    class_mask = np.expand_dims(class_mask, axis = -1)

    instance_mask = image_info['instance_mask']
    instance_mask = np.expand_dims(instance_mask, axis = 0)
    instance_mask = np.expand_dims(instance_mask, axis = -1)

    y = np.concatenate([class_mask, instance_mask], axis = -1)

    return x, y


def totuple(a):
    """
    Convert a numpy array to a tuple of tuples in the format of [(), (), ...]
    """
    try:
        return [tuple(i) for i in a]
    except TypeError:
        return a


def normalize(x):
    """
    Normalize input to be zero mean and divide it by its global maximum value. 
    """

    x = x - np.min(x, keepdims=False)
    x = x / (np.max(x, keepdims=False) + 1e-10)
    return np.copy(x)