import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from skimage.transform import resize


def resize_img(img, width, height):
    img = resize(
        img,
        [width, height],
        order=0,
        cval=0,
        mode='constant',
        anti_aliasing=False,
        preserve_range=True)
    return img


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
    counter = 0
    for value in mask_values:
        mask_buffer[mask == value] = counter
        counter += 1
    mask = mask_buffer.astype(int)
    return mask


def prep_image_for_model(image_info, params):
    output_size = params.OUTPUT_SIZE
    x = image_info['image']
    x = x * 2 - 1
    x = np.expand_dims(x, axis = 0)

    class_mask    = image_info['class_mask']
    class_mask    = resize_img(class_mask, output_size, output_size)
    class_mask    = np.expand_dims(class_mask, axis = 0)
    class_mask    = np.expand_dims(class_mask, axis = -1)

    instance_mask = image_info['instance_mask']
    instance_mask = resize_img(instance_mask, output_size, output_size)
    instance_mask = consecutive_integer(instance_mask)
    instance_mask = np.expand_dims(instance_mask, axis = 0)
    instance_mask = np.expand_dims(instance_mask, axis = -1)

    y = np.concatenate([class_mask, instance_mask], axis = -1)

    return x, y


def prep_half_pair_for_model(image_info, params):
    x, y = prep_image_for_model(image_info, params)
    output_size   = params.OUTPUT_SIZE
    embedding_dim = params.EMBEDDING_DIM
    empty_prev_emb = np.zeros((1, output_size, output_size, embedding_dim))
    x = [x, empty_prev_emb]
    y = np.concatenate([y, empty_prev_emb], axis = -1)
    return x, y


def prep_pair_for_model(image_info, params, prev_image_info, emb):
    emb = np.squeeze(emb)
    prev_instance_mask = prev_image_info['instance_mask']
    output_size = params.OUTPUT_SIZE
    prev_instance_mask = resize_img(prev_instance_mask, output_size, output_size)
    centers = get_masked_emb(emb, prev_instance_mask)
    x, y = prep_image_for_model(image_info, params)
    output_size   = params.OUTPUT_SIZE
    embedding_dim = params.EMBEDDING_DIM
    prev_emb = np.zeros((1, output_size, output_size, embedding_dim))
    prev_emb[0, 0, :centers.shape[0], :] = centers
    x = [x, prev_emb]
    y = np.concatenate([y, prev_emb], axis = -1)
    return x, y


def get_masked_emb(emb, all_instances):
    embedding_dim = emb.shape[-1]
    num_clusters = int(np.max(np.unique(all_instances)))
    centers = np.zeros((num_clusters, embedding_dim))
    for i in range(num_clusters):
        masked_emb = emb[all_instances == i]
        num_pixel = np.sum(all_instances == i)
        masked_emb = np.reshape(masked_emb, [num_pixel, embedding_dim])
        avg_emb = np.mean(masked_emb, axis = 0)
        centers[i, :] = avg_emb
    return centers


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


def visualize_history(loss_history, title):
    plt.figure(figsize=(20, 4))
    plt.plot(loss_history)
    plt.grid()
    plt.title(title)
    plt.show()


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def update_progress(progress, text=""):
    bar_length = 25
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0 or progress > 1:
        raise ValueError('Progress must be in [0.0, 1.0]')
    block = int(round(bar_length * progress))
    if in_ipynb():
        clear_output(wait=True)
        print(text)
        print("Progress: [{0}] {1:.1f}%".format(
            "#" * block + "-" * (bar_length - block), progress * 100))
    else:
        sys.stdout.write("\r Progress: [{0}] {1:.1f}% {2}".format(
            "#" * block + "-" * (bar_length - block), progress * 100, text))
        sys.stdout.flush()
