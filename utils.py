import os
import sys

import cv2
import matplotlib
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


def prep_image(image_info):
    x = image_info['image']
    # x = x.astype(np.float32)
    x = x * 2 - 1
    x = x[np.newaxis, ...]
    return x


def prep_class_mask(image_info):
    x = image_info['class_mask']
    x = x[np.newaxis, ..., np.newaxis]
    return x


def prep_occ_class_mask(image_info):
    x = image_info['occ_class_mask']
    x = x[np.newaxis, ..., np.newaxis]
    return x


def prep_instance_mask(image_info):
    x = image_info['instance_mask']
    x = x[np.newaxis, ..., np.newaxis]
    return x


def prep_occ_instance_mask(image_info):
    x = image_info['occ_instance_mask']
    x = x[np.newaxis, ..., np.newaxis]
    return x


def prep_optical_flow(image_info):
    x = image_info['optical_flow']
    x = x[np.newaxis, ...]
    return x


def prep_single_frame(image_info):
    image         = prep_image(image_info)
    class_mask    = prep_class_mask(image_info)
    instance_mask = prep_instance_mask(image_info)
    x = image
    y = np.concatenate((class_mask, instance_mask), axis = -1)
    return x, y


def prep_double_frame(prev_image_info, image_info):
    prev_image              = prep_image(               prev_image_info)
    prev_class_mask         = prep_class_mask(          prev_image_info)
    occ_prev_class_mask     = prep_occ_class_mask(      prev_image_info)
    prev_instance_mask      = prep_instance_mask(       prev_image_info)
    occ_prev_instance_mask  = prep_occ_instance_mask(   prev_image_info)
    optical_flow            = prep_optical_flow(        prev_image_info)
    image                   = prep_image(               image_info)
    class_mask              = prep_class_mask(          image_info)
    occ_class_mask          = prep_occ_class_mask(      image_info)
    instance_mask           = prep_instance_mask(       image_info)
    occ_instance_mask       = prep_occ_instance_mask(   image_info)
    x = np.concatenate((image, prev_image), axis = -1)
    y = np.concatenate((
        class_mask, 
        occ_class_mask, 
        prev_class_mask, 
        occ_prev_class_mask, 
        instance_mask, 
        occ_instance_mask, 
        prev_instance_mask, 
        occ_prev_instance_mask, 
        optical_flow), axis = -1)
    return x, y


def totuple(a):
    """
    Convert a numpy array to a tuple of tuples in the format of [(), (), ...]
    """
    try:
        return [tuple(i) for i in a]
    except TypeError:
        return a


def normalize(x, val_range=None):
    """
    Map x to [0, 1] using either its min and max or the given range.
    """
    if val_range:
        val_min, val_max = val_range
    else:
        val_min = np.min(x, keepdims=False)
        val_max = np.max(x, keepdims=False) + 1e-10
    x[x > val_max] = val_max
    x[x < val_min] = val_min
    x = (x - val_min) / (val_max - val_min)
    return np.copy(x)


def visualize_history(loss_history, title):
    plt.figure(figsize=(10, 2))
    plt.plot(loss_history[-2000:])
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


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def mask2bbox(mask, image_size):
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    x_center = (cmax + cmin) / (2 * image_size)
    y_center = (rmax + rmin) / (2 * image_size)
    width = (cmax - cmin) / image_size
    height = (rmax - rmin) / image_size
    bbox = [x_center, y_center, width, height]
    return bbox


def intersection(mask1, mask2):
    return np.sum((mask1 > 0) & (mask2 > 0))


def union(mask1, mask2):
    return np.sum((mask1 > 0) | (mask2 > 0))


def iou(mask1, mask2):
    mask1 = np.squeeze(mask1)
    mask2 = np.squeeze(mask2)
    i = intersection(mask1, mask2)
    u = union(mask1, mask2)
    return i / u


def images_to_video(image_path, video_path, fps, dimensions):
    image_list = os.listdir(image_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, dimensions)
    for image_name in image_list:
        image_full_path = os.path.join(image_path, image_name)
        image = cv2.imread(image_full_path)
        out.write(image)
    out.release()