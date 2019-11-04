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


def consecutive_integer(mask):
    """
    Convert input mask into consecutive integer values, starting from 0. 
    If the background class is missing to start with, we manually inject 
    a background pixel at [0, 0] so that the loss function will run properly. 
    We realize that this is suboptimal and will explore better solutions 
    in the future. 
    """

    mask_buffer = np.zeros(mask.shape)
    if (0 not in np.unique(mask)):
        mask[0, 0] = 0
    mask_values = np.unique(mask)
    counter = 0
    # this works because np.unique() sorts in ascending order
    for value in mask_values:
        mask_buffer[mask == value] = counter
        counter += 1
    mask = mask_buffer.astype(int)
    return mask


def prep_image(image_info):
    image = image_info['image']
    image = image * 2 - 1
    image = np.expand_dims(image, axis = 0)

    return image


def prep_class_mask(image_info, params):
    output_size = params.OUTPUT_SIZE

    class_mask    = image_info['class_mask']
    class_mask    = resize_img(class_mask, output_size, output_size)
    class_mask    = np.expand_dims(class_mask, axis = 0)
    class_mask    = np.expand_dims(class_mask, axis = -1)

    return class_mask


def prep_instance_mask(image_info, params):
    output_size = params.OUTPUT_SIZE

    instance_mask = image_info['instance_mask']
    instance_mask = resize_img(instance_mask, output_size, output_size)
    instance_mask = consecutive_integer(instance_mask)
    instance_mask = np.expand_dims(instance_mask, axis = 0)
    instance_mask = np.expand_dims(instance_mask, axis = -1)

    return instance_mask


def prep_embedding(emb, params):
    img_size = params.IMG_SIZE

    emb = np.squeeze(emb)
    emb = resize(emb, [img_size, img_size])
    emb = np.expand_dims(emb, axis = 0)

    return emb


def prep_identity_mask(image_info, params):
    output_size = params.OUTPUT_SIZE

    identity_mask = image_info['identity_mask']
    identity_mask = resize_img(identity_mask, output_size, output_size)
    identity_mask = np.expand_dims(identity_mask, axis = 0)
    identity_mask = np.expand_dims(identity_mask, axis = -1)

    return identity_mask


def prep_optical_flow(image_info, params):
    output_size = params.OUTPUT_SIZE

    optical_flow = image_info['optical_flow']
    optical_flow = resize_img(optical_flow, output_size, output_size)
    optical_flow = np.expand_dims(optical_flow, axis = 0)

    return optical_flow


def prep_single_frame(image_info, params):
    image         = prep_image(image_info)
    class_mask    = prep_class_mask(image_info, params)
    instance_mask = prep_instance_mask(image_info, params)

    x = image
    y = np.concatenate((class_mask, instance_mask), axis = -1)

    return x, y


def prep_double_frame(image_info, prev_image_info, params):
    image         = prep_image(image_info)
    class_mask    = prep_class_mask(image_info, params)
    identity_mask = prep_identity_mask(image_info, params)

    prev_image         = prep_image(prev_image_info)
    prev_class_mask    = prep_class_mask(prev_image_info, params)
    prev_identity_mask = prep_identity_mask(prev_image_info, params)

    x = np.concatenate((image, prev_image), axis = -1)
    y = np.concatenate((class_mask, prev_class_mask, 
        identity_mask, prev_identity_mask), axis = -1)
    return x, y


def prep_half_pair(image_info, params):
    img_size      = params.IMG_SIZE
    embedding_dim = params.EMBEDDING_DIM
    output_size   = params.OUTPUT_SIZE

    image         = prep_image(image_info)
    class_mask    = prep_class_mask(image_info, params)
    instance_mask = prep_instance_mask(image_info, params)

    empty_prev_emb           = np.zeros((1, img_size, img_size, embedding_dim))
    empty_prev_img           = np.zeros((1, img_size, img_size, 3))
    empty_prev_instance_mask = np.zeros((1, output_size, output_size, 1))
    empty_identity_mask      = np.zeros((1, output_size, output_size, 1))
    empty_prev_identity_mask = np.zeros((1, output_size, output_size, 1))
    empty_optical_flow       = np.zeros((1, output_size, output_size, 2))

    x = np.concatenate((image, empty_prev_img, empty_prev_emb), axis = -1)
    y = np.concatenate((class_mask, instance_mask, empty_prev_instance_mask, 
        empty_identity_mask, empty_prev_identity_mask, empty_optical_flow), axis = -1)
    
    return x, y


def prep_pair(image_info, prev_image_info, prev_emb, params):
    img_size      = params.IMG_SIZE
    output_size   = params.OUTPUT_SIZE
    embedding_dim = params.EMBEDDING_DIM

    image              = prep_image(image_info)
    prev_image         = prep_image(prev_image_info)
    prev_emb           = prep_embedding(prev_emb, params)

    class_mask         = prep_class_mask(image_info, params)
    instance_mask      = prep_instance_mask(image_info, params)
    prev_instance_mask = prep_instance_mask(prev_image_info, params)
    identity           = prep_identity_mask(image_info, params)
    prev_identity      = prep_identity_mask(prev_image_info, params)
    optical_flow       = prep_optical_flow(prev_image_info, params)

    x = np.concatenate((image, prev_image, prev_emb), axis = -1)
    y = np.concatenate((class_mask, instance_mask, prev_instance_mask, 
        identity, prev_identity, optical_flow), axis = -1)

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


def flow_to_rgb(flow):
    # read nonzero optical flow
    image_size = flow.shape[0]
    direction_hsv = np.zeros((image_size, image_size, 3))
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    vmap = np.logical_or(dx != 0, dy != 0)
    vx = dx[vmap]
    vy = dy[vmap]
    # define min and max
    mag_max = np.sqrt(2)
    mag_min = 0
    angle_max = np.pi
    angle_min = -np.pi
    angles = np.arctan2(vx, vy)
    magnitudes = np.sqrt(np.power(vx, 2) + np.power(vy, 2))
    # convert to hsv
    hue = normalize(angles, [angle_min, angle_max])
    saturation = normalize(magnitudes, [mag_min, mag_max])
    value = np.zeros(angles.shape) + 1
    H = direction_hsv[:, :, 0]
    S = direction_hsv[:, :, 1]
    V = direction_hsv[:, :, 2]
    H[vmap] = hue
    S[vmap] = saturation
    V[vmap] = value
    direction_rgb = matplotlib.colors.hsv_to_rgb(direction_hsv)
    return direction_rgb


def flows_to_video(flows, video_name, fps):
    # assumes `flows` contains square images in shape of (x, x, 3)
    images = []
    for flow in flows:
        image = flow_to_rgb(flow)
        image = image * 255
        image = image.astype(np.uint8)
        images.append(image)
    imgs_to_video(images, video_name, fps)
    return


def imgs_to_video(images, video_name, fps):
    # assumes `images` contains square images in shape of (x, x, 3)
    image_size = images[0].shape[0]
    video = cv2.VideoWriter(video_name, 0, fps, (image_size, image_size))
    for image in images:
        video.write(image)
    video.release()
    return
