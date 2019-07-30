import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from postprocessing import ETH_mean_shift
from utils import (normalize, prep_half_pair_for_model, prep_image_for_model,
                   prep_pair_for_model, resize_img)


def visualize(embedding_pred, embedding_dim, output_size, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_gt, colors, image):
    # pca on embedding purely for visualization, not for clustering
    embedding_pred_flat = np.reshape(embedding_pred, (-1, embedding_dim))
    embedding_pred_flat = StandardScaler().fit_transform(embedding_pred_flat)
    pca = PCA(n_components=3)
    pc_flat = pca.fit_transform(embedding_pred_flat)
    pc = np.reshape(pc_flat, (output_size, output_size, 3))
    pc = normalize(pc)

    # prepare predicted embeddings (front/back)
    show_mask = np.expand_dims(class_mask_int_pred > 0, axis=-1)
    embedding_masked = np.multiply(pc, show_mask)

    # show instance mask and predicted embeddings
    all_instances = np.zeros((output_size, output_size, 3))
    slice_0 = np.zeros((output_size, output_size))
    slice_1 = np.zeros((output_size, output_size))
    slice_2 = np.zeros((output_size, output_size))
    for i in range(int(np.max(cluster_all_class))):
        slice_0[cluster_all_class == i] = colors[i, 0]
        slice_1[cluster_all_class == i] = colors[i, 1]
        slice_2[cluster_all_class == i] = colors[i, 2]
    all_instances[:, :, 0] = slice_0
    all_instances[:, :, 1] = slice_1
    all_instances[:, :, 2] = slice_2

    instance_mask_gt_color = np.zeros((output_size, output_size, 3))
    for i in np.unique(instance_mask_gt):
        instance_mask_gt_color[instance_mask_gt == i] = colors[int(i), :]

    class_max = class_num - 1
    class_mask_int_pred_color = np.zeros((output_size, output_size, 3))
    class_mask_int_pred_color[:, :, 0] = class_mask_int_pred/class_max
    class_mask_int_pred_color[:, :, 1] = class_mask_int_pred/class_max
    class_mask_int_pred_color[:, :, 2] = class_mask_int_pred/class_max

    class_mask_int_gt_color = np.zeros((output_size, output_size, 3))
    class_mask_int_gt_color[:, :, 0] = class_mask_gt/class_max
    class_mask_int_gt_color[:, :, 1] = class_mask_gt/class_max
    class_mask_int_gt_color[:, :, 2] = class_mask_gt/class_max

    image = cv2.resize(image, (output_size, output_size))
    image = (image + 1)/2
    board = np.zeros((output_size, output_size*7, 3))
    board[:, (output_size*0):(output_size*1), :] = image
    board[:, (output_size*1):(output_size*2), :] = pc
    board[:, (output_size*2):(output_size*3), :] = embedding_masked
    board[:, (output_size*3):(output_size*4), :] = all_instances
    board[:, (output_size*4):(output_size*5), :] = instance_mask_gt_color
    board[:, (output_size*5):(output_size*6), :] = class_mask_int_pred_color
    board[:, (output_size*6):(output_size*7), :] = class_mask_int_gt_color

    plt.figure(figsize=(4 * 7, 4))
    plt.imshow(board)
    plt.show()


def single_eval(model, x, y, params):
    class_num                = params.NUM_CLASSES
    embedding_dim            = params.EMBEDDING_DIM
    ETH_mean_shift_threshold = params.ETH_MEAN_SHIFT_THRESHOLD
    output_size              = params.OUTPUT_SIZE
    task                     = params.TASK
    colors                   = params.COLORS

    if task == 'image':
        image = np.squeeze(x)
    elif task == 'sequence':
        image = np.squeeze(x[0])

    class_mask_gt = y[0, ..., 0]
    instance_mask_gt = y[0, ..., 1]

    outputs = model.predict(x)

    class_mask_pred = outputs[0, :, :, :class_num]
    embedding_pred = outputs[0, :, :, class_num:(class_num + embedding_dim)]

    ### Post-processing ###
    # separate the process for different non-background classes
    class_mask_int_pred = np.argmax(class_mask_pred, axis=-1)
    cluster_all_class = np.zeros((output_size, output_size))
    previous_highest_label = 0
    instance_to_class = []
    for j in range(class_num-1):
        class_mask_pred_slice = np.zeros((output_size, output_size))
        class_mask_pred_slice[class_mask_int_pred == j+1] = 1
        cluster = ETH_mean_shift(
            embedding_pred, class_mask_pred_slice, threshold=ETH_mean_shift_threshold)
        instance_to_class += [j+1] * np.max(cluster).astype(np.int)
        cluster[cluster != 0] += previous_highest_label
        filter_mask = class_mask_pred_slice > 0
        filter_template = np.zeros((output_size, output_size))
        filter_template[filter_mask] = 1
        cluster = np.multiply(cluster, filter_template)
        cluster_all_class += cluster
        previous_highest_label = np.max(cluster_all_class)

    visualize(embedding_pred, embedding_dim, output_size, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_gt, colors, image)

    if task == 'sequence':
        return embedding_pred


def sequence_eval(model, sequence, params):
    image_info = sequence[0]
    x, y = prep_half_pair_for_model(image_info, params)
    emb = single_eval(model, x, y, params)
    for i in range(1, len(sequence)):
        prev_image_info, image_info = sequence[i-1:i+1]
        x, y = prep_pair_for_model(image_info, params, prev_image_info, emb)
        _, emb = x
        _ = single_eval(model, x, y, params)
