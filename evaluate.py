import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from postprocessing import ETH_mean_shift
from utils import *


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
    
    num_instances = int(np.max(cluster_all_class))
    random_colors = np.random.rand((num_instances, 3))

    for i in range(num_instances):
        slice_0[cluster_all_class == i] = random_colors[i, 0]
        slice_1[cluster_all_class == i] = random_colors[i, 1]
        slice_2[cluster_all_class == i] = random_colors[i, 2]
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
    output_size              = params.OUTPUT_SIZE
    colors                   = params.COLORS

    outputs = model.predict(x)
    class_mask_pred = outputs[0, :, :, :class_num]
    embedding_pred  = outputs[0, :, :, class_num:(class_num + embedding_dim)]
    class_mask_int_pred = np.argmax(class_mask_pred, axis=-1)
    cluster_all_class = embedding_to_instance(embedding_pred, class_mask_pred, params)
    image = np.squeeze(x)
    class_mask_gt    = y[0, ..., 0]
    instance_mask_gt = y[0, ..., 1]
    visualize(embedding_pred, embedding_dim, output_size, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_gt, colors, image)


def eval_pair(model, pair, params):
    class_num                = params.NUM_CLASSES
    embedding_dim            = params.EMBEDDING_DIM
    output_size              = params.OUTPUT_SIZE
    image_size               = params.IMG_SIZE
    colors                   = params.COLORS

    images = np.zeros((image_size, image_size*2, 3))
    board = np.zeros((output_size*2, output_size*5, 3))
    image_info, prev_image_info = pair

    image              = image_info['image']
    identity_mask      = image_info['identity_mask']
    class_mask         = image_info['class_mask']

    prev_image         = prev_image_info['image']
    prev_identity_mask = prev_image_info['identity_mask']
    prev_class_mask    = prev_image_info['class_mask']
    optical_flow       = prev_image_info['optical_flow']

    images[:image_size, :, :, :] = image
    images[image_size:, :, :, :] = prev_image

    board[:output_size, :output_size]                    = identity_mask
    board[:output_size, output_size:(output_size*2)]     = class_mask
    board[:output_size, (output_size*2):(output_size*3)] = prev_identity_mask
    board[:output_size, (output_size*3):(output_size*4)] = prev_class_mask
    board[:output_size, (output_size*4):(output_size*5)] = optical_flow

    x, y = prep_double_frame(image_info, prev_image_info, params)
    outputs = model.predict(x)
    outputs = np.squeeze(outputs)

    class_mask_pred      = outputs[0, :, :, :class_num]
    class_mask_prev_pred = outputs[0, :, :, class_num:(class_num*2)]
    embedding_pred       = outputs[0, :, :, (class_num*2):(class_num*2 + embedding_dim)]
    embedding_prev_pred  = outputs[0, :, :, (class_num*2 + embedding_dim):((class_num*2 + embedding_dim*2))]
    optical_flow_pred    = outputs[0, :, :, (class_num*2 + embedding_dim*2):]

    combined_class_mask_pred = np.zeros((output_size, output_size*2))
    combined_embeding_pred   = np.zeros((output_size, output_size*2, embedding_dim))

    combined_class_mask_pred[:, :output_size] = class_mask_pred
    combined_class_mask_pred[:, output_size:] = class_mask_prev_pred
    combined_embeding_pred[:, :output_size, :] = embedding_pred
    combined_embeding_pred[:, output_size:, :] = embedding_prev_pred

    cluster_all_class = embedding_to_instance(combined_embeding_pred, combined_class_mask_pred, params)
    identity_mask_pred      = cluster_all_class[:, :output_size]
    prev_identity_mask_pred = cluster_all_class[:, output_size:]
    
    board[output_size:, :output_size]                    = identity_mask_pred
    board[output_size:, output_size:(output_size*2)]     = class_mask_pred
    board[output_size:, (output_size*2):(output_size*3)] = prev_identity_mask_pred
    board[output_size:, (output_size*3):(output_size*4)] = class_mask_prev_pred
    board[output_size:, (output_size*4):(output_size*5)] = optical_flow_pred


def embedding_to_instance(embedding, class_mask, params):
    output_size              = params.OUTPUT_SIZE
    class_num                = params.NUM_CLASSES
    ETH_mean_shift_threshold = params.ETH_MEAN_SHIFT_THRESHOLD

    class_mask_int = np.argmax(class_mask, axis=-1)
    cluster_all_class = np.zeros((output_size, output_size*2))
    previous_highest_label = 0
    instance_to_class = []
    for j in range(class_num-1):
        class_mask_slice = np.zeros((output_size, output_size*2))
        class_mask_slice[class_mask_int == j+1] = 1
        cluster = ETH_mean_shift(
            data=embedding, 
            mask=class_mask_slice, 
            threshold=ETH_mean_shift_threshold)
        instance_to_class += [j+1] * np.max(cluster).astype(np.int)
        cluster[cluster != 0] += previous_highest_label
        filter_mask = class_mask_slice > 0
        filter_template = np.zeros((output_size, output_size*2))
        filter_template[filter_mask] = 1
        cluster = np.multiply(cluster, filter_template)
        cluster_all_class += cluster
        previous_highest_label = np.max(cluster_all_class)

    return cluster_all_class