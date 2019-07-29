import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from postprocessing import ETH_mean_shift
from utils import normalize, prep_for_model


def single_eval(model, image_info, params):

    class_num                = params.NUM_CLASSES
    embedding_dim            = params.EMBEDDING_DIM
    ETH_mean_shift_threshold = params.ETH_MEAN_SHIFT_THRESHOLD
    img_size                 = params.IMG_SIZE

    image            = image_info['image']
    class_mask_gt    = image_info['class_mask']
    instance_mask_gt = image_info['instance_mask']

    # predictions
    img, _ = prep_for_model(image_info)
    x = model.predict(img)

    class_mask_pred = x[0, :, :, :class_num]
    embedding_pred  = x[0, :, :, class_num:(class_num + embedding_dim)]
    
    # use segmentation predicted by model
    class_mask_int_pred = np.argmax(class_mask_pred, axis=-1)
        
    ### Post-processing ###
    # separate the process for different non-background classes
    cluster_all_class = np.zeros((img_size, img_size))
    previous_highest_label = 0
    instance_to_class = []
    for j in range(class_num-1):
        class_mask_pred_slice = np.zeros((img_size, img_size))
        class_mask_pred_slice[class_mask_int_pred == j+1] = 1
        cluster = ETH_mean_shift(
            embedding_pred, class_mask_pred_slice, threshold=ETH_mean_shift_threshold)
        instance_to_class += [j+1] * np.max(cluster).astype(np.int)
        cluster[cluster != 0] += previous_highest_label
        filter_mask = class_mask_pred_slice > 0
        filter_template = np.zeros((img_size, img_size))
        filter_template[filter_mask] = 1
        cluster = np.multiply(cluster, filter_template)
        cluster_all_class += cluster
        previous_highest_label = np.max(cluster_all_class)

    # pca on embedding for better visualization
    embedding_pred_flat = np.reshape(embedding_pred, (-1, embedding_dim))
    num_pixels = embedding_pred_flat.shape[0]
    embedding_pred_flat = StandardScaler().fit_transform(embedding_pred_flat)
    pca = PCA(n_components=3)
    pc_flat = pca.fit_transform(embedding_pred_flat)
    pc = np.reshape(pc_flat, (img_size, img_size, 3))
    pc = normalize(pc)
    
    # prepare predicted embeddings (front/back)
    show_mask = np.expand_dims(class_mask_int_pred > 0, axis=-1)
    embedding_masked = np.multiply(pc, show_mask)
    
    # show instance mask and predicted embeddings
    random_colors = np.random.random((int(np.max(cluster_all_class)), 3))
    all_instances = np.zeros((img_size, img_size, 3))
    slice_0 = np.zeros((img_size, img_size))
    slice_1 = np.zeros((img_size, img_size))
    slice_2 = np.zeros((img_size, img_size))
    for i in range(int(np.max(cluster_all_class))):
        slice_0[cluster_all_class == i] = random_colors[i, 0]
        slice_1[cluster_all_class == i] = random_colors[i, 1]
        slice_2[cluster_all_class == i] = random_colors[i, 2]
    all_instances[:, :, 0] = slice_0
    all_instances[:, :, 1] = slice_1
    all_instances[:, :, 2] = slice_2

    instance_mask_gt_color = np.zeros((img_size, img_size, 3))
    for i in np.unique(instance_mask_gt):
        instance_mask_gt_color[instance_mask_gt == i] = np.random.random((3))
    instance_mask_gt_color_vertical = np.zeros((img_size*2, img_size, 3))
    instance_mask_gt_color_vertical[:img_size, :, :] = instance_mask_gt_color[:, :img_size, :]
    instance_mask_gt_color_vertical[img_size:, :, :] = instance_mask_gt_color[:, img_size:, :]

    class_mask_int_pred_color = np.zeros((img_size, img_size, 3))
    class_mask_int_pred_color[:, :, 0] = class_mask_int_pred/3
    class_mask_int_pred_color[:, :, 1] = class_mask_int_pred/3
    class_mask_int_pred_color[:, :, 2] = class_mask_int_pred/3

    board = np.zeros((img_size, img_size*6, 3))
    board[:, :(img_size), :] = image
    board[:, (img_size):(img_size*2), :] = pc
    board[:, (img_size*2):(img_size*3), :] = class_mask_int_pred_color
    board[:, (img_size*3):(img_size*4), :] = embedding_masked
    board[:, (img_size*4):(img_size*5), :] = all_instances
    board[:, (img_size*5):(img_size*6), :] = instance_mask_gt_color

    plt.figure(figsize=(4 * 6, 4))
    plt.imshow(board)
