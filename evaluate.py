import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from postprocessing import embedding_to_instance

from utils import *


def principal_component_analysis(embedding_pred, embedding_dim):
    width, height, _ = embedding_pred.shape
    embedding_pred_flat = np.reshape(embedding_pred, (-1, embedding_dim))
    embedding_pred_flat = StandardScaler().fit_transform(embedding_pred_flat)
    pca = PCA(n_components=3)
    pc_flat = pca.fit_transform(embedding_pred_flat)
    pc = np.reshape(pc_flat, (width, height, 3))
    pc = normalize(pc)

    return pc


def colorize_instances(instance_masks):
    # show instance mask and predicted embeddings
    width, height = instance_masks.shape
    instances_color = np.zeros((width, height, 3))
    # slice_0 = np.zeros((width, height))
    # slice_1 = np.zeros((width, height))
    # slice_2 = np.zeros((width, height))
    
    num_instances = int(np.max(instance_masks))
    random_colors = np.random.rand(num_instances, 3)

    for i in range(num_instances):
        instances_color[instance_masks == i] = random_colors[i, :]
    #     slice_0[instance_masks == i] = random_colors[i, 0]
    #     slice_1[instance_masks == i] = random_colors[i, 1]
    #     slice_2[instance_masks == i] = random_colors[i, 2]
    # instances_color[:, :, 0] = slice_0
    # instances_color[:, :, 1] = slice_1
    # instances_color[:, :, 2] = slice_2

    return instances_color


def colorize_class_mask(class_mask_int, class_num):
    class_max = class_num - 1
    width, height = class_mask_int.shape
    class_mask_int_color = np.zeros((width, height, 3))
    class_mask_int_color[:, :, 0] = class_mask_int/class_max
    class_mask_int_color[:, :, 1] = class_mask_int/class_max
    class_mask_int_color[:, :, 2] = class_mask_int/class_max

    return class_mask_int_color


def visualize(embedding_pred, embedding_dim, output_size, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_int_gt, image):
    # pca on embedding purely for visualization, not for clustering
    pc = principal_component_analysis(embedding_pred, embedding_dim)

    # prepare predicted embeddings (front/back)
    show_mask = np.expand_dims(class_mask_int_pred > 0, axis=-1)
    embedding_masked = np.multiply(pc, show_mask)

    instance_mask_pred_color = colorize_instances(cluster_all_class)
    instance_mask_gt_color   = colorize_instances(instance_mask_gt)

    # instance_mask_gt_color = np.zeros((output_size, output_size, 3))
    # for i in np.unique(instance_mask_gt):
    #     instance_mask_gt_color[instance_mask_gt == i] = colors[int(i), :]

    class_mask_int_pred_color = colorize_class_mask(class_mask_int_pred, class_num)
    class_mask_int_gt_color   = colorize_class_mask(class_mask_int_gt, class_num)

    image = cv2.resize(image, (output_size, output_size))
    image = (image + 1)/2
    board = np.zeros((output_size, output_size*7, 3))
    board[:, (output_size*0):(output_size*1), :] = image
    board[:, (output_size*1):(output_size*2), :] = pc
    board[:, (output_size*2):(output_size*3), :] = embedding_masked
    board[:, (output_size*3):(output_size*4), :] = instance_mask_pred_color
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

    outputs = model.predict(x)
    class_mask_pred = outputs[0, :, :, :class_num]
    embedding_pred  = outputs[0, :, :, class_num:(class_num + embedding_dim)]
    class_mask_int_pred = np.argmax(class_mask_pred, axis=-1)
    cluster_all_class = embedding_to_instance(embedding_pred, class_mask_pred, params)
    image = np.squeeze(x)
    class_mask_gt    = y[0, ..., 0]
    instance_mask_gt = y[0, ..., 1]
    visualize(embedding_pred, embedding_dim, output_size, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_gt, image)


def eval_pair(model, pair, params):
    class_num                = params.NUM_CLASSES
    embedding_dim            = params.EMBEDDING_DIM
    output_size              = params.OUTPUT_SIZE
    image_size               = params.IMG_SIZE

    images = np.zeros((image_size, image_size*2, 3))
    board = np.zeros((output_size*2, output_size*5, 3))
    image_info, prev_image_info = pair

    image                 = image_info['image']
    identity_mask_gt      = image_info['identity_mask']
    class_mask_gt         = image_info['class_mask']

    prev_image            = prev_image_info['image']
    prev_identity_mask_gt = prev_image_info['identity_mask']
    prev_class_mask_gt    = prev_image_info['class_mask']
    optical_flow_gt       = prev_image_info['optical_flow']

    images[:, :image_size, :] = image
    images[:, image_size:, :] = prev_image

    x, y = prep_double_frame(image_info, prev_image_info, params)
    outputs = model.predict(x)
    outputs = np.squeeze(outputs)

    class_mask_pred      = outputs[:, :, :class_num]
    prev_class_mask_pred = outputs[:, :, class_num:(class_num*2)]
    embedding_pred       = outputs[:, :, (class_num*2):(class_num*2 + embedding_dim)]
    prev_embedding_pred  = outputs[:, :, (class_num*2 + embedding_dim):((class_num*2 + embedding_dim*2))]
    optical_flow_pred    = outputs[:, :, (class_num*2 + embedding_dim*2):]

    combined_class_mask_pred = np.zeros((output_size, output_size*2))
    combined_embeding_pred   = np.zeros((output_size, output_size*2, embedding_dim))

    # argmax
    class_mask_pred_int = np.argmax(class_mask_pred, axis = -1)
    prev_class_mask_pred_int = np.argmax(prev_class_mask_pred, axis = -1)
    # resize to output_size
    class_mask_gt         = resize_img(class_mask_gt, output_size, output_size)
    prev_class_mask_gt    = resize_img(prev_class_mask_gt, output_size, output_size)
    identity_mask_gt      = resize_img(identity_mask_gt, output_size, output_size)
    prev_identity_mask_gt = resize_img(prev_identity_mask_gt, output_size, output_size)
    optical_flow_gt       = resize(optical_flow_gt, [output_size, output_size])

    combined_class_mask_pred[:, :output_size] = class_mask_pred_int
    combined_class_mask_pred[:, output_size:] = prev_class_mask_pred_int
    combined_embeding_pred[:, :output_size, :] = embedding_pred
    combined_embeding_pred[:, output_size:, :] = prev_embedding_pred

    cluster_all_class = embedding_to_instance(combined_embeding_pred, combined_class_mask_pred, params)
    identity_mask_pred      = cluster_all_class[:, :output_size]
    prev_identity_mask_pred = cluster_all_class[:, output_size:]

    # colorize for visualization
    combined_identity_mask_gt       = np.zeros((output_size, output_size*2))
    combined_identity_mask_gt[:, :output_size] = identity_mask_gt
    combined_identity_mask_gt[:, output_size:] = prev_identity_mask_gt
    combined_identity_mask_gt_color = colorize_instances(combined_identity_mask_gt)
    identity_mask_gt_color          = combined_identity_mask_gt_color[:, :output_size, :]
    prev_identity_mask_gt_color     = combined_identity_mask_gt_color[:, output_size:, :]
    class_mask_gt_color             = colorize_class_mask(class_mask_gt, class_num)
    prev_class_mask_gt_color        = colorize_class_mask(prev_class_mask_gt, class_num)
    optical_flow_gt_color           = flow_to_rgb(optical_flow_gt)

    identity_mask_pred_color        = colorize_instances(identity_mask_pred)
    prev_identity_mask_pred_color   = colorize_instances(prev_identity_mask_pred)
    class_mask_pred_color           = colorize_class_mask(class_mask_pred_int, class_num)
    prev_class_mask_pred_color      = colorize_class_mask(prev_class_mask_pred_int, class_num)
    optical_flow_pred_color         = flow_to_rgb(optical_flow_pred)

    board[:output_size, :output_size]                    = identity_mask_gt_color
    board[:output_size, output_size:(output_size*2)]     = class_mask_gt_color
    board[:output_size, (output_size*2):(output_size*3)] = prev_identity_mask_gt_color
    board[:output_size, (output_size*3):(output_size*4)] = prev_class_mask_gt_color
    board[:output_size, (output_size*4):(output_size*5)] = optical_flow_gt_color
    
    board[output_size:, :output_size]                    = identity_mask_pred_color
    board[output_size:, output_size:(output_size*2)]     = class_mask_pred_color
    board[output_size:, (output_size*2):(output_size*3)] = prev_identity_mask_pred_color
    board[output_size:, (output_size*3):(output_size*4)] = prev_class_mask_pred_color
    board[output_size:, (output_size*4):(output_size*5)] = optical_flow_pred_color

    plt.figure(figsize=(2*2, 2*2))
    plt.imshow(images)

    plt.figure(figsize=(2*5, 2*2))
    plt.imshow(board)
    plt.show()

