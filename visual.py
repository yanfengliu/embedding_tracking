import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from postprocessing import embedding_to_instance
import inference
import utils
import matplotlib


def principal_component_analysis(embedding_pred, embedding_dim):
    width, height, _ = embedding_pred.shape
    embedding_pred_flat = np.reshape(embedding_pred, (-1, embedding_dim))
    embedding_pred_flat = StandardScaler().fit_transform(embedding_pred_flat)
    pca = PCA(n_components=3)
    pc_flat = pca.fit_transform(embedding_pred_flat)
    pc = np.reshape(pc_flat, (width, height, 3))
    pc = utils.normalize(pc)

    return pc


def flow_to_rgb(flow):
    # read nonzero optical flow
    image_size = flow.shape[0]
    direction_hsv = np.zeros((image_size, image_size, 3))
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    # define min and max
    mag_max = np.sqrt(2)
    mag_min = 0
    angle_max = np.pi
    angle_min = -np.pi
    angles = np.arctan2(dx, dy)
    magnitudes = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    # convert to hsv
    hue = utils.normalize(angles, [angle_min, angle_max])
    saturation = utils.normalize(magnitudes, [mag_min, mag_max])
    value = np.zeros(angles.shape) + 1
    direction_hsv[:, :, 0] = hue
    direction_hsv[:, :, 1] = saturation
    direction_hsv[:, :, 2] = value
    direction_rgb = matplotlib.colors.hsv_to_rgb(direction_hsv)
    return direction_rgb


def imgs_to_video(images, video_name, fps):
    # assumes `images` contains square images in shape of (x, x, 3)
    height, width = images[0].shape[0:2]
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))
    for image in images:
        video.write(image)
    video.release()
    return


def flows_to_video(flows, video_name, fps):
    # assumes `flows` contains square images in shape of (x, x, 2)
    images = []
    for flow in flows:
        image = flow_to_rgb(flow)
        image = image * 255
        image = image.astype(np.uint8)
        images.append(image)
    imgs_to_video(images, video_name, fps)
    return


def float_to_uint8(data):
    data = data * 255
    data = data.astype(np.uint8)
    return data


def pair_embedding_to_video(sequence, model, params, video_name, fps):
    class_num     = params.NUM_CLASSES
    embedding_dim = params.EMBEDDING_DIM
    image_size    = params.IMG_SIZE
    OS            = params.OUTPUT_SIZE
    boards = []
    for i in range(len(sequence) - 1):
        prev_image = sequence[i]['image']
        image = sequence[i+1]['image']
        board = np.zeros((image_size * 2, image_size * 2, 3))
        x, _ = utils.prep_double_frame(sequence[i], sequence[i+1])
        outputs = model.predict(x)
        outputs = np.squeeze(outputs)
        embedding_pred = outputs[:, :, (class_num*2):(class_num*2 + embedding_dim)]
        prev_embedding_pred = outputs[:, :, (class_num*2 + embedding_dim):((class_num*2 + embedding_dim*2))]
        combined_embedding_pred = np.zeros((OS, OS*2, embedding_dim))
        combined_embedding_pred[:, :OS, :] = prev_embedding_pred
        combined_embedding_pred[:, OS:, :] = embedding_pred
        board[:image_size, :image_size, :] = prev_image
        board[:image_size, image_size:, :] = image
        pc = principal_component_analysis(combined_embedding_pred, embedding_dim)
        pc = utils.resize_img(pc, image_size, image_size*2)
        board[image_size:, image_size:, :] = pc[:, image_size:, :]
        board[image_size:, :image_size, :] = pc[:, :image_size, :]
        board = float_to_uint8(board)
        boards.append(board)
    imgs_to_video(boards, video_name, fps)
    return


def colorize_instances(instance_masks):
    # show instance mask and predicted embeddings
    width, height = instance_masks.shape
    instances_color = np.zeros((width, height, 3))
    
    num_instances = int(np.max(instance_masks))
    random_colors = np.random.rand(num_instances, 3)

    for i in range(num_instances):
        instances_color[instance_masks == i] = random_colors[i, :]

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
    OS = output_size
    # pca on embedding purely for visualization, not for clustering
    pc = principal_component_analysis(embedding_pred, embedding_dim)

    # prepare predicted embeddings (front/back)
    show_mask = np.expand_dims(class_mask_int_pred > 0, axis=-1)
    embedding_masked = np.multiply(pc, show_mask)

    instance_mask_pred_color = colorize_instances(cluster_all_class)
    instance_mask_gt_color   = colorize_instances(instance_mask_gt)

    class_mask_int_pred_color = colorize_class_mask(class_mask_int_pred, class_num)
    class_mask_int_gt_color   = colorize_class_mask(class_mask_int_gt, class_num)

    image = cv2.resize(image, (OS, OS))
    image = (image + 1)/2
    board = np.zeros((OS, OS*7, 3))
    board[:, (OS*0):(OS*1), :] = image
    board[:, (OS*1):(OS*2), :] = pc
    board[:, (OS*2):(OS*3), :] = embedding_masked
    board[:, (OS*3):(OS*4), :] = instance_mask_pred_color
    board[:, (OS*4):(OS*5), :] = instance_mask_gt_color
    board[:, (OS*5):(OS*6), :] = class_mask_int_pred_color
    board[:, (OS*6):(OS*7), :] = class_mask_int_gt_color

    plt.figure(figsize=(4 * 7, 4))
    plt.imshow(board)
    plt.show()


def single_eval(model, x, y, params):
    class_num       = params.NUM_CLASSES
    embedding_dim   = params.EMBEDDING_DIM
    OS              = params.OUTPUT_SIZE

    outputs = model.predict(x)
    class_mask_pred = outputs[0, :, :, :class_num]
    embedding_pred  = outputs[0, :, :, class_num:(class_num + embedding_dim)]
    class_mask_int_pred = np.argmax(class_mask_pred, axis=-1)
    cluster_all_class = embedding_to_instance(embedding_pred, class_mask_pred, params)
    image = np.squeeze(x)
    class_mask_gt    = y[0, ..., 0]
    instance_mask_gt = y[0, ..., 1]
    visualize(embedding_pred, embedding_dim, OS, class_mask_int_pred,
              cluster_all_class, instance_mask_gt, class_num, class_mask_gt, image)


def eval_pair(model, pair, params):
    nC          = params.NUM_CLASSES
    nD          = params.EMBEDDING_DIM
    OS          = params.OUTPUT_SIZE
    image_size  = params.IMG_SIZE

    images  = np.zeros((image_size, image_size*2, 3))
    board   = np.zeros((OS*2, OS*9, 3))

    prev_image_info, image_info = pair
    x, _ = utils.prep_double_frame(prev_image_info, image_info)
    inference_model = inference.InferenceModel(model, params)
    combined_embedding_pred, combined_class_mask_pred_int, cluster_all_class = inference_model.segment(x)

    image                       = image_info['image']
    id_mask_gt                  = image_info['instance_mask']
    occ_id_mask_gt              = image_info['occ_instance_mask']
    class_mask_gt_int           = image_info['class_mask']
    occ_class_mask_gt_int       = image_info['occ_class_mask']
    prev_image                  = prev_image_info['image']
    prev_id_mask_gt             = prev_image_info['instance_mask']
    occ_prev_id_mask_gt         = prev_image_info['occ_instance_mask']
    prev_class_mask_gt_int      = prev_image_info['class_mask']
    occ_prev_class_mask_gt_int  = prev_image_info['occ_class_mask']
    images[:, :image_size, :]   = prev_image
    images[:, image_size:, :]   = image

    combined_class_mask_gt_int      = np.zeros((OS, OS*4))
    combined_id_mask_gt             = np.zeros((OS, OS*4))

    # colorize id masks
    combined_id_mask_gt[:, (OS * 0):(OS * 1)] = id_mask_gt
    combined_id_mask_gt[:, (OS * 1):(OS * 2)] = occ_id_mask_gt
    combined_id_mask_gt[:, (OS * 2):(OS * 3)] = prev_id_mask_gt
    combined_id_mask_gt[:, (OS * 3):(OS * 4)] = occ_prev_id_mask_gt
    combined_id_mask_gt_color   = colorize_instances(combined_id_mask_gt)
    combined_id_mask_pred_color = colorize_instances(cluster_all_class)
    
    # colorize class masks
    combined_class_mask_gt_int[:, (OS * 0):(OS * 1)] = class_mask_gt_int
    combined_class_mask_gt_int[:, (OS * 1):(OS * 2)] = occ_class_mask_gt_int
    combined_class_mask_gt_int[:, (OS * 2):(OS * 3)] = prev_class_mask_gt_int
    combined_class_mask_gt_int[:, (OS * 3):(OS * 4)] = occ_prev_class_mask_gt_int
    combined_class_mask_gt_color    = colorize_class_mask(combined_class_mask_gt_int, nC)
    combined_class_mask_pred_color  = colorize_class_mask(combined_class_mask_pred_int, nC)

    # colorize embeddings
    pc = principal_component_analysis(combined_embedding_pred, nD)
    show_mask = np.expand_dims(combined_class_mask_pred_int > 0, axis=-1)
    emb_masked = np.multiply(pc, show_mask)

    # fill the display board
    board[:OS, (OS * 0):(OS * 4), :] = combined_id_mask_gt_color
    board[OS:, (OS * 0):(OS * 4), :] = combined_id_mask_pred_color
    board[:OS, (OS * 4):(OS * 8), :] = combined_class_mask_gt_color
    board[OS:, (OS * 4):(OS * 8), :] = combined_class_mask_pred_color

    # show visulizations
    plt.figure(figsize=(2*2, 2*2))
    plt.imshow(images)

    plt.figure(figsize=(2*9, 2*2))
    plt.imshow(board)

    plt.figure(figsize=(2*4, 2*2))
    plt.imshow(emb_masked)

    plt.show()