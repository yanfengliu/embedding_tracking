import os
import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw

import utils

int_to_shape = {
    1: "circle",
    2: "triangle",
    3: "rectangle"
}


def get_transform_params(image_size):
    angle = np.random.randint(0, 360)
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    x_shift = np.round((np.random.rand() * 0.8 + 0.1) * image_size)
    y_shift = np.round((np.random.rand() * 0.8 + 0.1) * image_size)
    offset = np.array([x_shift, y_shift])
    return R, offset


def get_circle_center(shape_size):
    radius = 0.25 * shape_size
    x1 = radius
    y1 = radius
    return x1, y1


def get_ellipse_center(shape_size):
    ellipse_x = (np.random.random() * 0.3 + 0.1) * shape_size
    ellipse_y = (np.random.random() * 0.3 + 0.1) * shape_size
    x1 = ellipse_x
    y1 = ellipse_y
    return x1, y1


def get_triangle_corners(shape_size):
    L = 0.3 * shape_size
    corners = [[0, 0], [L, 0], [0.5 * L, 0.866 * L], [0, 0]]
    return corners


def get_star_corners(shape_size):
    L = 0.2 * shape_size
    a = L/(1+np.sqrt(3))
    corners = [[0, L], [L-a, L+a], [L, 2*L], [L+a, L+a],
               [2*L, L], [L+a, L-a], [L, 0], [L-a, L-a], [0, L]]
    return corners


def get_rectangle_corners(shape_size):
    width = 0.1 * shape_size
    height = 0.8 * shape_size
    corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
    return corners


def get_square_corners(shape_size):
    width = 0.3 * shape_size
    height = width
    corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
    return corners


def get_shape(shape_choice_int, image_size, shape_size, identity):
    shape_info = {}
    shape_info['image_size'] = image_size
    shape_info['shape_size'] = shape_size
    shape_info['identity'] = identity
    shape_choice_str = int_to_shape[shape_choice_int]
    shape_info['shape_choice_int'] = shape_choice_int
    shape_info['shape_choice_str'] = shape_choice_str

    R, offset = get_transform_params(image_size)
    x_shift, y_shift = offset
    shape_info['offset'] = offset
    shape_info['rotation'] = R
    if (shape_choice_str == "circle" or shape_choice_str == "ellipse"):
        if shape_choice_str == "circle":
            x1, y1 = get_circle_center(shape_size)
        elif shape_choice_str == "ellipse":
            x1, y1 = get_ellipse_center(shape_size)
        else:
            raise ValueError('Invalid shape_choice_str value')
        shape_info['type'] = 'round'
        shape_info['x1'] = x1 + x_shift
        shape_info['y1'] = y1 + y_shift
    else:
        if (shape_choice_str == "triangle"):
            corners = get_triangle_corners(shape_size)
        elif (shape_choice_str == "star"):
            corners = get_star_corners(shape_size)
        elif (shape_choice_str == "rectangle"):
            corners = get_rectangle_corners(shape_size)
        elif (shape_choice_str == "square"):
            corners = get_square_corners(shape_size)
        else:
            raise ValueError('Invalid shape_choice_str value')

        corners = np.dot(corners, R) + offset
        shape_info['corners'] = corners
        shape_info['type'] = 'polygon'

    return shape_info


def get_shapes(shape_types, image_size, shape_sizes):
    shapes_info = []
    identity = 1
    for i in range(len(shape_types)):
        shape_choice_int = shape_types[i]
        shape_info = get_shape(shape_choice_int, image_size, shape_sizes[i], identity)
        shapes_info.append(shape_info)
        identity += 1

    return shapes_info


def round_corners(draw_img, shape_tuple, line_width):
    for point in shape_tuple:
        draw_img.ellipse((
            point[0] - 0.5*line_width,
            point[1] - 0.5*line_width,
            point[0] + 0.5*line_width,
            point[1] + 0.5*line_width),
            fill=(0, 0, 0))


def draw_shapes(shape_info, draws, counter):
    image_size       = shape_info['image_size']
    x_shift, y_shift = shape_info['offset']
    shape_choice_int = shape_info['shape_choice_int']

    draw_img         = draws['draw_img']
    draw_mask        = draws['draw_mask']
    draw_class_mask  = draws['draw_class_mask']
    draw_full_mask   = draws['draw_full_mask']

    line_width = int(0.01 * image_size)

    if (shape_info['type'] == 'round'):
        x0 = x_shift
        y0 = y_shift
        x1 = shape_info['x1']
        y1 = shape_info['y1']
        bbox = [x0, y0, x1, y1]

        draw_ellipse(draw=draw_img, bbox=bbox,
                     linewidth=line_width, image_or_mask='image')
        draw_ellipse(draw=draw_mask, bbox=bbox, linewidth=line_width,
                     image_or_mask='mask', mask_value=counter)
        draw_ellipse(draw=draw_class_mask, bbox=bbox, linewidth=line_width,
                     image_or_mask='mask', mask_value=int(shape_choice_int))
        draw_ellipse(draw=draw_full_mask, bbox=bbox, linewidth=line_width,
                     image_or_mask='mask', mask_value=1)
    else:
        corners = shape_info['corners']
        shape_tuple = utils.totuple(corners)
        draw_img.polygon(xy=shape_tuple, fill=(255, 255, 255), outline=0)
        draw_img.line(xy=shape_tuple, fill=(0, 0, 0), width=line_width)
        round_corners(draw_img, shape_tuple, line_width)
        draw_mask.polygon(xy=shape_tuple, fill=counter, outline=counter)
        draw_class_mask.polygon(xy=shape_tuple, fill=int(shape_choice_int), 
            outline=int(shape_choice_int))
        draw_full_mask.polygon(xy=shape_tuple, fill=1, outline=1)

    new_draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask
    }

    return new_draws


def draw_ellipse(draw, bbox, linewidth, image_or_mask, mask_value=None):
    if image_or_mask == 'image':
        for offset, fill in (linewidth/-2.0, 'black'), (linewidth/2.0, 'white'):
            left, top = [(value + offset) for value in bbox[:2]]
            right, bottom = [(value - offset) for value in bbox[2:]]
            draw.ellipse([left, top, right, bottom], fill=fill)
    elif image_or_mask == 'mask':
        offset = linewidth/-2.0
        left, top = [(value + offset) for value in bbox[:2]]
        right, bottom = [(value - offset) for value in bbox[2:]]
        draw.ellipse([left, top, right, bottom], fill=mask_value)

    return draw


def get_image_from_shapes(shapes, image_size):
    img           = Image.new(mode='RGB', size=(image_size, image_size), color=(255, 255, 255))
    mask          = Image.new(mode='I',   size=(image_size, image_size), color=0)
    full_mask     = Image.new(mode='I',   size=(image_size, image_size), color=0)
    class_mask    = Image.new(mode='I',   size=(image_size, image_size), color=0)

    draw_img        = ImageDraw.Draw(img)
    draw_mask       = ImageDraw.Draw(mask)
    draw_full_mask  = ImageDraw.Draw(full_mask)
    draw_class_mask = ImageDraw.Draw(class_mask)

    background_all_ones = np.array([
        (0, 0), (image_size, 0), (image_size, image_size), (0, image_size), (0, 0)])
    corners = utils.totuple(background_all_ones)
    draw_full_mask.polygon([tuple(p) for p in corners], fill=1, outline=1)
    full_masks_list = []
    full_masks_list.append(full_mask)

    draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask
    }
    
    counter = 1
    num_shapes = len(shapes)
    instance_to_class = np.zeros(shape=(num_shapes+1))
    velocities = np.zeros((num_shapes, 2))
    for i in range(num_shapes):
        shape_info = shapes[i]
        if 'velocity' in shape_info:
            velocities[i, :] = shape_info['velocity']
        full_mask = Image.new(mode='I', size=(image_size, image_size), color=0)
        draw_full_mask = ImageDraw.Draw(full_mask)
        draws['draw_full_mask'] = draw_full_mask
        draws = draw_shapes(shape_info, draws, counter)
        full_masks_list.append(full_mask)
        instance_to_class[i+1] = shape_info['shape_choice_int']
        counter = counter + 1

    image = np.asarray(img) / 255.0
    mask = np.asarray(mask)
    class_mask = np.asarray(class_mask)
    num_layer = len(np.unique(mask))
    stacked_full_masks = np.zeros((image_size, image_size, num_layer))
    full_masks = []
    for i in range(num_layer):
        full_mask = np.asarray(full_masks_list[i])
        stacked_full_masks[:, :, i] = full_mask
        if i > 0:
            full_masks.append(full_mask)
    mask_count = np.sum(stacked_full_masks, axis=2)
    occ_mask = np.zeros((image_size, image_size))
    update_idx = (mask_count >= 3)
    obj_idx_array = stacked_full_masks[update_idx]
    obj_idx_pool = np.linspace(0, num_layer-1, num_layer)
    obj_idx = np.multiply(obj_idx_pool, (obj_idx_array).astype(int))
    num_update_pixels = obj_idx.shape[0]
    obj_unique_idx = np.zeros((num_update_pixels, 1))
    for i in range(num_update_pixels):
        obj_unique_idx[i] = np.unique(obj_idx[i, :])[-2]
    occ_mask[update_idx] = np.squeeze(obj_unique_idx)
    occ_class_mask = np.zeros_like(class_mask)
    for i in range(num_layer):
        occ_class_mask[occ_mask == i] = instance_to_class[i]

    identities = [shape_info['identity'] for shape_info in shapes]
    classes = [shape_info['shape_choice_int'] for shape_info in shapes]
    # NOTE: The values of [x_center] [y_center] [width] [height] 
    # are normalized by the width/height of the image, so they are 
    # float numbers ranging from 0 to 1.
    bboxes = []
    for full_mask in full_masks:
        rmin, rmax, cmin, cmax = utils.mask2bbox(full_mask)
        x_center = (cmax + cmin) / (2 * image_size)
        y_center = (rmax + rmin) / (2 * image_size)
        width = (cmax - cmin) / image_size
        height = (rmax - rmin) / image_size
        bbox = [x_center, y_center, width, height]
        bboxes.append(bbox)

    image_info = {
        'image':                image,
        'instance_mask':        mask,
        'occ_instance_mask':    occ_mask,
        'class_mask':           class_mask,
        'occ_class_mask':       occ_class_mask,
        'full_masks':           full_masks,
        'velocities':           velocities,
        'classes':              classes,
        'bboxes':               bboxes
    }

    return image_info


def draw_flow(shape_info, draw):
    image_size       = shape_info['image_size']
    x_shift, y_shift = shape_info['offset']
    dx, dy           = shape_info['velocity']
    # add 100 to velocities to bypass the positive value constraint for draw
    velocity = (100 + dx, 100 + dy, 0)
    line_width = int(0.01 * image_size)

    if (shape_info['type'] == 'round'):
        x0 = x_shift
        y0 = y_shift
        x1 = shape_info['x1']
        y1 = shape_info['y1']
        bbox = [x0, y0, x1, y1]
        draw_ellipse(draw=draw, bbox=bbox, linewidth=line_width,
                     image_or_mask='mask', mask_value=velocity)
    else:
        corners = shape_info['corners']
        shape_tuple = utils.totuple(corners)
        draw.polygon(xy=shape_tuple, fill=velocity, outline=velocity)

    return draw


def get_flow_from_shapes(shapes_info, image_size):
    img = Image.new(mode='RGB', size=(image_size, image_size), color=(0, 0, 0))
    draw_img = ImageDraw.Draw(img)
    num = len(shapes_info)
    for i in range(num):
        shape_info = shapes_info[i]
        draw_img = draw_flow(shape_info, draw_img)
    flow = np.asarray(img, dtype = np.float)
    flow = np.copy(flow[:, :, :2])
    flow[flow != 0] -= 100
    max_v = int(0.1 * image_size)
    flow = flow / max_v
    return flow