import os
import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils import consecutive_integer, totuple

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
    identity         = shape_info['identity']
    x_shift, y_shift = shape_info['offset']
    shape_choice_int = shape_info['shape_choice_int']

    draw_img         = draws['draw_img']
    draw_mask        = draws['draw_mask']
    draw_class_mask  = draws['draw_class_mask']
    draw_identity    = draws['draw_identity']

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
        draw_ellipse(draw=draw_identity, bbox=bbox, linewidth=line_width,
                     image_or_mask='mask', mask_value=int(identity))
    else:
        corners = shape_info['corners']
        shape_tuple = totuple(corners)
        draw_img.polygon(xy=shape_tuple, fill=(255, 255, 255), outline=0)
        draw_img.line(xy=shape_tuple, fill=(0, 0, 0), width=line_width)
        round_corners(draw_img, shape_tuple, line_width)
        draw_mask.polygon(xy=shape_tuple, fill=counter, outline=counter)
        draw_class_mask.polygon(xy=shape_tuple, fill=int(shape_choice_int), 
            outline=int(shape_choice_int))
        draw_identity.polygon(xy=shape_tuple, fill=int(identity), outline=int(identity))

    new_draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask,
        'draw_identity': draw_identity
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
    class_mask    = Image.new(mode='I',   size=(image_size, image_size), color=0)
    identity_mask = Image.new(mode='I',   size=(image_size, image_size), color=0)

    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    draw_class_mask = ImageDraw.Draw(class_mask)
    draw_identity = ImageDraw.Draw(identity_mask)

    draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask,
        'draw_identity': draw_identity
    }

    counter = 1
    num = len(shapes)
    velocities = np.zeros((num, 2))
    for i in range(num):
        shape_info = shapes[i]
        velocities[i, :] = shape_info['velocity']
        draws = draw_shapes(shape_info, draws, counter)
        counter = counter + 1

    image = np.asarray(img) / 255.0
    mask = np.asarray(mask)
    mask = consecutive_integer(mask)
    class_mask = np.asarray(class_mask)
    identity_mask = np.asarray(identity_mask)

    image_info = {
        'image':         image,
        'instance_mask': mask,
        'class_mask':    class_mask,
        'identity_mask': identity_mask,
        'velocities':    velocities
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
        shape_tuple = totuple(corners)
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