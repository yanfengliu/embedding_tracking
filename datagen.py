import random

import numpy as np
from PIL import Image

import shapes
import utils

min_dist = 0
max_dist = 1

class ShapeDataGenerator:
    def __init__(self, num_shape, image_size, shape_sizes=None):
        self.num_shape = num_shape
        self.image_size = image_size
        if shape_sizes is None:
            self.shape_sizes = np.ones(shape=(num_shape, )) * image_size
        else:
            self.shape_sizes = shape_sizes
        self.shapes = None

    def init_shapes(self):
        shape_choices = [1, 2, 3]
        shape_types = np.random.choice(
            shape_choices, size=(self.num_shape), replace=True)
        self.shapes = shapes.get_shapes(shape_types, self.image_size, self.shape_sizes)
    
    def render_frame(self):
        image_info = shapes.get_image_from_shapes(self.shapes, self.image_size)
        return image_info


class ImageDataGenerator(ShapeDataGenerator):
    def __init__(self, num_shape, image_size):
        ShapeDataGenerator.__init__(self, num_shape, image_size)

    def get_image(self):
        self.init_shapes()
        image_info = self.render_frame()
        return image_info


class SequenceDataGenerator(ShapeDataGenerator):
    def __init__(self, num_shape, image_size, sequence_len, random_size=False, rotate_shapes=False):
        if random_size:
            shape_sizes = np.random.rand(num_shape) * 0.2 + 0.8
        else:
            shape_sizes = np.ones(shape=(num_shape, ))
        shape_sizes *= image_size
        ShapeDataGenerator.__init__(self, num_shape, image_size, shape_sizes)
        self.sequence_len = sequence_len
        self.shapes = None
        self.rotate_shapes = rotate_shapes

    def get_velocities(self):
        velocities = np.random.randint(
            low=int(-0.1 * self.image_size), high=int(0.1 * self.image_size), size=(self.num_shape, 2))
        for i in range(self.num_shape):
            self.shapes[i]['velocity'] = velocities[i]

    def move(self):
        for i in range(self.num_shape):
            velocity = self.shapes[i]['velocity']
            dx, dy = velocity
            shape_info = self.shapes[i]
            shape_info['offset'] += velocity
            if shape_info['type'] == 'round':
                shape_info['x1'] += dx
                shape_info['y1'] += dy
            else:
                shape_info['corners'] += velocity

    def bounce(self):
        for i in range(self.num_shape):
            dx, dy = self.shapes[i]['velocity']
            shape_info = self.shapes[i]
            if shape_info['type'] == 'round':
                x1, y1 = shape_info['x1'], shape_info['y1']
                if (x1 < 0.2 * self.image_size and dx < 0) or (
                    x1 > 1.0 * self.image_size and dx > 0):
                    dx = -dx
                if (y1 < 0.2 * self.image_size and dy < 0) or (
                    y1 > 1.0 * self.image_size and dy > 0):
                    dy = -dy
            else:
                corners = shape_info['corners']
                x_min = np.min(corners[:, 0])
                x_max = np.max(corners[:, 0])
                y_min = np.min(corners[:, 1])
                y_max = np.max(corners[:, 1])
                if (x_min < min_dist * self.image_size and dx < 0) or (
                    x_max > max_dist * self.image_size and dx > 0):
                    dx = -dx
                if (y_min < min_dist * self.image_size and dy < 0) or (
                    y_max > max_dist * self.image_size and dy > 0):
                    dy = -dy
            self.shapes[i]['velocity'] = [dx, dy]
    

    def get_rotation_velocity(self):
        self.Rs = []
        for _ in range(self.num_shape):
            if self.rotate_shapes:
                angle = random.randint(-10, 10)
                theta = (np.pi / 180.0) * angle
                R = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            else:
                R = np.array([[1, 0], [0, 1]])
            self.Rs.append(R)


    def rotate(self):
        for i in range(self.num_shape):
            shape_info = self.shapes[i]
            if shape_info['type'] != 'round':
                R = self.Rs[i]
                offset = shape_info['offset']
                corners = shape_info['corners']
                corners = corners - offset
                corners = np.dot(corners, R) + offset
                shape_info['corners'] = corners


    def render_flow(self):
        flow = shapes.get_flow_from_shapes(self.shapes, self.image_size)
        return flow


    def get_sequence(self):
        sequence = []
        self.init_shapes()
        self.get_velocities()
        self.get_rotation_velocity()
        for _ in range(self.sequence_len):
            self.move()
            self.rotate()
            image_info = self.render_frame()
            flow = self.render_flow()
            image_info['optical_flow'] = flow.astype(np.float32)
            sequence.append(image_info)
            self.bounce()
        return sequence
