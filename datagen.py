import os
import pickle

import cv2
import numpy as np
from PIL import Image

from shapes import get_image_from_shapes, get_flow_from_shapes, get_shapes
from utils import consecutive_integer, totuple


class ShapeDataGenerator:
    def __init__(self, num_shape, image_size):
        self.num_shape = num_shape
        self.image_size = image_size
        self.shapes = None

    def init_shapes(self):
        shape_choices = [1, 2, 3]
        shape_types = np.random.choice(
            shape_choices, size=(self.num_shape), replace=True)
        self.shapes = get_shapes(shape_types, self.image_size)
    
    def render_frame(self):
        image_info = get_image_from_shapes(self.shapes, self.image_size)
        return image_info


class ImageDataGenerator(ShapeDataGenerator):
    def __init__(self, num_shape, image_size):
        ShapeDataGenerator.__init__(self, num_shape, image_size)

    def get_image(self):
        self.init_shapes()
        image_info = self.render_frame()
        return image_info


class SequenceDataGenerator(ShapeDataGenerator):
    def __init__(self, num_shape, image_size, sequence_len):
        ShapeDataGenerator.__init__(self, num_shape, image_size)
        self.sequence_len = sequence_len
        self.shapes = None

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
                if (x1 < 0.1 * self.image_size and dx < 0) or (
                    x1 > 0.9 * self.image_size and dx > 0):
                    dx = -dx
                if (y1 < 0.1 * self.image_size and dy < 0) or (
                    y1 > 0.9 * self.image_size and dy > 0):
                    dy = -dy
            else:
                corners = shape_info['corners']
                x_min = np.min(corners[:, 0])
                x_max = np.max(corners[:, 0])
                y_min = np.min(corners[:, 1])
                y_max = np.max(corners[:, 1])
                if (x_min < 0.1 * self.image_size and dx < 0) or (
                    x_max > 0.9 * self.image_size and dx > 0):
                    dx = -dx
                if (y_min < 0.1 * self.image_size and dy < 0) or (
                    y_max > 0.9 * self.image_size and dy > 0):
                    dy = -dy
            self.shapes[i]['velocity'] = [dx, dy]

    
    def render_flow(self):
        flow = get_flow_from_shapes(self.shapes, self.image_size)
        return flow


    def get_sequence(self):
        sequence = []
        optical_flow = []
        self.init_shapes()
        self.get_velocities()
        for _ in range(self.sequence_len):
            self.move()
            flow = self.render_flow()
            image_info = self.render_frame()
            sequence.append(image_info)
            optical_flow.append(flow)
            self.bounce()
        return sequence, optical_flow
