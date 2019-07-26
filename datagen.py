import os
import pickle

import cv2
import numpy as np
from PIL import Image

from shapes import get_image_from_shapes, get_shapes
from utils import consecutive_integer, totuple


class ImageDataGenerator:
    def __init__(self, num_shape, image_size):
        self.num_shape = num_shape
        self.image_size = image_size
    
    def init_shapes(self):
        shape_choices = [1, 2, 3]
        shape_types = np.random.choice(
            shape_choices, size=(self.num_shape), replace=True)
        self.shapes = get_shapes(shape_types, self.image_size)
    
    def render_frame(self):
        image_info = get_image_from_shapes(self.shapes, self.image_size)
        return image_info
    
    def get_image(self):
        self.init_shapes()
        image_info = self.render_frame()
        return image_info


class SequenceDataGenerator:
    def __init__(self, num_shape, image_size, sequence_len):
        self.num_shape = num_shape
        self.image_size = image_size
        self.sequence_len = sequence_len

    def init_shapes(self):
        shape_choices = [1, 2, 3]
        shape_types = np.random.choice(
            shape_choices, size=(self.num_shape), replace=True)
        self.shapes = get_shapes(shape_types, self.image_size)

    def get_velocities(self):
        self.velocities = np.random.randint(
            low=int(-0.1 * self.image_size), high=int(0.1 * self.image_size), size=(self.num_shape, 2))

    def move(self):
        for i in range(self.num_shape):
            velocity = self.velocities[i]
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
            dx, dy = self.velocities[i]
            shape_info = self.shapes[i]
            if shape_info['type'] == 'round':
                x1, y1 = shape_info['x1'], shape_info['y1']
                if x1 < 0 or x1 > self.image_size:
                    dx = -dx
                if y1 < 0 or y1 > self.image_size:
                    dy = -dy
            else:
                corners = shape_info['corners']
                x_min = np.min(corners[:, 0])
                x_max = np.max(corners[:, 0])
                y_min = np.min(corners[:, 1])
                y_max = np.max(corners[:, 1])
                if x_min < 0 or x_max > self.image_size:
                    dx = -dx
                if y_min < 0 or y_max > self.image_size:
                    dy = -dy
            self.velocities[i] = [dx, dy]

    def render_frame(self):
        image_info = get_image_from_shapes(self.shapes, self.image_size)
        return image_info

    def render_sequence(self):
        sequence = []
        self.get_velocities()
        for _ in range(self.sequence_len):
            self.move()
            self.bounce()
            image_info = self.render_frame()
            sequence.append(image_info)
        return sequence

    def get_sequence(self):
        self.init_shapes()
        sequence = self.render_sequence()
        return sequence
