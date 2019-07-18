import os
import pickle

import cv2
import numpy as np
from keras.utils import to_categorical
from PIL import Image, ImageDraw
from scipy.special import binom
from skimage.transform import resize

from shapes import get_shapes, get_image_from_shapes
from utils import consecutive_integer, totuple


class DataGenerator:
    def __init__(self, num_shape, image_size, sequence_len):
        self.num_shape = num_shape
        self.image_size = image_size
        self.sequence_len = sequence_len
        self.init_shapes()

    def init_shapes(self):
        shape_choices = [1, 2, 3]
        shape_types = np.random.choice(
            shape_choices, size=(self.num_shape), replace=True)
        self.shapes = get_shapes(shape_types, self.image_size)

    def add_offset(self, offsets):
        for i in range(len(offsets)):
            offset = offsets[i]
            dx, dy = offset
            shape_info = self.shapes[i]
            shape_info['offset'] += offset
            try:
                shape_info['x1'] += dx
                shape_info['y1'] += dy
            except:
                shape_info['corners'] += offset

    def render_frame(self):
        image_info = get_image_from_shapes(self.shapes, self.image_size)
        return image_info

    def render_sequence(self):
        sequence = []
        velocities = np.random.randint(
            low=int(-0.05 * self.image_size), high=int(0.05 * self.image_size), size=(self.num_shape, 2))
        for _ in range(self.sequence_len):
            self.add_offset(velocities)
            image_info = self.render_frame()
            sequence.append(image_info)
        return sequence
