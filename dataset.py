import numpy as np
import random
import cv2
import os
import utils
from datagen import SequenceDataGenerator


class SequenceDataset():
    def __init__(self, sdg):
        self.sdg = sdg

    
    def gen_dataset(self, path, num_seq, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        for i in range(num_seq):
            seq = self.sdg.get_sequence()
            image_count = 0
            for info in seq:
                image = info['image']
                image = (image*255).astype(np.uint8)
                folder_path = os.path.join(path, f'seq_{i}', 'image')
                utils.mkdir_if_missing(folder_path)
                image_full_path = os.path.join(folder_path, f'{image_count:05d}.png')
                label_full_path = os.path.join(folder_path, f'{image_count:05d}.txt')
                image = image.astype(np.uint8)
                cv2.imwrite(image_full_path, image)
                label_txt = open(label_full_path, 'w')
                # [class] [identity] [x_center] [y_center] [width] [height]
                for j in range(len(info['classes'])):
                    class_int = info['classes'][j]
                    # for "towards real-time MOT" identity is from 0 to num_identities-1
                    identity = j
                    x_center, y_center, width, height = info['bboxes'][j]
                    ann = f'{class_int} {identity} {x_center} {y_center} {width} {height}\n'
                    label_txt.write(ann)
                label_txt.close() 
                image_count += 1