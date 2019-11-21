import os
import pickle
import random

import cv2
import numpy as np

import utils
from datagen import SequenceDataGenerator


class SequenceDataset():
    def __init__(self):
        pass

    
    def gen_dataset(self, path, num_seq, num_shape, image_size, sequence_len, random_size, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        num_unique_identities = 0
        for i in range(num_seq):
            sdg = SequenceDataGenerator(num_shape, image_size, sequence_len, random_size)
            utils.update_progress(i/num_seq)
            seq = sdg.get_sequence()
            # save entire sequence to a single pickle file
            pickle_folder_path = os.path.join(path, f'seq_{i}')
            utils.mkdir_if_missing(pickle_folder_path)
            pickle_full_path = os.path.join(pickle_folder_path, 'sequence.pickle')
            with open(pickle_full_path, 'wb') as handle:
                pickle.dump(seq, handle)
            image_count = 0
            for info in seq:
                image = info['image']
                image_folder_path   = os.path.join(path, f'seq_{i}', 'image')
                label_folder_path   = os.path.join(path, f'seq_{i}', 'labels_with_ids')
                utils.mkdir_if_missing(image_folder_path)
                utils.mkdir_if_missing(label_folder_path)
                image_full_path     = os.path.join(image_folder_path, f'{image_count:05d}.png')
                label_full_path     = os.path.join(label_folder_path, f'{image_count:05d}.txt')
                # save image
                image = (image*255).astype(np.uint8)
                cv2.imwrite(image_full_path, image)
                # save annotation txt
                label_txt = open(label_full_path, 'w')
                # [class] [identity] [x_center] [y_center] [width] [height]
                for j in range(len(info['classes'])):
                    class_int = info['classes'][j]
                    # for "towards real-time MOT" identity is from 0 to num_identities-1
                    identity = j + num_unique_identities
                    x_center, y_center, width, height = info['bboxes'][j]
                    ann = f'{class_int} {identity} {x_center} {y_center} {width} {height}\n'
                    label_txt.write(ann)
                label_txt.close() 
                image_count += 1
            # increment the total number of unique identities because they should not
            # be mixed up between sequences
            num_unique_identities += num_shape
