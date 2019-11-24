import json
import os
import pickle
import random

import cv2
import numpy as np

import utils
from datagen import SequenceDataGenerator


def fill_image_list(params):
    r"""
    Generate *.train and *.test file for training protocol of 
    "Towards-Realtime-MOT"
    """
    shapes_dir = os.path.join('dataset', f'{params.NUM_SHAPE}_shapes')
    for i in range(params.TRAIN_NUM_SEQ):
        train_image_list_path = os.path.join(shapes_dir, f'seq_{i}.train')
        f = open(train_image_list_path, 'w')
        image_list = os.listdir(f'{shapes_dir}/train/seq_{i}/images')
        for image_name in image_list:
            f.write(os.path.join('train', f'seq_{i}', 'images', image_name) + '\n')
        f.close()
    for i in range(params.TEST_NUM_SEQ):
        test_image_list_path = f'{shapes_dir}/seq_{i}.test'
        f = open(test_image_list_path, 'w')
        image_list = os.listdir(f'{shapes_dir}/test/seq_{i}/images')
        for image_name in image_list:
            f.write(os.path.join('test', f'seq_{i}', 'images', image_name) + '\n')
        f.close()


def gen_ccmcpe(params):
    r"""
    Create and fill the ccmcpe.json file to specify sets used for training and testing.
    Used for "Towards-Realtime-MOT"
    """
    ccmcpe = dict()
    
    dataset_dir = os.path.join(params.GITHUB_DIR, 'embedding_tracking', 'dataset')
    ccmcpe['root'] = os.path.join(dataset_dir, f'{params.NUM_SHAPE}_shapes')
    # list train sets
    train_seq_dict = dict()
    for i in range(params.TRAIN_NUM_SEQ):
        train_seq_path = os.path.join(dataset_dir, f'{params.NUM_SHAPE}_shapes/seq_{i}.train')
        train_seq_dict[f'seq_{i}'] = train_seq_path
    ccmcpe['train'] = train_seq_dict
    # list test sets
    test_seq_dict = dict()
    for i in range(params.TEST_NUM_SEQ):
        test_seq_path = os.path.join(dataset_dir, f'{params.NUM_SHAPE}_shapes/seq_{i}.test')
        test_seq_dict[f'seq_{i}'] = test_seq_path
    ccmcpe['test'] = test_seq_dict

    ccmcpe_json_path = os.path.join(params.GITHUB_DIR, 'Towards-Realtime-MOT', 'cfg', 'ccmcpe.json')
    with open(ccmcpe_json_path, 'w') as f:
        json_str = json.dumps(ccmcpe)
        f.write(json_str)


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
                image_folder_path   = os.path.join(path, f'seq_{i}', 'images')
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
                    # class_int = info['classes'][j]
                    class_int = 0
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


class SequenceDataLoader():
    def __init__(self, dataset_path, shuffle=False):
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.seq_list = os.listdir(self.dataset_path)
        np.random.shuffle(self.seq_list)
        self.num_seq = len(self.seq_list)
        self.current_seq = 0
    
    
    def get_next_sequence(self):
        if self.current_seq < self.num_seq - 1:
            self.current_seq += 1
        else:
            self.current_seq = 0
            if self.shuffle:
                np.random.shuffle(self.seq_list)
        seq_name = self.seq_list[self.current_seq]
        print(f'Next sequence: {seq_name}')
        pickle_full_path = os.path.join(self.dataset_path, seq_name, 'sequence.pickle')
        with open(pickle_full_path, 'rb') as handle:
            sequence = pickle.load(handle)
        return sequence