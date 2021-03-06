import json
import os
import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils import data

import utils
from datagen import ImageDataGenerator, SequenceDataGenerator


def get_test_videos(params):
    shapes_dir = os.path.join('dataset2', f'{params.NUM_SHAPE}_shapes')
    for i in range(params.TEST_NUM_SEQ):
        print(f'Making video for seq_{i} out of {params.TEST_NUM_SEQ}')
        image_path = f'{shapes_dir}/test/seq_{i}/images'
        video_path = f'{shapes_dir}/test/seq_{i}/video'
        utils.mkdir_if_missing(video_path)
        video_full_path = os.path.join(video_path, 'video.avi')
        utils.images_to_video(image_path, video_full_path, 5, (256, 256))


def fill_image_list(params):
    r"""
    Generate *.train and *.test file for training protocol of 
    "Towards-Realtime-MOT"
    """
    shapes_dir = os.path.join('dataset2', f'{params.NUM_SHAPE}_shapes')
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
    
    dataset_dir = os.path.join(params.GITHUB_DIR, 'embedding_tracking', 'dataset2')
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

    
    def gen_dataset(self, params, dataset_type, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        if dataset_type == 'train':
            num_seq = params.TRAIN_NUM_SEQ
            path = params.TRAIN_SET_PATH
        elif dataset_type == 'val':
            num_seq = params.VAL_NUM_SEQ
            path = params.VAL_SET_PATH
        elif dataset_type == 'test':
            num_seq = params.TEST_NUM_SEQ
            path = params.TEST_SET_PATH
        else:
            raise ValueError('dataset_type must be train, val, or test')

        for i in range(num_seq):
            sdg = SequenceDataGenerator(
                params.NUM_SHAPE, 
                params.IMG_SIZE, 
                params.SEQUENCE_LEN, 
                params.RANDOM_SIZE, 
                params.ROTATE_SHAPES)
            utils.update_progress(i/num_seq)
            seq = sdg.get_sequence()
            pickle_folder_path = os.path.join(path, f'seq_{i}')
            utils.mkdir_if_missing(pickle_folder_path)
            pickle_full_path = os.path.join(pickle_folder_path, 'sequence.pickle')
            with open(pickle_full_path, 'wb') as handle:
                pickle.dump(seq, handle)
            image_count = 0
            for info in seq:
                image = info['image']
                image_folder_path   = os.path.join(path, f'seq_{i}', 'images')
                utils.mkdir_if_missing(image_folder_path)
                image_full_path     = os.path.join(image_folder_path, f'{image_count:05d}.png')
                image = (image*255).astype(np.uint8)
                cv2.imwrite(image_full_path, image)
                image_count += 1


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
        pickle_full_path = os.path.join(self.dataset_path, seq_name, 'sequence.pickle')
        with open(pickle_full_path, 'rb') as handle:
            sequence = pickle.load(handle)
        return sequence


class FastImageDataset(data.Dataset):
  def __init__(self, params):
        self.params = params

  def __len__(self):
        return self.params.STEPS

  def __getitem__(self, index):
        generator = ImageDataGenerator(self.params.NUM_SHAPE, self.params.IMG_SIZE)
        image_info = generator.get_image()
        x, y = utils.prep_single_frame(image_info)
        x = np.squeeze(x)
        y = np.squeeze(y)
        return x, y


class FastSequenceDataset(data.Dataset):
  def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.seq_list = os.listdir(self.dataset_path)
        self.num_seq = len(self.seq_list)

  def __len__(self):
        return self.num_seq * 99

  def __getitem__(self, index):
        seq_id = index // 99
        image_id = index % 99
        seq_name = self.seq_list[seq_id]
        pickle_full_path = os.path.join(self.dataset_path, seq_name, 'sequence.pickle')
        with open(pickle_full_path, 'rb') as handle:
            sequence = pickle.load(handle)
        [prev_image_info, image_info] = sequence[image_id:image_id+2]
        x, y = utils.prep_double_frame(prev_image_info, image_info)
        x = np.squeeze(x)
        y = np.squeeze(y)
        return x, y
