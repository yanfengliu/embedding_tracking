"""
This module provides a unified class that handles the experiment workflow
"""


import os
import time

import keras
import keras.backend as K
import motmetrics as mm
import numpy as np
from IPython.display import clear_output

import datagen
import dataset
import embedding_model
import eval
import inference
import loss_functions
import postprocessing
import utils
import visual


class Experiment:
    def __init__(self, params):
        self.params = params
        self.starting_time = time.time()
        self.elapsed_time = 0
        utils.mkdir_if_missing(self.params.MODEL_SAVE_DIR)
        self.val_datagen = datagen.SequenceDataGenerator(
            num_shape = self.params.NUM_SHAPE,
            image_size = self.params.IMG_SIZE,
            sequence_len = self.params.SEQUENCE_LEN,
            random_size = True
            )
        self.train_data_loader = dataset.SequenceDataLoader(
            dataset_path=self.params.TRAIN_SET_PATH, shuffle=True)
        self.val_data_loader   = dataset.SequenceDataLoader(
            dataset_path=self.params.VAL_SET_PATH,   shuffle=False)
        self.test_data_loader  = dataset.SequenceDataLoader(
            dataset_path=self.params.TEST_SET_PATH,  shuffle=False)
    

    def load_latest_weight(self):
        save_files = os.listdir(self.params.MODEL_SAVE_DIR)
        self.latest_saved_epoch = 0
        if len(save_files) > 0:
            for filename in save_files:
                saved_epoch = int(filename.split('.')[0])
                if saved_epoch > self.latest_saved_epoch:
                    self.latest_saved_epoch = saved_epoch
        self.model_full_path = os.path.join(
            self.params.MODEL_SAVE_DIR,
            f'{self.latest_saved_epoch}.h5'
        )
        if len(save_files) > 0:
            print(f'Loading weights from {self.model_full_path}')
            self.model.load_weights(self.model_full_path)


    def init_model(self):
        self.model = embedding_model.SequenceEmbeddingModel(self.params)
        optim = keras.optimizers.Adam(lr = self.params.LEARNING_RATE)
        loss_function = loss_functions.sequence_loss_with_params(self.params)
        self.model.compile(optim, loss = loss_function)
        self.load_latest_weight()
        self.inference_model = inference.InferenceModel(self.model, self.params)
    

    def save_model(self):
        self.model_full_path = os.path.join(
            self.params.MODEL_SAVE_DIR,
            f'{self.epoch}.h5'
        )
        print(f'saving model at {self.model_full_path}')
        self.model.save_weights(self.model_full_path)
    

    def visual_val(self):
        clear_output(wait=True)
        self.elapsed_time = int(time.time() - self.starting_time)
        utils.visualize_history(
            self.loss_history, 
            f'loss, epoch: {self.epoch}, total step: {self.step}, total time: \
                {self.elapsed_time}, learning_rate: {self.get_learning_rate()}')
        sequence = self.val_datagen.get_sequence()
        pair = sequence[0:2]
        visual.eval_pair(self.model, pair, self.params)
    

    def train_on_sequence(self, sequence):
        for i in range(self.params.SEQUENCE_LEN - 1):
            self.step += 1
            [prev_image_info, image_info] = sequence[i:i+2]
            x, y = utils.prep_double_frame(prev_image_info, image_info)
            history = self.model.fit(x, y, batch_size = 1, verbose = False)
            latest_loss = history.history['loss'][-1]
            self.loss_history.append(latest_loss)
            if self.step % self.params.STEPS_PER_VISUAL == 0:
                self.visual_val()


    def eval(self, data_loader):
        print('Evaluating model')
        evaluator = eval.MaskTrackEvaluator(iou_threshold=self.params.IOU_THRESHOLD)
        for i in range(data_loader.num_seq):
            print(f'Sequence {i+1}/{data_loader.num_seq}')
            sequence = data_loader.get_next_sequence()
            tracks = self.inference_model.track_on_sequence(sequence)
            evaluator.eval_on_sequence(tracks, sequence)
        strsummary = evaluator.summarize()
        return strsummary


    def write_summary(self, strsummary, eval_type):
        txt_path = f'summary/{self.params.FEATURE_STRING}_{eval_type}.txt'
        print(f'Writing metrics summary to {txt_path}')
        with open(txt_path, "a") as f:
            f.write(f'Epoch: {self.epoch} \n')
            f.write(f'{strsummary} \n')
    

    def validate(self):
        strsummary = self.eval(self.val_data_loader)
        self.write_summary(strsummary, 'val')


    def test(self):
        if not hasattr(self, 'epoch'):
            raise AttributeError("There is no attribute 'epoch'")
        strsummary = self.eval(self.test_data_loader)
        self.write_summary(strsummary, 'test')
    

    def update_learning_rate(self):
        # only update once when 50% training is complete
        if self.epoch == int(0.5 * self.params.EPOCHS):
            K.set_value(self.model.optimizer.lr, 0.1 * self.params.LEARNING_RATE)
    

    def get_learning_rate(self):
        return K.get_value(self.model.optimizer.lr)


    def train_val_save(self):
        self.epoch = 0
        self.step = 0
        self.loss_history = []
        for epoch in range(self.latest_saved_epoch, self.params.EPOCHS):
            print(f'Training epoch {epoch+1}/{self.params.EPOCHS}')
            print(f'Learning rate: {self.get_learning_rate()}')
            self.epoch = epoch
            self.update_learning_rate()
            for _ in range(self.params.TRAIN_NUM_SEQ):
                sequence = self.train_data_loader.get_next_sequence()
                self.train_on_sequence(sequence)
            if (epoch + 1) % self.params.EPOCHS_PER_SAVE == 0:
                self.save_model()
                self.validate()


    def run(self):
        self.init_model()
        self.train_val_save()
        self.test()
