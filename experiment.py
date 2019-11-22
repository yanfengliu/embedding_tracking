import os

import motmetrics as mm
import numpy as np
from IPython.display import clear_output

import dataset
import postprocessing as pp
import utils
import visual
from datagen import SequenceDataGenerator
from embedding_model import SequenceEmbeddingModel, sequence_loss_with_params


class Experiment:
    def __init__(self, params):
        self.params = params
        utils.mkdir_if_missing(self.params.MODEL_SAVE_DIR)
        self.model_full_path = os.path.join(
            self.params.MODEL_SAVE_DIR, self.params.MODEL_SAVE_NAME)
        self.val_sdg = SequenceDataGenerator(
            num_shape = self.params.NUM_SHAPE,
            image_size = self.params.IMG_SIZE,
            sequence_len = self.params.SEQUENCE_LEN,
            random_size = True
            )
        self.train_data_loader = dataset.SequenceDataLoader(self.params.TRAIN_SET_PATH)
        self.test_data_loader  = dataset.SequenceDataLoader(self.params.TEST_SET_PATH)
    

    def init_model(self):
        self.model = SequenceEmbeddingModel(self.params)
        optim = Adam(lr = params.LEARNING_RATE)
        loss_function = sequence_loss_with_params(self.params)
        self.model.compile(optim, loss = loss_function)
    

    def save_model(self):
        print(f'saving model at {self.model_full_path}')
        self.model.save(self.model_full_path)
    

    def visual_val(self):
        clear_output(wait=True)
        visual.visualize_history(
            self.loss_history, f'loss, epoch: {self.epoch}, total step: {self.step}')
        sequence = self.val_sdg.get_sequence()
        pair = sequence[0:2]
        visual.eval_pair(self.model, pair, self.params)
    

    def train_on_sequence(self, sequence):
        for i in range(self.params.SEQUENCE_LEN - 1):
            self.step += 1
            [prev_image_info, image_info] = sequence[i:i+2]
            x, y = utils.prep_double_frame(image_info, prev_image_info)
            history = self.model.fit(x, y, batch_size = 1, verbose = False)
            latest_loss = history.history['loss'][-1]
            self.loss_history.append(latest_loss)
            if self.step % self.params.STEP_PER_VISUAL == 0:
                self.visual_val()
    

    def eval(self):
        pass


    def train_and_eval(self):
        self.epoch = 0
        self.step = 0
        self.loss_history = []
        self.metrics = {
            'map': [],
            'mota': []
        }
        for epoch in range(self.params.EPOCHS):
            self.epoch = epoch
            for _ in range(self.params.TRAIN_NUM_SEQ):
                sequence = self.train_data_loader.get_next_sequence()
                self.train_on_sequence(sequence)
            if (epoch + 1) % self.params.EPOCHS_PER_SAVE == 0:
                self.save_model()
                self.eval()
    

    def run(self):
        self.init_model()
        self.train_and_eval()
