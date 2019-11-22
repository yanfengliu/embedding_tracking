"""
This module provides a unified class that handles the experiment workflow
"""


import os

import keras
import motmetrics as mm
import numpy as np
from IPython.display import clear_output

import datagen
import dataset
import embedding_model
import loss_functions
import postprocessing
import utils
import visual
import eval
import inference


class Experiment:
    def __init__(self, params):
        self.params = params
        utils.mkdir_if_missing(self.params.MODEL_SAVE_DIR)
        self.model_full_path = os.path.join(
            self.params.MODEL_SAVE_DIR, self.params.MODEL_SAVE_NAME)
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
    

    def init_model(self):
        self.model = embedding_model.SequenceEmbeddingModel(self.params)
        optim = keras.optimizers.Adam(lr = self.params.LEARNING_RATE)
        loss_function = loss_functions.sequence_loss_with_params(self.params)
        self.model.compile(optim, loss = loss_function)
        self.inference_model = inference.InferenceModel(self.model, self.params)
    

    def save_model(self):
        print(f'saving model at {self.model_full_path}')
        self.model.save(self.model_full_path)
    

    def visual_val(self):
        clear_output(wait=True)
        visual.visualize_history(
            self.loss_history, f'loss, epoch: {self.epoch}, total step: {self.step}')
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
            if self.step % self.params.STEP_PER_VISUAL == 0:
                self.visual_val()
    
    
    def track_on_sequence(self, sequence):
        tracking_output = []
        for i in range(self.params.SEQUENCE_LEN - 1):
            [prev_image_info, image_info] = sequence[i:i+2]
            x, _ = utils.prep_double_frame(prev_image_info, image_info)
            tracking = self.inference_model.track(x)
            tracking_output.append(tracking)
        return tracking_output


    def eval(self):
        self.evaluator = eval.MaskTrackEvaluator(iou_threshold=self.params.IOU_THRESHOLD)
        self.val_track = []
        for _ in range(self.params.VAL_NUM_SEQ):
            sequence = self.val_data_loader.get_next_sequence()
            tracking_output = self.track_on_sequence(sequence)
            self.val_track.append(tracking_output)
            self.evaluator.eval_on_sequence(tracking_output, sequence)
        overall_scores = self.evaluator.summarize()
        return overall_scores


    def train_and_eval(self):
        self.epoch = 0
        self.step = 0
        self.loss_history = []
        self.metrics = []
        for epoch in range(self.params.EPOCHS):
            self.epoch = epoch
            for _ in range(self.params.TRAIN_NUM_SEQ):
                sequence = self.train_data_loader.get_next_sequence()
                self.train_on_sequence(sequence)
            if (epoch + 1) % self.params.EPOCHS_PER_SAVE == 0:
                self.save_model()
                scores = self.eval()
                self.metrics.append(scores)
    

    def test(self):
        # TODO: test on test set and report motmetrics
        pass


    def report_metrics(self):
        # TODO: fuse scores together and generate report
        pass


    def run(self):
        self.init_model()
        self.train_and_eval()
        self.test()
        self.report_metrics()
