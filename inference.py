"""
This module handles the post-processing logic
"""

import postprocessing


class InferenceModel:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    

    def predict(self, x):
        return self.model.predict(x)
    

    def segment(self, x):
        raw_output = self.predict(x)
        # TODO: use post-processing to get amodal instance segmentation
        return None


    def track(self, x):
        # TODO: use masks to track
        return None
