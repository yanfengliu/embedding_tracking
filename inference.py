"""
This module handles the post-processing logic
"""

import postprocessing
import utils
import numpy as np 


class InferenceModel:
    def __init__(self, model, params):
        self.model = model
        self.params = params
    

    def predict(self, x):
        return self.model.predict(x)
    

    def segment(self, x):
        nC = self.params.NUM_CLASSES
        nD = self.params.EMBEDDING_DIM
        OS = self.params.OUTPUT_SIZE
        outputs = self.predict(x)
        outputs = np.squeeze(outputs)
        combined_class_mask_pred = np.zeros((OS, OS*4, nC))
        combined_embedding_pred  = np.zeros((OS, OS*4, nD))
        for i in range(4):
            # channel wise slice copied to horizontal slice
            combined_class_mask_pred[:, (OS*i):(OS*(i+1)), :] = \
                outputs[:, :, :, (nC*0):(nC*(i+1))]
            combined_embedding_pred[:, (OS*i):(OS*(i+1)), :] = \
                outputs[:, :, :, (nC*4+nD*i):(nC*4+nD*(i+1))]
        combined_class_mask_pred_int = np.argmax(combined_class_mask_pred, axis = -1)
        cluster_all_class = postprocessing.embedding_to_instance(
            combined_embedding_pred, 
            combined_class_mask_pred_int, 
            self.params)
        return combined_embedding_pred, combined_class_mask_pred_int, cluster_all_class


    def track(self, x):
        # TODO: use masks to track
        OS = self.params.OUTPUT_SIZE
        _, _, cluster_all_class = self.predict(x)
        num_instance = np.max(cluster_all_class)
        full_masks = []
        for i in range(num_instance):
            cluster_all_class[:, (OS * 0):(OS * 1)]
            cluster_all_class[:, (OS * 1):(OS * 2)]
            cluster_all_class[:, (OS * 2):(OS * 3)]
            cluster_all_class[:, (OS * 3):(OS * 4)]

        return None
