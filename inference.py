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

        class_mask_pred             = outputs[:, :, :, (nC * 0):(nC * 1)]
        occ_class_mask_pred         = outputs[:, :, :, (nC * 1):(nC * 2)]
        prev_class_mask_pred        = outputs[:, :, :, (nC * 2):(nC * 3)]
        occ_prev_class_mask_pred    = outputs[:, :, :, (nC * 3):(nC * 4)]
        embedding_pred              = outputs[:, :, :, (nC * 4 + nD * 0):(nC * 4 + nD * 1)]
        occ_embedding_pred          = outputs[:, :, :, (nC * 4 + nD * 1):(nC * 4 + nD * 2)]
        prev_embedding_pred         = outputs[:, :, :, (nC * 4 + nD * 2):(nC * 4 + nD * 3)]
        occ_prev_embedding_pred     = outputs[:, :, :, (nC * 4 + nD * 3):(nC * 4 + nD * 4)]

        combined_class_mask_pred    = np.zeros((OS, OS*4, nC))
        combined_embedding_pred     = np.zeros((OS, OS*4, nD))

        # fill in value to the combined visualization
        combined_class_mask_pred[:, (OS * 0):(OS * 1), :]   = class_mask_pred         
        combined_class_mask_pred[:, (OS * 1):(OS * 2), :]   = occ_class_mask_pred     
        combined_class_mask_pred[:, (OS * 2):(OS * 3), :]   = prev_class_mask_pred    
        combined_class_mask_pred[:, (OS * 3):(OS * 4), :]   = occ_prev_class_mask_pred
        combined_embedding_pred[:, (OS * 0):(OS * 1), :]    = embedding_pred              
        combined_embedding_pred[:, (OS * 1):(OS * 2), :]    = occ_embedding_pred          
        combined_embedding_pred[:, (OS * 2):(OS * 3), :]    = prev_embedding_pred         
        combined_embedding_pred[:, (OS * 3):(OS * 4), :]    = occ_prev_embedding_pred     

        combined_class_mask_pred_int = np.argmax(combined_class_mask_pred, axis = -1)

        cluster_all_class = postprocessing.embedding_to_instance(
            combined_embedding_pred, 
            combined_class_mask_pred_int, 
            self.params)

        return combined_embedding_pred, combined_class_mask_pred_int, cluster_all_class


    def track(self, x):
        # TODO: use masks to track
        return None
