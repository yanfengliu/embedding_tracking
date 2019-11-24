"""
This module handles the detection and tracking logic
"""

import postprocessing
import utils
import numpy as np 
from target import Target


class InferenceModel:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.frames = []
    

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
                outputs[:, :, (nC*i):(nC*(i+1))]
            combined_embedding_pred[:, (OS*i):(OS*(i+1)), :] = \
                outputs[:, :, (nC*4+nD*i):(nC*4+nD*(i+1))]
        combined_class_mask_pred_int = np.argmax(combined_class_mask_pred, axis = -1)
        cluster_all_class = postprocessing.embedding_to_instance(
            combined_embedding_pred, 
            combined_class_mask_pred_int, 
            self.params)
        return combined_embedding_pred, combined_class_mask_pred_int, cluster_all_class


    def get_mask_pair(self, x):
        OS = self.params.OUTPUT_SIZE
        _, _, cluster_all_class = self.segment(x)
        num_instance = int(np.max(cluster_all_class))
        amodal_prev_masks = []
        amodal_masks = []
        for i in range(num_instance):
            mask_id = i + 1
            instance_mask           = cluster_all_class[:, (OS * 0):(OS * 1)]
            occ_instance_mask       = cluster_all_class[:, (OS * 1):(OS * 2)]
            prev_instance_mask      = cluster_all_class[:, (OS * 2):(OS * 3)]
            occ_prev_instance_mask  = cluster_all_class[:, (OS * 3):(OS * 4)]
            amodal_mask = np.logical_or(
                instance_mask == mask_id, 
                occ_instance_mask == mask_id)
            amodal_prev_mask = np.logical_or(
                prev_instance_mask == mask_id, 
                occ_prev_instance_mask == mask_id)
            if (np.sum(amodal_mask) > self.params.MASK_AREA_THRESHOLD and \
                np.sum(amodal_prev_mask) > self.params.MASK_AREA_THRESHOLD):
                amodal_masks.append(amodal_mask)
                amodal_prev_masks.append(amodal_prev_mask)
        return amodal_prev_masks, amodal_masks
    

    def update_track(self, x):
        masks_0, masks_1 = self.get_mask_pair(x)
        # step 1: initialize tracks with every mask in the first frame
        highest_id = -1
        if len(self.frames) == 0:
            frame_0 = []
            frame_1 = []
            for i in range(len(masks_0)):
                mask_0 = masks_0[i]
                mask_1 = masks_1[i]
                id = highest_id + 1
                highest_id += 1
                frame_0.append(Target(mask_0, id))
                frame_1.append(Target(mask_1, id))
            self.frames.append(frame_0)
            self.frames.append(frame_1)
        # step 2: match current frame with previous frame
        else:
            prev_frame = self.frames[-1]
            frame = []
            for target in prev_frame:
                matched = False
                for i in range(len(masks_0)):
                    mask_0 = masks_0[i]
                    mask_1 = masks_1[i]
                    iou = utils.iou(target.mask, mask_0)
                    if iou > self.params.IOU_THRESHOLD:
                        # mask_0 is already tracked in the previous frame
                        # so we only link mask_1
                        linked_target = Target(mask_1, target.id)
                        matched = True
                        frame.append(linked_target)
                        break
                # if there is no match between any previous mask and the
                # new detection, then start a new track
                if not matched:
                    id = highest_id + 1
                    highest_id += 1
                    new_target = Target(mask_1, id)
                    frame.append(new_target)
            self.frames.append(frame)


    def track_on_sequence(self, sequence):
        self.frames = []
        for i in range(len(sequence) - 1):
            [prev_image_info, image_info] = sequence[i:i+2]
            x, _ = utils.prep_double_frame(prev_image_info, image_info)
            self.update_track(x)
        return self.frames
