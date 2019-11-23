import motmetrics as mm
import numpy as np


class MaskTrackEvaluator:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
    

    def init_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
    

    def eval_on_sequence(self, hypothesis, gt_sequence):
        # TODO: match hypothesis with objects for MOT metrics
        return None
    

    def summarize(self):
        # TODO: generate overall report
        return None