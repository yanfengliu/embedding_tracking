import motmetrics as mm
import numpy as np
import utils
from target import Target


class MaskTrackEvaluator:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
        self.accs = []
        self.names = []
        self.seq_id = 0
    

    def gen_target_sequence(self, sequence):
        gt_sequence = []
        for i in range(len(sequence)):
            frame = []
            image_info = sequence[i]
            full_masks = image_info['full_masks']
            id = 0
            for full_mask in full_masks:
                frame.append(Target(full_mask, id))
                id += 1
            gt_sequence.append(frame)
        return gt_sequence


    def eval_on_sequence(self, dt_sequence, gt_sequence):
        gt_sequence = self.gen_target_sequence(gt_sequence)
        acc = mm.MOTAccumulator(auto_id=True)
        for i in range(len(dt_sequence)):
            dt_frame = dt_sequence[i]
            gt_frame = gt_sequence[i]
            num_dt = len(dt_frame)
            num_gt = len(gt_frame)
            dt_ids = [x.id for x in dt_frame]
            gt_ids = [x.id for x in gt_frame]
            dist_matrix = np.zeros((num_gt, num_dt))
            for j in range(num_gt):
                for k in range(num_dt):
                    gt_target = gt_frame[j]
                    dt_target = dt_frame[k]
                    dist_matrix[j, k] = utils.iou(gt_target.mask, dt_target.mask)
            acc.update(gt_ids, dt_ids, dist_matrix)
        self.accs.append(acc)
        self.names.append(f'seq_{self.seq_id}')
        self.seq_id += 1
    

    def summarize(self):
        metrics_names = ['num_frames', 'idf1', 'mota']
        print('Matching hypothesis with ground truth')
        print(f'Metrics: {metrics_names}')
        mh = mm.metrics.create()
        summary = mh.compute_many(
            self.accs, 
            metrics=metrics_names, 
            names=self.names,
            generate_overall=False
            )
        strsummary = mm.io.render_summary(
            summary, 
            formatters=mh.formatters, 
            namemap=mm.io.motchallenge_metric_names
        )
        return strsummary