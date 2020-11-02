import torch
import torch.nn as nn
from .utilities import iou_function
from ..models.utilities import convert_box_chw_to_vertices
from ..models.full import ANCHORS

class LossFunctions:
    def __init__(self, ground_truth, inference):
        self.ground_truth = ground_truth
        self.inference = inference

        self.objectness_thr = 0.5  # According to Mask R-CNN paper

        self.first_stage_num_samples = 256  # According to EfficientPS paper

    def ss_loss_max_pooling(self):
        logits = self.inference.semantic_logits
        (batch, num_classes, height, width) = logits.shape
        nlll = nn.NLLLoss(reduction='none')
        loss = nlll(logits, self.ground_truth.get_label_IDs())
        
        values, ind = torch.sort(loss.view(batch,-1), 1)

        middle = int(val.shape[1] / 2)
        top_quartile = int(middle + middle/2)

        top_quartile_values = val[:, top_quartile]
        mask = loss.ge(top_quartile_values.unsqueeze(-1).unsqueeze(-1)) * 4 / height / width

        return torch.sum(loss*mask)/batch

    def objectness_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        # How many samples, positive and negative, per bbox we sample
        num_samples_per_box_stage = self.first_stage_num_samples // len(gt_bb) // 4

        objectness_loss = 0
        for bb in gt_bb:
            for i in range(4):
                level_primitives = self.primitive_anchors[i]
                primitive_bb = level_primitives.anchors
                primitive_obj = level_primitives.objectness
                samples = random.sample(range(primitive_bb[0].shape[0]), num_samples_per_box_stage)

                edges_bb = convert_box_chw_to_vertices(primitive_bb[0])
                gt_objectness = iou_function(edges_bb, bb["box"]).ge(self.objectness_thr)
                partial_objectness_loss = gt_objectness * torch.log(primitive_obj) + (1 - gt_objectness) * torch.log(1 - primitive_obj)
                objectness_loss += partial_objectness_loss[samples]

                positive_indeces = gt_objectness[:,0]

                self._object_proposal_loss(level_primitives, bb["box"], positive_indeces)

        return loss

    def _object_proposal_loss(self, level_primitives, gt_bbox, positive_indeces):
        level_primitives.anchors

if __name__ == "__main__":
