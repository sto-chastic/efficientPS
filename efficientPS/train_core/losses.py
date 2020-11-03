import random

import torch
import torch.nn as nn

from ..dataset.dataset import DataSet
from ..dataset import LABELS_TO_ID, STUFF, THINGS, THINGS_TO_ID
from ..models import *
from ..models.full import FullModel, PSOutput
from ..models.utilities import (convert_box_chw_to_vertices,
                                convert_box_vertices_to_cwh)
from .utilities import intersection, iou_function

class LossFunctions:
    def __init__(self, ground_truth, inference):
        self.ground_truth = ground_truth
        self.inference = inference

        self.objectness_thr = 0.5  # According to Mask R-CNN paper

        self.first_stage_num_samples = 256   # According to EfficientPS paper
                                             # only sample 256 elements
        self.second_stage_num_samples = 512  # According to EfficientPS paper
                                             # only sample 512 elements

    def ss_loss_max_pooling(self):
        logits = self.inference.semantic_logits
        (batch, num_classes, height, width) = logits.shape
        nlll = nn.NLLLoss(reduction="none")
        loss = nlll(logits, self.ground_truth.get_label_IDs())

        values, ind = torch.sort(loss.view(batch, -1), 1)

        middle = int(val.shape[1] / 2)
        top_quartile = int(middle + middle / 2)

        top_quartile_values = val[:, top_quartile]
        mask = (
            loss.ge(top_quartile_values.unsqueeze(-1).unsqueeze(-1))
            * 4
            / height
            / width
        )

        return torch.sum(loss * mask) / batch

    def roi_proposal_objectness(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        # How many samples, positive and negative, per extraction level
        num_samples_per_stage = (
            self.first_stage_num_samples // 4
        )

        total_loss = 0.0
        for i in range(4):
            objectness = []
            level_primitives = self.inference.primitive_anchors[i]
            primitive_bb = level_primitives.anchors
            primitive_obj = level_primitives.objectness

            for bb in gt_bb:
                edges_bb = convert_box_chw_to_vertices(primitive_bb[0]* level_primitives.scale)
                gt_objectness = iou_function(edges_bb, bb["bbox"]).ge(
                    self.objectness_thr
                )

                objectness.append(
                    gt_objectness
                )

            objectness_stack = torch.stack(objectness)
            objectness_gt = objectness_stack.sum(0).ge(1)

            loss = self._bb_proposal_objectness(objectness_gt, primitive_obj[0])
            
            if objectness_gt.shape[0] > num_samples_per_stage:
                samples = random.sample(
                    range(objectness_gt.shape[0]), num_samples_per_stage
                )  # According to the paper, only sample 256 elements

                total_loss += torch.sum(loss[samples]) / num_samples_per_stage
            else:
                total_loss += torch.sum(loss)

        return total_loss

    @staticmethod
    def _bb_proposal_objectness(gt_objectness, objectness):
        partial_objectness_loss = gt_objectness * torch.log(objectness) + (
            ~gt_objectness
        ) * torch.log(1 - objectness)
        return -partial_objectness_loss


    def roi_proposal_regression(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        # How many samples, positive and negative, per extraction level
        num_samples_per_stage = (
            self.first_stage_num_samples // 4
        )

        total_loss = 0.0
        for i in range(4):
            positive_matches = []
            level_primitives = self.inference.primitive_anchors[i]
            primitive_bb = level_primitives.anchors
            primitive_transformations = level_primitives.transformations

            gt_bboxesl = []
            for bb in gt_bb:
                edges_bb = convert_box_chw_to_vertices(primitive_bb[0]* level_primitives.scale)
                gt_bboxesl.append(convert_box_vertices_to_cwh(bb["bbox"]))
                positive_match = iou_function(edges_bb, bb["bbox"]).ge(
                    self.objectness_thr
                )

                positive_matches.append(
                    positive_match
                )

            gt_bboxes_raw = torch.stack(gt_bboxesl).to(primitive_bb[0].device)
            positives_stack = torch.stack(positive_matches)
            objectness_gt = positives_stack.sum(0).ge(1)

            positives_index = positives_stack.nonzero()

            bboxes = primitive_bb[0].index_select(0, positives_index[:,1])
            gt_bboxes = gt_bboxes_raw.index_select(0, positives_index[:,0])
            transf = primitive_transformations[0].index_select(0, positives_index[:,1])

            loss = self._bb_proposal_regression(bboxes, gt_bboxes, transf)

            if bboxes.shape[0] > num_samples_per_stage:
                samples = random.sample(
                    range(bboxes.shape[0]), num_samples_per_stage
                )  # According to the paper, only sample 256 elements

                total_loss += torch.sum(loss[samples]) / num_samples_per_stage
            else:
                total_loss += torch.sum(loss)

        return total_loss 

    @staticmethod
    def _bb_proposal_regression(
        bboxes, gt_bboxes, transformations,
    ):
        def bb_parametrization(anchors):
            param = torch.zeros_like(anchors)
            param[:, :2] = (
                anchors[:, :2] - transformations[:, :2]
            ) / transformations[:, 2:]
            param[:, 2:] = torch.log(anchors[:, 2:] / transformations[:, 2:])
            return param

        gt_bboxes_param = bb_parametrization(gt_bboxes)
        bboxes_param = bb_parametrization(bboxes)

        l1 = nn.SmoothL1Loss(reduction="none")  # According to Fast R-CNN paper
        loss = l1(gt_bboxes_param, bboxes_param)

        return loss

    def classification_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        objectness = []
        ious = []
        proposed_bboxes = self.inference.proposed_bboxes
        inference_classes = self.inference.classes

        classesl = []
        for bb in gt_bb:
            edges_bb = convert_box_chw_to_vertices(proposed_bboxes[0])
            iou = iou_function(edges_bb, bb["bbox"])
            gt_objectness = iou.ge(self.objectness_thr)

            objectness.append(gt_objectness)
            ious.append(iou)
            classesl.append(THINGS_TO_ID[bb["label"]])

        classes = torch.tensor(classesl).to(iou.device)
        closest_iou = torch.stack(ious).argmax(dim=0)
        selected_class = classes.index_select(0, closest_iou)
        objectness_stack = torch.stack(objectness)
        objectness_gt = objectness_stack.sum(0).ge(1)

        gt_class = selected_class*objectness_gt  # Class 0 is empty bbox, background

        one_hot_targets = nn.functional.one_hot(gt_class, num_classes=len(THINGS)+1).permute(1, 0)

        nlll = nn.NLLLoss(reduction="none")
        
        loss = nlll(inference_classes.permute(0, 1, 2), gt_class.unsqueeze_(0))
        if loss.shape[0] > self.second_stage_num_samples:
            samples = random.sample(
                range(loss.shape[0]), self.second_stage_num_samples
            )  # According to the paper, only sample 512 elements

            total_loss = torch.sum(loss[samples]) / self.second_stage_num_samples
        else:
            total_loss = torch.sum(loss)

        return total_loss

    def regression_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()

        objectness = []
        ious = []
        proposed_bboxes = self.inference.proposed_bboxes
        inference_bboxes = self.inference.bboxes
        inference_classes = torch.exp(self.inference.classes)

        renorm_inference_classes = inference_classes[:, 1:, :] / torch.sum(inference_classes[:, 1:, :], dim=1)

        selected_inference_bb = inference_bboxes.index_select(1, renorm_inference_classes.argmax(dim=1)[0])
        # TODO(David): Fix this index select so that it does not broadcast
        gt_bboxesl = []
        for bb in gt_bb:
            edges_bb = convert_box_chw_to_vertices(proposed_bboxes[0])
            iou = iou_function(edges_bb, bb["bbox"])
            gt_objectness = iou.ge(self.objectness_thr)

            objectness.append(gt_objectness)
            ious.append(iou)
            gt_bboxesl.append(convert_box_vertices_to_cwh(bb["bbox"]))

        closest_iou = torch.stack(ious).argmax(dim=0)
        gt_bboxes_stack = torch.stack(gt_bboxesl).to(iou.device)
        gt_bboxes = gt_bboxes_stack.index_select(0, closest_iou)

        objectness_stack = torch.stack(objectness)
        objectness_gt = objectness_stack.sum(0).ge(1)

        

        one_hot_targets = nn.functional.one_hot(gt_class, num_classes=len(THINGS)+1).permute(1, 0)

        nlll = nn.NLLLoss(reduction="none")
        
        loss = nlll(inference_classes.permute(0, 1, 2), gt_class.unsqueeze_(0))
        if loss.shape[0] > self.second_stage_num_samples:
            samples = random.sample(
                range(loss.shape[0]), self.second_stage_num_samples
            )  # According to the paper, only sample 512 elements

            total_loss = torch.sum(loss[samples]) / self.second_stage_num_samples
        else:
            total_loss = torch.sum(loss)

        return total_loss





if __name__ == "__main__":
    anchors = ANCHORS.cuda()

    ds = DataSet(
        root_folder_inp="efficientPS/data/left_img/leftImg8bit/train",
        root_folder_gt="efficientPS/data/gt/gtFine/train",
        cities_list=["aachen"],
    )
    boxes = ds.samples_path[0].get_bboxes(scale=1 / 8)
    IDs = ds.samples_path[0].get_label_IDs()
    instances_IDs = ds.samples_path[0].get_instances_IDs()
    image = ds.samples_path[0].get_image(scale=1 / 8)

    full = FullModel(len(THINGS), len(STUFF), anchors, 0.6).cuda()
    out = full(image.unsqueeze(0).float().cuda())

    lf = LossFunctions(ds.samples_path[0], out)
    lf.roi_proposal_regression()
    lf.roi_proposal_objectness()
    lf.classification_loss()
    lf.regression_loss()
