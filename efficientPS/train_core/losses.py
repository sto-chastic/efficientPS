import random

import torch
import torch.nn as nn

from ..dataset.dataset import DataSet
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

        self.first_stage_num_samples = 256  # According to EfficientPS paper
                                            # only sample 256 elements

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

                total_loss += torch.sum(loss[samples])
            else:
                total_loss += torch.sum(loss)


        return total_loss / num_samples_per_stage / 4

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

                total_loss += torch.sum(loss[samples])
            else:
                total_loss += torch.sum(loss)

        return total_loss / num_samples_per_stage / 4

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
        gt_bb = self.ground_truth.get_bboxes()
        for bb in gt_bb:
            inference_edges_bb = convert_box_chw_to_vertices(self.inference.bboxes)
            positive_matches = iou_function(inference_edges_bb, bb["box"]).ge(
                self.objectness_thr
            )
    # @staticmethod
    # def _bb_proposal_objectness(gt_objectness, objectness, samples):
    #     partial_objectness_loss = gt_objectness * torch.log(primitive_obj) + (
    #         1 - gt_objectness
    #     ) * torch.log(1 - primitive_obj)
    #     return partial_objectness_loss[samples]

    # @staticmethod
    # def _bb_proposal_regression(
    #     anchors_bb, transformations, bb, positive_indeces
    # ):
    #     def bb_parametrization(anchors):
    #         param = torch.zeros_like(anchors)
    #         param[:, :2] = (
    #             anchors[:, :2] - transformations[:, :2]
    #         ) / transformations[:, 2:]
    #         param[:, 2:] = torch.log(anchors[:, 2:] / transformations[:, 2:])
    #         return param

    #     param_anchors = bb_parametrization(
    #         convert_box_vertices_to_cwh(anchors_bb)
    #     )

    #     bb_tensor = torch.zeros_like(anchors_bb) + convert_box_vertices_to_cwh(
    #         bb
    #     ).to(anchors_bb.device)
    #     param_bb = bb_parametrization(bb_tensor)

    #     l1 = nn.SmoothL1Loss(reduction="none")  # According to Fast R-CNN paper
    #     loss = l1(param_anchors, param_bb)

    #     return torch.sum(loss.index_select(0, positive_indeces)), torch.shape(
    #         positive_indeces
    #     )
    # def _first_stage_loss(self):
    #     # TODO(David): This assumes batchsize 1, extend later if required
    #     gt_bb = self.ground_truth.get_bboxes()

    #     # How many samples, positive and negative, per bbox we sample
    #     num_samples_per_box_stage = (
    #         self.first_stage_num_samples // len(gt_bb) // 4
    #     )

    #     objectness_loss = 0
    #     regression_loss = 0
    #     for bb in gt_bb:
    #         for i in range(4):
    #             level_primitives = self.primitive_anchors[i]
    #             primitive_bb = level_primitives.anchors
    #             primitive_obj = level_primitives.objectness
    #             primitive_transformations = level_primitives.transformations

    #             samples = random.sample(
    #                 range(primitive_bb[0].shape[0]), num_samples_per_box_stage
    #             )

    #             edges_bb = convert_box_chw_to_vertices(primitive_bb[0])
    #             gt_objectness = iou_function(edges_bb, bb["box"]).ge(
    #                 self.objectness_thr
    #             )

    #             positive_indeces = gt_objectness[:, 0].nonzero()

    #             objectness_loss += self._bb_proposal_objectness(
    #                 gt_objectness, primitive_obj, samples
    #             )

    #             regression_loss += self._bb_proposal_regression(
    #                 primitive_bb[0],
    #                 primitive_transformations[0],
    #                 bb["box"],
    #                 intersection(positive_indeces, samples),
    #             )

    #     return (
    #         objectness_loss / num_samples_per_box_stage * 4,
    #         regression_loss / num_samples_per_box_stage * 4,
    #     )



    # @staticmethod
    # def _bb_class_loss()




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

    full = FullModel(11, 8, anchors, 0.6).cuda()
    out = full(image.unsqueeze(0).float().cuda())

    lf = LossFunctions(ds.samples_path[0], out)
    lf.roi_proposal_regression()
    lf.roi_proposal_objectness()
