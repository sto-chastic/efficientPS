import random

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

from ..dataset.dataset import DataSet
from ..dataset import LABELS_TO_ID, STUFF, THINGS, THINGS_TO_THINGS_ID
from ..models import *
from ..models.full import FullModel, PSOutput
from ..models.utilities import (
    convert_box_chw_to_vertices,
    convert_box_vertices_to_cwh,
)
from .utilities import (
    intersection,
    iou_function,
    index_select2D,
    id_to_things_id_expanded,
    id_to_things_stuff_id
)


class LossFunctions:
    def __init__(self, ground_truth, inference):
        self.ground_truth = ground_truth
        self.inference = inference

        self.first_stage_objectness_thr = (
            0.7  # According to Faster R-CNN paper
        )
        self.first_stage_nonobjectness_thr = (
            0.3  # According to Faster R-CNN paper
        )
        self.objectness_thr = 0.5  # According to Mask R-CNN paper

        self.first_stage_num_samples = 256  # According to EfficientPS paper
        # only sample 256 elements
        self.second_stage_num_samples = 512  # According to EfficientPS paper
        # only sample 512 elements

    def ss_loss_max_pooling(self):
        raw_output = self.inference.semantic_logits
        (batch, num_classes, height, width) = raw_output.shape
        formated_labels = id_to_things_stuff_id(self.ground_truth.get_label_IDs())

        nlll = nn.NLLLoss(reduction="none")
        loss = nlll(
            raw_output, formated_labels.unsqueeze(0)
        )
        values, ind = torch.sort(loss.view(batch, -1), 1)

        middle = int(values.shape[1] / 2)
        top_quartile = int(middle + middle / 2)

        top_quartile_values = values[:, top_quartile]
        mask = (
            loss.ge(top_quartile_values.unsqueeze(-1).unsqueeze(-1)).float()
            * 4
            / (height + 1e-3)
            / (width + 1e-3)
        )
        return torch.sum(loss * mask) / batch

    def roi_proposal_objectness(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()
        if len(gt_bb) == 0:
            return 0.0
        # How many samples, positive and negative, per extraction level
        num_samples_per_stage = self.first_stage_num_samples // 4

        total_loss = 0.0
        number_samples = 0
        for i in range(4):
            objectness_true = []
            objectness_false = []
            ious = []
            level_primitives = self.inference.primitive_anchors[i]
            primitive_bb = level_primitives.anchors
            primitive_obj = level_primitives.objectness

            for bb in gt_bb:
                edges_bb = convert_box_chw_to_vertices(
                    primitive_bb[0] * level_primitives.scale
                )
                iou = iou_function(edges_bb, bb["bbox"])
                gt_objectness_true = iou.ge(self.first_stage_objectness_thr)

                objectness_true.append(gt_objectness_true)

                gt_objectness_false = ~iou.ge(
                    self.first_stage_nonobjectness_thr
                )

                objectness_false.append(gt_objectness_false)

            # Positive matches
            objectness_stack_true = torch.stack(objectness_true)
            objectness_gt_t_ind = objectness_stack_true.nonzero()[:, 1]

            positive_loss = self._bb_partial_proposal_objectness(
                primitive_obj[0][objectness_gt_t_ind]
            )

            # Negative matches
            objectness_stack_false = torch.stack(objectness_false)
            objectness_gt_f_ind = objectness_stack_false.nonzero()[:, 1]

            negative_loss = self._bb_partial_proposal_objectness(
                1-primitive_obj[0][objectness_gt_f_ind]
            )

            loss = torch.cat([positive_loss, negative_loss])

            if loss.shape[0] > num_samples_per_stage:
                # Sample half and half due to class imbalance
                num_t_samples = min(positive_loss.shape[0], num_samples_per_stage //2)
                samples_t = random.sample(
                    range(positive_loss.shape[0]), num_t_samples
                )  # According to the paper, only sample 256 elements
                num_f_samples = min(negative_loss.shape[0], num_samples_per_stage //2)
                samples_f = random.sample(
                    range(negative_loss.shape[0]), num_f_samples
                )  # According to the paper, only sample 256 elements

                total_loss = total_loss + torch.sum(positive_loss[samples_t])
                total_loss = total_loss + torch.sum(negative_loss[samples_f])
                number_samples += len(samples_t)
                number_samples += len(samples_f)
            else:
                total_loss = total_loss + torch.sum(loss)
                number_samples += loss.shape[0]

        return total_loss / (number_samples + 1e-3)

    @staticmethod
    def _bb_partial_proposal_objectness(pred_class):
        partial_objectness_loss = torch.log(pred_class)
        return -partial_objectness_loss

    def roi_proposal_regression(self):
        """Calculate the ROI proposal loss .
        """
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()
        if len(gt_bb) == 0:
            return 0.0
        # How many samples, positive and negative, per extraction level
        num_samples_per_stage = self.first_stage_num_samples // 4

        total_loss = 0.0
        number_samples = 0
        for i in range(4):
            positive_matches = []
            level_primitives = self.inference.primitive_anchors[i]
            primitive_bb = level_primitives.anchors
            primitive_transformations = level_primitives.transformations

            ious = []
            gt_bboxesl = []
            for bb in gt_bb:
                edges_bb = convert_box_chw_to_vertices(
                    primitive_bb[0] * level_primitives.scale
                )
                gt_bboxesl.append(convert_box_vertices_to_cwh(bb["bbox"]))
                iou = iou_function(edges_bb, bb["bbox"])
                positive_match = iou.ge(self.objectness_thr)

                ious.append(iou)
                positive_matches.append(positive_match)

            gt_bboxes_stack = torch.stack(gt_bboxesl).to(iou.device).float()
            positives_stack = torch.stack(positive_matches)

            if len(positives_stack.nonzero()) > 0:
                # Calculate with only the positiove matches (above threshold)
                positive_matches_gt = positives_stack.nonzero()[:, 0]
                positive_matches_inf = positives_stack.nonzero()[:, 1]

                loss = self._bb_regression(
                    primitive_bb[0][positive_matches_inf],
                    gt_bboxes_stack[positive_matches_gt],
                    primitive_transformations[0][positive_matches_inf],
                )

            # If no positive is found, then: "the anchor/anchors  with
            # the  highest  Intersection-over-Union (IoU) overlap with
            # a ground-truth box" - Faster-RCNN paper
            closest_iou = torch.stack(ious).argmax(dim=1)
            no_matching = (
                (~torch.sum(positives_stack, 1).gt(0)).nonzero().squeeze(1)
            )
            closest_iou_ind = closest_iou[no_matching]

            if len(positives_stack.nonzero()) > 0:
                loss = torch.cat(
                    [
                        loss,
                        self._bb_regression(
                            primitive_bb[0][closest_iou_ind],
                            gt_bboxes_stack[no_matching],
                            primitive_transformations[0][closest_iou_ind],
                        ),
                    ]
                )
            else:
                loss = self._bb_regression(
                    primitive_bb[0][closest_iou_ind],
                    gt_bboxes_stack[no_matching],
                    primitive_transformations[0][closest_iou_ind],
                )

            if loss.shape[0] > num_samples_per_stage:
                samples = random.sample(
                    range(loss.shape[0]), num_samples_per_stage
                )  # According to the paper, only sample 256 elements

                total_loss = total_loss + torch.sum(loss[samples])
                number_samples += num_samples_per_stage
            else:
                total_loss = total_loss + torch.sum(loss)
                number_samples += loss.shape[0]

        return total_loss / (number_samples + 1e-3)

    @staticmethod
    def _bb_regression(
        bboxes,
        gt_bboxes,
        transformations,
    ):
        """Compute the Box regression .
        """
        smooth = 1e-3

        def bb_parametrization(anchors):
            param = torch.zeros_like(anchors)
            param[:, :2] = (anchors[:, :2] - transformations[:, :2]) / (
                transformations[:, 2:] + smooth
            )

            log_arg = anchors[:, 2:] / (transformations[:, 2:] + smooth)
            param[:, 2:] = torch.log(torch.clamp(log_arg, min=smooth))
            return param

        gt_bboxes_param = bb_parametrization(gt_bboxes)
        bboxes_param = bb_parametrization(bboxes)

        l1 = nn.SmoothL1Loss(reduction="none")  # According to Fast R-CNN paper
        loss = l1(gt_bboxes_param, bboxes_param)

        return torch.sum(loss, dim=1)

    def classification_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()
        if len(gt_bb) == 0:
            return 0.0
        objectness = []
        ious = []
        proposed_bboxes = self.inference.proposed_bboxes
        inference_classes = self.inference.classes

        classesl = []
        for bb in gt_bb:
            edges_bb = convert_box_chw_to_vertices(proposed_bboxes[0])
            iou = iou_function(edges_bb, bb["bbox"])
            gt_objectness = iou.ge(self.objectness_thr)
            # Usually samples below T_H and above T_L are ignored, but
            # since T_H = T_L, we do not ignore samples

            objectness.append(gt_objectness)
            ious.append(iou)
            classesl.append(THINGS_TO_THINGS_ID[bb["label"]])

        # Selects the closest GT box for every output
        classes = torch.tensor(classesl).to(iou.device)
        closest_iou = torch.stack(ious).argmax(dim=0)
        selected_class = classes.index_select(0, closest_iou)

        objectness_stack = torch.stack(objectness)
        objectness_gt = objectness_stack.sum(0).ge(1)

        # If it is not a positive match, then the output should be 0
        gt_class = (
            selected_class * objectness_gt
        )  # Class 0 is empty bbox, background

        nlll = nn.NLLLoss(reduction="none")

        loss = nlll(inference_classes, gt_class.unsqueeze_(0))
        if loss.shape[1] > self.second_stage_num_samples:
            samples = random.sample(
                range(loss.shape[1]), self.second_stage_num_samples
            )  # According to the paper, only sample 512 elements

            total_loss = (
                torch.sum(loss[:, samples]) / (self.second_stage_num_samples  + 1e-3)
            )
        else:
            total_loss = torch.sum(loss) / (loss.shape[1] + 1e-3)

        return total_loss

    def regression_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()
        if len(gt_bb) == 0:
            return 0.0

        proposed_bboxes = self.inference.proposed_bboxes
        inference_bboxes = self.inference.bboxes
        inference_classes = self.inference.get_classes()

        selected_inference_bb = index_select2D(
            inference_bboxes.permute(0, 2, 1, 3),
            inference_classes.argmax(dim=1)[0],
        ).permute(1, 0, 2)

        objectness = []
        ious = []
        gt_bboxesl = []
        for bb in gt_bb:
            edges_bb = convert_box_chw_to_vertices(selected_inference_bb[0])
            iou = iou_function(edges_bb, bb["bbox"])
            gt_objectness = iou.ge(self.objectness_thr)
            # Usually samples below T_H and above T_L are ignored, but
            # since T_H = T_L, we do not ignore sample

            objectness.append(gt_objectness)
            ious.append(iou)
            gt_bboxesl.append(convert_box_vertices_to_cwh(bb["bbox"]))

        gt_bboxes_stack = torch.stack(gt_bboxesl).to(iou.device).float()
        objectness_stack = torch.stack(objectness)

        # Calculate with only the positiove matches (above threshold)
        positive_matches_gt = objectness_stack.nonzero()[:, 0]
        positive_matches_inf = objectness_stack.nonzero()[:, 1]

        loss = self._bb_regression(
            selected_inference_bb[0][positive_matches_inf],
            gt_bboxes_stack[positive_matches_gt],
            proposed_bboxes[0][positive_matches_inf],
        )

        # If no positive is found, then: "the anchor/anchors  with
        # the  highest  Intersection-over-Union (IoU) overlap with
        # a ground-truth box" - Faster-RCNN paper
        closest_iou = torch.stack(ious).argmax(dim=1)
        no_matching = (
            (~torch.sum(objectness_stack, 1).gt(0)).nonzero().squeeze(1)
        )
        closest_iou_ind = closest_iou[no_matching]

        loss = torch.cat(
            (
                loss,
                self._bb_regression(
                    selected_inference_bb[0][closest_iou_ind],
                    gt_bboxes_stack[no_matching],
                    proposed_bboxes[0][closest_iou_ind],
                ),
            )
        )

        if loss.shape[0] > self.second_stage_num_samples:
            samples = random.sample(
                range(loss.shape[0]), self.second_stage_num_samples
            )  # According to the paper, only sample 512 elements

            total_loss = (
                torch.sum(loss[samples]) / (self.second_stage_num_samples  + 1e-3)
            )
        else:
            total_loss = torch.sum(loss) / (loss.shape[0] + 1e-3)

        return total_loss

    @staticmethod
    def _cross_entropy(inference, gt):
        loss = gt * torch.log(inference) + (~gt) * torch.log(1 - inference)
        return -loss

    def mask_loss(self):
        # TODO(David): This assumes batchsize 1, extend later if required
        gt_bb = self.ground_truth.get_bboxes()
        if len(gt_bb) == 0:
            return 0.0

        mask_logits = self.inference.mask_logits
        proposed_bboxes = self.inference.proposed_bboxes
        gt_mask_seg = id_to_things_id_expanded(
            self.ground_truth.get_label_IDs()
        )
        gt_mask_seg = gt_mask_seg.to(mask_logits.device)

        objectness = []
        gt_bboxesl = []
        ious = []
        gt_classes = []
        for bb in gt_bb:
            edges_bb = convert_box_chw_to_vertices(proposed_bboxes[0])
            iou = iou_function(edges_bb, bb["bbox"])
            gt_classes.append(THINGS_TO_THINGS_ID[bb["label"]])
            gt_objectness = iou.ge(self.objectness_thr)

            objectness.append(gt_objectness)
            ious.append(iou)
            gt_bboxesl.append(convert_box_vertices_to_cwh(bb["bbox"]))

        gt_bboxes_stack = torch.stack(gt_bboxesl).to(iou.device).float()
        objectness_stack = torch.stack(objectness)

        gt_bboxes_stack = convert_box_chw_to_vertices(gt_bboxes_stack)

        # Calculate with only the positiove matches (above threshold)
        positive_matches_gt = objectness_stack.nonzero()[:, 0]
        positive_matches_inf = objectness_stack.nonzero()[:, 1]

        gt_bboxes_ = gt_bboxes_stack[positive_matches_gt]
        gt_classes_ = torch.tensor(gt_classes, device=iou.device)[
            positive_matches_gt
        ]

        gt_bb_classes = torch.cat(
            [gt_classes_.unsqueeze(1), gt_bboxes_], dim=1
        )

        gt_masks = self._extract_mask_from_gt(
            gt_mask_seg, gt_bb_classes
        ).float()

        non_null_pixels = 0
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        if len(positive_matches_inf) > 0:
            selected_mask = index_select2D(
                mask_logits.permute(0, 3, 4, 2, 1).index_select(
                    4, positive_matches_inf
                ),
                gt_classes_ - 1, # we don't count 0 class
            )

            loss = criterion(selected_mask, gt_masks)
            non_null_pixels += len(gt_masks.nonzero())

        # If no positive is found, then: "the anchor/anchors  with
        # the  highest  Intersection-over-Union (IoU) overlap with
        # a ground-truth box" - Faster-RCNN paper
        closest_iou = torch.stack(ious).argmax(dim=1)
        no_matching = (
            (~torch.sum(objectness_stack, 1).gt(0)).nonzero().squeeze(1)
        )

        if len(no_matching) > 0:
            gt_bboxes_ = gt_bboxes_stack[no_matching]
            gt_classes_ = torch.tensor(gt_classes, device=iou.device)[
                no_matching
            ]

            gt_bb_classes = torch.cat(
                [gt_classes_.unsqueeze(1), gt_bboxes_], dim=1
            )

            gt_masks = self._extract_mask_from_gt(
                gt_mask_seg, gt_bb_classes
            ).float()

            closest_iou_ind = closest_iou[no_matching]

            selected_mask = index_select2D(
                mask_logits.permute(0, 3, 4, 2, 1).index_select(
                    4, closest_iou_ind
                ),
                gt_classes_ - 1,
            )
            if len(positive_matches_inf) > 0:
                loss = torch.cat((loss, criterion(selected_mask, gt_masks)))
            else:
                loss = criterion(selected_mask, gt_masks)
            non_null_pixels = non_null_pixels + len(gt_masks.nonzero())

        return torch.sum(loss) / (non_null_pixels + 1e-3)

    @staticmethod
    def _extract_mask_from_gt(full_mask, bb_and_classes):
        binarize_threshold = 0.7
        roi = RoIAlign((28, 28), spatial_scale=1, sampling_ratio=-1)
        extracted = roi(full_mask.unsqueeze(1), bb_and_classes.float())
        return extracted.ge(binarize_threshold)

    def get_total_loss(self):
        self.losses_dict = {
            "ss_loss_max_pooling": self.ss_loss_max_pooling(), #
            "roi_proposal_objectness": self.roi_proposal_objectness(), #
            "roi_proposal_regression": self.roi_proposal_regression(), #
            "classification_loss": self.classification_loss(), #
            "regression_loss": self.regression_loss(),
            "mask_loss": self.mask_loss(),
        }
        print("Done loss ")
        loss = 0
        for v in self.losses_dict.values():
            loss = loss + v

        self.losses_dict["full_model"] = loss
        return loss


if __name__ == "__main__":
    anchors = ANCHORS.cuda()

    ds = DataSet(
        root_folder_inp="efficientPS/data/left_img/leftImg8bit/train",
        root_folder_gt="efficientPS/data/gt/gtFine/train",
        cities_list=["aachen"],
        crop=[128, 256],
    )
    boxes = ds.samples_path[0].get_bboxes()
    image = ds.samples_path[0].get_image()

    full = FullModel(len(THINGS), len(STUFF), anchors, 0.6).cuda()
    out = full(image.cuda())

    lf = LossFunctions(ds.samples_path[0], out)
    # lf.roi_proposal_regression()
    lf.ss_loss_max_pooling()
    lf.roi_proposal_objectness()
    lf.classification_loss()
    lf.regression_loss()
    lf.mask_loss()
