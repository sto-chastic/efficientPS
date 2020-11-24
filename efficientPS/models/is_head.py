import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torchvision.ops import RoIAlign, nms

from .utilities import (
    DepthSeparableConv2d,
    RegionProposalNetwork,
    convert_box_chw_to_vertices,
    conv_1x1_bn_custom_act,
    RegionProposalOutput,
)


class ROIFeatureExtraction(nn.Module):
    def __init__(self, anchors, nms_threshold, activation=nn.LeakyReLU):
        super(ROIFeatureExtraction, self).__init__()
        # An anchor is centered at the sliding window in question,
        # and is associated with a scale and aspectratio.
        # By  default  we  use  3  scales  and3 aspect ratios, yielding
        # k= 9anchors at each slidingposition.  For  a  convolutional
        # feature  map  of  a  size W×H(typically∼2,400),
        # there are WHk anchors in total.
        self.rps_p32 = RegionProposalNetwork(anchors, 32)
        self.rps_p16 = RegionProposalNetwork(anchors, 16)
        self.rps_p8 = RegionProposalNetwork(anchors, 8)
        self.rps_p4 = RegionProposalNetwork(anchors, 4)

        self.roi_align = RoIAlign((14, 14), spatial_scale=1, sampling_ratio=-1)

        self.nms_threshold = nms_threshold

    @staticmethod
    def checkpointed_nms(threshold, splits=100):
        def custom_forward(*inputs):
            div = math.ceil(inputs[0].shape[0] / splits)

            collected = []
            scores = []
            for i in range(div):
                input_data = inputs[0][i*splits:(i+1)*splits]
                input_scores = inputs[1][i*splits:(i+1)*splits]
                nms_indices = nms(
                    input_data,
                    input_scores,
                    iou_threshold=threshold,
                )
                collected.append(input_data.index_select(0, nms_indices))
                scores.append(input_scores.index_select(0, nms_indices))
            collected = torch.cat(collected, 0)
            scores = torch.cat(scores, 0)

            nms_indices = nms(
                collected,
                scores,
                iou_threshold=threshold,
            )

            return collected.index_select(0, nms_indices)
        return custom_forward

    def forward(self, p32, p16, p8, p4):
        batches = p32.shape[0]
        feature_inputs = {
            1: p4,
            2: p8,
            3: p16,
            4: p32,
        }  # Size descending order
        # Main and bottom-up
        proposal_outputs = [
            self.rps_p32(p32),
            self.rps_p16(p16),
            self.rps_p8(p8),
            self.rps_p4(p4),
        ]

        p32_anchors, p32_objectness = proposal_outputs[0].get_anch_obj()
        p16_anchors, p16_objectness = proposal_outputs[1].get_anch_obj()
        p8_anchors, p8_objectness = proposal_outputs[2].get_anch_obj()
        p4_anchors, p4_objectness = proposal_outputs[3].get_anch_obj()

        def apply_to_batches(l, operation, *args):
            return [operation(x, *args) for x in l]

        def zip_apply_to_batches(l1, l2, operation, *args):
            return [operation(x1, x2, *args) for x1, x2 in zip(l1, l2)]

        ### Scale properly

        scaled_anchors = [
            apply_to_batches(p4_anchors, torch.mul, 4),
            apply_to_batches(p8_anchors, torch.mul, 8),
            apply_to_batches(p16_anchors, torch.mul, 16),
            apply_to_batches(p32_anchors, torch.mul, 32),
        ]

        scores = [
            p4_objectness,
            p8_objectness,
            p16_objectness,
            p32_objectness,
        ]

        ### Sort each bounding box by its correct extraction level

        def get_channel(anchors):
            """
            This function selects from which level the features
            are extracted based on the size of the box
            """
            return torch.max(
                torch.ones_like(anchors[:, 2]),
                torch.min(
                    torch.ones_like(anchors[:, 2]) * 4,
                    torch.floor(
                        3
                        + torch.log2(
                            torch.sqrt(anchors[:, 2] * anchors[:, 3]) / 224
                        )
                    ),
                ),
            )

        new_level_calculation = [
            apply_to_batches(p4_anchors, get_channel),
            apply_to_batches(p8_anchors, get_channel),
            apply_to_batches(p16_anchors, get_channel),
            apply_to_batches(p32_anchors, get_channel),
        ]

        anchors_per_level = {
            1: [],
            2: [],
            3: [],
            4: [],
        }
        scores_per_level = {
            1: [],
            2: [],
            3: [],
            4: [],
        }

        # Nested loop but with short ranges. Could be vectorized later
        for original_level in range(4):
            for new_level in range(1, 5):

                def sort_anchors_by_level(sc_anchors, calculated_level_nums):
                    return sc_anchors[
                        calculated_level_nums.eq(new_level).nonzero()
                    ]

                anchors_per_level[new_level].append(
                    zip_apply_to_batches(
                        scaled_anchors[original_level],
                        new_level_calculation[original_level],
                        sort_anchors_by_level,
                    )
                )
                scores_per_level[new_level].append(
                    zip_apply_to_batches(
                        scores[original_level],
                        new_level_calculation[original_level],
                        sort_anchors_by_level,
                    )
                )

        ### Extract features

        def prepare_boxes(anchors, level):
            scale = 2 * 2 ** level
            vertices_anchors = anchors
            vertices_anchors /= scale
            return torch.cat(
                (
                    torch.zeros(vertices_anchors.shape[0])
                    .to(vertices_anchors.device)
                    .unsqueeze(-1),
                    vertices_anchors,
                ),
                1,
            )

        extractions_by_batch = []
        extracting_anchors_by_batch = []
        for b in range(batches):
            joined_extractions = []
            extracting_anchors = []
            for level in range(1, 5):
                if len(anchors_per_level[level]) == 0:
                    continue
                joined_anchors_per_level_l = []
                joined_scores_per_level_l = []
                for anch, sc in zip(
                    anchors_per_level[level], scores_per_level[level]
                ):
                    if anch[b].shape[0] == 0:
                        continue
                    joined_anchors_per_level_l.append(anch[b].squeeze(1))
                    joined_scores_per_level_l.append(sc[b].squeeze(1))

                if len(joined_anchors_per_level_l) == 0:
                    continue
                joined_anchors_per_level = torch.cat(
                    joined_anchors_per_level_l, 0
                )
                joined_scores_per_level = torch.cat(
                    joined_scores_per_level_l, 0
                )

                print("nms {} boxes".format(len(joined_anchors_per_level)))
                joined_anchors_per_level = checkpoint.checkpoint(
                    self.checkpointed_nms(self.nms_threshold), 
                    convert_box_chw_to_vertices(joined_anchors_per_level),
                    joined_scores_per_level
                )

                extractions = checkpoint.checkpoint(
                    self.roi_align,
                    feature_inputs[level][b].unsqueeze(0),
                    prepare_boxes(joined_anchors_per_level, level)
                )

                joined_extractions.append(extractions)
                extracting_anchors.append(joined_anchors_per_level)

            if len(joined_extractions) != 0:
                extractions_by_batch.append(torch.cat(joined_extractions, 0))
            else:
                print("Warning: No extractions made at this level.")
                extractions_by_batch.append(joined_extractions)

            if len(extracting_anchors) != 0:
                extracting_anchors_by_batch.append(
                    torch.cat(extracting_anchors, 0)
                )
            else:
                print("Because there were no proposal anchors.")
                extracting_anchors_by_batch.append(extracting_anchors)

        return (
            torch.stack(extractions_by_batch),
            torch.stack(extracting_anchors_by_batch),
            proposal_outputs,
        )
        # For the following part use 1d convolution with size and stride (256*14*14) to go through all the proposals


class InstanceSegmentationHead(nn.Module):
    def __init__(
        self, num_things, anchors, nms_threshold, activation=nn.LeakyReLU
    ):
        super(InstanceSegmentationHead, self).__init__()
        self.num_things = num_things

        self.roi_features = ROIFeatureExtraction(
            anchors, nms_threshold, activation
        )

        self.core_fc = self.make_fully_connected_module(activation)
        self.fc_classes = self.make_classes_output(num_things)
        self.fc_bb = self.make_bb_output(num_things)

        self.mask = self.make_mask_segmentation(num_things)

    def make_fully_connected_module(self, activation):
        fully_connected = [
            nn.Conv1d(256 * 14 * 14, 1024, 1, 1),
            nn.BatchNorm1d(1024),
            activation(inplace=True),
            nn.Conv1d(1024, 1024, 1, 1),
            nn.BatchNorm1d(1024),
            activation(inplace=True),
        ]
        return nn.Sequential(*fully_connected)

    def make_classes_output(self, num_things, activation=nn.LogSoftmax):
        convolutions = [
            nn.Conv1d(1024, num_things + 1, 1, 1),
            nn.BatchNorm1d(num_things + 1),
            activation(dim=1),
        ]
        return nn.Sequential(*convolutions)

    def make_bb_output(self, num_things):
        convolutions = [
            nn.Conv1d(1024, 4 * num_things, 1, 1),
            nn.BatchNorm1d(4 * num_things),
        ]
        return nn.Sequential(*convolutions)

    def make_mask_segmentation(self, num_things):
        convolutions = [
            DepthSeparableConv2d(256, 256),
            DepthSeparableConv2d(256, 256),
            DepthSeparableConv2d(256, 256),
            DepthSeparableConv2d(256, 256),
            nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0),
            conv_1x1_bn_custom_act(256, num_things, None),
        ]
        return nn.Sequential(*convolutions)

    def forward(self, p32, p16, p8, p4):
        (
            extracted_features_,
            proposed_bboxes,
            primitive_anchors,
        ) = self.roi_features(p32, p16, p8, p4)
        shape_ = extracted_features_.shape
        extracted_features = (
            extracted_features_.view(shape_[0], shape_[1], -1)
            .permute(0, 2, 1)
            .contiguous()
        )
        core = self.core_fc(extracted_features)

        classes = (
            self.fc_classes(core)
            .view(shape_[0], self.num_things + 1, 1, shape_[1])
            .squeeze(2)
        )

        bboxes_correction = self.fc_bb(core).view(
            shape_[0], self.num_things, 4, shape_[1]
        )
        bboxes = torch.zeros_like(bboxes_correction)
        proposed = proposed_bboxes.permute(0, 2, 1)

        bboxes[:, :, :2, :] = (
            proposed[:, :2, :] + bboxes_correction[:, :, :2, :]
        )
        bboxes[:, :, 2:, :] = proposed[:, 2:, :] * torch.exp(
            bboxes_correction[:, :, 2:, :]
        )


        def masks_by_subbatches(inputs, splits=200):
            div = math.ceil(inputs.shape[0] / splits)

            collected = []
            for i in range(div):
                elements = inputs[i*splits:(i+1)*splits]
                print("SUBMasks num: {}".format(len(elements)))
                collected.append(
                    checkpoint.checkpoint(
                        self.mask,
                        elements
                    )
                )

            return torch.cat(collected, 0)

        masks = []
        for b in range(shape_[0]):
            elements = extracted_features_[b]
            print("Masks num: {}".format(len(elements)))
            masks.append(
                    masks_by_subbatches(elements)
            )
        masks = torch.stack(masks)

        return (
            classes[:75],
            bboxes,
            masks[:75],
            proposed_bboxes,
            primitive_anchors,
        )


if __name__ == "__main__":
    anchors = torch.tensor(
        [[1.0, 1.0, 220.0, 320.0], [1.0, 1.0, 320.0, 220.0]]
    ).cuda()

    ish = InstanceSegmentationHead(8, anchors, 0.3).cuda()
    output = ish(
        torch.rand(1, 256, 32, 64).cuda(),
        torch.rand(1, 256, 64, 128).cuda(),
        torch.rand(1, 256, 128, 256).cuda(),
        torch.rand(1, 256, 256, 512).cuda(),
    )
    print("output", output)
