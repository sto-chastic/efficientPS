import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign, nms

from .utilities import DepthSeparableConv2d, RegionProposalNetwork, convert_box_chw_to_vertices


class InstanceSegmentationHead(nn.Module):
    def __init__(self, anchors, nms_threshold, activation=nn.LeakyReLU):
        super(InstanceSegmentationHead, self).__init__()
        # An anchor is centered at the sliding windowin question,
        # and is associated with a scale and aspectratio.
        # By  default  we  use  3  scales  and3 aspect ratios, yielding
        # k= 9anchors at each slidingposition.  For  a  convolutional  
        # feature  map  of  a  sizeW×H(typically∼2,400), there are WHk anchors intotal.
        self.rps_p32 = RegionProposalNetwork(anchors/32, nms_threshold)
        self.rps_p16 = RegionProposalNetwork(anchors/16, nms_threshold)
        self.rps_p8 = RegionProposalNetwork(anchors/8, nms_threshold)
        self.rps_p4 = RegionProposalNetwork(anchors/4, nms_threshold)

        self.roi_align = RoIAlign((14, 14), spatial_scale=1, sampling_ratio=-1)

        self.nms_threshold = nms_threshold

    def forward(self, p32, p16, p8, p4):
        batches = p32.shape[0]
        feature_inputs = {
            1:p32,
            2:p16,
            3:p8,
            4:p4
        }
        # Main and bottom-up
        p32_anchors, p32_objectness = self.rps_p32(p32)
        p16_anchors, p16_objectness = self.rps_p16(p16)
        p8_anchors, p8_objectness = self.rps_p8(p8)
        # p4_anchors, p4_objectness = self.rps_p4(p4)

        def apply_to_batches(l, operation, *args):
            return [operation(x, *args) for x in l]

        def zip_apply_to_batches(l1, l2, operation, *args):
            return [operation(x1, x2, *args) for x1, x2 in zip(l1, l2)]

        # Scale properly
        scaled_anchors = [
            apply_to_batches(p32_anchors, torch.mul, 32),
            apply_to_batches(p16_anchors, torch.mul, 16),
            apply_to_batches(p8_anchors, torch.mul, 8),
            # apply_to_batches(p4_anchors, torch.mul, 4),
        ]

        scores = [
            p32_objectness,
            p16_objectness,
            p8_objectness,
            # p4_objectness,
        ]

        def get_channel(anchors):
            return torch.max(torch.ones_like(anchors[:, 2]), torch.min(torch.ones_like(anchors[:, 2])*4, torch.floor(3 + torch.log2(torch.sqrt(anchors[:,2]*anchors[:,3])/224))))

        level_calculation = [
            apply_to_batches(p32_anchors, get_channel),
            apply_to_batches(p16_anchors, get_channel),
            apply_to_batches(p8_anchors, get_channel),
            # apply_to_batches(p4_anchors, get_channel),
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

        # Nested loop but with a short ranges. Could be vectorized later
        for l in range(3):
            for nl in range(1, 4):
                def sort_anchors_by_level(sc_anchors, level_num):
                    return sc_anchors[level_num.eq(nl).nonzero()]
                
                anchors_per_level[nl].append(
                    zip_apply_to_batches(scaled_anchors[l], level_calculation[l], sort_anchors_by_level)
                )
                scores_per_level[nl].append(
                    zip_apply_to_batches(scores[l], level_calculation[l], sort_anchors_by_level)
                )
                
        def select_channels(p, n):
            return p.index_select(0, n.long()).unsqueeze(1)

        def prepare_boxes(anchors):
            return torch.cat((torch.zeros(anchors.shape[0]).to(anchors.device).unsqueeze(-1), anchors), 1)

        extractions_by_batch = []
        for b in range(batches):
            joined_extractions = []
            for nl in range(1, 4):
                if len(anchors_per_level[nl]) == 0:
                    continue
                joined_anchors_per_level_l = []
                joined_scores_per_level_l = []
                for anch, sc in zip(anchors_per_level[nl], scores_per_level[nl]):
                    if anch[b].shape[0] == 0:
                        continue
                    joined_anchors_per_level_l.append(anch[b].squeeze(1))
                    joined_scores_per_level_l.append(sc[b].squeeze(1))

                joined_anchors_per_level = torch.cat(joined_anchors_per_level_l, 0)
                joined_scores_per_level = torch.cat(joined_scores_per_level_l, 0)

                nms_indices = nms(
                    convert_box_chw_to_vertices(joined_anchors_per_level),
                    joined_scores_per_level,
                    iou_threshold=self.nms_threshold
                )

                joined_anchors_per_level = joined_anchors_per_level.index_select(0, nms_indices)

                joined_extractions.append(
                    self.roi_align(
                        feature_inputs[nl][b].unsqueeze(0),
                        prepare_boxes(joined_anchors_per_level)
                    ).squeeze_(),
                )

            extractions_by_batch.append(torch.cat(joined_extractions, 0))
        return torch.stack(extractions_by_batch)


if __name__ == "__main__":
    anchors = torch.tensor([[1.0, 1.0, 220.0, 320.0], [1.0, 1.0, 320.0, 220.0]]).cuda()
    ish = InstanceSegmentationHead(anchors, 0.6).cuda()
    extracted_features = ish(
        torch.rand(1, 256, 32, 64).cuda(),
        torch.rand(1, 256, 64, 128).cuda(),
        torch.rand(1, 256, 128, 256).cuda(),
        torch.rand(1, 256, 256, 512).cuda(),
    )
    print("extracted_features", extracted_features.shape)
