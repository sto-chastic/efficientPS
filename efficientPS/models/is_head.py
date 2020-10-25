import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign, nms

from .utilities import DepthSeparableConv2d, RegionProposalNetwork


class InstanceSegmentationHead(nn.Module):
    def __init__(self, anchors, iou_threshold, activation=nn.LeakyReLU):
        super(InstanceSegmentationHead, self).__init__()
        # An anchor is centered at the sliding windowin question,
        # and is associated with a scale and aspectratio.
        # By  default  we  use  3  scales  and3 aspect ratios, yielding
        # k= 9anchors at each slidingposition.  For  a  convolutional  
        # feature  map  of  a  sizeW×H(typically∼2,400), there are WHk anchors intotal.
        self.rps_p32 = RegionProposalNetwork(anchors/32, iou_threshold)
        self.rps_p16 = RegionProposalNetwork(anchors/16, iou_threshold)
        self.rps_p8 = RegionProposalNetwork(anchors/8, iou_threshold)
        self.rps_p4 = RegionProposalNetwork(anchors/4, iou_threshold)

        self.roi_align = RoIAlign((14, 14), spatial_scale=1, sampling_ratio=-1)

        self.iou_threshold = iou_threshold

    def forward(self, p32, p16, p8, p4):
        batches = p32.shape[0]
        # Main and bottom-up
        p32_anchors, p32_objectness = self.rps_p32(p32)
        p16_anchors, p16_objectness = self.rps_p16(p16)
        p8_anchors, p8_objectness = self.rps_p8(p8)
        # p4_anchors, p4_objectness = self.rps_p4(p4)

        def apply_to_batches(l, operation, *args):
            return [operation(x, *args) for x in l]

        def zip_to_batches(l1, l2, operation, *args):
            return [operation(x1, x2, *args) for x1, x2 in zip(l1, l2)]

        p32_anchors = apply_to_batches(p32_anchors, torch.mul, 32)
        p16_anchors = apply_to_batches(p16_anchors, torch.mul, 16)
        p8_anchors = apply_to_batches(p8_anchors, torch.mul, 8)
        # p4_anchors = apply_to_batches(p4_anchors, torch.mul, 4)

        def get_channel(anchors):
            return torch.max(torch.ones_like(anchors[:, 2]), torch.min(torch.ones_like(anchors[:, 2])*4, torch.floor(3 + torch.log2(torch.sqrt(anchors[:,2]*anchors[:,3] * 32**2)/224))))

        p32_N = apply_to_batches(p32_anchors, get_channel)
        p16_N = apply_to_batches(p16_anchors, get_channel)
        p8_N = apply_to_batches(p8_anchors, get_channel)
        # p4_N = apply_to_batches(p4_anchors, get_channel)

        def select_channels(p, n):
            return p.index_select(0, n.long()).unsqueeze(1)

        def prepare_boxes(anchors):
            return torch.cat((torch.arange(anchors.shape[0]).to(anchors.device).unsqueeze(-1), anchors), 1)

        joined_extractions = []
        for i in range(batches):
            temp_list = [
                self.roi_align(
                    select_channels(p32[i], p32_N[i]),
                    prepare_boxes(p32_anchors[i])
                ).squeeze_(),
                self.roi_align(
                    select_channels(p16[i], p16_N[i]),
                    prepare_boxes(p16_anchors[i])
                ).squeeze_(),
                self.roi_align(
                    select_channels(p8[i], p8_N[i]),
                    prepare_boxes(p8_anchors[i])
                ).squeeze_(),
                # self.roi_align(
                #     select_channels(p4[i], p4_N[i]),
                #     prepare_boxes(p4_anchors[i])
                # ).squeeze_(),
            ]
            joined_extractions.append(
                torch.cat(temp_list, 0)
            )
        return torch.stack(joined_extractions)
        # for i in range(batches):
        #     start_ind = 0
        #     for j in range(4):
        #         end_ind = start_ind + joined_levels[i][j].shape[0]
        #         roi_features = self.roi_align(joined_levels[i][j][start_ind:end_ind], score)
        #         start = end_ind
        # joined_anchors = []
        # joined_score = []
        # joined_levels = []
        # joined_inputs = []
        # for i in range(batches):
        #     joined_anchors.append(
        #         torch.cat((
        #             p32_anchors[i],
        #             p16_anchors[i],
        #             p8_anchors[i],
        #             # p4_anchors[i],
        #         ), dim=0)
        #     ) # [ax4, bx4 ... bath size]
        #     joined_score.append(
        #         torch.cat((
        #             p32_objectness[i],
        #             p16_objectness[i],
        #             p8_objectness[i],
        #             # p4_objectness[i],
        #         ), dim=0)
        #     ) # [ax4, bx4 ... bath size]

        #     joined_levels.append(
        #         [
        #             p32[i].index_select(0, p32_N[i].long()),
        #             p16[i].index_select(0, p16_N[i].long()),
        #             p8[i].index_select(0, p8_N[i].long()),
        #             # p4[i].index_select(0, p4_N[i].long()),
        #         ]
        #     ) # [ax4, bx4 ... bath size]

        # nms_indeces = zip_to_batches(joined_anchors, joined_score, nms, self.iou_threshold)
        # nms_boxes = [torch.index_select(boxes, 0, indices) for boxes, indices in zip(joined_anchors, nms_indeces)]

        # for i in range(batches):
        #     start_ind = 0
        #     for j in range(4):
        #         end_ind = start_ind + joined_levels[i][j].shape[0]
        #         roi_features = self.roi_align(joined_levels[i][j][start_ind:end_ind], score)
        #         start = end_ind

        # return p32, p16, p8, p4


if __name__ == "__main__":
    anchors = torch.tensor([[-2.0, -2.0, 22.0, 22.0], [-2.0, -2.0, 22.0, 22.0]]).cuda()
    ish = InstanceSegmentationHead(anchors, 0.8).cuda()
    extracted_features = ish(
        torch.rand(1, 256, 32, 64).cuda(),
        torch.rand(1, 256, 64, 128).cuda(),
        torch.rand(1, 256, 128, 256).cuda(),
        torch.rand(1, 256, 256, 512).cuda(),
    )
    print("extracted_features", extracted_features.shape)
