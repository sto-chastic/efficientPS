from torchvision.ops import nms
import torch
import cv2

from .utilities import *
from ..models import *
from ..models.full import PSOutput, FullModel
from ..dataset.dataset import DataSet
from ..dataset import LABELS_TO_ID, STUFF, THINGS, THINGS_TO_THINGS_ID


def panoptic_fusion_module(
    ps_output, confidence_thresh=0.1, nms_threshold=0.5, probe_name=None
):
    """
    if probe_name, it saves the panoptic fusion image
    """
    n_things = len(THINGS)
    n_stuff = len(STUFF)
    og_size = (
        ps_output.semantic_logits.shape[2],
        ps_output.semantic_logits.shape[3],
    )
    batches = ps_output.mask_logits.shape[0]
    (
        masked_logits,
        class_pred,
        confidence,
        bboxes,
    ) = ps_output.pick_mask_class_conf_bboxes()

    masked_logits_f = masked_logits.cpu().detach().numpy()
    bboxes_f = bboxes.permute(0, 2, 1).cpu().detach().numpy()
    class_pred_f = class_pred.unsqueeze(1).cpu().detach().numpy()
    confidence_f = confidence.unsqueeze(1).cpu().detach().numpy()
    for b in range(batches):
        # masked_logitsc = ps_output.mask_logits[0,0,...].cpu().detach().numpy()
        masked_logits = masked_logits_f[b]

        # bboxes = create_uniform_bbox_pred(n_things, **bbox_data)
        bboxes = (
            ps_output.bboxes[0, ..., 0].permute(1, 0).cpu().detach().numpy()
        )
        bboxes = bboxes_f[b]

        # class_pred = create_class_pred(n_things)
        class_predc = (
            ps_output.classes[0, :8, 0].unsqueeze(0).cpu().detach().numpy()
        )
        class_pred = class_pred_f[b]

        # confidence = create_confidence_levels(n_things)
        confidencec = (
            ps_output.classes[0, :8, 0].unsqueeze(0).cpu().detach().numpy()
        )
        confidence = confidence_f[b]

        # filter all the masked logits that are less than the threshold
        filtered_logits, filtered_conf, og_indices = filter_on_confidence(
            confidence, masked_logits, confidence_thresh
        )
        # sort based on confidence levels
        sorted_logits, sorted_conf, og_indices = sort_by_confidence(
            filtered_logits, filtered_conf, og_indices
        )
        # scale the masked logits by bounding box and pad to original image size
        bboxes = bboxes[:, og_indices]
        MLa = scale_pad_logits_with_bbox(sorted_logits, og_size, bboxes)

        nms_indices = nms(
            torch.tensor(bboxes).permute(1, 0),
            torch.tensor(filtered_conf),
            nms_threshold,
        )
        nms_indices = nms_indices.int().numpy()
        bboxes = bboxes[:, nms_indices]
        MLa = MLa[nms_indices]

        # semantic segmentation route
        semantic_probs = torch.exp(ps_output.semantic_logits[b])

        semantic_logit = (
            torch.log(
                torch.exp(semantic_probs) / (1 - torch.exp(semantic_probs))
            )
            .cpu()
            .detach()
            .numpy()
        )
        semantic_prediction = logit_prediction(semantic_logit)

        # assume the stuff is in the initial channels, and the things in the final ones
        semantic_logit_things = semantic_logit[: n_things + 1]
        semantic_logit_stuff = semantic_logit[-n_stuff:]

        filtered_classes = class_pred[:, og_indices[nms_indices]][0]
        MLb = zero_out_nonbbox(semantic_logit_things[filtered_classes], bboxes)

        fusion = panoptic_fusion(MLa, MLb)

        intermediate_logits = merge_fusion_semantic_things(
            fusion.data.numpy(), semantic_logit_stuff
        )

        intermediate_prediction = logit_prediction(intermediate_logits)
        filled_canvas = fill_canvas(
            intermediate_prediction, filtered_classes, n_stuff
        )

        if probe_name is not None:
            import matplotlib.pyplot as plt

            plt.imsave(probe_name, filled_canvas, dpi=300)

        return filled_canvas, intermediate_logits


if __name__ == "__main__":
    anchors = ANCHORS.cuda()
    device = torch.device("cuda")

    ds = DataSet(
        root_folder_inp="efficientPS/data/left_img/leftImg8bit/train",
        root_folder_gt="efficientPS/data/gt/gtFine/train",
        cities_list=["aachen"],
        crop=[64, 128],
        device=device,
    )
    boxes = ds.samples_path[0].get_bboxes()
    image = ds.samples_path[0].get_image()

    full = FullModel(len(THINGS), len(STUFF), anchors, 0.6).cuda()
    out = full(image.cuda())

    panoptic_fusion_module(out)
