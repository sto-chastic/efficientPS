from .utils import *
from ..models import *
from ..models.full import PSOutput, FullModel
from ..dataset.dataset import DataSet
from ..dataset import LABELS_TO_ID, STUFF, THINGS, THINGS_TO_THINGS_ID

# # instance segmentation route
# # fake dataset in the architecture input here
# masked_logits = create_simple_logit(n_things, logit_size, logit_size)
# bboxes = create_uniform_bbox_pred(n_things, **bbox_data)
# class_pred = create_class_pred(n_things)
# confidence = create_confidence_levels(n_things)

def panoptic_fusion_module(ps_output):
    n_things = len(THINGS)
    n_stuff = len(STUFF)
    confidence_thresh = 0.05
    og_size = (ps_output.semantic_logits.shape[2], ps_output.semantic_logits.shape[3])



    # masked_logits = ps_output.mask_logits
    # bboxes = ps_output.bboxes
    # confidence = ps_output.classes 
    # class_pred = ps_output.classes
    masked_logits = ps_output.mask_logits[0,0,...].cpu().detach().numpy()
    bboxes = ps_output.bboxes[0, ..., 0].squeeze().permute(1, 0).cpu().detach().numpy()
    confidence = ps_output.classes[0, :8, 0].unsqueeze(0).cpu().detach().numpy()
    class_pred = ps_output.classes[0, :8, 0].unsqueeze(0).cpu().detach().numpy()
    
    confidence = np.exp(confidence)
    
    # filter all the masked logits that are less than the threshold
    filtered_logits, filtered_conf, og_indices = filter_on_confidence(confidence, masked_logits, confidence_thresh)
    # sort based on confidence levels
    sorted_logits, sorted_conf, og_indices = sort_by_confidence(filtered_logits, filtered_conf, og_indices)
    # scale the masked logits by bounding box and pad to original image size
    MLa = scale_pad_logits_with_bbox(sorted_logits, og_size, bboxes)

    ################################################################################################
    #### FILTER THE MLa WITH OVERLAP THRESH USING YOUR IOU FUNCTION to get a new MLa ####
    #### MAKE SURE TO TRACK THE OG_INDICES WHEN YOU DO THE OVERLAP THRESHOLD FUNCTION ####
    ################################################################################################

    # semantic segmentation route
    semantic_logit = create_simple_logit(n_things+n_stuff, og_size[0], og_size[1])
    semantic_prediction = logit_prediction(semantic_logit)

    # assume the stuff is in the initial channels, and the things in the final ones
    semantic_logit_things = semantic_logit[:n_things]
    semantic_logit_stuff = semantic_logit[-n_stuff:]

    MLb = zero_out_nonbbox(semantic_logit_things[og_indices], bboxes[:, og_indices])

    fusion = panoctic_fusion(MLa, MLb)

    intermediate_logits = merge_fusion_semantic_things(semantic_logit_stuff, fusion.data.numpy())

    intermediate_prediction = logit_prediction(intermediate_logits)

    print(intermediate_prediction.shape)


if __name__ == "__main__":
    anchors = ANCHORS.cuda()
    device = torch.device("cuda")

    ds = DataSet(
        root_folder_inp="efficientPS/data/left_img/leftImg8bit/train",
        root_folder_gt="efficientPS/data/gt/gtFine/train",
        cities_list=["aachen"],
        crop=[128, 256],
        device=device,
    )
    boxes = ds.samples_path[0].get_bboxes()
    image = ds.samples_path[0].get_image()

    full = FullModel(len(THINGS), len(STUFF), anchors, 0.6).cuda()
    out = full(image.cuda())

    panoptic_fusion_module(out)