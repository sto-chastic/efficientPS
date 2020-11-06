from fake_data import *
from panoctic_ops import *

n_things = 5
n_stuff = 3
logit_size = 4
confidence_thresh = 0.05
og_size = (20, 22)  # original image size
bbox_data = {'x':4, 'y':4, 'w':8, 'h':10}

# instance segmentation route
# fake dataset in the architecture input here
masked_logits = create_simple_logit(n_things, logit_size, logit_size)
bboxes = create_uniform_bbox_pred(n_things, **bbox_data)
class_pred = create_class_pred(n_things)
confidence = create_confidence_levels(n_things)
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

