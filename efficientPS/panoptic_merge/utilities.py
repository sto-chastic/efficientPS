import numpy as np
from torch import Tensor, zeros, stack, sigmoid
from torch.nn.functional import interpolate
from typing import List, Tuple

from ..dataset import *


def filter_on_confidence(
    confidence: np.ndarray, logits: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered_logits = []
    corresponding_confidence = []
    corresponding_indices = []
    confidence = confidence.ravel()
    for i, el in enumerate(logits):
        if confidence[i] >= thresh:
            filtered_logits.append(el)
            corresponding_confidence.append(confidence[i])
            corresponding_indices.append(i)
    return (
        np.array(filtered_logits),
        np.array(corresponding_confidence),
        np.array(corresponding_indices),
    )


def sort_by_confidence(
    logits: np.ndarray, confidence: np.ndarray, og_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_conf_inds = confidence.argsort()
    sorted_conf = confidence[sorted_conf_inds[::-1]]
    sorted_logits = logits[sorted_conf_inds[::-1]]
    sorted_og_indices = og_indices[sorted_conf_inds[::-1]]
    return sorted_logits, sorted_conf, sorted_og_indices


def scale_pad_logits_with_bbox(
    logits: np.ndarray, og_size: tuple, bboxes: np.ndarray
) -> Tensor:
    out_tensor = []
    for (i, array) in enumerate(logits):
        tensor = Tensor(array)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        x, y, w, h = bboxes[:, i]
        x, y, h, w = (
            round(x),
            round(y),
            round(h),
            round(w),
        )
        if h * w == 0:
            repadded_tensor = zeros(og_size[0], og_size[1])
            out_tensor.append(repadded_tensor)
            continue
        scaled_tensor = interpolate(
            tensor, (h, w)
        )  # pick your own interpolation method here
        scaled_tensor = scaled_tensor.squeeze(0).squeeze(0)

        eff_x = round(x - w // 2)
        eff_y = round(y - h // 2)

        repadded_tensor = zeros(og_size[0], og_size[1])
        repadded_tensor = overlay_image_alpha(
            repadded_tensor, scaled_tensor, (eff_x, eff_y)
        )

        out_tensor.append(repadded_tensor)
    return stack(out_tensor)


def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # channels = img.shape[2]

    # for c in range(channels):
    img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o]

    return img


def logit_prediction(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=0)


def zero_out_nonbbox(logits: np.ndarray, bboxes: np.ndarray) -> Tensor:
    out_tensor = []
    for (i, array) in enumerate(logits):
        tensor = Tensor(array)
        x, y, w, h = bboxes[:, i]
        h, w = int(h), int(w)

        eff_x = int(x - w // 2)
        eff_y = int(y - h // 2)

        # adjust for patially out of scope bboxes
        if eff_x < 0:
            w = int(w + eff_x)
            eff_x = 0

        if eff_y < 0:
            h = int(h + eff_y)
            eff_y = 0

        eff_tensor = zeros(array.shape[0], array.shape[1])

        # do nothing with out of scope bboxes
        if h <= 0 and w <= 0:
            out_tensor.append(eff_tensor)
            continue

        eff_tensor[eff_y : eff_y + h, eff_x : eff_x + w] = tensor[
            eff_y : eff_y + h, eff_x : eff_x + w
        ]
        out_tensor.append(eff_tensor)
    return stack(out_tensor)


def panoptic_fusion(MLa: Tensor, MLb: Tensor) -> Tensor:
    return (sigmoid(MLa) + sigmoid(MLb)) * (MLa + MLb)


def merge_fusion_semantic_things(array1: np.ndarray, array2: np.ndarray):
    return np.concatenate((array1, array2))


def fill_canvas(intermediate_prediction, filtered_classes, n_stuff):
    # Fill with things instances
    mask = intermediate_prediction >= n_stuff
    intermediate_prediction_things = mask * intermediate_prediction

    canvas = np.take(filtered_classes, intermediate_prediction)
    canvas = np.take(np.array(THINGS_STUFF_TO_ID), canvas)
    # Fill with stuff
    intermediate_prediction_stuff = ~mask * intermediate_prediction
    intermediate_prediction_stuff = np.take(
        np.array(THINGS_STUFF_TO_ID), intermediate_prediction_stuff
    )

    return canvas + intermediate_prediction_stuff
