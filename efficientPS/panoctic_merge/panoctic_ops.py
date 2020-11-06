import numpy as np
from torch import Tensor, zeros, stack, sigmoid
from torch.nn.functional import interpolate
from typing import List, Tuple


def filter_on_confidence(
    confidence: np.ndarray, logits: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # not vectorized - sorry not sorry
    filtered_logits = []
    corresponding_confidence = []
    corresponding_indices = []
    confidence = confidence.ravel()
    for i, el in enumerate(logits):
        if confidence[i] >= thresh:
            filtered_logits.append(el)
            corresponding_confidence.append(confidence[i])
            corresponding_indices.append(i)
    return np.array(filtered_logits), np.array(corresponding_confidence), np.array(corresponding_indices)


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
    # also not vectorized, sorry
    out_tensor = []
    for (i, array) in enumerate(logits):
        tensor = Tensor(array)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        x, y, w, h = bboxes[:, i]
        h, w = int(h), int(w)
        scaled_tensor = interpolate(
            tensor, (h, w)
        )  # pick your own interpolation method here
        scaled_tensor = scaled_tensor.squeeze()

        eff_x = int(x - w // 2)
        eff_y = int(y - h // 2)

        # adjust for patially out of scope bboxes
        if eff_x < 0:
            w = int(w + eff_x)
            eff_x = 0

        if eff_y < 0:
            h = int(h + eff_y)
            eff_y = 0
        
        repadded_tensor = zeros(og_size[0], og_size[1])

        # do nothing with out of scope bboxes
        if h <= 0 and w <= 0:
            out_tensor.append(repadded_tensor)
            continue
        
        repadded_tensor[eff_y : eff_y + h, eff_x : eff_x + w] = scaled_tensor[
            eff_y : eff_y + h, eff_x : eff_x + w
        ]
        out_tensor.append(repadded_tensor)
    return stack(out_tensor)


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
        
        eff_tensor[eff_y : eff_y + h, eff_x : eff_x + w] = tensor[
            eff_y : eff_y + h, eff_x : eff_x + w
        ]
        out_tensor.append(eff_tensor)
    return stack(out_tensor)


def panoctic_fusion(MLa: Tensor, MLb: Tensor) -> Tensor:
    return (sigmoid(MLa) + sigmoid(MLb)) * (MLa + MLa)

def merge_fusion_semantic_things(array1: np.ndarray, array2: np.ndarray):
    return np.concatenate((array1, array2))