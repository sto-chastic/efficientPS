import torch
from ..dataset import THINGS_TO_THINGS_ID, ID_TO_LABELS, THINGS_STUFF_TO_LABEL


def iou_function(inputs, target):
    def custom_max(tensor, value):
        return torch.where(
            tensor > value, tensor, torch.ones_like(tensor) * value
        )

    def custom_min(tensor, value):
        return torch.where(
            tensor < value, tensor, torch.ones_like(tensor) * value
        )

    xA = custom_max(inputs[:, 0], target[0][0])
    yA = custom_max(inputs[:, 1], target[0][1])
    xB = custom_min(inputs[:, 2], target[1][0])
    yB = custom_min(inputs[:, 3], target[1][1])

    interArea = custom_max(xB - xA, 0) * custom_max(yB - yA, 0)

    boxAArea = (inputs[:, 2] - inputs[:, 0]) * (inputs[:, 3] - inputs[:, 1])
    boxBArea = (target[1][0] - target[0][0]) * (target[1][1] - target[0][1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)

    return iou


def intersection(t1, t2):
    indices = torch.zeros_like(t1)
    for elem in t2:
        indices = indices | (t1 == elem)
    intersection = t1[indices]


def index_select2D(original, index1, index2=None):
    #  Not in Pytorch, proposed here:
    #  discuss.pytorch.org/t/how-to-select-index-over-two-dimension/10098/4
    if not index2:
        index2 = torch.arange(0, index1.shape[0])
    return torch.cat(
        [original[..., x, y].unsqueeze(0) for x, y in zip(index1, index2)]
    )


def id_to_things_stuff_id(id_):
    things_stuff_id = torch.zeros_like(id_) * torch.zeros(
        (len(THINGS_STUFF_TO_LABEL), 1, 1)
    ).to(id_.device)

    for k, v in ID_TO_LABELS.items():
        if v[0] in THINGS_STUFF_TO_LABEL:
            new_id = THINGS_STUFF_TO_LABEL[v[0]]

            things_stuff_id[new_id, ...] = id_.eq(k)*new_id

    return torch.sum(things_stuff_id, dim=0).long()


def id_to_things_id_expanded(id_):
    things_id = torch.zeros_like(id_) * torch.zeros(
        (len(THINGS_TO_THINGS_ID) + 1, 1, 1)
    ).to(id_.device)

    for k, v in ID_TO_LABELS.items():
        if v[0] in THINGS_TO_THINGS_ID:
            new_id = THINGS_TO_THINGS_ID[v[0]]

            things_id[new_id, ...] = id_.eq(k)

    return things_id
