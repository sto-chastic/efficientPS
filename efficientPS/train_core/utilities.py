import torch


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
