import numpy as np
np.random.seed(0)

def create_simple_logit(c: int, x: int, y: int):
    """
    @ brief - creates a masked logit of size x, y
    This isnt an actual logit, just simple fake data to work with
    """
    fake_logit = np.zeros((c, x, y))
    for i, c in enumerate(fake_logit):
        fake_logit[i, :, :] = i
    return fake_logit

def create_uniform_bbox_pred(c:int, x:int, y:int, w:int, h:int):
    """
    @brief - create bounding box data
    since its fake data make everything uniform
    """
    out_array = np.zeros((4, c))
    for (i, el) in enumerate([x, y, w, h]):
        out_array[i, :] = el
    return out_array

def create_class_pred(n_classes: int):
    out = np.random.rand(1, n_classes)
    temp_step = (out - out.min()) / (out.max() - out.min())
    return temp_step/temp_step.sum()

def create_confidence_levels(n_classes: int):
    out = np.random.rand(1, n_classes)
    temp_step = (out - out.min()) / (out.max() - out.min())
    return temp_step/temp_step.sum()

