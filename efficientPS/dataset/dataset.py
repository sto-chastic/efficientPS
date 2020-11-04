import torch
import os
import json

import cv2

from .utilities import polygons_to_bboxes
from . import *

class PSSamples:
    def __init__(
        self,
        image_path,
        gt_polygons_path,
        gt_label_IDs_path,
        gt_instance_IDs_path,
    ):
        self.image_path = (image_path,)
        self.gt_polygons_path = (gt_polygons_path,)
        self.gt_label_IDs_path = (gt_label_IDs_path,)
        self.gt_instance_IDs_path = (gt_instance_IDs_path,)

    def get_bboxes(self, scale=1 / 2):
        filtered_bboxes = []
        with open(self.gt_polygons_path[0]) as f:
            data = json.load(f)

        for element in data["objects"]:
            if element["label"] in THINGS:
                filtered_bboxes.append(
                    {
                        "label": element["label"],
                        "bbox": polygons_to_bboxes(element["polygon"], scale),
                    }
                )

        return filtered_bboxes

    def get_label_IDs(self):
        image = cv2.imread(self.gt_label_IDs_path[0], cv2.IMREAD_GRAYSCALE)
        torch_image = torch.tensor(image)
        return torch_image.unsqueeze(0)

    def get_instances_IDs(self):
        # TODO(David): get labels from here 
        image = cv2.imread(self.gt_instance_IDs_path[0], cv2.IMREAD_GRAYSCALE)
        torch_image = torch.tensor(image)
        return torch_image.unsqueeze(0)

    def get_image(self, scale=1 / 2):
        image = cv2.imread(self.gt_instance_IDs_path[0])
        if scale != 1:
            image_dims = image.shape
            width = int(image_dims[0] * scale)
            height = int(image_dims[1] * scale)
            dim = (width, height)
            image = cv2.resize(image, dim)
        torch_image = torch.tensor(image)
        return torch_image.permute(2, 0, 1)


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder_inp,
        root_folder_gt,
        cities_list,
    ):
        self.root_folder_inp = root_folder_inp
        self.root_folder_gt = root_folder_gt
        self.cities_list = cities_list
        self.samples_path = self.find_instances()

    def find_instances(self):
        samples_path = []
        for city in self.cities_list:
            root_img = os.path.join(self.root_folder_inp, city)
            root_gt = os.path.join(self.root_folder_gt, city)
            for path_to_file in os.listdir(root_img):
                name = path_to_file.replace("_leftImg8bit.png", "")
                ps = PSSamples(
                    os.path.join(
                        root_img, "{}{}".format(name, "_leftImg8bit.png")
                    ),
                    os.path.join(
                        root_gt, "{}{}".format(name, "_gtFine_polygons.json")
                    ),
                    os.path.join(
                        root_gt, "{}{}".format(name, "_gtFine_labelIds.png")
                    ),
                    os.path.join(
                        root_gt, "{}{}".format(name, "_gtFine_instanceIds.png")
                    ),
                )
                samples_path.append(ps)
        return samples_path

    def __getitem__(self, index: int):
        return self.samples_path[index]

    def __len__(self) -> int:
        return len(self.samples_path)


if __name__ == "__main__":
    ds = DataSet(
        root_folder_inp="efficientPS/data/left_img/leftImg8bit/train",
        root_folder_gt="efficientPS/data/gt/gtFine/train",
        cities_list=["aachen"],
    )
    boxes = ds.samples_path[0].get_bboxes()
    IDs = ds.samples_path[0].get_label_IDs()
    instances_IDs = ds.samples_path[0].get_instances_IDs()
    image = ds.samples_path[0].get_image()
    pass
