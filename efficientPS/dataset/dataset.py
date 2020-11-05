import torch
import os
import json

import cv2
import random

from .utilities import polygons_to_bboxes
from . import *

class PSSamples:
    def __init__(
        self,
        image_path,
        gt_polygons_path,
        gt_label_IDs_path,
        gt_instance_IDs_path,
        crop = (1024, 2048)
    ):
        self.crop = crop
        self.get_cropped_area()
        self.image_path = image_path
        self.gt_polygons_path = gt_polygons_path
        self.gt_label_IDs_path = gt_label_IDs_path
        self.gt_instance_IDs_path = gt_instance_IDs_path

    def get_cropped_area(self):
        self.x1 = random.randint(0, 2048-self.crop[1])
        self.y1 = random.randint(0, 1024-self.crop[0])

        self.x2 = self.x1+self.crop[1]
        self.y2 = self.y1+self.crop[0]

        self.offset = [self.x1, self.y1]
        

    def get_bboxes(self,):
        filtered_bboxes = []
        with open(self.gt_polygons_path) as f:
            data = json.load(f)

        for element in data["objects"]:
            if element["label"] in THINGS:
                potential_box = {
                    "label": element["label"],
                    "bbox": polygons_to_bboxes(element["polygon"], offset=self.offset),
                }
                if (
                    potential_box["bbox"][0][0] >= 0.0 or
                    potential_box["bbox"][0][1] >= 0.0 or
                    potential_box["bbox"][1][0] < self.x2 or
                    potential_box["bbox"][1][1] < self.y2
                ):
                    filtered_bboxes.append(
                        potential_box
                    )

        return filtered_bboxes

    def get_label_IDs(self):
        image = cv2.imread(self.gt_label_IDs_path, cv2.IMREAD_GRAYSCALE)
        torch_image = torch.tensor(image[self.y1:self.y2, self.x1:self.x2])
        return torch_image.unsqueeze(0).long()

    def get_image(self, scale = 1):
        image = cv2.imread(self.image_path)
        image = image[self.y1:self.y2, self.x1:self.x2]
        if scale != 1:
            image_dims = image.shape
            width = int(image_dims[0] * scale)
            height = int(image_dims[1] * scale)
            dim = (width, height)
            image = cv2.resize(image, dim)
        torch_image = torch.tensor(image)
        return torch_image.permute(2, 0, 1).float()


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder_inp,
        root_folder_gt,
        cities_list,
        crop=(1024, 2048)
    ):
        self.root_folder_inp = root_folder_inp
        self.root_folder_gt = root_folder_gt
        self.cities_list = cities_list
        self.crop = crop
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
                    crop = self.crop
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
