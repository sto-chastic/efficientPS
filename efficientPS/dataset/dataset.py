import torch
import os
import json

import cv2

from .utilities import polygons_to_bboxes

LABELS = {
    # Class: ID, Color
    0: ["unlabeled", (0, 0, 0)],
    1: ["ego vehicle", (0, 0, 0)],
    2: ["rectification border", (0, 0, 0)],
    3: ["out of roi", (0, 0, 0)],
    4: ["static", (0, 0, 0)],
    5: ["dynamic", (111, 74, 0)],
    6: ["ground", (81, 0, 81)],
    7: ["road", (128, 64, 128)],
    8: ["sidewalk", (244, 35, 232)],
    9: ["parking", (250, 170, 160)],
    10: ["rail track", (230, 150, 140)],
    11: ["building", (70, 70, 70)],
    12: ["wall", (102, 102, 156)],
    13: ["fence", (190, 153, 153)],
    14: ["guard rail", (180, 165, 180)],
    15: ["bridge", (150, 100, 100)],
    16: ["tunnel", (150, 120, 90)],
    17: ["pole", (153, 153, 153)],
    18: ["polegroup", (153, 153, 153)],
    19: ["traffic light", (250, 170, 30)],
    20: ["traffic sign", (220, 220, 0)],
    21: ["vegetation", (107, 142, 35)],
    22: ["terrain", (152, 251, 152)],
    23: ["sky", (70, 130, 180)],
    24: ["person", (220, 20, 60)],
    25: ["rider", (255, 0, 0)],
    26: ["car", (0, 0, 142)],
    27: ["truck", (0, 0, 70)],
    28: ["bus", (0, 60, 100)],
    29: ["caravan", (0, 0, 90)],
    30: ["trailer", (0, 0, 110)],
    31: ["train", (0, 80, 100)],
    32: ["motorcycle", (0, 0, 230)],
    33: ["bicycle", (119, 11, 32)],
    -1: ["license plate", (0, 0, 142)],
}

STUFF = [
    "road",
    "sidewalk",
    "building",
    "fence",
    "wall",
    "vegetation",
    "terrain",
    "sky",
    "pole",
    "traffic sign",
    "traffic ligth",
]

THINGS = [
    "car",
    "bicycle",
    "bus",
    "truck",
    "train",
    "motorcycle",
    "person",
    "rider",
]


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

    def get_bboxes(self):
        filtered_bboxes = []
        with open(self.gt_polygons_path[0]) as f:
            data = json.load(f)

        for element in data["objects"]:
            if element["label"] in THINGS:
                filtered_bboxes.append(
                    {
                        "label": element["label"],
                        "bbox": polygons_to_bboxes(element["polygon"]),
                    }
                )

        return filtered_bboxes

    def get_label_IDs(self):
        image = cv2.imread(self.gt_label_IDs_path[0], cv2.IMREAD_GRAYSCALE)
        torch_image = torch.tensor(image)
        return torch_image.unsqueeze(0)

    def get_instances_IDs(self):
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
