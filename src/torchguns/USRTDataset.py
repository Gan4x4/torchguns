from .BoundingBoxDataset import BoundingBoxDataset
from .YOLODataset import YOLODataset
import xml.etree.ElementTree as ElementTree
import torch
from torch import Tensor
import os
from torchvision.datasets.utils import download_and_extract_archive
import warnings


class USRTDataset(BoundingBoxDataset):
    classes = ['Handgun', 'Short_rifle', 'Knife']
    class_filter = ['Handgun', 'Short_rifle']  # firearms only
    original_url = [
        'https://uses0-my.sharepoint.com/:u:/g/personal/jsalazar_us_es/Ee7yqsE68U9PhnNHZneIuTABfTX5P9iVClJyxIKORfBJvg?e=VpXVtT']
    urls = ['https://ml.gan4x4.ru/hse/torchguns/USRT.zip']
    name = "USRT"

    def boxes(self, n, image=None):
        bboxs = []
        filename = self.getfilename(n)
        #
        xml = ElementTree.parse(filename)
        objects = xml.findall('object')
        for o in objects:
            # if o.find('name').text in self.weapon_types:
            class_name = o.find('name').text
            assert class_name in self.classes, f"Class {class_name} not found"
            class_num = self.classes.index(class_name)
            bbox_obj = o.find('bndbox')
            bbox = [
                class_num,
                bbox_obj.find('xmin').text,
                bbox_obj.find('ymin').text,
                bbox_obj.find('xmax').text,
                bbox_obj.find('ymax').text]
            bbox = list(map(float, bbox))  # str ->float
            bboxs.append(bbox)
        return Tensor(bboxs)

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        return img_path.replace('.jpg', '.xml')

    def download(self, root, train=None):
        """
            Validation and train part in one archive
        """
        if train is not None:
            warnings.warn("USRT dataset hasn't train/test split")
        download_and_extract_archive(self.urls[0], root)  # images

    def get_root(self, base_path):
        return f"{base_path}{os.sep}{self.name}{os.sep}"


class USRTDatasetWithPersons(USRTDataset):
    classes = ['Person', 'Handgun', 'Short_rifle', 'Knife', ]
    class_filter = ['Person', 'Handgun', 'Short_rifle']

    def boxes(self, n, image=None):
        original_boxes = super().boxes(n, image)
        filename = self.getfilename(n)
        parts = filename.split(os.sep)
        new_filename = parts[-1].replace(".xml", ".txt")
        parts[-1] = "persons"
        parts.append(new_filename)
        new_path = os.sep.join(parts)
        h, w = YOLODataset.extract_hw(image)
        person_boxes = YOLODataset.read_yolo_file(new_path, h, w)
        all_boxes = torch.cat((original_boxes, Tensor(person_boxes)), dim=0)
        return all_boxes
