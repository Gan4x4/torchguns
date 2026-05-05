from .BoundingBoxDataset import BoundingBoxDataset
from .YOLODataset import YOLODataset
import xml.etree.ElementTree as ElementTree
import torch
from torch import Tensor
import os
from torchvision.datasets.utils import download_url, extract_archive
import zipfile


class USRTDataset(BoundingBoxDataset):
    classes = ['Handgun', 'Short_rifle', 'Knife']
    class_filter = ['Handgun', 'Short_rifle']
    original_url = [
        'https://uses0-my.sharepoint.com/:u:/g/personal/jsalazar_us_es/Ee7yqsE68U9PhnNHZneIuTABfTX5P9iVClJyxIKORfBJvg?e=VpXVtT']
    urls = ['https://ml.gan4x4.ru/hse/torchguns/USRT.zip']
    name = "USRT"

    def boxes(self, n, image=None):
        bboxs = []
        filename = self.getfilename(n)
        xml = ElementTree.parse(filename)
        objects = xml.findall('object')
        for o in objects:
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
            bboxs.append(list(map(float, bbox)))
        return Tensor(bboxs) if bboxs else torch.empty((0, 5))

    def getfilename(self, n):
        img_path = self.image_paths[n]
        return img_path.replace('.jpg', '.xml')

    def download(self, root, train=None):
        os.makedirs(root, exist_ok=True)
        name = "USRT.zip"
        path = os.path.join(root, name)
        if os.path.exists(path) and not zipfile.is_zipfile(path):
            os.remove(path)
        download_url(self.urls[0], root=root, filename=name)
        if not zipfile.is_zipfile(path):
            raise RuntimeError(f"Downloaded file is not a zip file: {path}")
        extract_archive(path, root)

    def get_root(self, base_path):
        return f"{base_path}{os.sep}{self.name}{os.sep}"

    def get_weapon_classes(self):
        return [self.classes.index('Handgun'), self.classes.index('Short_rifle')]


class USRTDatasetWithPersons(USRTDataset):
    classes = ['Person', 'Handgun', 'Short_rifle', 'Knife']
    class_filter = ['Person', 'Handgun', 'Short_rifle']

    def weapon_boxes(self, n):
        bboxs = []
        xml = ElementTree.parse(self.getfilename(n))
        for o in xml.findall('object'):
            class_name = o.find('name').text
            assert class_name in USRTDataset.classes, f"Class {class_name} not found"
            # Keep base USRT ids for weapons, then shift by +1 because Person occupies class 0 here.
            class_num = USRTDataset.classes.index(class_name) + 1
            bbox_obj = o.find('bndbox')
            bbox = [
                class_num,
                bbox_obj.find('xmin').text,
                bbox_obj.find('ymin').text,
                bbox_obj.find('xmax').text,
                bbox_obj.find('ymax').text,
            ]
            bboxs.append(list(map(float, bbox)))
        return Tensor(bboxs) if bboxs else torch.empty((0, 5))

    def boxes(self, n, image=None):
        original_boxes = self.weapon_boxes(n)
        filename = self.getfilename(n)
        parts = filename.split(os.sep)
        new_filename = parts[-1].replace(".xml", ".txt")
        parts[-1] = "persons"
        parts.append(new_filename)
        new_path = os.sep.join(parts)
        h, w = YOLODataset.extract_hw(image)
        person_boxes = YOLODataset.read_yolo_file(new_path, h, w) if os.path.exists(new_path) else []
        person_boxes = Tensor(person_boxes) if person_boxes else torch.empty((0, 5))
        return torch.cat((person_boxes, original_boxes), dim=0)

    def get_person_classes(self):
        return [self.classes.index('Person')]

    def get_weapon_classes(self):
        return [self.classes.index('Handgun'), self.classes.index('Short_rifle')]
