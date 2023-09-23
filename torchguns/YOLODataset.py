from .BoundingBoxDataset import BoundingBoxDataset
import csv
import os
from torch import Tensor
import PIL

class YOLODataset(BoundingBoxDataset):
    postfix = ".txt"

    # Override
    def boxes(self, n, image):
        filename = self.getfilename(n)
        height, width = YOLODataset.extract_hw(image)
        boxes = YOLODataset.read_yolo_file(filename, height, width)
        return Tensor(boxes)

    @staticmethod
    def read_yolo_file(filename, height, width):
        boxes = []
        with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                class_num = int(row[0])
                xywh = list(map(float, row[1:]))
                assert len(xywh) == 4
                xyxy = YOLODataset.xywh2xyxy(xywh, width, height)
                boxes.append([class_num] + list(xyxy))
        return boxes

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        return self.root + os.sep + f"frame_{n:06d}{self.postfix}"

    @staticmethod
    def xywh2xyxy(bbox, width, height):
        x, y, w, h = bbox
        x1 = int(width * (x - w / 2))
        x2 = int(width * (x + w / 2))
        y1 = int(height * (y - h / 2))
        y2 = int(height * (y + h / 2))
        return x1, y1, x2, y2
    @staticmethod
    def extract_hw(img):
        if img is None:
            height = 1
            width = 1
        elif isinstance(img,Tensor):
            height, width = img.shape[1:]
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise ValueError("Unsupported image type")

        return height, width
