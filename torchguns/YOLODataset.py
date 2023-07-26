from .BoundingBoxDataset import BoundingBoxDataset
import csv
import os
from torch import Tensor


class YOLODataset(BoundingBoxDataset):
    postfix = ".txt"

    # Override
    def boxes(self, n, image):
        boxes = []
        filename = self.getfilename(n)
        with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                class_num = int(row[0])
                xywh = list(map(float, row[1:]))
                assert len(xywh) == 4
                height, width = image.shape[1:]
                xyxy = YOLODataset.xywh2xyxy(xywh, width, height)
                boxes.append([class_num] + list(xyxy))
        return Tensor(boxes)

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
