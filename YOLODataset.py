from .BoundingBoxDataset import BoundingBoxDataset
import csv

class YOLODataset(BoundingBoxDataset):

    def __init__(self, folder):
        self.dir = folder
        self.postfix = ".txt"

    # Override
    def getboxes(self, n):
        boxes = []
        filename = self.getfilename(n)
        with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                class_num = row[0]
                xywh = list(map(float, row[1:]))
                assert len(xywh) == 4
                xyxy = self.xywh2xyxy(xywh)
                boxes.append(class_num + xyxy)
        return boxes

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        return self.folder + f"frame_{n:06d}{self.postfix}"

    def xywh2xyxy(self, bbox):
        x, y, w, h = bbox
        x1 = int(self.width * (x - w / 2))
        x2 = int(self.width * (x + w / 2))
        y1 = int(self.height * (y - h / 2))
        y2 = int(self.height * (y + h / 2))
        return x1, y1, x2, y2
