from .YOLODataset import YOLODataset
from .VideoDataset import VideoDataset
import os
import torch


class HSESubset(VideoDataset):
    classes = ['Person', 'Gun']
    class_filter = ['Person', 'Gun']

    def boxes(self, n, image=None):
        gun_boxes = super().boxes(n, image)
        if len(gun_boxes) > 0:
            gun_boxes[:, 0] = self.classes.index("Gun")
        filename = self.getfilename(n)
        new_path = filename.replace('obj_train_data', 'persons')
        if not os.path.exists(new_path):
            return gun_boxes
        h, w = YOLODataset.extract_hw(image)
        person_boxes = YOLODataset.read_yolo_file(new_path, h, w)
        return torch.cat((gun_boxes, torch.Tensor(person_boxes)), dim=0)

    def get_person_classes(self):
        return [self.classes.index('Person')]

    def get_weapon_classes(self):
        return [self.classes.index('Gun')]
