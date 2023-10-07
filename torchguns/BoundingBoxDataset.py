from typing import Optional, Callable

from torchvision.datasets import VisionDataset
from torchvision.io import read_image
import torch
import os
from torchvision.io.image import ImageReadMode
import pandas as pd


class BoundingBoxDataset(VisionDataset):
    valid_images = [".jpg", ".gif", ".png", ".jpeg"]
    classes = []
    class_filter = []
    return_class_num = False

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.image_paths = self.get_image_paths(self.root)

    def get_image_paths(self, root):
        search_path = root
        paths = []
        for f in os.listdir(search_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in self.valid_images:
                continue
            paths.append(search_path + os.sep + f)
        paths.sort()
        if len(paths) == 0:
            raise FileNotFoundError(f'No images found in {search_path}')
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, n):
        img_tensor = self.image(n)
        boxes = self.boxes(n, img_tensor)
        boxes = self.filter(boxes)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.target_transform:
            boxes = self.target_transform(boxes)
        return img_tensor, boxes

    def filter(self, boxes):
        filtered_boxes = self.filter_by_class(boxes)
        if len(filtered_boxes) > 0 and not self.return_class_num:
            filtered_boxes = filtered_boxes[:, 1:]
        return filtered_boxes

    def filter_by_class(self, boxes):

        if len(self.class_filter) == 0:
            return boxes
        out = []
        class_nums_to_save = self.names2nums()
        for b in boxes:
            if b[0] in class_nums_to_save:
                out.append(b)
        if len(out) > 0:
            out = torch.stack(out)
        else:
            out = torch.empty((0, 4))
        return out

    def names2nums(self):
        nums = []
        for n in self.class_filter:
            assert n in self.classes
            nums.append(self.classes.index(n))
        return nums

    def image(self, n):
        """
            Return  image on n place in PIL format
        """
        path = self.image_paths[n]
        return read_image(path, ImageReadMode.RGB)

    def boxes(self, n) -> torch.Tensor:
        """
            Must return bbox list for image of n place in format
            [class_num, x1, y1, x2, y2]
        """
        raise NotImplementedError

    def get_weapon_classes(self):
        return [0]

    def get_person_classes(self):
        return []

    def to_pandas(self):
        l = len(self)
        data = []
        for i in range(l):
            boxes = self.boxes(i, None)
            f_nums = torch.full(size=(boxes.shape[0], 1), fill_value=1)
            boxes_with_num = torch.cat((f_nums, boxes),dim=1)
            data.append(boxes_with_num)
            data = torch.stack(data).squeeze(0)
            df = pd.DataFrame(data = data, columns=['frame_num', 'class_num', 'cx','cy','w','h'])  # , 'image_path','data_path']
        return df
