from typing import Optional, Callable

from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torch import Tensor
from glob import glob
import os


class BoundingBoxDataset(VisionDataset):
    valid_images = [".jpg", ".gif", ".png", ".jpeg"]
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
        len(self.images)

    def __getitem__(self, n):
        img_tensor = self.image(n)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        boxes = self.boxes(n, img_tensor)
        if self.target_transform:
            boxes = self.target_transform(boxes)
        return img_tensor, boxes

    def image(self, n):
        """
            Return  image on n place in PIL format
        """
        path = self.image_paths[n]
        return read_image(path)

    def boxes(self, n) -> Tensor:
        """
            Must return bbox list for image on n place in format
            [class_num, x1, y1, x2, y2]
        """
        raise NotImplementedError
