from torchvision.datasets import VisionDataset
from torch import Tensor
class BoundingBoxDataset(VisionDataset):

    def __len__(self):
        pass

    def __getitem__(self, n):
        pass

    def getboxes(self, n) -> Tensor:
        """
            Must return bbox list for image on n place in format
            [class_num, x1, y1, x2, y2]
        """
        pass

