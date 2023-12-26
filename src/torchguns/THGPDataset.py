from .YOLODataset import YOLODataset
from torchvision.datasets.utils import download_and_extract_archive
import os


class THGPDataset(YOLODataset):
    classes = ['hand', 'gun', 'phone']
    class_filter = ['gun']
    url = "https://ml.gan4x4.ru/hse/torchguns/THGP.zip"
    name = "THGP"

    def get_root(self, base_path):
        if self.train:
            postfix = "train2017"
        else:
            postfix = "val2017"
        return f"{base_path}{os.sep}{self.name}{os.sep}images{os.sep}{postfix}{os.sep}"

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        ann_path = img_path.replace("images", "labels")  # TODO refactor to replace only last occurrence
        ann_path = ann_path.replace("png", "txt")
        return ann_path

    def download(self, path, train=True):
        """
            Validation and train part in one archive
        """
        download_and_extract_archive(
            self.url,
            path
        )


class HGPDataset(THGPDataset):
    url = "https://ml.gan4x4.ru/hse/torchguns/HGP.zip"
    name = "HGP"

    def get_root(self, base_path):
        root = super().get_root(base_path)
        return root.replace("THGP", self.name)
