from .YOLODataset import YOLODataset
from torchvision.datasets.utils import download_and_extract_archive
import os
class YouTubeGddDataset(YOLODataset):
    classes = ['person', 'gun']
    class_filter = ['gun']
    gdrive_urls = ["https://drive.google.com/file/d/1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa",
                    "https://github.com/UCAS-GYX/YouTube-GDD/blob/main/labels.zip"]
    urls = ["https://ml.gan4x4.ru/hse/torchguns/YouTube-GDD/YouTube-GDD.zip",
            "https://ml.gan4x4.ru/hse/torchguns/YouTube-GDD/YouTube-GDD_labels.zip"]


    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        ann_path = img_path.replace("images", "labels")  # TODO refactor to replace only last occurrence
        ann_path = ann_path.replace("jpg", "txt")
        return ann_path

    def download(self, path, train=True):
        """
            Validation and train part in one archive
        """
        download_and_extract_archive(self.urls[0],path) # images
        download_and_extract_archive(self.urls[1],path+os.sep + "YouTube-GDD") # labels

        if train == True:
            postfix = "train"
        else:
            postfix = "val"
        return f"{path}{os.sep}YouTube-GDD{os.sep}images{os.sep}{postfix}{os.sep}"
