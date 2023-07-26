from .YOLODataset import YOLODataset


class THGPDataset(YOLODataset):

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        ann_path = img_path.replace("images", "labels")  # TODO refactor to replace only last occurrence
        ann_path = ann_path.replace("png", "txt")
        return ann_path
