from .BoundingBoxDataset import BoundingBoxDataset
import xml.etree.ElementTree as ElementTree
from torch import Tensor

class MockAttackDataset(BoundingBoxDataset):
    classes = ['Handgun', 'Short_rifle', 'Knife']
    firearms = ['Handgun', 'Short_rifle']

    def boxes(self, n, image):
        bboxs = []
        filename = self.getfilename(n)
        #
        xml = ElementTree.parse(filename)
        objects = xml.findall('object')
        for o in objects:
            # if o.find('name').text in self.weapon_types:
            class_name = o.find('name').text
            assert class_name in self.classes, f"Class {class_name} not found"
            class_num = self.classes.index(class_name)
            bbox_obj = o.find('bndbox')
            bbox = [
                class_num,
                bbox_obj.find('xmin').text,
                bbox_obj.find('ymin').text,
                bbox_obj.find('xmax').text,
                bbox_obj.find('ymax').text]
            bbox = list(map(float, bbox))  # str ->float
            bboxs.append(bbox)
        return Tensor(bboxs)

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        return img_path.replace('.jpg', '.xml')
