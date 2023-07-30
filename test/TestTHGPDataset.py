import unittest
from torchguns.THGPDataset import THGPDataset
from .utils import draw
from tqdm import tqdm

class TestTHGPDataset(unittest.TestCase):

    def test_empty_bbox_list(self):
        ds = THGPDataset("/home/anton/Code/MIEM/Weapon/TYolov5/Datasets/THGP/images/val2017")
        #ds.class_filter = []
        im, bbox = ds[780]
        self.assertEqual(len(bbox), 0)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    def test_getitem(self):
        ds = THGPDataset("test/data/thgp_dataset/images/val2017")
        ds.class_filter = []
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    def test_len(self):
        ds = THGPDataset("test/data/thgp_dataset/images/val2017")
        self.assertEqual(len(ds), 1)

    def test_scan(self):
        ds = THGPDataset("/home/anton/Code/MIEM/Weapon/TYolov5/Datasets/HGP/images/val2017")
        for i, (im, bbox) in enumerate(ds):
            if len(im) >0:
                print(im.shape)
                self.assertEqual(im.shape[0], 3, ds.image_paths[i])
            #print(bbox.shape,i,ds.image_paths[i])
            #if bbox.shape[0] > 0:
            message = f'Invalid bbox {bbox} item {i}'
            self.assertEqual(bbox.shape[1] , 4) , message


    def get_weapon_clases(self):
        return self.classes.index("gun")

    def get_person_clases(self):
        return []


if __name__ == '__main__':
    unittest.main()
