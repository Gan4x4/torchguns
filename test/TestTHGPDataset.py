import unittest
# from torchguns.THGPDataset import THGPDataset
from .utils import draw
from tqdm import tqdm
import shutil
from torchguns.THGPDataset import THGPDataset


class TestTHGPDataset(unittest.TestCase):
    path = "test/data/THGP/images/val2017"

    @unittest.skip("Time consuming operation")
    def test_load_THGP_train2017(self):
        """
            Load real train part of THGP dataset
        """
        shutil.rmtree("test/out/THGP", ignore_errors=True)
        ds = THGPDataset("test/out", download=True, train=True)
        self.assertEqual(5000, len(ds))

    def test_load_HGP_train2017(self):
        """
            Load real train part of HGP dataset
        """
        shutil.rmtree("test/out/HGP", ignore_errors=True)
        ds = THGPDataset("test/out", download=True, train=True)
        self.assertEqual(1989, len(ds))


    @unittest.skip("Time consuming operation")
    def test_load_val2017(self):
        """
            Load real validation part of THGP dataset
        """
        shutil.rmtree("test/out/THGP", ignore_errors=True)
        ds = THGPDataset("test/out", download=True, train=False)
        self.assertEqual(960, len(ds))

    def test_getitem(self):
        ds = THGPDataset(self.path)
        ds.class_filter = []
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/thgp_1.jpg")

    def test_empty_bbox_list(self):
        ds = THGPDataset(self.path)
        self.assertCountEqual(ds.class_filter,['gun'])
        im, bbox = ds[1] # item number 780 in real dataset this item has no guns bbox
        self.assertEqual(len(bbox), 0)
        pil = draw(im, bbox)
        pil.save("test/out/thgp_780.jpg")

    """

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

"""

if __name__ == '__main__':
    unittest.main()
