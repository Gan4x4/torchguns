import unittest
from torchguns.utils import draw
from tqdm import tqdm
import shutil
from torchguns.THGPDataset import HGPDataset, THGPDataset


class TestTHGPDataset(unittest.TestCase):
    path = "test/data"

    def test_load_THGP_train2017(self):
        """
            Load real train part of THGP dataset
        """
        shutil.rmtree("test/out/THGP", ignore_errors=True)
        ds = THGPDataset("test/out", download=True, train=True)
        self.assertEqual(5000, len(ds))

    def test_load_THGP_val2017(self):
        """
            Load real validation part of THGP dataset
        """
        shutil.rmtree("test/out/THGP", ignore_errors=True)
        ds = THGPDataset("test/out", download=True, train=False)
        self.assertEqual(960, len(ds))

    def test_load_HGP_train2017(self):
        """
            Load real train part of HGP dataset
        """
        shutil.rmtree("test/out/HGP", ignore_errors=True)
        ds = HGPDataset("test/out", download=True, train=True)
        self.assertEqual(1989, len(ds))

    def test_load_HGP_val2017(self):
        """
            Load real validation part of THGP dataset
        """
        shutil.rmtree("test/out/HGP", ignore_errors=True)
        ds = HGPDataset("test/out", download=True, train=False)
        self.assertEqual(210, len(ds))

    def test_patch(self):
        ds = HGPDataset("test/out", download=False, train=False)
        self.assertEqual(210, len(ds))

    def test_getitem(self):
        ds = THGPDataset(self.path)
        ds.class_filter = []
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/thgp_1.jpg")

    def test_empty_bbox_list(self):
        ds = THGPDataset(self.path)
        self.assertCountEqual(ds.class_filter, ['gun'])
        im, bbox = ds[1]  # item number 780 in real dataset this item has no guns bbox
        self.assertEqual(len(bbox), 0)
        pil = draw(im, bbox)
        pil.save("test/out/thgp_780.jpg")


if __name__ == '__main__':
    unittest.main()
