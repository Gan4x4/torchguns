import unittest
from torchguns.USRTDataset import USRTDataset
from torchguns.USRTDataset import USRTDatasetWithPersons
from torchguns.utils import draw
import shutil

# Pytorch wrapper to this dataset: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset


class TestUSRTDataset(unittest.TestCase):
    path_download = "test/out/"
    path_to_sample = "test/data/USRT/images/val"
    path_to_extract = "test/out/USRT"

    def tearDown(self):
        shutil.rmtree(self.path_to_extract, ignore_errors = True)
    #@unittest.skip("Time consuming operation")
    def test_load_USRT_train(self):
        """
            Load real train part of YouTube-GDD dataset
        """
        ds = USRTDataset(self.path_download, download=True)
        self.assertEqual(5149, len(ds))

    def test_getitem(self):
        ds = USRTDataset("test/data/USRT")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/usrt_0.jpg")


"""
    def test_getitem(self):
        ds = USRTDataset("test/data/USRT")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    #@unittest.skip("Used only to find all weapon types")
    @unittest.skip("For speedup debugging")
    def test_scan(self):
        ds = USRTDataset("/home/anton/Code/MIEM/Weapon/gunrec/data/USRT")
        w = 0
        nw = 0
        for im, bbox in ds:
            if len(bbox):
                w += 1
            else:
                nw += 1
        print("Weapon",w,"No weapon",nw)

    def test_person(self):
        ds = USRTDatasetWithPersons("test/data/USRT")
        im, bbox = ds[0]
        #self.assertEqual(len(bbox), 2)  # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    @unittest.skip("Uncompleted")
    def test_pandas(self):
        ds = USRTDataset("test/data/USRT")
        x = ds.to_pandas()
        print(x.head())

"""

if __name__ == '__main__':
    unittest.main()
