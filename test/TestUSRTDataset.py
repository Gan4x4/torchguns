import unittest
from torchguns.USRTDataset import USRTDataset
from torchguns.USRTDataset import USRTDatasetWithPersons
from .utils import draw


# Pytorch wrapper to this dataset: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset


class TestMockAttackDataset(unittest.TestCase):
    @unittest.skip("For speedup debugging")
    def test_getitem(self):
        ds = USRTDataset("test/data/USRT")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    @unittest.skip("Used only to find all weapon types")
    def test_scan(self):
        ds = USRTDataset("/home/anton/Code/MIEM/Weapon/gunrec/data/USRT")
        for im, bbox in ds:
            # print(len(bbox))
            pass

    def test_person(self):
        ds = USRTDatasetWithPersons("test/data/USRT")
        im, bbox = ds[0]
        #self.assertEqual(len(bbox), 2)  # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")



if __name__ == '__main__':
    unittest.main()
