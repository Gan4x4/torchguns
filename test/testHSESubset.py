import unittest
from src.HSESubset import HSESubset
from .utils import draw

class HSESubsetTest(unittest.TestCase):

    @unittest.skip("")
    def test_base(self):
        ds = HSESubset("test/data/HSEDataset/store_07")
        for i, (img, bbox) in enumerate(ds):

            pil = draw(img, bbox)
            pil.save(f"test/out/tmp{i}.jpg")

    def test_persons(self):
        ds = HSESubset("/home/anton/Code/MIEM/Weapon/gunrec/data/miem_test/office_03",build_cache=False)
        ds.desired_frames = 20
        ds.build_cache()
        for i, (img, bbox) in enumerate(ds):
            pil = draw(img, bbox)
            pil.save(f"test/out/tmp{i}.jpg")


if __name__ == '__main__':
    unittest.main()






