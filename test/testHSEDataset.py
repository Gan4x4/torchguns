import unittest
from src.HSESubset import HSESubset
from src.HSEDataset import HSEDataset
from .utils import draw

class HSESubsetTest(unittest.TestCase):

    path_download = "test/out/"
    path_to_sample = "test/data/HSE"
    path_to_extract = "test/out/HSE"

    def test_load_HSE_train(self):
        ds = HSEDataset(self.path_download, download=True, train=True)
        self.assertEqual(19, len(ds.sub_datasets))

    def test_load_HSE_test(self):
        ds = HSEDataset(self.path_download, download=True, train=False)
        self.assertEqual(7, len(ds.sub_datasets))


"""
    def test_persons(self):
        ds = HSESubset("/home/anton/Code/MIEM/Weapon/gunrec/data/miem_test/office_03",build_cache=False)
        ds.desired_frames = 20
        ds.build_cache()
        for i, (img, bbox) in enumerate(ds):
            pil = draw(img, bbox)
            pil.save(f"test/out/tmp{i}.jpg")
"""

if __name__ == '__main__':
    unittest.main()






