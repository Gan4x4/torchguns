import unittest
from torchguns.MockAttackDataset import MockAttackDataset
from .utils import draw

# Pytorch wrapper to this dataset: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset


class TestMockAttackDataset(unittest.TestCase):
    def test_getitem(self):
        ds = MockAttackDataset("test/data/mock_attack_dataset")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)   # Two bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")
    @unittest.skip("Used only to find all weapon types")
    def test_scan(self):
        ds = MockAttackDataset("/home/anton/Code/MIEM/Weapon/gunrec/data/EtiquetadosDataset")
        for im, bbox in ds:
            #print(len(bbox))
            pass


if __name__ == '__main__':
    unittest.main()
