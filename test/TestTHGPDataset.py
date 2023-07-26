import unittest
from torchguns.THGPDataset import THGPDataset
from .utils import draw


class TestTHGPDataset(unittest.TestCase):
    classes = ["phone", "gun", "hand"]

    def test_getitem(self):
        ds = THGPDataset("test/data/thgp_dataset/images/val2017")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")


if __name__ == '__main__':
    unittest.main()
