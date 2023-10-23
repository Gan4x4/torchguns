import unittest
from src.YOLODataset import YOLODataset
from .utils import draw


class TestYOLODataset(unittest.TestCase):
    def test_getitem(self):
        ds = YOLODataset("test/data/yolo_dataset")
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 6)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")


if __name__ == '__main__':
    unittest.main()
