import unittest
from torchguns.YOLODataset import YOLODataset

class TestYOLODataset(unittest.TestCase):
    def test_getitem(self):
        ds = YOLODataset("test/data/yolo_dataset")
        for im, bbox in ds:
            self.assertEqual(len(bbox), 6)  # add assertion here


if __name__ == '__main__':
    unittest.main()
