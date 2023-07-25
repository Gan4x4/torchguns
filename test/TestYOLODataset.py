import unittest
from torchguns.YOLODataset import YOLODataset


class TestYOLODataset(unittest.TestCase):
    def test_getitem(self):
        ds = YOLODataset("test/data/yolo_dataset")
        for im, bbox in ds:
            self.assertEqual(len(bbox), 6)  # add assertion here
            pil = TestYOLODataset.draw(im, bbox)
            pil.save("test/out/tmp.jpg")
            break

    @staticmethod
    def draw(im, bbox):
        demo_im = draw_bounding_boxes(im, bbox[:, 1:], width=2)
        pil = to_pil_image(demo_im)
        return pil


if __name__ == '__main__':
    unittest.main()
