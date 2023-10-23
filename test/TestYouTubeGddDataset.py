import unittest
from src.YouTubeGddDataset import YouTubeGddDataset
from .utils import draw
from tqdm import tqdm

class TestYouTubeGddDataset(unittest.TestCase):

    def test_getitem(self):
        ds = YouTubeGddDataset("test/data/youtube-gdd_dataset/images")
        ds.class_filter = []
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")

    def test_len(self):
        ds = YouTubeGddDataset("test/data/youtube-gdd_dataset/images")
        self.assertEqual(len(ds), 1)


if __name__ == '__main__':
    unittest.main()
