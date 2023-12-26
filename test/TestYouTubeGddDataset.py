import unittest
from torchguns.YouTubeGddDataset import YouTubeGddDataset
from torchguns.utils import draw
from tqdm import tqdm
import shutil
class TestYouTubeGddDataset(unittest.TestCase):

    path_download = "test/out/"
    path_to_sample = "test/data"
    path_to_extract = "test/out/YouTube-GDD"


    def setUp(self):
        pass

    def tearDown(self):
        shutil.rmtree(self.path_to_extract, ignore_errors = True)

    #@unittest.skip("Time consuming operation")
    def test_load_GDD_train(self):
        """
            Load real train part of YouTube-GDD dataset
        """
        ds = YouTubeGddDataset(self.path_download, download=True, train=True)
        self.assertEqual(4000, len(ds))

    #@unittest.skip("Time consuming operation")
    def test_load_GDD_val(self):
        """
            Load real train part of YouTube-GDD dataset
        """
        ds = YouTubeGddDataset(self.path_download, download=True, train=False)
        self.assertEqual(500, len(ds))

    def test_getitem(self):
        ds = YouTubeGddDataset("test/data")
        ds.class_filter = []
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 2)  # add assertion here
        pil = draw(im, bbox)
        pil.save("test/out/gdd_0.jpg")


if __name__ == '__main__':
    unittest.main()
