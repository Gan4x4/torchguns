import unittest
from torchguns.VideoDataset import VideoDataset
from glob import glob
from torchguns.utils import draw
import os


def delete_jpeg(folder):
    jpegs = glob(folder + "/*.jpg")
    for f in jpegs:
        os.remove(f)
class TestVideoDataset(unittest.TestCase):
    folder = "test/data/HSEDataset/store_07"

    @unittest.skip("fff")
    def test_build_cache(self):
        delete_jpeg(self.folder+"/obj_train_data")
        VideoDataset(self.folder, build_cache=True)


    #@unittest.skip("fff")
    def test_smoke(self):
        ds = VideoDataset(self.folder, build_cache=False)
        im, bbox = ds[0]
        self.assertEqual(len(bbox), 1)  # One bbox
        pil = draw(im, bbox)
        pil.save("test/out/tmp.jpg")


if __name__ == '__main__':
    unittest.main()
