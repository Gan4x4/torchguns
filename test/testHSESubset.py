import unittest
from torchguns.HSESubset import HSESubset
from torchguns.utils import draw


class HSESubsetTest(unittest.TestCase):
    path_download = "test/out/"
    path_to_sample = "test/data/HSE/store_07"

    def test_base(self):
        ds = HSESubset(self.path_to_sample)
        img, bbox = ds[0]
        self.assertIsNotNone(img)
        # pil = draw(img, bbox)
        # pil.save(f"test/out/tmp_{ds.name}_1.jpg")

    def test_desired_frames(self):
        ds = HSESubset(self.path_to_sample, desired_frames=3)
        self.assertEqual(3, len(ds))

    def test_fps(self):
        ds = HSESubset(self.path_to_sample, fps=0.15)
        self.assertEqual(8, len(ds))

    def test_fps_and_desired_frames_together(self):
        ds = HSESubset(self.path_to_sample, fps=0.15, desired_frames=100)
        self.assertWarns(Warning)
        self.assertEqual(8, len(ds))


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
