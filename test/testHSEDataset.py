import unittest
from torchguns.HSEDataset import HSEDataset

class HSESubsetTest(unittest.TestCase):
    path_download = "test/out/"
    path_to_sample = "test/data/HSE"

    def test_load_HSE_train(self):
        ds = HSEDataset(self.path_download, download=True, train=True)
        self.assertEqual(19, len(ds.sub_datasets))

    def test_load_HSE_test(self):
        ds = HSEDataset(self.path_download, download=True, train=False)
        self.assertEqual(7, len(ds.sub_datasets))

    def test_desired_frames(self):
        ds = HSEDataset(self.path_download,
                        download=True,
                        train=False,
                        desired_frames=10  # extract only 10 frames from each video
                        )
        for d in ds.sub_datasets.values():
            self.assertEqual(10, len(d), msg=d.name)


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
