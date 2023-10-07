import unittest
from torchguns.HSESubset import HSESubset
from .utils import draw

class HSESubsetTest(unittest.TestCase):


    def test_base(self):
        ds = HSESubset("test/data/HSEDataset/store_07")
        for i, (img, bbox) in enumerate(ds):
            #print(len(bbox))
            pil = draw(img, bbox)
            pil.save(f"test/out/tmp{i}.jpg")









if __name__ == '__main__':
    unittest.main()






