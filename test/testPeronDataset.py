import unittest
from src.PersonDataset import PersonDataset
from src.USRTDataset import USRTDatasetWithPersons
from .utils import draw

class PersonDatasetTest(unittest.TestCase):

    def test_indexing(self):
        ds = USRTDatasetWithPersons("test/data/USRT")
        ds.class_filter  = ['Handgun', 'Short_rifle']
        pds = PersonDataset(ds)
        self.assertEqual(3, len(pds))  # add assertion here

    def test_base(self):
        ds = USRTDatasetWithPersons("test/data/USRT")
        ds.class_filter = ['Handgun', 'Short_rifle']
        pds = PersonDataset(ds)
        for i, (img, bbox) in enumerate(pds):
            #print(len(bbox))
            pil = draw(img, bbox)
            pil.save(f"test/out/tmp{i}.jpg")







if __name__ == '__main__':
    unittest.main()






