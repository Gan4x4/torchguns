import unittest
from torchguns.PersonDataset import PersonDataset
from torchguns.USRTDataset import USRTDatasetWithPersons
from torchguns.HSESubset import  HSESubset
from .utils import draw

class PersonDatasetTest(unittest.TestCase):

    @unittest.skip("For article")
    def test_base(self):
        ds = HSESubset("test/data/HSE/store_07", build_cache=False)
        #ds.class_filter = ['Handgun', 'Short_rifle']
        img, bbox = ds[4]
        pds = PersonDataset(ds)
        pil = draw(img, bbox, 6)#,["red", "green"])
        pil.save(f"test/out/hse_store_07_orig_{0}.jpg")

        for i, (img, bbox) in enumerate(pds):
            #print(len(bbox))
            pil = draw(img, bbox, 3, "red")
            pil.save(f"test/out/hse_store_07_person_{i}.jpg")

    def test_base(self):
        name = "street_05"
        frame_num = 4
        ds = HSESubset(f"test/out/hse_test/{name}", build_cache=True)
        # ds.class_filter = ['Handgun', 'Short_rifle']
        img, bbox = ds[frame_num]
        pil = draw(img, bbox, 4,["red", "green",  "green"])
        pil.save(f"test/out/hse_{name}_orig_{frame_num}.jpg")
        pds = PersonDataset(ds)

        for i, (img, bbox) in enumerate(pds):
            # print(len(bbox))
            if pds._get_frame_num(i) == frame_num:
                pil = draw(img, bbox, 3, "red")
                pil.save(f"test/out/hse_{name}_{frame_num}_person_{i}.jpg")
            elif pds._get_frame_num(i) > frame_num:
                break


"""
    TODO copy txt files from 
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

"""





if __name__ == '__main__':
    unittest.main()






