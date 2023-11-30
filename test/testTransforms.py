import unittest
from torchguns.HSESubset import HSESubset
from torchvision.transforms import v2, Resize, RandomCrop, Compose
from torchguns.utils import draw

class TransformsTestCase(unittest.TestCase):
    path_download = "test/out/"
    path_to_sample = "test/data/HSE/store_07"
    def test_v2_rotation(self):
        ds = HSESubset(self.path_to_sample, build_cache= False)
        img, bbox = ds[0]
        d_img = draw(img, bbox)
        d_img.save("test/out/transform1.jpg")

        transforms = v2.Compose([
            # v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomRotation(90)
        ])

        ds.transforms = transforms
        t_img, t_bbox = ds[0]

        d_img = draw(t_img, t_bbox)
        d_img.save("test/out/transform2.jpg")
        self.assertFalse(bbox.allclose(t_bbox))



    def test_backward_compatibility(self):
        ds = HSESubset(self.path_to_sample, build_cache=False)
        img, bbox = ds[0]

        image_transform = Compose([
            # v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomCrop(10)
        ])

        ds.transform = image_transform
        t_img, t_bbox = ds[0]
        self.assertTrue(bbox.allclose(t_bbox))

    def test_wrong_usage(self):
        ds = HSESubset(self.path_to_sample, build_cache=False)
        one_image_transform = Compose([
            RandomCrop(10)
        ])
        ds.transforms = one_image_transform
        with self.assertRaises(Exception) as context:
            t_img, t_bbox = ds[0]



if __name__ == '__main__':
    unittest.main()
