import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from torchguns.PersonDataset import PersonDataset
from torchguns.USRTDataset import USRTDatasetWithPersons


class DummyDataset:
    class_filter = ['Person', 'Gun']

    def __init__(self, images, boxes, person=(5,), weapon=(9,)):
        self.images = images
        self.boxes_by_num = boxes
        self.person = list(person)
        self.weapon = list(weapon)

    def __len__(self):
        return len(self.images)

    def image(self, i):
        return self.images[i]

    def boxes(self, i, img):
        return self.boxes_by_num[i].clone()

    def get_person_classes(self):
        return self.person

    def get_weapon_classes(self):
        return self.weapon

    def __getitem__(self, i):
        raise AssertionError('PersonDataset should use raw image()/boxes()')


class SizeDataset(DummyDataset):
    def __init__(self):
        images = [torch.zeros((3, 100, 100)), torch.zeros((3, 200, 50))]
        super().__init__(images, [torch.empty((0, 5)), torch.empty((0, 5))])

    def boxes(self, i, img):
        h, w = img.shape[1:]
        return torch.tensor([[5., 0., 0., float(w), float(h)]])


class PersonDatasetTest(unittest.TestCase):
    def test_extract_persons_uses_frame_image(self):
        pds = PersonDataset(SizeDataset())
        self.assertTrue(torch.equal(pds.persons[1, 1:5], torch.tensor([0., 0., 50., 200.])))

    def test_empty_persons(self):
        ds = DummyDataset([torch.zeros((3, 10, 10))], [torch.empty((0, 5))])
        pds = PersonDataset(ds)
        self.assertEqual(0, len(pds))
        self.assertEqual((0, 6), tuple(pds.persons.shape))

    def test_class_filter_is_unchanged(self):
        ds = DummyDataset([torch.zeros((3, 10, 10))], [torch.tensor([[5., 0., 0., 5., 5.]])])
        before = list(ds.class_filter)
        PersonDataset(ds)
        self.assertEqual(before, ds.class_filter)

    def test_getitem_uses_raw_image_and_boxes(self):
        ds = DummyDataset(
            [torch.zeros((3, 100, 100))],
            [torch.tensor([
                [5., 10., 10., 60., 80.],
                [9., 20., 20., 30., 30.],
            ])],
        )
        pds = PersonDataset(ds)
        img, boxes = pds[0]
        self.assertEqual((3, 84, 75), tuple(img.shape))
        self.assertEqual((1, 4), tuple(boxes.shape))

    def test_unarmed_shape(self):
        ds = DummyDataset([torch.zeros((3, 100, 100))], [torch.tensor([[5., 10., 10., 60., 80.]])])
        pds = PersonDataset(ds)
        _, boxes = pds[0]
        self.assertEqual((0, 4), tuple(boxes.shape))

    def test_get_squared_person_bbox(self):
        ds = DummyDataset([torch.zeros((3, 100, 100))], [torch.tensor([[5., 10., 10., 60., 80.]])])
        pds = PersonDataset(ds)
        self.assertEqual([0, 3, 75, 87], pds.get_squared_person_bbox(0))

    def test_upscale(self):
        ds = DummyDataset(
            [torch.zeros((3, 100, 100))],
            [torch.tensor([
                [5., 10., 10., 60., 80.],
                [9., 20., 20., 30., 30.],
            ])],
        )
        pds = PersonDataset(ds)
        restored = pds.upscale(0, pds[0][1])
        self.assertTrue(torch.allclose(restored, torch.tensor([[20., 20., 30., 30.]])))

    def test_nearby_weapon_is_not_clipped(self):
        ds = DummyDataset(
            [torch.zeros((3, 100, 100))],
            [torch.tensor([
                [5., 10., 10., 60., 80.],
                [9., 80., 20., 90., 30.],
            ])],
        )
        pds = PersonDataset(ds)
        img, boxes = pds[0]
        self.assertEqual([0, 3, 91, 87], pds.get_squared_person_bbox(0))
        self.assertEqual((3, 84, 91), tuple(img.shape))
        self.assertTrue(torch.allclose(boxes, torch.tensor([[80., 17., 90., 27.]])))
        self.assertTrue(torch.allclose(pds.upscale(0, boxes), torch.tensor([[80., 20., 90., 30.]])))

    def test_usrt_with_persons(self):
        sample = Path(__file__).resolve().parent / 'data' / 'USRT'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'USRT'
            root.mkdir()
            stem = 'Cam1-From09-23-00To10-03-25_Segment_0_x264_frame_29'
            shutil.copy2(sample / f'{stem}.jpg', root / f'{stem}.jpg')
            shutil.copy2(sample / f'{stem}.xml', root / f'{stem}.xml')
            person_dir = root / 'persons'
            person_dir.mkdir()
            person_dir.joinpath(f'{stem}.txt').write_text('0 0.572917 0.370370 0.104167 0.370370\n')
            ds = USRTDatasetWithPersons(tmp)
            pds = PersonDataset(ds)
            self.assertEqual(1, len(pds))
            img, boxes = pds[0]
            self.assertTrue(img.shape[1] > 0 and img.shape[2] > 0)
            self.assertEqual((1, 4), tuple(boxes.shape))


if __name__ == '__main__':
    unittest.main()
