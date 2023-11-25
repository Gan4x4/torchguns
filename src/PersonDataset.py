from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from .YOLODataset import YOLODataset


class PersonDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.persons = self.extract_persons()
        if "Person" in self.base_dataset.class_filter:
            self.base_dataset.class_filter.remove("Person")

    def extract_persons(self):
        persons = []
        l = len(self.base_dataset)
        img, _ = self.base_dataset[0]
        for i in range(l):
            boxes = self.base_dataset.boxes(i, img)
            for b in boxes:
                if b[0] == 0:  # Person
                    boxes_with_num = torch.cat((b, torch.Tensor([i])))
                    persons.append(boxes_with_num)
        return torch.stack(persons)

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, n):
        person_bb = self.persons[n]
        base_num = int(person_bb[-1].item())
        pil, boxes = self.base_dataset[base_num]
        # align  the person box
        p_box = self.expand(person_bb[1:5], pil)
        x1, y1, x2, y2 = p_box.int().tolist()

        # height, width = YOLODataset.extract_hw(tensor_img=pil)
        # x1 *= width
        # x2 *= width
        # y1 *= height
        # y2 *= height

        weapon_bb = boxes  # because of filter
        patch = transforms.functional.crop(pil, y1, x1, height=y2 - y1, width=x2 - x1)

        if len(weapon_bb):
            overlapped_weapon_bb = self.get_overlapped_weapon_bbox(p_box, weapon_bb)
        else:
            overlapped_weapon_bb = torch.Tensor([])  # unarmed person

        # rescale
        overlapped_weapon_bb = self.rescale_weapon(p_box, overlapped_weapon_bb)

        return patch, overlapped_weapon_bb

    def expand(self, person_bb, img):
        px1, py1, px2, py2 = person_bb

        h = py2 - py1
        w = px2 - px1
        deltas = torch.Tensor([-h * 0.1, -w * 0.1, h * 0.1, w * 0.1])
        ih, iw = YOLODataset.extract_hw(img)
        # to square
        if h > w:
            d = (h - w) / 2
            deltas[0] -= d
            deltas[2] += d
        if w > h:
            d = (w - h) / 2
            deltas[1] -= d
            deltas[3] += d

        p = person_bb + deltas
        p[2] = min(iw, p[2])
        p[3] = min(ih, p[3])
        p[p < 0] = 0
        return p

    def get_overlapped_weapon_bbox(self, person_bb, weapon_bb):
        person_bb = person_bb.unsqueeze(0)
        iou = torchvision.ops.box_iou(person_bb, weapon_bb)
        index = iou > 0
        overlapped = weapon_bb[index[0]]
        return overlapped

    def rescale_weapon(self, person_bb, weapon_bbs):
        px1, py1, px2, py2 = person_bb
        height = py2 - py1
        width = px2 - px1
        out = []
        for x1, y1, x2, y2 in weapon_bbs:
            x1n = max(0, x1 - px1)
            y1n = max(0, y1 - py1)
            x2n = min(width, x2 - px1)
            y2n = min(height, y2 - py1)
            out.append([x1n, y1n, x2n, y2n])
        return torch.Tensor(out)

    def _get_frame_num(self, n):
        return int(self.persons[n][-1].item())
