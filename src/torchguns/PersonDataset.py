from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from .YOLODataset import YOLODataset


class PersonDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.person_classes = set(map(int, self.base_dataset.get_person_classes()))
        self.weapon_classes = set(map(int, self.base_dataset.get_weapon_classes()))
        self.persons = self.extract_persons()

    def extract_persons(self):
        persons = []
        for i in range(len(self.base_dataset)):
            img = self.base_dataset.image(i)
            boxes = self.base_dataset.boxes(i, img)
            for b in boxes:
                if int(b[0].item()) in self.person_classes:
                    persons.append(torch.cat((b, b.new_tensor([i]))))
        return torch.stack(persons) if persons else torch.empty((0, 6))

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, n):
        base_num = self._get_frame_num(n)
        pil = self.base_dataset.image(base_num)
        boxes = self.base_dataset.boxes(base_num, pil)
        weapon_bb = self.weapon_boxes(boxes)
        p_box, overlapped_weapon_bb = self.get_crop_box(n, pil, weapon_bb)
        x1, y1, x2, y2 = p_box.int().tolist()
        patch = transforms.functional.crop(pil, y1, x1, height=y2 - y1, width=x2 - x1)
        return patch, self.rescale_weapon(p_box, overlapped_weapon_bb)

    def get_squared_person_bbox(self, n):
        base_num = self._get_frame_num(n)
        img = self.base_dataset.image(base_num)
        boxes = self.base_dataset.boxes(base_num, img)
        p_box, _ = self.get_crop_box(n, img, self.weapon_boxes(boxes))
        return p_box.int().tolist()

    def expand(self, person_bb, img):
        px1, py1, px2, py2 = person_bb
        h = py2 - py1
        w = px2 - px1
        deltas = torch.Tensor([-w * 0.1, -h * 0.1, w * 0.1, h * 0.1])
        ih, iw = YOLODataset.extract_hw(img)
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

    def get_crop_box(self, n, img, weapon_bb):
        p_box = self.expand(self.persons[n][1:5], img)
        if len(weapon_bb) == 0:
            return p_box, torch.empty((0, 4))
        m = max((p_box[2] - p_box[0]).item(), (p_box[3] - p_box[1]).item()) * 0.1
        search_box = p_box + p_box.new_tensor([-m, -m, m, m])
        weapon_bb = self.get_overlapped_weapon_bbox(search_box, weapon_bb)
        return self.fit_weapon_margin(p_box, weapon_bb, img), weapon_bb

    def fit_weapon_margin(self, p_box, weapon_bbs, img):
        # Keep overlapped weapons fully inside crop.
        if len(weapon_bbs) == 0:
            return p_box
        ih, iw = YOLODataset.extract_hw(img)
        x1, y1, x2, y2 = p_box.tolist()
        for wx1, wy1, wx2, wy2 in weapon_bbs.tolist():
            mx = (wx2 - wx1) * 0.1
            my = (wy2 - wy1) * 0.1
            x1 = min(x1, wx1 - mx)
            y1 = min(y1, wy1 - my)
            x2 = max(x2, wx2 + mx)
            y2 = max(y2, wy2 + my)
        return torch.tensor([max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)], dtype=p_box.dtype)

    def weapon_boxes(self, boxes):
        out = [b[1:5] for b in boxes if int(b[0].item()) in self.weapon_classes]
        return torch.stack(out) if out else torch.empty((0, 4))

    def get_overlapped_weapon_bbox(self, person_bb, weapon_bb):
        person_bb = person_bb.unsqueeze(0)
        iou = torchvision.ops.box_iou(person_bb, weapon_bb)
        index = iou > 0
        return weapon_bb[index[0]]

    def rescale_weapon(self, person_bb, weapon_bbs):
        if len(weapon_bbs) == 0:
            return torch.empty((0, 4))
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

    def upscale(self, n, boxes: torch.Tensor):
        base_num = self._get_frame_num(n)
        pil = self.base_dataset.image(base_num)
        weapon_bb = self.weapon_boxes(self.base_dataset.boxes(base_num, pil))
        p_box, _ = self.get_crop_box(n, pil, weapon_bb)
        x1, y1, x2, y2 = p_box.int().tolist()
        shift = torch.Tensor([x1, y1, x1, y1])
        return boxes + shift
