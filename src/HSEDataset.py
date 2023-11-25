from .HSESubset import HSESubset
import os
from torch.utils.data import ConcatDataset
from torchvision.datasets.utils import download_and_extract_archive
from glob import glob
from tqdm import tqdm
from typing import Optional, Callable

class HSEDataset(ConcatDataset):
    url = "https://ml.gan4x4.ru/hse/torchguns/hse_dataset/"
    def __init__(self,
                 folder=None,
                 train: Optional[bool] = False,
                 download: Optional[bool] = False,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 url=None,
                 exclude=[],
                 build_cache=False):

        self.transform = transform
        self.name = 'hse_' + ('train' if train else 'test')
        self.build_cache = build_cache

        if folder is None:
            folder = os.getcwd()

        if download:
            if url is None:
                url = f"{self.url}{self.name}.zip"
            download_and_extract_archive(url, folder)

        ds_folder = folder + os.sep + self.name
        paths = self.find_dirs(ds_folder, exclude)
        self.sub_datasets = self.create_sub_datasets(paths)
        super().__init__(self.sub_datasets.values())

    def find_dirs(self, folder, exclude):
        dirs = glob(f"{folder}/*/", recursive=True)
        final = []
        exclude = set(exclude)
        for d in dirs:
            parts = set(d.split(os.sep))
            if len(parts.intersection(exclude)) == 0:
                final.append(d)
        return sorted(final)

    def create_sub_datasets(self, paths):
        datasets = {}
        for i, path in enumerate(paths):
            ds = HSESubset(path, transform=self.transform, build_cache=self.build_cache)
            datasets[ds.name] = ds
        return datasets

"""
    def find_video(self, folder):
        types = ('mp4', 'wmv')  # the tuple of file types
        files_grabbed = []
        for t in types:
            files_grabbed.extend(glob(folder + "*." + t))
        assert len(files_grabbed) == 1, f"Multiple videos in dir dir"
        return files_grabbed[0]

    def build_cache(self, verbose=True):
        labels = []
        for i, (_, label) in tqdm(enumerate(self), disable=not verbose):
            labels.append(int(label))
        return labels

    def get_labels(self, verbose=True):
        labels = self.update_cache(verbose=verbose)
        return BaseDataset.split_labels(labels)

    def detect_persons(self, detector):
        for key in self.sub_datasets.keys():
            ds = self.sub_datasets[key]
            pdy = PersonDetector(ds, detector)
            pdy()  # call

    def info(self, perfix="Total"):
        for k in self.sub_datasets.keys():
            summary = self.sub_datasets[k].info(k)
            print(summary)

        l, p, n = self.get_labels(verbose=False)
        print(
            f"Full dataset Positive: {len(p) / len(l):0.2f}% Negative {len(n) / len(l):.2f} %  Num samples: {len(l)}")

    def remove_unused(self):
        cnt = 0
        for d in self.sub_datasets.values():
            cnt += d.remove_unused()
        return cnt

    def bboxes(self, idx):
        # TODO: rewrite this code copied from ConcatDataset
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].bboxes(sample_idx)
"""