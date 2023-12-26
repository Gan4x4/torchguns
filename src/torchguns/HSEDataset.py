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
                 # build_cache=False,
                 **kwargs):

        self.transform = kwargs.get('transform', None)
        self.name = 'hse_' + ('train' if train else 'test')
        #self.build_cache = build_cache

        if folder is None:
            folder = os.getcwd()

        if download:
            if url is None:
                url = f"{self.url}{self.name}.zip"
            download_and_extract_archive(url, folder)

        ds_folder = folder + os.sep + self.name
        paths = self.find_dirs(ds_folder, exclude)
        self.sub_datasets = self.create_sub_datasets(paths, kwargs)
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

    def create_sub_datasets(self, paths, kwargs):
        datasets = {}
        for i, path in enumerate(paths):
            try:
                ds = HSESubset(path, **kwargs)
            except Exception as e:
                print(f"Error on loading data from {path}")
                raise e

            datasets[ds.name] = ds
        return datasets

