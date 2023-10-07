from .YOLODataset import YOLODataset
import os
import json
from typing import Optional, Callable
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
class VideoDataset(YOLODataset):
    """ Wrapper of single videofile dataset """

    classes = ["Gun"]
    class_filter = ['Gun']
    settings = {}

    def __init__(
            self,
            video_folder: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            build_cache = True
    ) -> None:
        self.video_folder = video_folder
        self.settings = {'desired_frames': None}
        self.load_settings(video_folder + os.sep + "settings.json")
        self.labels_dir = video_folder + "/obj_train_data/"
        cap = self.open_video()
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if build_cache:
            self.build_cache()
        super().__init__(self.labels_dir, transforms, transform, target_transform)


    def find_video(self):
        mp4_files = glob(self.video_folder+"/*.mp4")
        wmv_files = glob(self.video_folder + "/*.wmv")
        files = mp4_files + wmv_files
        if len(files) > 1:
            raise Exception(f"Found multiple video files in {self.video_folder}")
        if len(files) < 1:
            raise Exception(f"Video not found in {self.video_folder}")
        return files[0]

    def open_video(self):
        video_path = self.find_video()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open {video_path}")
        return cap

    def build_cache(self):
        """ Extract frames from video"""
        cap = self.open_video()
        skip = self.calculate_skip()
        frames_extracted = 0
        postfix = "jpg"
        frame_list = range(0,self.total_frames,int(skip))
        for frame_num in tqdm(frame_list, desc=f"Build cache for {self.name}"):
            filename = self.labels_dir + f"frame_{frame_num:06d}.{postfix}"
            if not os.path.isfile(filename):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                res, frame_bgr = cap.read()
                if res:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.save(filename)
                if not res or not img:
                    raise Exception(f"Can't read frame {frame_num}")
            frames_extracted += 1

    def calculate_skip(self):
        if not self.desired_frames:
            return 1
        skip = (self.total_frames -1) / self.desired_frames
        return max(skip, 1)

    @property
    def desired_frames(self):
        return self.settings['desired_frames']

    @desired_frames.setter
    def desired_frames(self,value):
        self.settings['desired_frames'] = value


    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        ann_path = img_path.replace("jpg", "txt")
        ann_path = ann_path.replace(os.sep + os.sep, os.sep)
        return ann_path

    def load_settings(self, fn):
        if os.path.exists(fn):
            with open(fn) as fp:
                data = json.load(fp)
                # merge settings
                self.settings = {**self.settings, **data}

    def save_settings(self, fn):
        with open(fn, 'w') as outfile:
            json.dump(self.settings, outfile)
    @property
    def name(self):
        parts = self.video_folder.split(os.sep)
        parts = list(filter(None, parts))
        name = parts[-1].strip()
        assert len(name) > 0
        return name

