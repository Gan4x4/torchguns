from .YOLODataset import YOLODataset
import os
import json
from typing import Optional, Callable
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import warnings
import numpy as np


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
            build_cache=True,
            ** kwargs # settings like fps an desired frames
    ) -> None:
        self.video_folder = video_folder
        # self.settings = {'desired_frames': None}
        self.load_settings(video_folder + os.sep + "settings.json", kwargs)
        self.labels_dir = video_folder + "/obj_train_data/"
        cap = self.open_video()
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if build_cache:
            self.build_cache()
        super().__init__(self.labels_dir, transforms, transform, target_transform)

    def find_video(self):
        mp4_files = glob(self.video_folder + "/*.mp4")
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
        frame_list = self.get_frame_list()
        for frame_num in tqdm(frame_list, desc=f"Build cache for {self.name}"):
            self.extract_frame(frame_num, cap)
        cap.release()
        self.delete_unused_frames(frame_list)
        #self.image_paths = self.get_image_paths(self.root)

    def get_frame_list(self):
        skip = self.calculate_skip()
        frame_list = np.arange(skip, self.total_frames, skip)
        frame_list = frame_list.astype(int)
        if self.desired_frames:
            assert len(frame_list) == self.desired_frames
        return frame_list.tolist()

    def extract_frame(self, frame_num, cap):
        filename = self.get_frame_path(frame_num)
        if not os.path.isfile(filename):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            res, frame_bgr = cap.read()
            if res:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(filename)
            if not res or not img:
                raise Exception(f"Can't read frame {frame_num}")
        return filename

    def get_frame_path(self, frame_num):
        postfix = "jpg"
        filename = self.labels_dir + f"frame_{frame_num:06d}.{postfix}"
        return filename

    def delete_unused_frames(self, frames_to_preserve):
        for frame_num in range(self.total_frames):
            if frame_num not in frames_to_preserve:
                fn = self.get_frame_path(frame_num)
                if os.path.isfile(fn):
                    os.unlink(fn) #print("deleted", fn)

    def calculate_skip(self):
        if self.fps is None and self.desired_frames is None:
            # Use all frames
            return 1
        assert self.fps is None or self.desired_frames is None

        # By FPS
        if self.fps:
            if self.fps > self.video_fps:
                warnings.warn(f"Impossible increase fps from {self.video_fps} to {self.fps}")
            skip = self.video_fps / self.fps
            return skip

        # By desired_frames
        skip = (self.total_frames - 1) / self.desired_frames
        return max(skip, 1)

    @property
    def desired_frames(self):
        return self.settings['desired_frames']

    @desired_frames.setter
    def desired_frames(self, value):
        self.settings['desired_frames'] = value
        self.settings['fps'] = None  # fps and desired_frames are conflicting properties

    @property
    def fps(self):
        return self.settings['fps']

    @fps.setter
    def fps(self, value):
        self.settings['desired_frames'] = None  # fps and desired_frames are conflicting properties
        self.settings['fps'] = value

    def getfilename(self, n):
        """ get file with annotations for image on n place """
        img_path = self.image_paths[n]  # TODO refactor to method call
        ann_path = img_path.replace("jpg", "txt")
        ann_path = ann_path.replace(os.sep + os.sep, os.sep)
        return ann_path

    def load_settings(self, fn, user_defined_settings):
        if os.path.exists(fn):
            with open(fn) as fp:
                self.settings = json.load(fp)
                #self.settings = {**self.settings, **data}
        # merge settings python 9+
        self.settings = self.settings | user_defined_settings
        self.setup_defaults()

    def setup_defaults(self):
        defaults = ['desired_frames', 'fps']
        for d in defaults:
            if d not in self.settings:
                self.settings[d] = None
        if (self.desired_frames is not None) and (self.fps is not None):
            warnings.warn(f"FPS and desired_frames are conflicting properties so desired_frames will be ignored")
            self.settings['desired_frames'] = None



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
