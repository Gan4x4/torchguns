# Torchguns
PyTorch wrapper for various dataset for weapon detections


## Example Usage

The following sections give basic examples of what you can do with Torchguns.


### Basic usage 

Load classic dataset consisting of images
```python
from torchguns import THGPDataset


dataset = THGPDataset(folder="",
                          test=False,
                          download=True)

for image, bbox in dataset:
    # image - Pillow image object
    # bbox Torch tensor of shape [N,4] where N - number of bbox on image

```

Load dataset based on single video file

```python
from torchguns import HSESubset

dataset = HSESubset("path_to_folder_with_video_and_labels", )
for image, bbox in dataset:
    # image - Pillow image object
    # bbox Torch tensor of shape [N,4] where N - number of bbox on image
 
```

To load full HSEDataset consisting of 26 videos.
```python
from torchguns import HSEDataset

hse_dataset = HSEDataset(root = "", train = True, download = True)
 
```
Under the hood HSEDataset is a subclass of [ConcatDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset)


### Extended usage
Each part of full HSEDataset it's an object od HSESubset class. 
This parts can be accessed through `sub_datasets` property 

```python
    hse_dataset.sub_datasets # dictionary of HSESubset objects

```


All dataset compatible with Pytorch [transforms](https://pytorch.org/vision/0.16/auto_examples/transforms/plot_transforms_getting_started.html?highlight=transforms) mechanism

```python
from torchguns import USRTDataset
from from torchvision.transforms import v2
import torch 

transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
  
dataset = USRTDataset(root = "usrt_folder",  transform = transforms)
 
```


For classes derived from VideoDataset you can set the frame rate of the frames extracted from a particular video.  

```python
dataset = HSESubset("path_to_folder_with_video_and_labels",
                    fps = 2)
```

Also possibly to fix tonal nuber or frames fo extraction. In that case fps must be `None`
It's convenient when you want balance data from sources with different length and frame rate

```python
dataset = HSESubset("path_to_folder_with_video_and_labels",
                    fps = None,
                    desired_frames = 100)
```


Possible  to converting a standard full-frame bounding-box dataset into a "Person dataset", if dataset contains bounding boxes for "person" class.
In converted dataset each data item is squared image patch containing the content of a single person's bounding box
and a list of scaled weapon bounding boxes that intersect with this patch.

```python
from torchguns import HSESubset, PersonDataset
street_05 = HSESubset("test/data/HSE/street_05", desired_frames = 100)
print(len(street_05)) # 100 frames
person_street_05 = PersonDataset(street_05)
print(len(person_street_05)) # 246 patch along with bbox
```



## Datasets

| Date | Name                                  | First Author                                                                                    | Article                                                                                                                                                   | Description                                                                                                                                                                                        | Link                                                                                                                                                                  
|------|---------------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2021 | Hands Guns and Phones (HGP)           | [Mario Alberto Duran-Vega](https://arxiv.org/search/cs?searchtype=author&query=Duran-Vega,+M+A) | [TYolov5: A Temporal Yolov5 Detector Based on Quasi-Recurrent Neural Networks for Real-Time Handgun Detection in Video](https://arxiv.org/abs/2111.08867) | 2199 images (1989/210 train/test) of people using guns or phones                                                                                                                                   | [download](https://drive.google.com/file/d/138Zp7MuchcS4He6LBFSTow5q97BwnpWv)                                                                                         |
| 2021 | Temporal Hands Guns and Phones (THGP) | [Mario Alberto Duran-Vega](https://arxiv.org/search/cs?searchtype=author&query=Duran-Vega,+M+A) | [TYolov5: A Temporal Yolov5 Detector Based on Quasi-Recurrent Neural Networks for Real-Time Handgun Detection in Video](https://arxiv.org/abs/2111.08867) | 5960 video frames (5000 for training and 960 for testing) 720 × 720 pixels. This dataset contains 20 videos of shooting drills, 20 videos of armed robberies, and 10 videos of people making calls. | [download](https://drive.google.com/file/d/1hF7Vr6g0fG56Oy3Jdnm2t9Y3TK9W9bn4)                                                                                         |
| 2023 | YouTube-GDD                           | [Yongxiang Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu,+Y)                       | [YouTube-GDD: A challenging gun detection dataset with rich contextual information](https://arxiv.org/abs/2203.04129)                                     | 5000 images from from 343 high-definition YouTube videos,  16064 instances of gun and 9046 instances of person are annotated                                                                       | [data](https://drive.google.com/file/d/1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa) [code](https://github.com/UCAS-GYX/YouTube-GDD)                                             |
| 2020 | Mock attack dataset (USRT)            | [Salazar-González, Jose L.](https://www.scopus.com/authid/detail.uri?authorId=57219090287)      | [Real-time gun detection in CCTV: An open problem. ](https://doi.org/10.1016/j.neunet.2020.09.013)                                                        | Frames form three cam (607+3511+1031) are мanually annotated and collected during a mock attack at 2 FPS.                                                                                          | [data](https://uses0-my.sharepoint.com/:u:/g/personal/jsalazar_us_es/Ee7yqsE68U9PhnNHZneIuTABfTX5P9iVClJyxIKORfBJvg?e=VpXVtT) [code](https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset) |
|2023|HSE Dataset| --                                                                                              | --                                                                                                                                                        | dataset includes 26 videos                                                                                                                                                                         |[code](https://github.com/Gan4x4/torchguns)|