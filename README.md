# Torchguns
PyTorch wrapper for various dataset for weapon detections


## Example Usage

The following sections give basic examples of what you can do with Torchguns.


### Basic usage 
```python
from torchguns import HSEDataset

hse_dataset = HSEDataset(folder="", test=False, download=True)

for image, bbox in hse_dataset:
    # image - Pillow image object
    # bbox Torch tensor of shape [N,4] where N - number of bbox on image

```

## Datasets

| Date | Name                         | First Author                                                                                    | Article                                                                                                                                                   | Description                                                                                                                                                                                         |Link
|------|------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| 2021 | Hands Guns and Phones (HGP)  | [Mario Alberto Duran-Vega](https://arxiv.org/search/cs?searchtype=author&query=Duran-Vega,+M+A) | [TYolov5: A Temporal Yolov5 Detector Based on Quasi-Recurrent Neural Networks for Real-Time Handgun Detection in Video](https://arxiv.org/abs/2111.08867) | 2199 images (1989/210 train/test) of people using guns or phones                                                                                                                                    |[download](https://drive.google.com/file/d/138Zp7MuchcS4He6LBFSTow5q97BwnpWv) |
| 2021 | Temporal Hands Guns and Phones (THGP) | [Mario Alberto Duran-Vega](https://arxiv.org/search/cs?searchtype=author&query=Duran-Vega,+M+A) | [TYolov5: A Temporal Yolov5 Detector Based on Quasi-Recurrent Neural Networks for Real-Time Handgun Detection in Video](https://arxiv.org/abs/2111.08867) | 5960 video frames (5000 for training and 960 for testing) 720 × 720 pixels. This dataset contains 20 videos of shooting drills, 20 videos of armed robberies, and 10 videos of people making calls. |[download](https://drive.google.com/file/d/1hF7Vr6g0fG56Oy3Jdnm2t9Y3TK9W9bn4)                             |
|2023|YouTube-GDD|[Yongxiang Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu,+Y)|[YouTube-GDD: A challenging gun detection dataset with rich contextual information](https://arxiv.org/abs/2203.04129)| [data](https://drive.google.com/file/d/1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa) </br>[code](https://github.com/UCAS-GYX/YouTube-GDD)                                                                      |
