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