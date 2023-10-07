from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def draw(im, bbox):
    if len(bbox):
        demo_im = draw_bounding_boxes(im, bbox[:, -4:], width=2, colors="green")
        pil = to_pil_image(demo_im)
    else:
        pil = to_pil_image(im)
    return pil
