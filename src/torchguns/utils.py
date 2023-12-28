from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import PIL

def draw(im, bbox, width = 2, colors = "green"):
    if len(bbox):
        if  isinstance(im, PIL.Image.Image):
            im = pil_to_tensor(im)
        demo_im = draw_bounding_boxes(im, bbox[:, -4:], width=width, colors= colors)
        pil = to_pil_image(demo_im)
    else:
        pil = to_pil_image(im)
    return pil
