from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def draw(im, bbox):
    demo_im = draw_bounding_boxes(im, bbox[:, -4:], width=2)
    pil = to_pil_image(demo_im)
    return pil
