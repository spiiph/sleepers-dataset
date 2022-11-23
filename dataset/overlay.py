import logging

from PIL import Image

from dataset.mask import Mask

logger = logging.getLogger("dataset.overlay")


def save_image_with_mask(image: str | Image.Image,
                         mask: str | Mask,
                         dest: str
                         ) -> Image.Image:
    """
    Save an image with a mask overlayed.

    Load an image and a binary mask, then save the image with the mask overlayed to a new location.
    The image will be converted to RGB format.

    :param image_path: Path to the image file.
    :param mask_path: Path to the image file with the binary mask.
    :param dest: Path to store the new image with the mask overlayed.
    :return: Image with the mask overlayed.
    """
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(mask, str):
        mask = Mask(mask)

    image_with_mask = mask.overlay(image, colour=(255, 0, 0))
    image_with_mask.save(dest)

    return image_with_mask


def save_image_with_polygons(image: str | Image.Image,
                             mask: str | Mask,
                             dest: str
                             ) -> Image.Image:
    """
    Save an image with polygons overlayed.

    Load an image, then save the image with the provided polygons overlayed to a new location.
    The image will be converted to RGB format.

    :param image_path: Path to the image file.
    :param mask_path: Path to the image file with the binary mask.
    :param image_with_polygons_path: Path to store the new image with the polygons overlayed.
    :return: Image with the mask overlayed.
    """
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(mask, str):
        mask = Mask(mask)

    image_with_polygons = mask.overlay_polygons(image, colour=(255, 0, 0))
    image_with_polygons.save(dest)

    return image_with_polygons
