import logging

from os.path import basename, join, splitext
from typing import Sequence

import numpy as np

from PIL import Image

from dataset.mask import Mask


logger = logging.getLogger("dataset.patches")


class Patches:
    """
    Class to split an image and optionally a corresponding mask into a set of patches
    """

    def __init__(self, image_path: str, mask_path: str | None = None):
        """
        Initialize a Patches object

        :param image_path: Path to image to split into patches
        :param mask_path: Path to image containing segmentation masks corresponding to the image
            in image_path
        """
        logger.debug(f"Creating Patches object with image_path = {image_path} and "
                     f"mask_path = {mask_path}")
        self.image_path: str = image_path
        self.mask_path: str | None = mask_path

    def write_patches(self, dest: str, splits: int = 8, store_empty: bool = False) -> None:
        """
        Split image and corresponding mask into patches and store the result

        Split the image and the corresponding mask, if it is defined, into a number of patches
        defined by `splits`, and store the result in `dest`, under subdirectories `images`
        and `masks`, respectively.

        :param dest: Destination directory for the patches
        :param splits: Number of patches to split the image into
        :param store_empty: If a mask is defined, store patches also for which no mask exists
        """
        logger.debug(f"Splitting {self.image_path} into {splits} patches and storing them in "
                     f"{dest}/images")
        if self.mask_path is not None:
            logger.debug(f"Also splitting {self.mask_path} and storing into {dest}/masks")
            logger.debug(f"{'Storing' if store_empty else 'Not storing'} empty masks")

        image_patches: list[Image.Image]
        # NOTE: Using Sequence because of a bug with list[x | None] in MyPy
        mask_patches: Sequence[Image.Image]
        empty_masks: list[bool] = [False] * splits

        logger.debug(f"Splitting image into {splits} patches")
        image_patches = self._split_image(Image.open(self.image_path), splits)
        if self.mask_path is not None:
            logger.debug(f"Splitting mask into {splits} patches")
            mask_patches = self._split_image(Image.open(self.mask_path), splits)
            empty_masks = [
                not Mask(data=np.asarray(patch)).has_polygons()
                for patch in mask_patches
            ]

        base, ext = splitext(basename(self.image_path))
        for image_patch, mask_patch, mask_is_empty, idx in \
                zip(image_patches, mask_patches, empty_masks, range(0, len(image_patches))):
            patch_filename = f"{base}_{idx}{ext}"

            if self.mask_path is None:
                logger.debug(f"Writing patch {idx} for {self.image_path}")
                self._write_patch(join(dest, "images", patch_filename), image_patch)
            elif not mask_is_empty or store_empty:
                logger.debug(f"Writing patch {idx} for {self.image_path}")
                self._write_patch(join(dest, "images", patch_filename), image_patch)
                logger.debug(f"Writing patch {idx} for {self.mask_path}")
                self._write_patch(join(dest, "masks", patch_filename), mask_patch)
            else:
                logger.debug(f"Empty mask for image {self.image_path} at index {idx}. Skipping.")

    @staticmethod
    def _write_patch(path: str, patch: Image.Image) -> None:
        logger.debug(f"Patch size is {patch.width} x {patch.height} and mode is {patch.mode}")
        patch.save(path)

    @staticmethod
    def _split_image(image: Image.Image, splits: int) -> list[Image.Image]:
        logger.debug(f"Image size is {image.width} x {image.height} and mode is {image.mode}")

        # Convert the image to a NumPy array and determine the patch width
        image_data = np.asarray(image)
        patch_width = image_data.shape[1] // splits
        logger.debug(f"Patch width is {patch_width}")

        # Split image into patches
        patches_data = [
            image_data[:, x:x+patch_width]
            for x in range(0, image_data.shape[1], patch_width)
        ]

        # Convert back to image objects
        patches = [Image.fromarray(patch, mode=image.mode) for patch in patches_data]

        # For palette mode images, the palette needs to be set explicitly from the original
        if image.mode == "P":
            for patch in patches:
                patch.putpalette(image.getpalette())  # type: ignore

        return patches
