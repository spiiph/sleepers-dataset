import logging

from os.path import basename, join, splitext
from typing import Sequence

import numpy as np

from PIL import Image


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


    def write_patches(self, dest_path: str, splits: int = 8, store_empty: bool = False) -> None:
        """
        Split image and corresponding mask into patches and store the result

        Split the image and the corresponding mask, if it is defined, into a number of patches
        defined by `splits`, and store the result in `dest_path`, under subdirectories `images`
        and `masks`, respectively.

        :param dest_path: Destination directory for the patches
        :param splits: Number of patches to split the image into
        :param store_empty: If a mask is defined, store patches also for which no mask exists
        """
        logger.info(f"Splitting {self.image_path} into {splits} patches and storing them in "
                    f"{dest_path}/images")
        if self.mask_path is not None:
            logger.info(f"Also splitting {self.mask_path} and storing into {dest_path}/masks")
            logger.info(f"{'Storing' if store_empty else 'Not storing'} empty masks")

        image_patches: list[Image.Image]
        # NOTE: Using Sequence because of a bug with list[x | None] in MyPy
        mask_patches: Sequence[Image.Image | None]
        empty_masks: list[bool]

        logger.debug(f"Splitting image into {splits} patches")
        image_patches, _ = self._split_image(Image.open(self.image_path), splits)
        if self.mask_path is not None:
            logger.debug(f"Splitting mask into {splits} patches")
            mask_patches, empty_masks = self._split_image(Image.open(self.mask_path), splits)
        else:
            logger.debug("Constructing empty masks")
            mask_patches = [None] * len(image_patches)
            empty_masks = [False] * len(image_patches)

        base, ext = splitext(basename(self.image_path))
        for image_patch, mask_patch, mask_is_empty, idx in \
                zip(image_patches, mask_patches, empty_masks, range(0, len(image_patches))):
            patch_filename = f"{base}_{idx}{ext}"

            if mask_is_empty and not store_empty:
                logger.debug(f"Empty mask for image {self.image_path} at index {idx}. Skipping.")
                continue

            logger.debug(f"Writing patch {idx} for {self.image_path}")
            self._write_patch(join(dest_path, "images", patch_filename), image_patch)
            if mask_patch is not None:
                logger.debug(f"Writing patch {idx} for {self.mask_path}")
                self._write_patch(join(dest_path, "masks", patch_filename), mask_patch)


    @staticmethod
    def _write_patch(path: str, patch: Image.Image) -> None:
        logger.debug(f"Patch size is {patch.width} x {patch.height} and mode is {patch.mode}")
        patch.save(path)


    @staticmethod
    def _split_image(image: Image.Image, splits: int) -> tuple[list[Image.Image], list[bool]]:
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

        # Determine empty status of the patches
        is_empty = [not patch.any() for patch in patches_data]

        # Convert back to image objects
        patches = [Image.fromarray(patch, mode=image.mode) for patch in patches_data]

        # For palette mode images, the palette needs to be set explicitly from the original
        if image.mode == "P":
            for patch in patches:
                patch.putpalette(image.getpalette()) # type: ignore

        return patches, is_empty
