import logging
import os
import shutil

from os.path import join

logger = logging.getLogger("dataset.manage")


def split(image_dir: str,
          mask_dir: str,
          dest_dir: str,
          splits: dict[str, list]
          ) -> None:
    """
    Split a dataset into train, validation, and test

    Copy a set of images and corresponding masks to the destination directory, distributing them
    according to a train-validation-test split.

    :param image_dir: Path to image files
    :param mask_dir: Path to binary mask files
    :param dest_dir: Destination directory
    :param splits: Dict of split and corresponding list of files
    """

    def copy_files(src: str, dest: str, files: list[str]) -> None:
        # Create a list of files in the source directory
        file_list = os.listdir(src)

        for entry in files:
            # Find candidate files in the source directory for each entry in the provided file
            # list
            candidates = [f for f in file_list if f.startswith(entry)]
            if not candidates:
                logger.warning(f"Found no file in {src} for entry {entry}")
                continue

            # Pick the first candidate (there should seldom if ever be more than one)
            file = candidates[0]
            try:
                logger.debug(f"Copying {join(src, file)} to {join(dest, file)}")
                shutil.copy(join(src, file), join(dest, file))
            except FileNotFoundError as e:
                logger.fatal(f"File does not exist: {e}")
                raise

    for split_dir, split_files in splits.items():
        split_image_dir = join(dest_dir, split_dir, "images")
        split_mask_dir = join(dest_dir, split_dir, "masks")
        logger.info(f"Creating destination directories {split_image_dir} and {split_mask_dir}")
        os.makedirs(split_image_dir)
        os.makedirs(split_mask_dir)
        logger.info(f"Copying selected files from {image_dir} to {split_image_dir}")
        copy_files(image_dir, split_image_dir, split_files)
        logger.info(f"Copying selected files from {mask_dir} to {split_mask_dir}")
        copy_files(mask_dir, split_mask_dir, split_files)
