import logging
import os
import shutil

from os.path import exists, isdir, join, splitext
from typing import Sequence

import yaml

from PIL import Image

from dataset.mask import Mask
from dataset.overlay import save_image_with_mask, save_image_with_polygons
from dataset.patches import Patches


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


def create_patches(src: str, dest: str, splits: int = 8) -> None:
    """
    Split the images and masks of a dataset into patches

    :param src: Path to source directory. Must contain '{images,masks}' directories.
    :param dest: Path to destination directory
    :param splits: Number of patches to split the images into
    """
    image_dir = join(src, "images")
    mask_dir = join(src, "masks")

    # Create destination directories
    for dest_subdir in [join(dest, d) for d in ["images", "masks"]]:
        if not isdir(dest_subdir):
            logger.info(f"Creating directory {dest_subdir}")
            os.makedirs(dest_subdir)

    # Create patches for all images with a corresponding mask
    for filename in os.listdir(image_dir):
        image_path = join(image_dir, filename)
        mask_path = join(mask_dir, filename)
        if not exists(mask_path):
            logger.warning(f"Image {image_path} has no corresponding mask {mask_path}. "
                           "Skipping.")
            continue
        patches = Patches(image_path, mask_path)
        patches.write_patches(dest, splits=splits)


def create_overlays_labels(src: str, dest: str) -> None:
    """
    Create mask and polygon overlays and polygon labels for a dataset

    :param src: Path to source directory. Must contain '{images,masks}' directories.
    :param dest: Path to destination directory
    """
    image_dir = join(src, "images")
    mask_dir = join(src, "masks")
    mask_overlay_dir = join(dest, "mask_overlays")
    polygon_overlay_dir = join(dest, "polygon_overlays")
    label_dir = join(dest, "labels")

    # Create destination directories
    for dest_subdir in [mask_overlay_dir, polygon_overlay_dir, label_dir]:
        if not isdir(dest_subdir):
            logger.info(f"Creating directory {dest_subdir}")
            os.makedirs(dest_subdir)

    # Create overlays for each image and mask
    for filename in os.listdir(image_dir):
        image = Image.open(join(image_dir, filename))
        mask = Mask(join(mask_dir, filename))

        logger.debug(f"Saving mask overlays to {mask_overlay_dir}")
        save_image_with_mask(image, mask, join(mask_overlay_dir, filename))

        logger.debug(f"Saving polygon overlays to {polygon_overlay_dir}")
        save_image_with_polygons(image, mask, join(polygon_overlay_dir, filename))

        base, _ = splitext(filename)
        logger.debug(f"Saving labels to {join(label_dir, f'{base}.txt')}")
        mask.save_polygons(join(label_dir, f"{base}.txt"))


def to_yolo(src: str,
            dest: str,
            subdirs: Sequence = ("train", "val"),
            splits: int = 8
            ) -> None:
    """
    Convert a dataset with images and binary masks to a YOLOv7 segmentation dataset

    Convert the dataset by traversing the provided sub-directories in the source directory,
    and for each pair of images and masks:
        - Split the image into patches
        - Split the mask into patches
        - Convert mask patches into polygons
        - Overlay image patches with masks
        - Overlay image patches with polygons

    :param src: Path to source directory. Must contain '{subdirs}/{images,masks}'.
        directories.
    :param dest: Path to destination directory. Will contain
        '{subdirs}/{images,labels,images_with_masks,images_with_polygons}' directories.
    :param subdirs: Sub-directories to traverse in the source directory (default: ["train",
        "val"])
    :param splits: Number of patches to split the images into
    """
    for subdir in subdirs:
        # Create patches for images and masks and place them in dest
        dest_dir = join(dest, subdir)
        if isdir(dest_dir):
            logger.info(f"Destination directory {dest_dir} already exists. "
                        "Skipping patch creation.")
        else:
            logger.info(f"Splitting images and masks from {subdir} into patches")
            create_patches(join(src, subdir), join(dest_dir), splits=splits)

        # Create overlays and labels from images and masks
        logger.info(f"Creating overlays and labels for {subdir}")
        create_overlays_labels(join(dest, subdir), join(dest, subdir))

    # Create the YOLO dataset specification file
    dataset = {
        "names": ["crack"],
        "nc": 1,
        "train": f"{dest}/train",
        "val": f"{dest}/val"
    }

    with open(join(dest, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(dataset, f)
