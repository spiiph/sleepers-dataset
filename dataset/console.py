import argparse
import logging
import os
import shutil
import sys
import textwrap

from dataclasses import dataclass, field
from os.path import basename, exists, isdir, join

from PIL import Image

from dataset.overlay import save_image_with_mask, save_image_with_polygons
from dataset.mask import Mask
from dataset.patches import Patches
from dataset.manage import split, to_yolo, check_labels


LOG_LEVELS = list(logging._nameToLevel.keys())[:-1]  # pylint: disable=protected-access


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Logger setup for console applications

    :param name: Name of the logger
    :param level: Logging level (default: INFO)
    :return: Instance of Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def hello_world(args: argparse.Namespace) -> int:
    """
    Hello World entry point for testing

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """

    logger = setup_logger("dataset", level=args.log_level)
    logger.debug("Hello World!")
    logger.info("Hello World!")
    logger.warning("Hello World!")
    logger.error("Hello World!")
    logger.fatal("Hello World!")

    return True


def create_patches(args: argparse.Namespace) -> bool:
    """
    Create patches from one image and one mask

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running create-patches with the following parameters:")
    logger.info(f"  image: {args.image}")
    logger.info(f"  mask: {args.mask}")
    logger.info(f"  dest: {args.dest}")

    for subdir in ["images", "masks"]:
        if not isdir(join(args.dest, subdir)):
            logger.debug(f"Creating missing directory {join(args.dest, subdir)}")
            os.makedirs(join(args.dest, subdir))

    patches = Patches(args.image, args.mask)
    patches.write_patches(args.dest, splits=8, store_empty=False)

    return True


def test_polygons(args: argparse.Namespace) -> bool:
    """
    Test the conversion of masks to contours and polygons

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running test-contours-and-polygons with the following parameters:")
    logger.info(f"  mask: {args.mask}")

    mask = Mask(args.mask)
    logger.info("Contours (not normalized):")
    for contour in mask.to_contours(normalize=False):
        logger.info(f">>> {contour.T[0].tolist()}")
        logger.info(f">>> {contour.T[1].tolist()}")
    logger.info("Contours (normalized):")
    for contour in mask.to_contours(normalize=True):
        logger.info(f">>> {contour.T[0].tolist()}")
        logger.info(f">>> {contour.T[1].tolist()}")
    logger.info("Polygons:")
    for polygon in mask.to_polygons(normalize=True):
        logger.info(f">>> {polygon}")

    return True


def overlay_image(args: argparse.Namespace) -> bool:
    """
    Save copies of an image with mask and polygons overlayed

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running overlay-image with the following parameters:")
    logger.info(f"  image: {args.image}")
    logger.info(f"  mask: {args.mask}")
    logger.info(f"  dest: {args.dest}")

    image = Image.open(args.image)
    mask = Mask(args.mask)

    for subdir in ["images_with_masks", "images_with_polygons"]:
        if not isdir(join(args.dest, subdir)):
            logger.debug(f"Creating missing directory {join(args.dest, subdir)}")
            os.makedirs(join(args.dest, subdir))

    mask_dest = join(args.dest, "mask_overlays", basename(args.image))
    logger.info(f"Saving image with mask to {mask_dest}")
    save_image_with_mask(image, mask, mask_dest)

    polygon_dest = join(args.dest, "polygon_overlays", basename(args.image))
    logger.info(f"Saving image with mask to {polygon_dest}")
    save_image_with_polygons(image, mask, polygon_dest)

    return True


def split_dataset(args: argparse.Namespace) -> bool:
    """
    Split a segmentation dataset into training, validation, and test datasets

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running split with the following parameters:")
    logger.info(f"  source: {args.source}")
    logger.info(f"  dest: {args.dest}")

    if not isdir(args.source):
        logger.fatal(f"Source directory {args.source} does not exist.")
        return False

    for subdir in [join(args.source, d) for d in ["images", "masks", "splits"]]:
        if not isdir(subdir):
            logger.fatal(f"Source subdirectory {subdir} does not exist.")
            return False

    if exists(args.dest):
        logger.info(f"Destination directory {args.dest} exists. Removing")
        shutil.rmtree(args.dest)

    def read_split(filename: str) -> list[str]:
        with open(filename, encoding="utf-8") as f:
            return [line.strip() for line in f]

    try:
        train = read_split(join(args.source, "splits", "train.txt"))
        val = read_split(join(args.source, "splits", "val.txt"))
        test = read_split(join(args.source, "splits", "test.txt"))
    except FileNotFoundError as e:
        logger.fatal(f"File does not exist: {e}")
        return False

    splits = {
        "train": train,
        "val": val,
        "test": test
    }
    split(join(args.source, "images"), join(args.source, "masks"), args.dest, splits)

    return True


def convert_to_yolo(args: argparse.Namespace) -> bool:
    """
    Convert an image dataset with binary masks to a dataset for YOLO segmentation

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running yolo with the following parameters:")
    logger.info(f"  source: {args.source}")
    logger.info(f"  dest: {args.dest}")
    logger.info(f"  patches: {args.patches}")

    if not isdir(args.source):
        logger.fatal(f"Source directory {args.source} does not exist.")
        return False

    for subdir in [join(args.source, d) for d in ["train", "val"]]:
        if not isdir(subdir):
            logger.fatal(f"Source subdirectory {subdir} does not exist.")
            return False

    if exists(args.dest):
        logger.warning(f"Destination directory {args.dest} exists. Trying to continue conversion.")

    to_yolo(args.source, args.dest, subdirs=("train", "val"), splits=args.patches)

    return True


def check_polygon_labels(args: argparse.Namespace) -> bool:
    """
    Check the validity of the polygon labels in a directory

    :param args: Console application arguments
    :return: Status indicating if the function succeeded or failed
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running check-labels with the following parameters:")
    logger.info(f"  source: {args.source}")

    if not isdir(args.source):
        logger.fatal(f"Source directory {args.source} does not exist.")
        return False

    check_labels(args.source)

    return True


@dataclass
class Argument:
    """
    Class to define reusable arguments for ArgumentParser
    """
    args: list = field()
    kwargs: dict = field()


def main() -> None:
    """
    Entry point for all sub-commands
    """
    parser = argparse.ArgumentParser(description="Run sleeper-dataset commands")
    subparsers = parser.add_subparsers(help="Command to run")

    log_level = Argument(
        args=["--log-level", "-l"],
        kwargs={
            "type": str,
            "choices": LOG_LEVELS,
            "default": "INFO"
        }
    )

    # Set up the hello parser
    hello_parser = subparsers.add_parser(
        "hello",
        help="Run the hello command"
    )
    hello_parser.set_defaults(func=hello_world)
    hello_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Setup the create-patches parser
    patches_parser = subparsers.add_parser(
        "create-patches",
        help="Run the create-patches command"
    )
    patches_parser.add_argument(
        "image",
        help="Image file"
    )
    patches_parser.add_argument(
        "mask",
        help="Mask file"
    )
    patches_parser.add_argument(
        "dest",
        help="Destination directory"
    )
    patches_parser.set_defaults(func=create_patches)
    patches_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the test-polygons parser
    polygons_parser = subparsers.add_parser(
        "test-polygons",
        help="Run the test-polygons command"
    )
    polygons_parser.add_argument(
        "mask",
        help="Mask file"
    )
    polygons_parser.set_defaults(func=test_polygons)
    polygons_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the overlay-image parser
    overlay_parser = subparsers.add_parser(
        "overlay-image",
        help="Run the overlay-image command"
    )
    overlay_parser.add_argument(
        "image",
        help="Image file"
    )
    overlay_parser.add_argument(
        "mask",
        help="Mask file"
    )
    overlay_parser.add_argument(
        "dest",
        help="Destination directory"
    )
    overlay_parser.set_defaults(func=overlay_image)
    overlay_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the split parser
    split_parser = subparsers.add_parser(
        "split",
        help="Run the split command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Split a dataset into training, validation, and test datasets

        The split command expects that the source directory contains the sub-directories
        'images', 'masks', and 'splits'. 'images' contains the images of the dataset while
        'masks' contains the corresponding binary masks. 'splits' contains the three text files
        'train.txt', 'val.txt', and 'test.txt' describing how the dataset should be split.
        """)
    )
    split_parser.add_argument(
        "source",
        help="Source directory"
    )
    split_parser.add_argument(
        "dest",
        help="Destination directory"
    )
    split_parser.set_defaults(func=split_dataset)
    split_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the yolo parser
    yolo_parser = subparsers.add_parser(
        "yolo",
        help="Run the yolo command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Convert a dataset with binary masks to polygon labels

        The yolo command converts a dataset of images and masks into one of images and labels,
        where label files are text files containing class ID and normalized polygon coordinates.
        It first splits images and masks into patches to reduce the image width and allow for a
        larger resolution to be used during training.

        The yolo command expects that the source directory contains the subdirectories 'train'
        and 'val', which in turn contain the subdirectories 'images' and 'masks'. In the
        destination directory, it creates the corresponding subdirectories 'train' and 'val', and
        under each of these the subdirectories 'images', 'masks', 'labels', 'mask_overlays', and
        'polygon_overlays'.
        """)
    )
    yolo_parser.add_argument(
        "source",
        help="Source directory"
    )
    yolo_parser.add_argument(
        "dest",
        help="Destination directory"
    )
    yolo_parser.add_argument(
        "--patches", "-p",
        type=int,
        default=8,
        help="Number of patches to split each image into"
    )
    yolo_parser.set_defaults(func=convert_to_yolo)
    yolo_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the check labels parser
    check_parser = subparsers.add_parser(
        "check-labels",
        help="Run the check-labels command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Check the validity of the polygon labels in a directory

        Check that the label files in a directory contains at least one polygon, that the first
        value of each line is the class label, that the polygons consist of 3 or more points, and
        that the number of coordinate values are even.
        """)
    )
    check_parser.add_argument(
        "source",
        help="Source directory"
    )
    check_parser.set_defaults(func=check_polygon_labels)
    check_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Parse arguments
    args = parser.parse_args()

    # Call sub-command
    status = args.func(args)
    sys.exit(0 if status else -1)
