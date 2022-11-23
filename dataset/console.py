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
from dataset.manage import split


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

    save_image_with_mask(image,
                         mask,
                         join(args.dest, "images_with_masks", basename(args.image)))
    save_image_with_polygons(image,
                             mask,
                             join(args.dest, "images_with_polygons", basename(args.image)))

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
            logger.fatal(f"Source sub-directory {subdir} does not exist.")
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

        The split expects that the source directory contains the sub-directories 'images',
        'masks', and 'splits'. 'images' contains the images of the dataset while 'masks' contains
        the corresponding binary masks. 'splits' contains the three text files 'train.txt',
        'val.txt', and 'test.txt' describing how the dataset should be split.
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

    # Parse arguments
    args = parser.parse_args()

    # Call sub-command
    status = args.func(args)
    sys.exit(0 if status else -1)
