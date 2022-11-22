import argparse
import logging
import os

from dataclasses import dataclass, field
from os.path import join

from dataset import patches, mask


LOG_LEVELS = list(logging._nameToLevel.keys())[:-1]  # pylint: disable=protected-access


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Logger setup for console applications
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def hello_world(args: argparse.Namespace) -> None:
    """
    Hello World entry point for testing
    """

    logger = setup_logger("dataset", level=args.log_level)
    logger.debug("Hello World!")
    logger.info("Hello World!")
    logger.warning("Hello World!")
    logger.error("Hello World!")
    logger.fatal("Hello World!")


def create_patches(args: argparse.Namespace) -> None:
    """
    Create patches from one image and one mask
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running create-patches with the following parameters:")
    logger.info(f"  image: {args.image}")
    logger.info(f"  mask: {args.mask}")
    logger.info(f"  dest: {args.dest}")

    for subdir in ["images", "masks"]:
        if not os.path.exists(join(args.dest, subdir)):
            logger.debug(f"Creating missing directory {join(args.dest, subdir)}")
            os.makedirs(join(args.dest, subdir))

    patches_obj = patches.Patches(args.image, args.mask)
    patches_obj.write_patches(args.dest, splits=8, store_empty=False)


def test_polygons(args: argparse.Namespace) -> None:
    """
    Test the conversion of masks to contours and polygons
    """
    logger = setup_logger("dataset", level=args.log_level)
    logger.info("Running test-contours-and-polygons with the following parameters:")
    logger.info(f"  mask: {args.mask}")

    mask_obj = mask.Mask(args.mask)
    logger.info("Contours (not normalized):")
    for contour in mask_obj.to_contours(normalize=False):
        logger.info(f">>> {contour.T[0].tolist()}")
        logger.info(f">>> {contour.T[1].tolist()}")
    logger.info("Contours (normalized):")
    for contour in mask_obj.to_contours(normalize=True):
        logger.info(f">>> {contour.T[0].tolist()}")
        logger.info(f">>> {contour.T[1].tolist()}")
    logger.info("Polygons:")
    for polygon in mask_obj.to_polygons(normalize=True):
        logger.info(f">>> {polygon}")


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
        help="Run the hello sub-command"
    )
    hello_parser.set_defaults(func=hello_world)
    hello_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Setup the create-patches parser
    patches_parser = subparsers.add_parser(
        "create-patches",
        help="Run the create-patches sub-command"
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
        help="Destination dir"
    )
    patches_parser.set_defaults(func=create_patches)
    patches_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Set up the test-polygons parser
    polygons_parser = subparsers.add_parser(
        "test-polygons",
        help="Run the teset-polygons sub-command"
    )
    polygons_parser.add_argument(
        "mask",
        help="Mask file"
    )
    polygons_parser.set_defaults(func=test_polygons)
    polygons_parser.add_argument(*log_level.args, **log_level.kwargs)

    # Parse arguments
    args = parser.parse_args()

    args.func(args)
