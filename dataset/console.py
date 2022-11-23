import argparse
import logging
import os

from os.path import join

from dataset import patches


LOG_LEVELS = list(logging._nameToLevel.keys())[:-1] # pylint: disable=protected-access


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Logger setup for console applications
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def hello_world() -> None:
    """
    Hello World entry point for testing
    """
    print("Hello world!")


def create_patches() -> None:
    """
    Create patches from one image and one mask
    """
    parser = argparse.ArgumentParser(description="Test patches")
    parser.add_argument(
        "image",
        help="Image file"
    )
    parser.add_argument(
        "mask",
        help="Mask file"
    )
    parser.add_argument(
        "dest",
        help="Destination dir"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=LOG_LEVELS,
        default="INFO"
    )

    args = parser.parse_args()

    logger = setup_logger("dataset", level=args.log_level)
    logger.info(f"Running {__name__} with the following parameters:")
    logger.info(f"  image: {args.image}")
    logger.info(f"  mask: {args.mask}")
    logger.info(f"  dest: {args.dest}")

    for subdir in ["images", "masks"]:
        if not os.path.exists(join(args.dest, subdir)):
            logger.debug(f"Creating missing directory {join(args.dest, subdir)}")
            os.makedirs(join(args.dest, subdir))

    patch = patches.Patches(args.image, args.mask)
    patch.write_patches(args.dest, splits=8, store_empty=False)
