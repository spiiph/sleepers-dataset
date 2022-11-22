import logging

import cv2  # type: ignore
import numpy as np

from PIL import Image

logger = logging.getLogger("dataset.mask")


class Mask:
    """
    Class to represent a segmentation annotation as a binary mask

    NOTE: This class can only handle images with two colours, in palette mode.
    """

    def __init__(self, mask_path: str):
        """
        Initialize a Mask object

        :param mask_path: Path to mask
        """
        logger.debug(f"Creating Mask object with mask_path = {mask_path}")

        self.mask_path: str = mask_path
        self._data: np.ndarray | None = None
        self._contours: list[np.ndarray] | None = None
        self._norm: np.ndarray

    def _load_data(self) -> np.ndarray:
        """
        Load the image data for the mask and construct the image norm

        Load the image data for the mask as a NumPy array. Construct the image norm out of the
        image width and height.

        :return: The image data of the mask as a
        """
        if self._data is None:
            self._data = np.asarray(Image.open(self.mask_path))
            self._norm = np.array([self._data.shape[1], self._data.shape[0]])

        return self._data

    def to_contours(self, normalize: bool = True) -> list[np.ndarray]:
        """
        Convert the masks to contours

        :param normalize: Normalize the contour to the width and height of the image
        :return: List of NumPy arrays, one for each identified contour
        """
        self._load_data()

        if self._contours is None:
            contours, _ = cv2.findContours(self._data, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            self._contours = [
                contour.reshape(contour.shape[0], contour.shape[2])
                for contour in contours
            ]

        if normalize:
            return [contour / self._norm for contour in self._contours]

        return self._contours

    def to_polygons(self, normalize: bool = True) -> list[list[int | float]]:
        """
        Convert the contours to YOLO polygons

        :param normalize: Normalize the polygons to the width and height of the image
        :return: List of flat lists of the format [class, x1, y1, x2, y2, ...]
        """
        return [
            [0] + contour.flatten().tolist()
            for contour in self.to_contours(normalize=normalize)
        ]
