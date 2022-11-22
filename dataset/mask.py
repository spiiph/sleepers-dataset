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
        self._data: np.ndarray
        self._data_loaded: bool = False
        self._contours: list[np.ndarray]
        self._contours_loaded: bool = False
        self._norm: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """
        Property to return the mask data

        :return: The image data of the mask as a NumPy array

        """
        return self._load_data()

    def _load_data(self) -> np.ndarray:
        """
        Load the image data for the mask and construct the image norm

        Load the image data for the mask as a NumPy array. Construct the image norm out of the
        image width and height.

        :return: The image data of the mask as a NumPy array
        """
        if not self._data_loaded:
            self._data = np.asarray(Image.open(self.mask_path))
            self._norm = np.array([self._data.shape[1], self._data.shape[0]])

        return self._data

    def to_contours(self, normalize: bool = True) -> list[np.ndarray]:
        """
        Convert the masks to contours

        :param normalize: Normalize the contour to the width and height of the image
        :return: List of NumPy arrays, one for each identified contour
        """
        if not self._contours_loaded:
            contours, _ = cv2.findContours(self.data, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
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

    def overlay(self,
                image: Image.Image,
                colour: tuple[int, int, int] = (255, 255, 255)
                ) -> Image.Image:
        """
        Overlay the mask on an image by painting it with the given colour

        :param image: Image on which to overlay the mask
        :param colour: RGB colour specification as a 3-tuple (default: (255, 255, 255))
        :return: Image with the mask overlayed
        """
        assert len(colour) == 3, "'colour' must be a tuple of 3'"

        # Convert the image to RGB and extract its data
        image_data = np.asarray(image.convert("RGB"))

        # Create the overlay from the binary mask by stacking RGB values
        overlay_data = np.stack(
            [self.data * colour[0], self.data * colour[1], self.data * colour[2]],
            -1
        )

        # Where overlay is non-zero, select that value, otherwise the original image value
        result = np.where(overlay_data > 0, overlay_data, image_data)
        # result = np.bitwise_or(image_data, overlay_data)
        return Image.fromarray(result)

    def overlay_polygons(self,
                         image: Image.Image,
                         colour: tuple[int, int, int] = (255, 255, 255)
                         ) -> Image.Image:
        """
        Overlay the contours of a mask on an image by painting it with the given colour

        :param image: Image on which to overlay the mask
        :param colour: RGB colour specification as a 3-tuple (default: (255, 255, 255))
        :return: Image with the mask overlayed
        """
        assert len(colour) == 3, "'colour' must be a tuple of 3'"

        result = np.asarray(image.convert("RGB"))
        contours = self.to_contours(normalize=False)

        # Draw contours on the image
        cv2.polylines(result,
                      contours,
                      isClosed=True,
                      color=colour,
                      thickness=5,
                      lineType=cv2.FILLED)

        return Image.fromarray(result)
