from typing import Any, Optional
import cv2
from cv2.typing import MatLike
import numpy as np


def convert_image_to_grayscale(img: MatLike):
    """
    Convert RGB matrix to grayscale matrix
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_grayscale_to_blacknwhite(grayscale_img: MatLike):
    """
    Convert to absolute black and white matrix (0 and 255 only) from grayscale matrix
    """
    return cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def invert_image(blacknwhite_img: MatLike):
    """
    Convert a black-white to a white-black image (0 -> 255 && 255 -> 0) from the matrix
    """
    return cv2.bitwise_not(blacknwhite_img)


def dilate_image(img: MatLike) -> MatLike:
    """
    Thicken the lines for contour detection
    """
    return cv2.dilate(img, None, iterations=5)


def find_contours(img: MatLike):
    """
    Emphasize all contour lines
    """
    contours, _ = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_rectangles(contours):
    """
    Indicates rectangle lines
    """
    rectangular_contours: list[MatLike] = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangular_contours.append(approx)
    return rectangular_contours


def find_boundaries(rectangular_contours):
    """
    Find the boundaries of the table object
    """
    max_area = 0
    boundaries_contour: MatLike = None
    for contour in rectangular_contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            boundaries_contour = contour
    return boundaries_contour


def get_corners_from_mat(points: MatLike):
    """
    @ Return: float[top-left, top-right, bottom-right, bottom-left]
    """
    points = points.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def order_boundaries(boundaries_contour):
    """
    Order boundaries to top-left, top-right, bottom-right, bottom-left
    """
    ordered_boundaries = get_corners_from_mat(boundaries_contour)
    return ordered_boundaries


def draw_boundaries(original_img, boundaries):
    img = original_img.copy()
    return cv2.drawContours(img, [boundaries], -1, (0, 255, 0), 3)


