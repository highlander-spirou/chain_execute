from typing import Any, Optional
import cv2
from cv2.typing import MatLike
import numpy as np


def calculate_distance(p1, p2):
    """
    Distance between two coordinates
    """
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


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


class ImgPreprocessor:
    """
    Class with static methods for `Image preprocessing`, and a callable method to run the flow of process
    """
    def __init__(self, img: Optional[MatLike] = None) -> None:
        self.img = img

    @staticmethod
    def convert_image_to_grayscale(img: MatLike):
        """
        Convert RGB matrix to grayscale matrix
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def threshold_image(grayscale_image: MatLike):
        """
        Convert to absolute black and white matrix (0 and 255 only) from grayscale matrix
        """
        return cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def invert_image(blacknwhite_img: MatLike):
        """
        Convert a black-white to a white-black image (0 -> 255 && 255 -> 0) from the matrix
        """
        return cv2.bitwise_not(blacknwhite_img)

    @staticmethod
    def dilate_image(img: MatLike) -> MatLike:
        """
        Thicken the lines for contour detection
        """
        return cv2.dilate(img, None, iterations=5)

    def __call__(self):
        if self.img is not None:
            grayscale_img = self.convert_image_to_grayscale(self.img)
            blacknwhite_img = self.threshold_image(grayscale_img)
            inverted_img = self.invert_image(blacknwhite_img)
            dilated_img = self.dilate_image(inverted_img)
            return dilated_img
        else:
            raise Exception("Image is None, cannot perform preprocessing")


class LocateTableBoundaries:
    """
    Class with static methods for `Locate the boundaries of table`, a callable method to run the flow of process, 
    and method to draw the boundaries contour
    
    @ Params: 
    - A dilated image matrix
    - Original image [Optional]
    """

    def __init__(self, img: MatLike, original_img: Optional[MatLike]) -> None:
        self.img = img
        self.original_img = original_img

    @staticmethod
    def find_contours(img: MatLike):
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def find_rectangles(contours):
        rectangular_contours: list[MatLike] = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rectangular_contours.append(approx)
        return rectangular_contours

    @staticmethod
    def find_boundaries(rectangular_contours):
        max_area = 0
        boundaries_contour: MatLike = None
        for contour in rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                boundaries_contour = contour
        return boundaries_contour

    @staticmethod
    def order_boundaries(boundaries_contour):
        """
        Order boundaries to top-left, top-right, bottom-right, bottom-left
        """
        ordered_boundaries = get_corners_from_mat(boundaries_contour)
        return ordered_boundaries

    def __call__(self):
        contours = self.find_contours(self.img)
        rectangular_contours = self.find_rectangles(contours)
        boundaries_contour = self.find_boundaries(rectangular_contours)
        ordered_boundaries = self.order_boundaries(boundaries_contour)
        return ordered_boundaries

    def draw_boundaries(self):
        contours = self.find_contours(self.img)
        rectangular_contours = self.find_rectangles(contours)
        boundaries_contour = self.find_boundaries(rectangular_contours)
        img = self.original_img.copy()
        return cv2.drawContours(img, [boundaries_contour], -1, (0, 255, 0), 3)


class IsolateImage:
    """
    Center the image on a 3D plane, and at padding the the image.

    @ Params:
    - ordered_boundaries

    """

    def __init__(self, img, ordered_boundaries) -> None:
        self.img = img
        self.ordered_boundaries = ordered_boundaries

    def calculate_width_height(self):
        """
        Calculate the width and height base on the determined ordered boundaries coordinates
        """
        image_width = int(self.img.shape[1] * 0.9)
        
        boundaries_width = calculate_distance(self.ordered_boundaries[0], self.ordered_boundaries[1])
        boundaries_height = calculate_distance(self.ordered_boundaries[0], self.ordered_boundaries[3])

        aspect_ratio = boundaries_height / boundaries_width

        self.new_image_width = image_width
        self.new_image_height = int(self.new_image_width * aspect_ratio)
    
    def apply_perspective_transform(self):
        pts1 = np.float32(self.ordered_boundaries)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_corrected_image: MatLike = cv2.warpPerspective(self.img, matrix, (self.new_image_width, self.new_image_height))
        return perspective_corrected_image

    def __call__(self):
        self.calculate_width_height()
        return self.apply_perspective_transform()

class TableExtractor:
    """
    OpenCV's + PyTesseract table OCR. This is a forked from https://livefiredev.com/how-to-extract-table-from-image-in-python-opencv-ocr/
    """

    def __init__(self, img_path) -> None:
        self.img_path = img_path
        self.read_img()

    # post init

    def read_img(self):
        self.img = cv2.imread(f'./imgs/{self.img_path}')

    def preprocess_img(self):
        return ImgPreprocessor(self.img)()

    def locate_boundaries(self, dilated_img):
        boundaries_locator = LocateTableBoundaries(
            img=dilated_img, original_img=self.img)
        boundaries_locator()


if __name__ == '__main__':
    img = cv2.imread('./imgs/test_3.jpg')
    img_processor = ImgPreprocessor(img)
    preprocessed_img = img_processor()
    boundaries_locator = LocateTableBoundaries(preprocessed_img, img)
    boundaries = boundaries_locator()
    print(boundaries)
    # # corrected_image = IsolateImage(img, boundaries)()
    output = boundaries_locator.draw_boundaries()
    
    # # img1 = boundaries_locator.draw_boundaries()
    cv2.imwrite('./output/test_1.png', output)
