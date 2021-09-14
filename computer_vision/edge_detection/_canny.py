import numpy as np
import cv2
import os
import argparse

from sobel import sobel_edge_detection
from gaussian_smoothing import gaussian_blur

import matplotlib.pyplot as plt


def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()

    return output


def threshold(image, low, high, weak, verbose=False):
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()

    return output



def set_edges(image, weak, row_range, col_range):
    g_im = image.copy()
    moves = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]]
    for row in row_range:
        for col in col_range:
            if g_im[row, col] == weak:
                isOk = False
                for move in moves:
                    if g_im[row + move[0], col + move[1]] == 255:
                        isOk = True
                g_im[row, col] = 255 if isOk else 0
    return g_im

def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = set_edges(image, weak, range(1, image_row), range(1, image_col))
    bottom_to_top = set_edges(image, weak, range(image_row - 1, 0, -1), range(image_col - 1, 0, -1))
    right_to_left = set_edges(image, weak, range(1, image_row), range(image_col - 1, 0, -1))
    left_to_right = set_edges(image, weak, range(image_row - 1, 0, -1), range(1, image_col))

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255

    return final_image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread("/home/bogdan/Main/4course/practice/machine_learning/computer_vision/edge_detection/images/1.png")

    if image is None:
        raise ValueError("Image is none")

    blurred_image = gaussian_blur(image, kernel_size=9, verbose=False)

    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, edge_filter, convert_to_degree=True,
                                                                  verbose=args["verbose"])

    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])

    weak = 50

    new_image = threshold(new_image, 5, 20, weak=weak, verbose=args["verbose"])

    new_image = hysteresis(new_image, weak)

    plt.imshow(new_image, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.show()
