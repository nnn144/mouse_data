# useful packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timeit

from lib.image_transform import translation_mat, trans_image
from skimage.transform import radon, rescale

"""
Generate 4 different type of U-shaped objects. All these 4 types are
not symmetric.

This module contains all the functions to generate U-shaped objects.
Some image transformation methods are used.

Note that image rotation function is not used for generating the
basic U-shaped objects. But this function is useful for the data
augmentation.
"""

def compute_sinogram(image):
    """Compute the sinogram of the image using Radon transform"""
    theta = np.linspace(0., 360., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    
    return sinogram


def u_shaped_object():
    """
    Say "U" in the middle
    This u-shaped object constructed by 5 squares, each square is 10 x 10 pixels
    Create one square and do transformation, then we can get rest of the squares
    
    the initial square is it the top-left corner
    6x6= 36 points, 2D image, so 72 values
    suppose: point 1 to 6 is the first row
                  7 to 12 is the second. and so on...
    using 3 columns because it's in homogeneous coordinates
    """
    # Initial image. All zeros now
    u_shape = np.zeros((64, 64))
    # place the small squre on the top left corner of the image
    square = np.ones((36, 3))
    for i in range(36):
        # x-coordinate for point i
        square[i][0] = i % 6
        # x-coordinate for point i
        square[i][1] = i // 6
        # change the correspoding values in u-shaped object
        # u_shape[i % 6, i // 6] = 1
    
    # do translation
    # center square
    center_square = np.dot(translation_mat(29, 35), square.T).astype(int)
    for j in center_square[1]:
        for i in center_square[0]:
            u_shape[i, j] = 1
    
    "left side of the u-shaped object"
    # bottom left one = center square move 6 pixel left
    bottom_left_square = np.dot(translation_mat(-6, 0), center_square).astype(int)
    for j in bottom_left_square[1]:
        for i in bottom_left_square[0]:
            u_shape[i, j] = 1
    # upper left = bottom left one move 6 pixel up
    upper_left_square = np.dot(translation_mat(0, -6), bottom_left_square).astype(int)
    for j in upper_left_square[1]:
        for i in upper_left_square[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more = np.dot(translation_mat(0, -6), upper_left_square).astype(int)
    for j in upper_left_square_more[1]:
        for i in upper_left_square_more[0]:
            u_shape[i, j] = 1
    
    "right side of the u-shaped object"
    # bottom right = center square move 6 pixel right
    bottom_right_square = np.dot(translation_mat(6, 0), center_square).astype(int)
    for j in bottom_right_square[1]:
        for i in bottom_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square = np.dot(translation_mat(0, -6), bottom_right_square).astype(int)
    for j in upper_right_square[1]:
        for i in upper_right_square[0]:
            u_shape[i, j] = 1

    return u_shape


def disconnected_Ushaped_object():
    """
    Similar to the basic U-shaped object
    But using a smaller square to generate the whole object
    From (6, 6) to (4, 4)
    
    Shifting offset is the same. So the object looks like a
    disconnected U shape
    """
    # Initial image. All zeros now
    u_shape = np.zeros((64, 64))
    # place the small squre on the top left corner of the image
    square = np.ones((16, 3))
    for i in range(16):
        # x-coordinate for point i
        square[i][0] = i % 4
        # x-coordinate for point i
        square[i][1] = i // 4
        # change the correspoding values in u-shaped object
        # u_shape[i % 6, i // 6] = 1
    
    # do translation
    # center square
    center_square = np.dot(translation_mat(29, 35), square.T).astype(int)
    for j in center_square[1]:
        for i in center_square[0]:
            u_shape[i, j] = 1
    
    "left side of the u-shaped object"
    # bottom left one = center square move 6 pixel left
    bottom_left_square = np.dot(translation_mat(-6, 0), center_square).astype(int)
    for j in bottom_left_square[1]:
        for i in bottom_left_square[0]:
            u_shape[i, j] = 1
    # upper left = bottom left one move 6 pixel up
    upper_left_square = np.dot(translation_mat(0, -6), bottom_left_square).astype(int)
    for j in upper_left_square[1]:
        for i in upper_left_square[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more = np.dot(translation_mat(0, -6), upper_left_square).astype(int)
    for j in upper_left_square_more[1]:
        for i in upper_left_square_more[0]:
            u_shape[i, j] = 1
    
    "right side of the u-shaped object"
    # bottom right = center square move 6 pixel right
    bottom_right_square = np.dot(translation_mat(6, 0), center_square).astype(int)
    for j in bottom_right_square[1]:
        for i in bottom_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square = np.dot(translation_mat(0, -6), bottom_right_square).astype(int)
    for j in upper_right_square[1]:
        for i in upper_right_square[0]:
            u_shape[i, j] = 1
    
    return u_shape


def thin_Ushaped_object():
    """
    Similar to the previous one, a smaller square is used,
    But shifting offset is changed, so the object is not disconnceted
    """
    # Initial image
    u_shape = np.zeros((64, 64))
    # place the small squre on the top left corner of the image
    square = np.ones((16, 3))
    for i in range(16):
        # x-coordinate for point i
        square[i][0] = i % 4
        # x-coordinate for point i
        square[i][1] = i // 4
        # change the correspoding values in u-shaped object
        # u_shape[i % 6, i // 6] = 1
    
    # do translation
    # center square
    center_square = np.dot(translation_mat(30, 40), square.T).astype(int)
    for j in center_square[1]:
        for i in center_square[0]:
            u_shape[i, j] = 1
    
    "left side of the u-shaped object"
    # bottom left one = center square move 6 pixel left
    bottom_left_square = np.dot(translation_mat(-4, 0), center_square).astype(int)
    for j in bottom_left_square[1]:
        for i in bottom_left_square[0]:
            u_shape[i, j] = 1
    # upper left = bottom left one move 6 pixel up
    upper_left_square = np.dot(translation_mat(0, -4), bottom_left_square).astype(int)
    for j in upper_left_square[1]:
        for i in upper_left_square[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more = np.dot(translation_mat(0, -4), upper_left_square).astype(int)
    for j in upper_left_square_more[1]:
        for i in upper_left_square_more[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more2 = np.dot(translation_mat(0, -4), upper_left_square_more).astype(int)
    for j in upper_left_square_more2[1]:
        for i in upper_left_square_more2[0]:
            u_shape[i, j] = 1
    
    "right side of the u-shaped object"
    # bottom right = center square move 6 pixel right
    bottom_right_square = np.dot(translation_mat(4, 0), center_square).astype(int)
    for j in bottom_right_square[1]:
        for i in bottom_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square = np.dot(translation_mat(0, -4), bottom_right_square).astype(int)
    for j in upper_right_square[1]:
        for i in upper_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square_more = np.dot(translation_mat(0, -4), upper_right_square).astype(int)
    for j in upper_right_square_more[1]:
        for i in upper_right_square_more[0]:
            u_shape[i, j] = 1
    
    return u_shape


def v_shaped_object():
    """
    Similar to the previous one, but now y-offset is not the same for all the shiftings.
    In order to generate V-shaped object, some will be smaller and some will be larger.
    """
    # Initial image
    u_shape = np.zeros((64, 64))
    # place the small squre on the top left corner of the image
    square = np.ones((16, 3))
    for i in range(16):
        # x-coordinate for point i
        square[i][0] = i % 4
        # x-coordinate for point i
        square[i][1] = i // 4
        # change the correspoding values in u-shaped object
        # u_shape[i % 6, i // 6] = 1
    
    # do translation
    # center square
    center_square = np.dot(translation_mat(30, 40), square.T).astype(int)
    for j in center_square[1]:
        for i in center_square[0]:
            u_shape[i, j] = 1
    
    "left side of the u-shaped object"
    # bottom left one = center square move 6 pixel left
    bottom_left_square = np.dot(translation_mat(-2, -4), center_square).astype(int)
    for j in bottom_left_square[1]:
        for i in bottom_left_square[0]:
            u_shape[i, j] = 1
    # upper left = bottom left one move 6 pixel up
    upper_left_square = np.dot(translation_mat(-2, -4), bottom_left_square).astype(int)
    for j in upper_left_square[1]:
        for i in upper_left_square[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more = np.dot(translation_mat(-2, -4), upper_left_square).astype(int)
    for j in upper_left_square_more[1]:
        for i in upper_left_square_more[0]:
            u_shape[i, j] = 1
    # one more step, longer than the other side
    upper_left_square_more2 = np.dot(translation_mat(-2, -4), upper_left_square_more).astype(int)
    for j in upper_left_square_more2[1]:
        for i in upper_left_square_more2[0]:
            u_shape[i, j] = 1
    
    "right side of the u-shaped object"
    # bottom right = center square move 6 pixel right
    bottom_right_square = np.dot(translation_mat(2, -4), center_square).astype(int)
    for j in bottom_right_square[1]:
        for i in bottom_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square = np.dot(translation_mat(2, -4), bottom_right_square).astype(int)
    for j in upper_right_square[1]:
        for i in upper_right_square[0]:
            u_shape[i, j] = 1
    # upper right = bottom right one move 6 pixel up
    upper_right_square_more = np.dot(translation_mat(2, -4), upper_right_square).astype(int)
    for j in upper_right_square_more[1]:
        for i in upper_right_square_more[0]:
            u_shape[i, j] = 1
    
    return u_shape

def summary():
    """Return all four basic types of the U-shaped object and their sinograms"""
    images = np.asarray([u_shaped_object(),
                  disconnected_Ushaped_object(),
                  thin_Ushaped_object(),
                  v_shaped_object()])
    sinograms = np.asarray([compute_sinogram(u_shaped_object()),
                    compute_sinogram(disconnected_Ushaped_object()),
                    compute_sinogram(thin_Ushaped_object()),
                    compute_sinogram(v_shaped_object())])
    
    return images, sinograms
