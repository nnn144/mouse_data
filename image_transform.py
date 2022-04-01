import cv2
import numpy as np
import matplotlib.pyplot as plt

def translation_mat(x, y):
    """
    This matrix is used for image shifting
    
    Parameters:
    x ---- x shifing offset
    y ---- y shifing offset
    
    Return:
    mat ---- translation matrix
    """
    mat = np.array([[1, 0, x],
               [0, 1, y],
               [0, 0, 1]])
    return mat

def translation(image, tx, ty):
    """
    Calculate the translated image.
    
    Parameters:
    image    ---- input image
    tx      ---- x shifting offset
    ty      ---- y shifting offset
    
    Return:
    trans_img ---- translated image
    
    Notes:
    The returned image will be displayed by matplot.
    Color order is different between opencv and matplot
    OpenCV: BRG, Matplot: RGB
    Need re-order before return if the image is colorful
    """
    # generate translation matrix
    trans_mat_homo = translation_mat(tx, ty)
    # opencv doesn't require the homogeneous coordinates
    trans_mat = trans_mat_homo[0:2, :].astype(dtype='float32')
    # get the size of the image
    rows, cols = image.shape
    # warping
    trans_img = cv2.warpAffine(image, trans_mat, (cols,rows))
    # re-order the image, so that it can be displayed correctly by matplot
    #b, r, g = cv2.split(trans_img)
    #trans_img = cv2.merge(r, g, b)
    # return, it is an np.ndarray
    return trans_img

def rotation(image, rot_angle):
    """Return rotated image given rotation angle"""
    rows, cols = image.shape
    
    rot_M = cv2.getRotationMatrix2D((cols/2,rows/2), rot_angle, 1)
    rotated_img = cv2.warpAffine(image, rot_M, (cols,rows))
    
    return rotated_img

def scaling(image, scale_factor):
    """ Uniform scaling """
    copy_img = np.array(image)
    rows, cols = copy_img.shape
    
    scale_mat = np.array([[scale_factor, 0, 0],
                          [0, scale_factor, 0]]).astype("float32")
    temp_img = cv2.warpAffine(copy_img, scale_mat, (rows,cols))
    return temp_img

def shearing(image, shear_factor):
    """ Image shearing """
    copy_img = np.array(image)
    rows, cols = copy_img.shape
    
    sx, sy = shear_factor
    shear_mat = np.array([[1, sx, 0],
                          [sy, 1, 0]]).astype("float32")
    temp_img = cv2.warpAffine(copy_img, shear_mat, (rows,cols))
    return temp_img

def flipping(image, flag):
    """ Flip the image and return it """
    if flag == 0:
        # vertically
        return np.flipud(image)
    elif flag == 1:
        # horizontally
        return np.fliplr(image)
    elif flag == 2:
        # both vertically and horizontally 
        return np.flipud(np.fliplr(image))
    elif flag == 3:
        # no flip, return original image
        return image
    else:
        print("ERROR: Wrong flag. Should be 0, 1, 2, 3")
        return

def image_plot(r, c, img_set, num):
    """ Randomly plot certain number of images from img_set """
    rnd_idx = np.random.randint(num, size=c)
    fig, ax = plt.subplots(r, c, figsize=(c*5, r*5))
    for i in range(r):
        for j in range(c):
            k = rnd_idx[j]
            ax[i, j].imshow(img_set[i][k])
            ax[i, j].axis("off")
    plt.show()
    return

def image_hplot(img_set: list, names: list):
    """ Plot all the images in the img_set. Each ROW is one set """
    # check if num of names is the same as the num of sets
    if len(img_set) != len(names):
        print(f"ERROR: number of sets is {len(img_set)}, but only {len(names)} names")
        return
    n_sets = len(img_set)
    n_images = len(img_set[0])
    # figure to plot has "n_sets" rows and "n_images" columns
    # each row is one set
    fig, ax = plt.subplots(n_sets, n_images, figsize=(n_images*5, n_sets*5))
    for i in range(n_sets):
        for j in range(n_images):
            ax[i, j].imshow(img_set[i][j])
            ax[i, j].set_title(f"{names[i]} {j}")
            ax[i, j].axis("off")
    plt.show()
    return

def image_vplot(img_set: list, names: list):
    """ Plot all the images in the img_set. Each COLUMN is one set """
    # check if num of names is the same as the num of sets
    if len(img_set) != len(names):
        print(f"ERROR: number of sets is {len(img_set)}, but only {len(names)} names")
        return
    n_sets = len(img_set)
    n_images = len(img_set[0])
    # figure to plot has "n_sets" columns and "n_images" rows
    # each row is one set
    fig, ax = plt.subplots(n_images, n_sets, figsize=(n_sets*5, n_images*5))
    for i in range(n_images):
        for j in range(n_sets):
            ax[i, j].imshow(img_set[j][i])
            ax[i, j].set_title(f"{names[j]} {i}")
            ax[i, j].axis("off")
    plt.show()
    return
