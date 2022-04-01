import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.transform import iradon

# linear normalization
def norm(arr, a=0., b=255.):
    arr_norm = np.zeros(arr.shape)
    arr_norm = cv2.normalize(arr, arr_norm, a, b, cv2.NORM_MINMAX)
    return arr_norm

def normalizing(img_set, a=0., b=255.):
    for i, img in enumerate(img_set):
        norm_img = norm(img, a, b)
        norm_img = np.clip(norm_img, a_min=0, a_max=None)
        img_set[i] = norm_img
    return img_set

def normalize_individual(img_set, a=0., b=255.):
    for i, img in enumerate(img_set):
        norm_img = np.clip(img, a_min=0, a_max=None)
        noisy_img = norm_img + np.random.poisson(norm_img)
        img_set[i] = np.clip(norm(noisy_img, a, b), a_min=0, a_max=None)
    return img_set

def mask_creation(img_set):
    """ Using Ostu threshold """
    mask = []
    for i in range(len(img_set)):
        val = threshold_otsu(img_set[i])
        mask.append((img_set[i]>val).astype("int32"))
    return np.array(mask)

def min_bbox(image_mask):
    """ Use the mask to find the bounding box """
    rows, cols = image_mask.shape[:2]
    l, t = 10000, 10000 # left and top boundary
    r, b = 0, 0         # right and bottom
    for i in range(rows):
        for j in range(cols):
            if image_mask[i, j] > 0:
                # update l boundary (min column)
                if j < l:
                    l = j
                # update t boundary (min row)
                if i < t:
                    t = i
                # update r boundary (max column)
                if j > r:
                    r = j
                # update b boundary (max row)
                if i > b:
                    b = i
    # return the boundary index
    return l, r, t, b

def bounding_box(mask_set, kk=5):
    """ 
    Create bounding box based on the mask
    First create a min bounding box, then extend it
    """
    bbox_list2 = []
    for i in range(len(mask_set)):
        m = mask_set[i]
        pp, qq = 0, mask_set.shape[1]
        temp = np.zeros(m.shape)
        if m.sum() > 0:
            l, r, t, b = min_bbox(m)
            # make sure won't be out of bound afer extension
            a = max(t-kk, 0)
            b = min(b+kk, qq)

            c = max(l-kk, 0)
            d = min(r+kk, qq)
            # set all vales within the box to be one
            temp[a:b, c:d] = 1
        bbox_list2.append(temp)
    return bbox_list2

def compare_with_mask(original, prediction, mask, pixel_range=255):
    k = mask.sum()
    
    origin_vec = original[mask.astype("bool")]
    pred_vec = prediction[mask.astype("bool")]
    
    # MSE and NRMSE
    # since out of mask are all zeors, we can use l2 norm (euclidean distance)
    # to compute the mse
    dist = np.linalg.norm(origin_vec-pred_vec)
    mse_val = dist**2 / k
    nrmse = sqrt(mse_val) / (origin_vec.max())

    # SSIM
    ssim_val = ssim(origin_vec, pred_vec, data_range=origin_vec.max()-origin_vec.min())

    # PSNR
    psnr_val = 20 * np.log10(255) - 10 * np.log10(mse_val)

    return mse_val, nrmse, ssim_val, psnr_val
