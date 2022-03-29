import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from itertools import chain, product
from math import log10, sqrt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon

class Table:
  """ Constructor """
  def __init__(self, img_set, col_index, y_pred_list):
    self.num_img = len(img_set)
    self.img_set = img_set

    # row index: image i
    self.text1 = ["image {}".format(i) for i in range(self.num_img)]
    self.text1 = self.text1 + ["mean", "std"]
    # part of column index
    self.text2 = ["NRMSE", "SSIM", "PSNR"]

    self.text3 = col_index
    self.pred_list = y_pred_list
    self.num_parr = len(y_pred_list)
    try:
      self.indice, self.data = self._build()
      self.df = self.get()
    except TypeError:
      print ("Failed to create the table data.")
  
  """
  Build the table frame
  """
  def _build(self):
    if len(self.text3) != len(self.pred_list):
      # print error message
      print ("ERROR: Column index and the number of arrays are not matched.")
      return
    elif not all([self.num_img==len(l) for l in self.pred_list]):
      print ("Number of images:", self.num_img, end=" ")
      print ("Number of images for each pred set:", [len(l) for l in self.pred_list])
      print ("Prediction array doesn't have the same length as the real array.")
      return

    # construct the merged cells for column index based
    # it will be something like: "NRMSE", "NRMSE", ..., "SSIM", "SSIM", ..., "PSNR", "PSNR",...
    temp = [[i]*len(self.text3) for i in self.text2]
    arr1 = list(chain.from_iterable(temp))

    # construct the merged cells for column index
    # it will repeat text3, repeating times is based on text2
    temp = [self.text3]*len(self.text2)
    arr2 = list(chain.from_iterable(temp))

    # combine arr1 and arr2 to match each other, naming "measurement" and "recon method"
    arr = [arr1, arr2]
    tup = list(zip(*arr))
    indice = pd.MultiIndex.from_tuples(tup, names=["Measurement", "Recon Method"])

    # create the array for the table data
    # num of rows should be the same as the input img_set
    # num of cols is the product of len(text2) and len(text3)
    # need extra two rows for the mean and std of each column
    data = np.zeros((self.num_img+2, len(self.text2)*len(self.text3)))

    return indice, data
  
  """
  A private function used to compare the difference between images
  return mse, nrmse, ssim, and psnr
  """
  def _compare(self, original, prediction, pixel_range=255):
    mse_val = mean_squared_error(original, prediction)
    nrmse = sqrt(mse_val) / (original.max())
    ssim_val = ssim(original, prediction, data_range=pixel_range)
    max_pixel = pixel_range
    psnr_val = 20 * log10(max_pixel / sqrt(mse_val))

    return mse_val, nrmse, ssim_val, psnr_val

  """
  Similar to the above function, but it only measure the mask area
  """
  def _compare_with_mask(self, original, prediction, mask, pixel_range=255):
    k = int(mask.sum()) # number of data points to measure
    origin_vec = original[mask.astype("bool")]
    pred_vec = prediction[mask.astype("bool")]

    # MSE and NRMSE
    # since out of mask are all zeors, we can use l2 norm (euclidean distance)
    # to compute the mse
    #dist = np.linalg.norm(origin_vec-pred_vec)
    #mse_val = dist**2 / k
    mse_val = mean_squared_error(origin_vec, pred_vec)
    nrmse = sqrt(mse_val) / (origin_vec.max())

    # SSIM
    ssim_val = ssim(origin_vec, pred_vec,
                    data_range=origin_vec.max()-origin_vec.min())

    # PSNR
    psnr_val = 20*np.log10(255) - 10*np.log10(mse_val)

    return mse_val, nrmse, ssim_val, psnr_val

  """
  get the table, return Pandas dataFrame
  """
  def get(self):
    # compare to the real image
    for i in range(self.num_img):
      for k, y_pred in enumerate(self.pred_list):
        # compute the statistics
        _, nrmse, ssim, psnr = self._compare(self.img_set[i], y_pred[i])
        # save to data
        self.data[i, k] = nrmse
        self.data[i, k+self.num_parr] = ssim
        self.data[i, k+self.num_parr*2] = psnr
    
    # compute the average and std for each column of self.data
    for i in range(self.data.shape[1]):
      self.data[self.num_img, i] = np.average(self.data[:, i])
      self.data[self.num_img+1, i] = np.std(self.data[:, i])
    
    # create the dataFrame
    self.df = pd.DataFrame(self.data, index=self.text1, columns=self.indice)
    return self.df
  
  """
  get the table, return Pandas dataFrame
  """
  def get_with_mask(self, mask):
    # compare to the real image
    for i in range(self.num_img):
      for k, y_pred in enumerate(self.pred_list):
        # compute the statistics
        _, nrmse, ssim, psnr = self._compare_with_mask(self.img_set[i], y_pred[i], mask[i])
        # save to data
        self.data[i, k] = nrmse
        self.data[i, k+self.num_parr] = ssim
        self.data[i, k+self.num_parr*2] = psnr
    
    # compute the average and std for each column of self.data
    for i in range(self.data.shape[1]):
      self.data[self.num_img, i] = np.average(self.data[:, i])
      self.data[self.num_img+1, i] = np.std(self.data[:, i])

    # create the dataFrame
    self.df_mask = pd.DataFrame(self.data, index=self.text1, columns=self.indice)
    return self.df_mask
  
  def plot(self, width=20, height=15, loc1=0, loc2=0, loc3=0):
    """ Plot the three curves: NRMSE, SSIM, PSNR """
    num_img = self.num_img
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width, height))
    label = self.text3
    data = self.data

    for i in range(len(label)):
      ax1.plot(np.arange(num_img), data[:num_img, i], '--o', label=label[i])
    ax1.legend(loc=loc1, prop={'size': 15})
    ax1.set_title("(a) NRMSE", size=25)

    for i in range(len(label)):
      ax2.plot(np.arange(num_img), data[:num_img, i+len(label)], '--o', label=label[i])
    ax2.legend(loc=loc2, prop={'size': 15})
    ax2.set_title("(b) SSIM", size=25)

    for i in range(len(label)):
      ax3.plot(np.arange(num_img), data[:num_img, i+2*len(label)], '--o', label=label[i])
    ax3.legend(loc=loc3, prop={'size': 15})
    ax3.set_title("(c) PSNR", size=25)
    plt.show()
