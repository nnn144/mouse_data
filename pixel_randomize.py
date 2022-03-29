import time
import numpy as np

def beta_para(mu, rng):
  """
  From wiki: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
  """
  # to make sure v > 0, sigma shoule be: mu(1-mu) > sigma > 0
  rate = rng.integers(10, 90)/100
  sigma = mu*(1-mu) * rate
  
  v = mu*(1-mu)/sigma - 1 # v = alpha + beta
  alpha = mu*v
  beta = (1-mu)*v
  return alpha, beta

def noise_adding_beta(image):
  """
  Randomize the pixel value based on the Beta distribution
  """
  # random generator
  seed = int(time.time()*10**10 % 1000000)
  rng = np.random.default_rng(seed)
  # max pixel value
  mx = image.max() + image.mean()
  
  # go through every pixel in the image
  rows, cols = image.shape
  for i in range(rows):
    for j in range(cols):
      if image[i, j] > 6:
        # normalized since the beta distribution mean < 1
        mu = image[i, j]/mx
        # generate the parameters of the beta distribution
        a, b = beta_para(mu, rng)
        if a < 0:
          # make sure it's greater than zero
          print(mu, image[i, j], mx)
        image[i, j] = rng.beta(a, b) * mx
  return image

def noise_adding(image):
  """
  Randomize the pixel value based on Normal (Gaussian) distribution
  """
  # random generator
  seed = int(time.time()*10**10 % 1000000)
  rng = np.random.default_rng(seed)
  
  # go through every pixel in the image
  rows, cols = image.shape
  for i in range(rows):
    for j in range(cols):
      if image[i, j] > 6:
        # get the mu and sigma
        mu = image[i, j]
        rate = rng.integers(100, 301)/1000
        sigma = mu*rate
        # get the random value from the gaus distr given mu and sigma
        image[i, j] = rng.normal(mu, sigma)
  return image

def noise_adding2(image):
  """
  Randomize the pixel value based on Normal (Gaussian) distribution.
  This one will change all the pixels that have the same values, not change them one by one
  """
  rows, cols = image.shape
  
  # random generator
  seed = int(time.time()*10**10 % 1000000)
  rng = np.random.default_rng(seed)
  
  # find the unique valus in the image
  vals = np.unique(image)
  # create a new image, intialize to zero
  new_img = image.reshape(-1)
  img_copy = image.reshape(-1)
  for v in vals[vals>0]:
    # val will be the mean of the normal distribution
    # 
    rate = rng.integers(100, 301)/1000
    mu = v if v > 6 else 6
    sigma = mu*rate
    
    # assign the random values to the image
    indices = np.argwhere(img_copy==v)[:, 0]
    new_img[indices] = rng.normal(mu, sigma, size=len(indices))
  
  # reshape and clip the negative values
  new_img = new_img.reshape(rows, cols)
  new_img = np.clip(new_img, a_min=0, a_max=None)
  return new_img
