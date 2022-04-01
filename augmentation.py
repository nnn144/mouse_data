import numpy as np

from lib.image_transform import translation, rotation, scaling, shearing, flipping
from lib.pixel_randomize import noise_adding, noise_adding2

def aug_one(img_data, repeat=4, seed=99991111, noisy=False):
  aug_image_list = []

  rows, cols = img_data[0].shape

  shear_range = np.arange(-2, 3)/10
  scale_range = np.arange(8, 11)/10
  translate_range = np.arange(-16, 16, 2)
  rot_angle_range = np.arange(0, 360, 10)

  rng = np.random.default_rng(seed)
  shear_factor_list = []
  tr_list = []
  angle_list = []
  
  # do augmentation for every image in image tr_list
  for _ in range(repeat):
    for img in img_data:
      # get the copy of the image
      temp_img = np.array(img)
      
      # adding noise
      if noisy:
        temp_img = noise_adding2(temp_img)

      # randomly select the affine parameters
      shear_factor = rng.choice(shear_range, size=2)
      scale_factor = rng.choice(scale_range)
      (x, y) = rng.choice(translate_range, size=2)
      rot_angle = rng.choice(rot_angle_range)
      flipping_flag = rng.integers(4)
      
      # flag for flipping
      temp_img = flipping(temp_img, flipping_flag)
      # scaling
      temp_img = scaling(temp_img, scale_factor)
      # shearing
      temp_img = shearing(temp_img, shear_factor)
      # rotation
      temp_img = rotation(temp_img, rot_angle)
      # translation
      temp_img = translation(temp_img, x, y)

      # append to the list
      aug_image_list.append(temp_img)
  
  print (f"After doing augmentation we got: {len(aug_image_list)} augmented images.")
  return np.array(aug_image_list)

def aug_two(noisy_data, gated_data, repeat=4, seed=99991111):
  aug_image_list = []
  aug_sino_list = []

  rows, cols = gated_data[0].shape

  shear_range = range(0, 5)
  translate_range = np.arange(-16, 16, 2)
  rot_angle_range = range(0, 360, 10)

  rng = np.random.default_rng(seed)
  shear_scale_list = []
  tr_list = []
  angle_list = []
  
  # do augmentation for every image in image tr_list
  for _ in range(repeat):
    for gated_img, noisy_img in zip(gated_data, noisy_data):
      # get the copy of the image
      temp_gated_img = np.array(gated_img)
      temp_noisy_img = np.array(noisy_img)
      
      # adding noise
      #temp_gated_img = noise_adding(temp_gated_img)
      #temp_noisy_img = noise_adding(temp_noisy_img)

      # randomly select the affine parameters
      shear_scale = rng.choice(shear_range, size=2)
      (x, y) = rng.choice(translate_range, size=2)
      rot_angle = rng.choice(rot_angle_range, size=1)[0]
      
      # flag for flipping
      flag = rng.integers(4)
      temp_gated_img = filp_image(temp_gated_img, flag)
      temp_noisy_img = filp_image(temp_noisy_img, flag)

      #shearing
      shear_scale_x, shear_scale_y = shear_scale
      shear_mat = np.array([[1, shear_scale_x/10, 0],
                           [shear_scale_y/10, 1, 0]]).astype("float32")
      #temp_gated_img = cv2.warpAffine(temp_gated_img, shear_mat, (rows,cols))
      #temp_noisy_img = cv2.warpAffine(temp_noisy_img, shear_mat, (rows,cols))

      # rotation
      temp_gated_img = rotate_image(temp_gated_img, rot_angle)
      temp_noisy_img = rotate_image(temp_noisy_img, rot_angle)

      # translation
      temp_gated_img = trans_image(temp_gated_img, x, y)
      temp_noisy_img = trans_image(temp_noisy_img, x, y)

      # append to the list
      aug_image_list.append(temp_gated_img)
      aug_sino_list.append(temp_noisy_img)

  assert len(aug_image_list) == len(aug_sino_list)
  #np.save("./phantom/aug_phantom/aug_training_phantom_images_cropped", aug_image_list)
  #np.save("./phantom/aug_phantom/aug_training_phantom_sinograms_cropped", aug_sino_list)
  print ("After doing augmentation we got:", len(aug_sino_list),
         "augmented sinograms and", len(aug_image_list), "augmented images.")

  return np.array(aug_sino_list), np.array(aug_image_list)
