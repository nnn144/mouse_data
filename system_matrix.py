import os
import struct
import time
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

class SystemMatrix:
    # constructor: create sparse tensor
    def __init__(self, fsysmat):
        self.f = open(fsysmat, "rb")
        # read Nsinogram ==> number of ROWs of the system matrix
        self.nsino = struct.unpack('i', self.f.read(4))[0]
        # read Nvolume ==> number of COLUMNs of the system matrix
        self.nvol = struct.unpack('i', self.f.read(4))[0]

        # store the elements of the system matrix
        self.st_index = []
        self.st_elem = []
        self.mat_index = []
        self.mat_elem = []
        for k in range(self.nsino):
            # read row index
            row_index = struct.unpack('i', self.f.read(4))[0]
            assert k == row_index

            # read the number of NON-zero values
            non_zeros = struct.unpack('i', self.f.read(4))[0]

            # read the index of the non-zero value
            index_list = struct.unpack('i'*non_zeros, self.f.read(4*non_zeros))
            #self.mat_index.append(index_list)
            # store the element index to create sparse tensor
            for col_index in index_list:
                self.st_index.append([row_index, col_index])

            # read the non-zero values
            val_list = struct.unpack('f'*non_zeros, self.f.read(4*non_zeros))
            #self.mat_elem.append(val_list)
            for val in val_list:
                self.st_elem.append(np.float32(val))
        # close file
        self.f.close()
        
        # create sparse tensor for the system matrix
        self.st_sm = tf.SparseTensor(indices=self.st_index,
                                values=self.st_elem,
                                dense_shape=(self.nsino, self.nvol))
        #self.st_sm = tf.sparse.to_dense(self.st_sm)

    # do forward/backward projection
    def fpbp(self, sinogram, volume, flag):
        if flag != "fp" and flag != "bp":
            print("Got invalid flag: {}. Flag should be fp or bp.".format(flag))
            return
        
        # matrix multiplication
        for k in range(self.nsino):
            for i, non_zero_index in enumerate(self.mat_index[k]):
                if flag == "fp":
                    sinogram[k] += volume[non_zero_index] * self.mat_elem[k][i]
                elif flag == "bp":
                    volume[non_zero_index] += sinogram[k] * self.mat_elem[k][i]

        if flag == "fp":
            return sinogram
        elif flag == "bp":
            return volume
    
    # do forward/backward projection using tensorflow:
    def tf_fpbp(self, data, flag):
        if flag != "fp" and flag != "bp":
            print("Got invalid flag: {}. Flag should be fp or bp.".format(flag))
            return
        
        # forward projection => sino = volume * sm
        if flag == "fp":
            product = tf.sparse.sparse_dense_matmul(self.st_sm, data)
            
        # backward projection => volume = sm * sinogram
        elif flag == "bp":
            product = tf.sparse.sparse_dense_matmul(self.st_sm, data,
                                                    adjoint_a=True, adjoint_b=True)
        
        # convert tensor to np.ndarray and return
        return product

# sm = SystemMatrix("./data/sm_mouse_vol_64x64x1_noatn")
# img_set = np.load("./data/total_images_cropped.npy").astype("float32").reshape((26, -1, 1))

# s = time.time()
# #sess = tf.Session()
# #with sess.as_default():
# for img in img_set:
#     sm.tf_fpbp(img, "fp").numpy()
# e = time.time()
# print (e-s)

# sinogram = np.zeros((64*64))
# s = time.time()
# for img in img_set:
#     sm.fpbp(sinogram, img, "fp")
# e = time.time()
# print (e-s)
