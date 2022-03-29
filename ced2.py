import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Softmax
from tensorflow.keras.layers import BatchNormalization, Activation, ReLU, LeakyReLU
from tensorflow.keras.models import Model

from math import log10, sqrt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

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

class Early_Stop():
    def __init__(self, patient, min_delta=0):
        """
        Parameter:
        - patient: Number of epochs with no improvement after which training will be stopped.
        """
        # store "patient"
        self.patient = patient
        #  a flag to check if best_loss has been updated or not
        self.is_changed = False
        # initial best_loss
        self.best_val_loss = float("inf")
        # min change, less than it will count as no change
        self.min_delta = min_delta
        # count the number of epochs has been stored
        self.count = 0

    def __is_changed(self, val_loss):
        """
        Check if the best val loss needs update
        """
        delta = abs(self.best_val_loss - val_loss)
        if self.best_val_loss > val_loss and self.min_delta <= delta:
            self.__update(val_loss)
            return True
        else:
            return False

    def __update(self, val_loss):
        """
        Update the best val loss. count and loss_list also need reset
        """
        self.best_val_loss = val_loss
        self.count = 1
        return

    def stop_training(self, val_loss):
        """
        If val_loss does decrease, no need to stop. -> Return Falss
        otherwise consider stop.                    -> Return True if necessary
        """
        if self.__is_changed(val_loss):
            return False
        # cannot update best_val_loss, check if it's the "patient" number of epochs
        if self.count < self.patient:
            # if NOT YET "patient" number of epochs, increase the count
            self.count += 1
            return False
        else:
            # if now it's the "patient" number of epochs, check if no more changed
            print(f"Stop because loss didn't update during the past {self.patient} epochs.")
            return True

class Attention(tf.keras.layers.Layer):
    def __init__(self, filters=1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.filters = filters
        
        self.query_conv = Conv2D(filters, kernel_size=1, padding='same')
        self.key_conv = Conv2D(filters, kernel_size=1, padding='same')
        self.value_conv = Conv2D(filters, kernel_size=1, padding='same')
        
        self.gamma = K.zeros(shape=(1,))

    def call(self, x):
        # get the size
        batchsize, width, height, c = x.shape
        
        # original
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        # reshape
        qt = K.reshape(q, [-1, width*height, c])
        kt = K.reshape(k, [-1, width*height, c])
        vt = K.reshape(v, [-1, width*height, c])
        
        s = K.batch_dot(K.permute_dimensions(qt, [0, 2, 1]), kt)
        scores = K.batch_dot(vt, K.softmax(s))
        scores = K.reshape(scores, [-1, width, height, c])
        
        return x + self.gamma * scores

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"filters" : self.filters})
        return config

def build_encoder(img_shape, kernels, base, attention=False):
    input_img = Input(shape=img_shape)

    e1 = Conv2D(filters=base, kernel_size=kernels[0], strides=(1, 2), padding='same')(input_img)
    e1 = BatchNormalization(momentum=0.8)(e1)
    e1 = LeakyReLU(0.3)(e1)
    e1 = Conv2D(filters=base, kernel_size=kernels[0], strides=(1, 1), padding='same')(e1)
    e1 = BatchNormalization(momentum=0.8)(e1)
    e1 = LeakyReLU(0.3)(e1)
    e1 = Conv2D(filters=base, kernel_size=kernels[0], strides=(1, 1), padding='same')(e1)
    e1 = BatchNormalization(momentum=0.8)(e1)
    e1 = LeakyReLU(0.3)(e1)

    e2 = Conv2D(filters=base*2, kernel_size=kernels[0], strides=(2, 2), padding='same')(e1)
    e2 = BatchNormalization(momentum=0.8)(e2)
    e2 = LeakyReLU(0.3)(e2)
    e2 = Conv2D(filters=base*2, kernel_size=kernels[0], strides=(1, 1), padding='same')(e2)
    e2 = BatchNormalization(momentum=0.8)(e2)
    e2 = LeakyReLU(0.3)(e2)
    e2 = Conv2D(filters=base*2, kernel_size=kernels[0], strides=(1, 1), padding='same')(e2)
    e2 = BatchNormalization(momentum=0.8)(e2)
    e2 = LeakyReLU(0.3)(e2)
#     if attention:
#         e2 = Attention(base*2)(e2)

    e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(2, 2), padding='same')(e2)
    e3 = BatchNormalization(momentum=0.8)(e3)
    e3 = LeakyReLU(0.3)(e3)
    e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(1, 1), padding='same')(e3)
    e3 = BatchNormalization(momentum=0.8)(e3)
    e3 = LeakyReLU(0.3)(e3)
    e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(1, 1), padding='same')(e3)
    e3 = BatchNormalization(momentum=0.8)(e3)
    e3 = LeakyReLU(0.3)(e3)
#     if attention:
#         e3 = Attention(base*4)(e3)

    e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(2, 2), padding='same')(e3)
    e4 = BatchNormalization(momentum=0.8)(e4)
    e4 = LeakyReLU(0.3)(e4)
    e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(1, 1), padding='same')(e4)
    e4 = BatchNormalization(momentum=0.8)(e4)
    e4 = LeakyReLU(0.3)(e4)
    e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(1, 1), padding='same')(e4)
    e4 = BatchNormalization(momentum=0.8)(e4)
    e4 = LeakyReLU(0.3)(e4)
#     if attention:
#         e4 = Attention(base*8)(e4)

    e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(2, 2), padding='same')(e4)
    e5 = BatchNormalization(momentum=0.8)(e5)
    e5 = LeakyReLU(0.3)(e5)
    e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(1, 1), padding='same')(e5)
    e5 = BatchNormalization(momentum=0.8)(e5)
    e5 = LeakyReLU(0.3)(e5)
    e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(1, 1), padding='same')(e5)
    e5 = BatchNormalization(momentum=0.8)(e5)
    e5 = LeakyReLU(0.3)(e5)
#     if attention:
#         e5 = Attention(base*16)(e5)

    e6 = Conv2D(filters=base*32, kernel_size=kernels[1], strides=(2, 2), padding='same')(e5)
    e6 = BatchNormalization(momentum=0.8)(e6)
    e6 = LeakyReLU(0.3)(e6)
    e6 = Conv2D(filters=base*32, kernel_size=kernels[1], strides=(1, 1), padding='same')(e6)
    e6 = BatchNormalization(momentum=0.8)(e6)
    e6 = LeakyReLU(0.3)(e6)
    e6 = Conv2D(filters=base*32, kernel_size=kernels[1], strides=(1, 1), padding='same')(e6)
    e6 = BatchNormalization(momentum=0.8)(e6)
    e6 = LeakyReLU(0.3)(e6)
    if attention:
        e6 = Attention(base*32)(e6)

    e7 = Conv2D(filters=base*64, kernel_size=kernels[1], strides=(2, 2), padding='same')(e6)
    e7 = BatchNormalization(momentum=0.8)(e7)
    e7 = LeakyReLU(0.3)(e7)
    e7 = Conv2D(filters=base*64, kernel_size=kernels[1], strides=(1, 1), padding='same')(e7)
    e7 = BatchNormalization(momentum=0.8)(e7)
    e7 = LeakyReLU(0.3)(e7)
    e7 = Conv2D(filters=base*64, kernel_size=kernels[1], strides=(1, 1), padding='same')(e7)
    e7 = BatchNormalization(momentum=0.8)(e7)
    e7 = LeakyReLU(0.3)(e7)
    if attention:
        e7 = Attention(base*64)(e7)

    e_out = Flatten()(e6)
    e_out = Dropout(0.2)(e_out)
    e_out = Reshape((1, 1, 2048))(e_out)

    model = Model(inputs=input_img, outputs=e_out)
    model.summary()

    return model

# generater
def build_decoder(input_shape, kernels, base, flag, attention=False):

    z_vec = Input(shape=input_shape)

    if flag == 0:
        d1 = UpSampling2D(size=(2, 2))(z_vec)
        d1 = Conv2D(filters=base*32, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
    else:
        d1 = Conv2DTranspose(filters=base*32, kernel_size=kernels, strides=(2, 2), padding='same')(z_vec)
    d1 = Conv2D(filters=base*32, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d1 = LeakyReLU(0.3)(d1)
    d1 = Conv2D(filters=base*32, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d1 = LeakyReLU(0.3)(d1)
    d1 = Conv2D(filters=base*32, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d1 = LeakyReLU(0.3)(d1)
    if attention:
        d1 = Attention(base*32)(d1)

    if flag == 0:
        d2 = UpSampling2D()(d1)
        d2 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
    else:
        d2 = Conv2DTranspose(filters=base*16, kernel_size=kernels, strides=(2, 2), padding='same')(d1)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(0.3)(d2)
    d2 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(0.3)(d2)
    d2 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(0.3)(d2)
    if attention:
        d2 = Attention(base*16)(d2)

    if flag == 0:
        d3 = UpSampling2D()(d2)
        d3 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
    else:
        d3 = Conv2DTranspose(filters=base*8, kernel_size=kernels, strides=(2, 2), padding='same')(d2)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(0.3)(d3)
    d3 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(0.3)(d3)
    d3 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(0.3)(d3)
    if attention:
        d3 = Attention(base*8)(d3)

    if flag == 0:
        d4 = UpSampling2D()(d3)
        d4 = Conv2D(filters=base*4, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
    else:
        d4 = Conv2DTranspose(filters=base*4, kernel_size=kernels, strides=(2, 2), padding='same')(d3)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(0.3)(d4)
    d4 = Conv2D(filters=base*4, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(0.3)(d4)
    d4 = Conv2D(filters=base*4, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(0.3)(d4)
    if attention:
        d4 = Attention(base*4)(d4)

    if flag == 0:
        d5 = UpSampling2D()(d4)
        d5 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    else:
        d5 = Conv2DTranspose(filters=base*2, kernel_size=kernels, strides=(2, 2),
                             padding='same')(d4)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(0.3)(d5)
    d5 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(0.3)(d5)
    d5 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(0.3)(d5)
    if attention:
        d5 = Attention(base*2)(d5)

    if flag == 0:
        d6 = UpSampling2D()(d5)
        d6 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d6)
    else:
        d6 = Conv2DTranspose(filters=base, kernel_size=kernels, strides=(2, 2),
                             padding='same')(d5)
    d6 = BatchNormalization(momentum=0.8)(d6)
    d6 = LeakyReLU(0.3)(d6)
    d6 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d6)
    d6 = BatchNormalization(momentum=0.8)(d6)
    d6 = LeakyReLU(0.3)(d6)
    d6 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d6)
    d6 = BatchNormalization(momentum=0.8)(d6)
    d6 = LeakyReLU(0.3)(d6)
    if attention:
        d6 = Attention(base)(d6)

    d_out = Conv2D(filters=8, kernel_size=kernels, strides=(1, 1), padding='same')(d6)
    d_out = BatchNormalization(momentum=0.8)(d_out)
    d_out = Conv2D(filters=4, kernel_size=kernels, strides=(1, 1), padding='same')(d_out)
    d_out = BatchNormalization(momentum=0.8)(d_out)
    d_out = Conv2D(filters=1, kernel_size=kernels, strides=(1, 1), padding='same')(d_out)
    d_out = BatchNormalization(momentum=0.8)(d_out)
    d_out = ReLU()(d_out)
    if attention:
        d_out = Attention(1)(d_out)

    d_out = Dropout(0.2)(d_out)
    model = Model(inputs=z_vec, outputs=d_out)
    model.summary()

    return model

def build_intermediate(input_shape):
    vec = Input(shape=input_shape)

    i0 = Dense(128)(vec)
    i1 = Dense(64)(i0)
    i1 = Dense(32)(i1)
    i1 = Dense(64)(i1)
    i1 = Dense(128)(i1)

    out = Dense(max(input_shape))(i1)

    model = Model(inputs=vec, outputs=out)
    model.summary()

    return model

def build_ced(img_shape, base=16, flag=0, attention=False):
    """ Create encoder and decoder """
    encoder = build_encoder(img_shape=img_shape,
                            kernels=[(4, 4), (2, 2)],
                            base=base,
                            attention=attention)
    # input shape of decoder is the same as output shape of encoder
    dec_input_shape = encoder.layers[-1].output_shape[1:]
    #intermediate = build_intermediate(input_shape=dec_input_shape)
    decoder = build_decoder(input_shape=dec_input_shape,
                            kernels=(2, 2),
                            base=base,
                            flag=flag,
                            attention=attention)

    # Build CED (combined model)
    input_img = Input(shape=img_shape)
    encodings = encoder(input_img)
    #medium = intermediate(encodings)
    recon_img = decoder(encodings)
    model = Model(inputs=input_img, outputs=recon_img)

    return model

def sample_images_mask(model, x_test, y_test, y_fbp, epoch, path, mask):
    recons = model.predict(x_test)

    r, c = 3, len(x_test)
    fig, axs = plt.subplots(r, c, figsize=(c*5, r*5))

    for j in range(c):
        fbp = y_fbp[j, :, :]
        real = y_test[j, :, :, 0]
        pred = recons[j, :, :, 0]

        axs[0,j].imshow(real, cmap='gray')
        axs[0,j].set_title("Real {}".format(j))
        axs[0,j].axis('off')

        axs[1,j].imshow(pred)
        axs[1,j].set_title("Recon {}".format(j))
        axs[1,j].axis('off')
        # add information about the statistics to the figure
        _, b, c, d = compare_with_mask(real, pred, mask[j])
        msg = "NRMSE: {:.03f}\nSSIM: {:.03f}\nPSNR: {:.03f}".format(b, c, d)
        axs[1,j].text(1, 16, msg, fontsize=20, color='w')

        axs[2,j].imshow(fbp)
        axs[2,j].set_title("FBP {}".format(j))
        axs[2,j].axis('off')
        # add information about the statistics to the figure
        _, b, c, d = compare_with_mask(real, fbp, mask[j])
        msg = "NRMSE: {:.03f}\nSSIM: {:.03f}\nPSNR: {:.03f}".format(b, c, d)
        axs[2,j].text(1, 16, msg, fontsize=20, color='w')

    fig.savefig("{}result_iter{:03}.png".format(path, epoch))
    plt.close()
    return

#build_ced(img_shape=(64, 64, 1), flag=1)
