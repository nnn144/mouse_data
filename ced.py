import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, ReLU, LeakyReLU
from tensorflow.keras.models import Model

def build_encoder(img_shape, kernels, base):
    input_img = Input(shape=img_shape)
    
    e1 = Conv2D(filters=base, kernel_size=kernels[0], strides=(1, 1), padding='same')(input_img)
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
    
    e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(2, 2), padding='same')(e2)
    e3 = BatchNormalization(momentum=0.8)(e3)
    e3 = LeakyReLU(0.3)(e3)
    e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(1, 1), padding='same')(e3)
    e3 = BatchNormalization(momentum=0.8)(e3)
    e3 = LeakyReLU(0.3)(e3)
#     e3 = Conv2D(filters=base*4, kernel_size=kernels[0], strides=(1, 1), padding='same')(e3)
#     e3 = BatchNormalization(momentum=0.8)(e3)
#     e3 = LeakyReLU(0.3)(e3)
    
    e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(2, 2), padding='same')(e3)
    e4 = BatchNormalization(momentum=0.8)(e4)
    e4 = LeakyReLU(0.3)(e4)
    e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(1, 1), padding='same')(e4)
    e4 = BatchNormalization(momentum=0.8)(e4)
    e4 = LeakyReLU(0.3)(e4)
#     e4 = Conv2D(filters=base*8, kernel_size=kernels[1], strides=(1, 1), padding='same')(e4)
#     e4 = BatchNormalization(momentum=0.8)(e4)
#     e4 = LeakyReLU(0.3)(e4)
    
    e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(2, 2), padding='same')(e4)
    e5 = BatchNormalization(momentum=0.8)(e5)
    e5 = LeakyReLU(0.3)(e5)
    e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(1, 1), padding='same')(e5)
    e5 = BatchNormalization(momentum=0.8)(e5)
    e5 = LeakyReLU(0.3)(e5)
#     e5 = Conv2D(filters=base*16, kernel_size=kernels[1], strides=(1, 1), padding='same')(e5)
#     e5 = BatchNormalization(momentum=0.8)(e5)
#     e5 = LeakyReLU(0.3)(e5)

    model = Model(inputs=input_img, outputs=e5)
    model.summary()

    return model

# generater
def build_decoder(input_shape, kernels, base, flag):
    z_vec = Input(shape=input_shape)

    d1 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(z_vec)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d1 = LeakyReLU(0.3)(d1)
    d1 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d1 = LeakyReLU(0.3)(d1)
#     d1 = Conv2D(filters=base*16, kernel_size=kernels, strides=(1, 1), padding='same')(d1)
#     d1 = BatchNormalization(momentum=0.8)(d1)
#     d1 = LeakyReLU(0.3)(d1)
    
    if flag == 0:
        d2 = UpSampling2D()(d1)
        d2 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
    else:
        d2 = Conv2DTranspose(filters=base*8, kernel_size=kernels, strides=(2, 2), padding='same')(d1)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(0.3)(d2)
    d2 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(0.3)(d2)
#     d2 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d2)
#     d2 = BatchNormalization(momentum=0.8)(d2)
#     d2 = LeakyReLU(0.3)(d2)

    if flag == 0:
        d3 = UpSampling2D()(d2)
        d3 = Conv2D(filters=base*8, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
    else:
        d3 = Conv2DTranspose(filters=base*8, kernel_size=kernels, strides=(2, 2), padding='same')(d2)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(0.3)(d3)
    d3 = Conv2D(filters=base*4, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(0.3)(d3)
#     d3 = Conv2D(filters=base*4, kernel_size=kernels, strides=(1, 1), padding='same')(d3)
#     d3 = BatchNormalization(momentum=0.8)(d3)
#     d3 = LeakyReLU(0.3)(d3)

    if flag == 0:
        d4 = UpSampling2D()(d3)
        d4 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
    else:
        d4 = Conv2DTranspose(filters=base*2, kernel_size=kernels, strides=(2, 2), padding='same')(d3)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(0.3)(d4)
    d4 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(0.3)(d4)
#     d4 = Conv2D(filters=base*2, kernel_size=kernels, strides=(1, 1), padding='same')(d4)
#     d4 = BatchNormalization(momentum=0.8)(d4)
#     d4 = LeakyReLU(0.3)(d4)

    if flag == 0:
        d5 = UpSampling2D()(d4)
        d5 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    else:
        d5 = Conv2DTranspose(filters=base, kernel_size=kernels, strides=(2, 2), padding='same')(d4)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(0.3)(d5)
    d5 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    d5 = BatchNormalization(momentum=0.8)(d5)
    d5 = LeakyReLU(0.3)(d5)
#     d5 = Conv2D(filters=base, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
#     d5 = BatchNormalization(momentum=0.8)(d5)
#     d5 = LeakyReLU(0.3)(d5)
    
    d_out = Conv2D(filters=1, kernel_size=kernels, strides=(1, 1), padding='same')(d5)
    d_out = BatchNormalization(momentum=0.8)(d_out)
    d_out = LeakyReLU(0.3)(d_out)

    model = Model(inputs=z_vec, outputs=d_out)
    model.summary()

    return model

def build_ced(img_shape, base, flag):
    """ Create encoder and decoder """
    encoder = build_encoder(img_shape=img_shape,
                            kernels=[(5, 5), (3, 3)],
                            base=base)
    # input shape of decoder is the same as output shape of encoder
    input_shape = encoder.layers[-1].output_shape[1:]
    decoder = build_decoder(input_shape=input_shape,
                            kernels=(3, 3),
                            base=base,
                            flag=flag)

    # Build CED (combined model)
    input_img = Input(shape=img_shape)
    encodings = encoder(input_img)
    recon_img = decoder(encodings)
    model = Model(inputs=input_img, outputs=recon_img)
    
    return model

def sample_images(model, x_test, y_test, epoch, path):
    r, c = 2, 5
    idx = np.random.choice(len(x_test), size=c, replace=False)

    recons = model.predict(x_test[idx])
    
    fig, axs = plt.subplots(r, c, figsize=(20, 10))
    for j in range(c):
        axs[0,j].imshow(recons[j,:,:,0], cmap='gray')
        axs[0,j].set_title("Recon {}".format(i))
        axs[0,j].axis('off')
        
        axs[1,j].imshow(y_test[idx[j],:,:,0], cmap='gray')
        axs[1,j].set_title("Real {}".format(i))
        axs[1,j].axis('off')
    
    fig.savefig("{}result_iter{:04}.png".format(path, epoch))
    plt.close()
    return

#print ("Functions.")