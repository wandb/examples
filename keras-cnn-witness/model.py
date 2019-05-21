from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, \
    concatenate

conv2d_args = dict(
    activation='relu',
    padding='same',
    kernel_initializer='he_normal')

def unet(image_size):
    input_size = (image_size[0], image_size[1], 3)
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, **conv2d_args)(inputs)
    conv1 = Conv2D(64, 3, **conv2d_args)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, **conv2d_args)(pool1)
    conv2 = Conv2D(128, 3, **conv2d_args)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, **conv2d_args)(pool2)
    conv3 = Conv2D(256, 3, **conv2d_args)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, **conv2d_args)(pool3)
    conv4 = Conv2D(512, 3, **conv2d_args)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, **conv2d_args)(pool4)
    conv5 = Conv2D(1024, 3, **conv2d_args)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, **conv2d_args)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, **conv2d_args)(merge6)
    conv6 = Conv2D(512, 3, **conv2d_args)(conv6)

    up7 = Conv2D(256, 2, **conv2d_args)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3,up7], axis=3)

    conv7 = Conv2D(256, 3, **conv2d_args)(merge7)
    conv7 = Conv2D(256, 3, **conv2d_args)(conv7)

    up8 = Conv2D(128, 2, **conv2d_args)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(128, 3, **conv2d_args)(merge8)
    conv8 = Conv2D(128, 3, **conv2d_args)(conv8)

    up9 = Conv2D(64, 2, **conv2d_args)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(64, 3, **conv2d_args)(merge9)
    conv9 = Conv2D(64, 3, **conv2d_args)(conv9)
    conv9 = Conv2D(2, 3, **conv2d_args)(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    return Model(input=inputs, output=conv10)
