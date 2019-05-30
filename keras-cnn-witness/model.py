from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, \
    concatenate

conv2d_args = dict(
    activation='relu',
    padding='same',
    kernel_initializer='he_normal')

identity = lambda x: x

def DownTier(conv_size, dropout=None, pool_size=None):
    """
    Create a down layer that returns the final layer and also the across
    outputs for merging.
    """
    def create_tier(inputs):
        conv_layer = Conv2D(conv_size, 3, **conv2d_args)(inputs)
        conv_layer = Conv2D(conv_size, 3, **conv2d_args)(conv_layer)
        final_layer, across_layers = conv_layer, { 'conv': conv_layer }

        if dropout:
            drop_layer = Dropout(dropout)(final_layer)
            final_layer = across_layers['drop'] = drop_layer

        if pool_size:
            pool_layer = MaxPooling2D(pool_size=pool_size)(final_layer)
            final_layer = across_layers['pool'] = pool_layer

        return final_layer, across_layers
    return create_tier

def UpTier(conv_size):
    """
    Create an up layer merging from across and down.
    """
    def create_tier(down_inputs, across_inputs):
        up_layer = UpSampling2D(size=(2, 2))(down_inputs)
        up_conv_layer = Conv2D(conv_size, 2, **conv2d_args)(up_layer)
        merge_layer = concatenate([across_inputs, up_conv_layer], axis=3)
        final_layer = Conv2D(conv_size, 3, **conv2d_args)(merge_layer)
        final_layer = Conv2D(conv_size, 3, **conv2d_args)(final_layer)
        return final_layer
    return create_tier

def unet(image_size):
    inputs = Input((image_size[0], image_size[1], 3))

    # 2 TIER

    # down1, down1_across = DownTier(64, pool_size=(2, 2))(inputs)
    # down2 = DownTier(128, dropout=0.5)(down1)[0]

    # up3 = UpTier(64)(down2, down1_across['conv'])

    # 3 TIER

    down1, down1_across = DownTier(16, pool_size=(2, 2))(inputs)
    down2, down2_across = DownTier(32, dropout=0.5, pool_size=(2, 2))(down1)
    down3 = DownTier(64, dropout=0.5)(down2)[0]

    up4 = UpTier(32)(down3, down2_across['drop'])
    up5 = UpTier(16)(up4, down1_across['conv'])

    # 4 TIER

    # down1, down1_across = DownTier(128, pool_size=(2, 2))(inputs)
    # down2, down2_across = DownTier(256, pool_size=(2, 2))(down1)
    # down3, down3_across = DownTier(512, dropout=0.5, pool_size=(2, 2))(down2)
    # down4 = DownTier(1024, dropout=0.5)(down3)[0]

    # up5 = UpTier(512)(down4, down3_across['drop'])
    # up6 = UpTier(256)(up5, down2_across['conv'])
    # up7 = UpTier(128)(up6, down1_across['conv'])

    # 5 TIER

    # down1, down1_across = DownTier(64, pool_size=(2, 2))(inputs)
    # down2, down2_across = DownTier(128, pool_size=(2, 2))(down1)
    # down3, down3_across = DownTier(256, pool_size=(2, 2))(down2)
    # down4, down4_across = DownTier(512, dropout=0.5, pool_size=(2, 2))(down3)
    # down5 = DownTier(1024, dropout=0.5)(down4)[0]

    # up6 = UpTier(512)(down5, down4_across['drop'])
    # up7 = UpTier(256)(up6, down3_across['conv'])
    # up8 = UpTier(128)(up7, down2_across['conv'])
    # up9 = UpTier(64)(up8, down1_across['conv'])

    # conv9 = Conv2D(2, 3, **conv2d_args)(up9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(up5)

    return Model(input=inputs, output=conv10)
