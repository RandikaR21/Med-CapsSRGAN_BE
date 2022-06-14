# import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, \
    Lambda, Reshape, Activation, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from sr_model.common import pixel_shuffle, normalize_01, denormalize_m11, normalize_m11

LR_SIZE = 24
HR_SIZE = 96


def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def res_block_enhanced(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16, withBatchNorm=True):
    x_in = Input(shape=(None, None, 1))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        if withBatchNorm:
            x = res_block(x, num_filters)
        else:
            x = res_block_enhanced(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(1, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    sr_resnet_generator = Model(x_in, x)
    # tf.keras.utils.plot_model(sr_resnet_generator, show_shapes=True, to_file="SRResNet_Generator.png")
    # sr_resnet_generator.summary()
    return sr_resnet_generator


generator = sr_resnet


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def capsule_discriminator():
    img = Input(shape=(HR_SIZE, HR_SIZE, 1))

    x = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', name='conv1')(img)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)
    x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
    x = Lambda(squash, name='primarycap_squash')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Flatten()(x)
    uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    c = Activation('softmax', name='softmax_digitcaps1')(uhat)
    c = Dense(160)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps2')(s_j)
    c = Dense(160)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation('softmax', name='softmax_digitcaps3')(s_j)
    c = Dense(160)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    pred = Dense(1, activation='sigmoid')(s_j)
    discriminator = Model(img, pred)
    # tf.keras.utils.plot_model(discriminator, show_shapes=True, to_file="Capsule_Discriminator.png")
    # capsule_discriminator.summary()
    return discriminator


def original_discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def original_discriminator(num_filters=64):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 1))
    x = Lambda(normalize_m11)(x_in)

    x = original_discriminator_block(x, num_filters, batchnorm=False)
    x = original_discriminator_block(x, num_filters, strides=2)

    x = original_discriminator_block(x, num_filters * 2)
    x = original_discriminator_block(x, num_filters * 2, strides=2)

    x = original_discriminator_block(x, num_filters * 4)
    x = original_discriminator_block(x, num_filters * 4, strides=2)

    x = original_discriminator_block(x, num_filters * 8)
    x = original_discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
