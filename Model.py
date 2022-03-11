from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

def conv_block(x, size, dropout):
    # Convolutional layer.
    conv = layers.Conv3D(size, (3, 3, 3), kernel_initializer='he_uniform', padding="same")(x)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv3D(size, (3, 3, 3), kernel_initializer='he_uniform', padding="same")(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv

def gating_signal(input, out_size):
    # resize the down layer feature map into the same dimension as the up layer feature map
    # using 1x1 conv
    # :return: the gating feature map with the same dimension of the up layer feature map
    x = layers.Conv3D(out_size, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(input)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)  # (None, 8, 8, 8, 128)
    shape_g = K.int_shape(gating)  # (None, 4, 4, 4, 128)
    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), kernel_initializer='he_uniform', padding='same')(
        x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv3D(inter_shape, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(gating)
    upsample_g = layers.Conv3DTranspose(inter_shape, (3, 3, 3),
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2],
                                                 shape_theta_x[3] // shape_g[3]),
                                        kernel_initializer='he_uniform', padding='same')(phi_g)  # 16
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv3D(1, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling3D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(
        sigmoid_xg)  # 32
    upsample_psi = repeat_elem(upsample_psi, shape_x[4])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv3D(shape_x[4], (1, 1, 1), kernel_initializer='he_uniform', padding='same')(y)
    return result


# Parameters for model
img_height = X_train.shape[1]  # 64
img_width = X_train.shape[2]  # 64
img_depth = X_train.shape[3]  # 64
img_channels = X_train.shape[4]  # 12
input_shape = (img_height, img_width, img_depth, img_channels)


def UNet_3D_Model(input_shape):
    # network structure
    filter_numb = 16  # number of filters for the first layer
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Contraction path:
    # DownRes 1, convolution + pooling
    conv_64 = conv_block(inputs, filter_numb, dropout=0.10)
    pool_32 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_64)
    # DownRes 2
    conv_32 = conv_block(pool_32, 2 * filter_numb, dropout=0.15)
    pool_16 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_32)
    # DownRes 3
    conv_16 = conv_block(pool_16, 4 * filter_numb, dropout=0.20)
    pool_8 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_16)
    # DownRes 4
    conv_8 = conv_block(pool_8, 8 * filter_numb, dropout=0.25)
    pool_4 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_8)
    # DownRes 5, convolution only

    conv_4 = conv_block(pool_4, 16 * filter_numb, dropout=0.30)

    # Upsampling layers
    up_8 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(conv_4)
    up_8 = layers.concatenate([up_8, conv_8])
    up_conv_8 = conv_block(up_8, 8 * filter_numb, dropout=0.25)
    # UpRes 7
    up_16 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_8)
    up_16 = layers.concatenate([up_16, conv_16])
    up_conv_16 = conv_block(up_16, 4 * filter_numb, dropout=0.20)
    # UpRes 8
    up_32 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32])
    up_conv_32 = conv_block(up_32, 2 * filter_numb, dropout=0.15)
    # UpRes 9
    up_64 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64])
    up_conv_64 = conv_block(up_64, filter_numb, dropout=0.10)

    # final convolutional layer
    conv_final = layers.Conv3D(1, (1, 1, 1))(up_conv_64)
    conv_final = layers.Activation('linear')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[conv_final], name="UNet_3D_Model")
    model.summary()
    return model


# Test if everything is working ok.
model = UNet_3D_Model(input_shape)
print(model.input_shape)
print(model.output_shape)

# Parameters for model
img_height = X_train.shape[1]  # 64
img_width = X_train.shape[2]  # 64
img_depth = X_train.shape[3]  # 64
img_channels = X_train.shape[4]  # 12
input_shape = (img_height, img_width, img_depth, img_channels)


def Attention_UNet_3D_Model(input_shape):
    # network structure
    filter_numb = 16  # number of filters for the first layer
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_64 = conv_block(inputs, filter_numb, dropout=0.10)
    pool_32 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_64)
    # DownRes 2
    conv_32 = conv_block(pool_32, 2 * filter_numb, dropout=0.15)
    pool_16 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_32)
    # DownRes 3
    conv_16 = conv_block(pool_16, 4 * filter_numb, dropout=0.20)
    pool_8 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_16)
    # DownRes 4
    conv_8 = conv_block(pool_8, 8 * filter_numb, dropout=0.25)
    pool_4 = layers.MaxPooling3D((2, 2, 2), padding="same")(conv_8)
    # DownRes 5, convolution only

    conv_4 = conv_block(pool_4, 16 * filter_numb, dropout=0.30)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_8 = gating_signal(conv_4, 8 * filter_numb)
    att_8 = attention_block(conv_8, gating_8, 8 * filter_numb)
    up_8 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(conv_4)
    up_8 = layers.concatenate([up_8, att_8])
    up_conv_8 = conv_block(up_8, 8 * filter_numb, dropout=0.25)
    # UpRes 7
    gating_16 = gating_signal(up_conv_8, 4 * filter_numb)
    att_16 = attention_block(conv_16, gating_16, 4 * filter_numb)
    up_16 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, 4 * filter_numb, dropout=0.20)
    # UpRes 8
    gating_32 = gating_signal(up_conv_16, 2 * filter_numb)
    att_32 = attention_block(conv_32, gating_32, 2 * filter_numb)
    up_32 = layers.UpSampling3D((2, 2, 2), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, 2 * filter_numb, dropout=0.15)
    # UpRes 9
    gating_64 = gating_signal(up_conv_32, filter_numb)
    att_64 = attention_block(conv_64, gating_64, filter_numb)
    up_64 = layers.UpSampling3D(size=(2, 2, 2), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, filter_numb, dropout=0.10)

    # final convolutional layer
    conv_final = layers.Conv3D(1, (1, 1, 1))(up_conv_64)
    conv_final = layers.Activation('linear')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[conv_final], name="Attention_UNet_3D_Model")
    model.summary()
    return model


# Test if everything is working ok.
model = Attention_UNet_3D_Model(input_shape)
print(model.input_shape)
print(model.output_shape)

###############################################################################
# 3. TRAIN AND VALIDATE THE CNN MODEL #########################################
###############################################################################

# Fit the model

epochs = 64
batch_size = 4
steps_per_epoch = len(X_train) // batch_size
val_steps_per_epoch = len(X_val) // batch_size
metrics = ['accuracy', 'mae']
loss = 'mean_squared_error'
LR = 0.001
optimizer = tf.keras.optimizers.Adam(LR)

model = load_model('saved_model/UNet_best_model.epoch25-loss0.01.hdf5')
#model = UNet_3D_Model(input_shape=input_shape)
# model = Attention_UNet_3D_Model(input_shape=input_shape)
# model = Res_UNet_3D_Model(input_shape=input_shape)
# model = Attention_Res_UNet_3D_Model(input_shape=input_shape)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy', 'mae'])
# model.compile(optimizer = optimizer, loss=loss, metrics=metrics)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

## TO PREVENT OVERFITTING: Use early stopping method to solve model over-fitting problem
early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', verbose=1)
# The patience parameter is the amount of epochs to check for improvement

## Checkpoint: ModelCheckpoint callback saves a model at some interval.
#checkpoint_filepath = "saved_model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"  # File name includes epoch and validation loss.
checkpoint_filepath = 'saved_model/UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
# checkpoint_filepath = 'saved_model/Att_UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
# checkpoint_filepath = 'saved_model/Res_UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
# checkpoint_filepath = 'saved_model/Att_Res_UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min',  # #Use Mode = max for accuracy and min for loss.
                                                )
## CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = tf.keras.callbacks.CSVLogger('my_logs.csv', separator=',', append=False)

# Train the model
import time
start = time.time()
# start1 = datetime.now()
history = model.fit(X_train, Y_train,
                    steps_per_epoch=steps_per_epoch,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stopping, checkpoint, log_csv],
                    validation_data=(X_val, Y_val),
                    validation_steps=val_steps_per_epoch,
                    shuffle=False,
                    )

finish = time.time()
# stop = datetime.now()
# Execution time of the model
print('total execution time in seconds is: ', finish - start)
# print(history.history.keys())
print('Training has been finished successfully')

