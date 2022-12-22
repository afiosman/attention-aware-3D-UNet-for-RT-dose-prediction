## KPB: Attention U-Net for accurate radiotherapy dose prediction 
# @author: Alexander F.I. Osman, April 2021

"""
This code demonstrates an attention U-Net model for voxel-wise dose prediction in radiation therapy.
The model is trained, validated, and tested using the OpenKBP—2020 AAPM Grand Challenge dataset.
"""

##########################################################

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import random
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

###############################################################################
# 1. LOAD DATA AND PREFORM PRE-PROCESSING #####################################
###############################################################################

combined_X = np.load('saved_data/combined_X.npy')
combined_Y = np.load('saved_data/combined_Y.npy')

X_train, X_test, Y_train, Y_test = train_test_split(combined_X, combined_Y, test_size=0.20, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=1)

###############################################################################
# 2. BUILD THE MODEL ARCHITECTURE #############################################
###############################################################################

# For consistency
# Since the neural network starts with random initial weights, the results of this
# example will differ slightly every time it is run. The random seed is set to avoid
# this randomness. However this is not necessary for your own applications.
seed = 42
np.random.seed = seed

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
epochs = 120
batch_size = 4
steps_per_epoch = len(X_train) // batch_size
val_steps_per_epoch = len(X_val) // batch_size
metrics = ['accuracy', 'mae']
loss = 'mean_squared_error'
LR = 0.001
optimizer = tf.keras.optimizers.Adam(LR)

# model = UNet_3D_Model(input_shape=input_shape)
model = Attention_UNet_3D_Model(input_shape=input_shape)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy', 'mae'])
# model.compile(optimizer = optimizer, loss=loss, metrics=metrics)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

## TO PREVENT OVERFITTING: Use early stopping method to solve model over-fitting problem
early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', verbose=1)
# The patience parameter is the amount of epochs to check for improvement

## Checkpoint: ModelCheckpoint callback saves a model at some interval.
# checkpoint_filepath = 'saved_model/UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
checkpoint_filepath = 'saved_model/Att_UNet_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min',  # #Use Mode = max for accuracy and min for loss.
                                                )
"""
# Decaying learning rate
reduce_lr = tf.keras.callbacks.callback_reduce_lr_on_plateau(
    monitor = "val_loss", 
    factor = 0.1, 
    patience = 10, 
    verbose = 0,
    mode = c("auto", "min", "max"),
    min_delta = 1e-04,
    cooldown = 0,
    min_lr = 0)
"""
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

## Plot training history
## LEARNING CURVE: plots the graph of the training loss vs.validation
# loss over the number of epochs.
def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('average training loss and validation loss')
    plt.ylabel('mean-squared error')
    plt.xlabel('epoch')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.show()
plot_history(history)

# Save Model
model.save('saved_model/dose_pred_Att_Unet3D_model_120epochs.hdf5')

## Evaluating the model
train_loss, train_acc, train_acc1 = model.evaluate(X_train, Y_train, batch_size = 8)
val_loss, test_acc, train_acc2 = model.evaluate(X_val, Y_val, batch_size = 8)
print('Train: %.3f, Test: %.3f' % (train_loss, val_loss))

###############################################################################
# 4. MAKE PREDICTIONS ON TEST DATASET #########################################
###############################################################################

# load test data >> for DVH plot and visualization
X_test = np.load('saved_data/X_test.npy')
Y_test = np.load('saved_data/Y_test.npy')

# Set compile=False as we are not loading it for training, only for prediction.
# Loading
#new_model = load_model('saved_model/UNet_best_model.epoch113-loss0.00.hdf5')
new_model = load_model('saved_model/Att_UNet_best_model.epoch112-loss0.00.hdf5')

# Check its architecture
new_model.summary()

# Predict on the test set: Let us see how the model generalize by using the test set.
predict_test = new_model.predict(X_test, verbose=1, batch_size = 4)

#Processing the negative values
predict_test2 = []
for pt_id in range(len(X_test)):
    print("pt_id: ", pt_id)
    predict_test1 = predict_test[pt_id].astype('float32')
    id1 = X_test[pt_id,:,:,:,0].astype(int)
    predict_test1[predict_test1 <=0] = 0
    predict_test2.append(np.array(predict_test1))
predict_test2 = (np.array(predict_test2)).astype('float32')

predict_test = predict_test2

img_height = X_test.shape[1] # 64
img_width = X_test.shape[2] # 64
img_depth = X_test.shape[3] # 64
img_channels = X_test.shape[4] # 1
predict_test = np.reshape(predict_test, (len(predict_test), img_height, img_width, img_depth))
real_test = np.reshape(Y_test, (len(Y_test), img_height, img_width, img_depth))

# Renormalize the dose to original scale by multipying with the prescription (70 Gy)
predict_test = predict_test * 70
real_test = real_test * 70

# Axial, Sagital, & Coronal views
image_number = 42
slice_number = 11
fig = plt.figure()
grid = plt.GridSpec(3, 4, wspace = .15, hspace = .15)
exec (f"plt.subplot(grid{[0]})")
plt.imshow(X_test[image_number,:,:,slice_number,11], cmap='gray')
plt.title('ct'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[1]})")
plt.imshow(predict_test[image_number,:,:,slice_number], cmap='jet')
plt.title('Predicted Test'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[2]})")
plt.imshow(real_test[image_number,:,:,slice_number], cmap='jet')
plt.title('GT'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[3]})")
plt.imshow((predict_test[image_number,:,:,slice_number] - real_test[image_number,:,:,slice_number]), cmap='jet')
plt.title('residual'),
plt.colorbar(label='Gy'), plt.axis('off')

image_number = 51
slice_number = 30
exec (f"plt.subplot(grid{[4]})")
plt.imshow(X_test[image_number,:,slice_number,:,11].T, cmap='gray')
plt.title('ct'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[5]})")
plt.imshow(predict_test[image_number,:,slice_number,:].T, cmap='jet')
plt.title('Predicted Test'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[6]})")
plt.imshow(real_test[image_number,:,slice_number,:].T, cmap='jet')
plt.title('GT'),
plt.colorbar(label='Gy'), plt.axis('off')
exec (f"plt.subplot(grid{[7]})")
plt.imshow((predict_test[image_number,:,slice_number,:] - real_test[image_number,:,slice_number,:]).T, cmap='jet')
plt.title('residual'),
plt.colorbar(label='Gy'), plt.axis('off')

image_number = 51
slice_number = 28
exec (f"plt.subplot(grid{[8]})")
plt.imshow(X_test[image_number,slice_number,:,:,11].T, cmap='gray')
plt.title('ct'),
plt.colorbar(), plt.axis('off')
exec (f"plt.subplot(grid{[9]})")
plt.imshow(predict_test[image_number,slice_number,:,:].T, cmap='jet')
plt.title('Predicted Test'),
plt.colorbar(label='Gy'), plt.axis('off')
exec (f"plt.subplot(grid{[10]})")
plt.imshow(real_test[image_number,slice_number,:,:].T, cmap='jet')
plt.title('GT'),
plt.axis('off')
plt.colorbar(label='Gy')
exec (f"plt.subplot(grid{[11]})")
plt.imshow((predict_test[image_number,slice_number,:,:] - real_test[image_number,slice_number,:,:]).T, cmap='jet')
plt.title('residual'),
plt.axis('off')
plt.colorbar(label='Gy')
plt.show()

###############################################################################
###############################################################################
