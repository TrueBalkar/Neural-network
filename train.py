import pandas as pd
import numpy as np
import cv2
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
import argparse


# Basic UNet model has over 31M+ params, so we need to simplify it
# in order to train it faster and avoid overfitting
def unet_model_simplified(input_shape=(768, 768, 3)):
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Contracting Path
    conv1 = layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(8, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)

    # Expansive Path
    up6 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv4], axis=-1)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv3], axis=-1)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = layers.concatenate([up8, conv2], axis=-1)
    conv8 = layers.Conv2D(16, 3, activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = layers.concatenate([up9, conv1], axis=-1)
    conv9 = layers.Conv2D(8, 3, activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(8, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def jacard_coef(y_true, y_pred):
    """
    Jacard coefficient for model metrics.
    """
    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    return (intersection + 1.0) / (union + 1.0)


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


parser = argparse.ArgumentParser()
parser.add_argument("DatasetPath", help="Path to the dataset.")
parser.add_argument("SaveModelPath", help="Path to save the model.")
args = parser.parse_args()

if not os.path.isdir(args.DatasetPath):
    print(f'The directory {args.DatasetPath} does not exist')
    exit()

if not os.path.isdir(args.SaveModelPath):
    print(f'The directory {args.SaveModelPath} does not exist')
    exit()

path = args.DatasetPath

masks = pd.read_csv(fr"{path}/train_ship_segmentations_v2.csv")
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)


# Function to decode mask
def rle_decode(mask_rle, IMG_SIZE = (768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 255 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(IMG_SIZE[0]*IMG_SIZE[1])
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(IMG_SIZE).T


# Create directories for images and masks used for training
try:
    os.mkdir(path + "/data/")
    os.mkdir(path + "/data/masks/")
    os.mkdir(path + "/data/images/")
except:
    pass

# Search for all images with number of ships greater than 0
ids_of_images = unique_img_ids.loc[unique_img_ids.ships > 0, 'ImageId'].tolist()

# Go through all of those images, decode masks, save all the masks and images to created directories
for ImageId in ids_of_images:
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    all_masks = np.zeros((768, 768))

    if type(img_masks[0]) != float:
        buffer = ""
        for i in img_masks:
            buffer = buffer + " " + i
        img_masks = buffer
        all_masks += rle_decode(img_masks, (768, 768))

    print("Writing: " + ImageId)
    cv2.imwrite(path + "/data/masks/" + ImageId, all_masks)
    shutil.copy(path + "/train_v2/" + ImageId, path + "/data/images")


# Create the U-Net model
model = unet_model_simplified()

# Display the model summary
model.summary()


# Get images and masks paths in dataframe
train_data_dir = path + '/data'
train_images_dir = train_data_dir + '/images'
train_masks_dir = train_data_dir + '/masks'

file_list = os.listdir(train_images_dir)
df = pd.DataFrame({'image_path': [os.path.join(train_images_dir, img) for img in file_list],
                   'mask_path': [os.path.join(train_masks_dir, img) for img in file_list]})

image_size = (768, 768)

# Create an ImageDataGenerator for data augmentation, normalization and validation_split
data_gen_args = dict(rescale=1./255, validation_split=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Create data generators for training images and masks
seed = 1
image_generator = image_datagen.flow_from_dataframe(
    df,
    x_col='image_path',
    target_size=image_size,
    class_mode=None,
    seed=seed,
    batch_size=32
)

mask_generator = image_datagen.flow_from_dataframe(
    df,
    x_col='mask_path',
    target_size=image_size,
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    batch_size=32
)

train_generator = zip(image_generator, mask_generator)


model.compile(optimizer='adam',
              loss=dice_coef_loss,
              metrics=[jacard_coef, Precision(), Recall()])

history = model.fit(train_generator, epochs=2, steps_per_epoch=len(image_generator))

model.save(args.SaveModelPath + "/model_4.h5")
