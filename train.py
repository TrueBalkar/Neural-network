import numpy as np
import pandas as pd
import os
import cv2
import shutil
from keras.layers import Conv2D, Input, Dropout, concatenate, MaxPooling2D, Conv2DTranspose
from keras import Model
from tensorflow.keras.utils import to_categorical
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU

print("Enter path to the dataset: ")
path = input()

# Load data from csv file, extract encoded pixels to create masks for training
masks = pd.read_csv(fr"{path}/train_ship_segmentations_v2.csv")
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)


# Function to get mask from encoded pixels
def rle_decode(mask_rle, IMG_SIZE=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 255 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(IMG_SIZE[0]*IMG_SIZE[1])
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(IMG_SIZE).T


# Create directories for images and masks used for training
os.mkdir(path + "/data/")
os.mkdir(path + "/data/masks/")
os.mkdir(path + "/data/images/")

# Search for all images with number of ships greater than 13
ids_of_images = unique_img_ids.loc[unique_img_ids.ships > 13, 'ImageId'].tolist()

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

# Use minmaxscaler instead of just dividing by 255 while patching
scaler = MinMaxScaler()

# Define root directory from which all the images and masks will be taken
root_directory = path + '/data/'

# Define size of patched images
# 256 stands for 256x256 pixels
patch_size = 256

# Read images from repsective 'images' subdirectory
# And then divide all images into patches of 256x256x3.
image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        images_list = os.listdir(path)
        for i, image_name in enumerate(images_list):
            image = cv2.imread(path + "/" + image_name, 1)
            SIZE_X = 768  # Width of input images
            SIZE_Y = 768  # Height of input images
            image = Image.fromarray(image)
            image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            image = np.array(image)

            # Extract patches from each image
            print("Now patchifying image:", path + "/" + image_name)
            # Step = patch_size for no overlaping
            patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]

                    # Use minmaxscaler instead of just dividing by 255.
                    single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

                    single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
                    image_dataset.append(single_patch_img)


# Do the same for masks
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        masks_list = os.listdir(path)
        for i, mask_name in enumerate(masks_list):
            mask = cv2.imread(path + "/" + mask_name, 1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            SIZE_X = 768  # Width of input images
            SIZE_Y = 768  # Height of input images
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
            mask = np.array(mask)

            # Extract patches from each image
            print("Now patchifying mask:", path + "/" + mask_name)
            patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    single_patch_mask = single_patch_mask[0]  # Drop the extra unecessary dimension that patchify adds.
                    mask_dataset.append(single_patch_mask)


# Convert images and masks into numpy array
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Define colors of objects in masks
Ship = np.array((255, 255, 255))
NotShip = np.array((0, 0, 0))

# Create numpy array for labels with the same shape as masks
label = single_patch_mask


# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Ship, axis=-1)] = 0
    label_seg[np.all(label == NotShip, axis=-1)] = 1

    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels

    return label_seg


labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)


def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    """
    UNet model for problems with 2 and more classes.

    :param n_classes: Number of classes to classify.
    :param IMG_HEIGHT: Height of the input image.
    :param IMG_WIDTH: Width of the input image.
    :param IMG_CHANNELS: Number of channels of the image.
    :return: tf.keras.Model
    """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def jacard_coef(y_true, y_pred):
    """
    Jacard coefficient for model metrics.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

# Parameters for model
weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss=dice_loss, metrics=metrics)
# model.summary()

# Train the model
history1 = model.fit(X_train, y_train,
                     batch_size=32,
                     verbose=1,
                     epochs=20,
                     validation_data=(X_test, y_test),
                     shuffle=True)

# Save the model
print("Enter path for saving model: ")
lol = input()

model.save(lol + "/model_4.h5")

# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

# Using built in keras function for IoU
# n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
