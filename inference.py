from keras.models import load_model
import segmentation_models as sm
from keras import backend as K
import cv2
from PIL import Image
from patchify import patchify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ImagePath", help="Path to the picture (.jpg).")
parser.add_argument("ModelPath", help="Path to the model (.h5).")
parser.add_argument("TrainShipSegmentationCSVPath", help="Path to train_ship_segmentations_v2.csv file.")
args = parser.parse_args()

path = Path(args.ImagePath)

if args.ImagePath[-4:] != ".jpg":
    print("Please enter .jpg file")
    exit()
if not path.is_file():
    print(f'The file {args.ImagePath} does not exist')
    exit()

path = Path(args.ModelPath)

if args.ModelPath[-3:] != ".h5":
    print("Please enter .h5 file")
    exit()
if not path.is_file():
    print(f'The file {args.ModelPath} does not exist')
    exit()

path = Path(args.TrainShipSegmentationCSVPath)

if args.TrainShipSegmentationCSVPath[-4:] != ".csv":
    print("Please enter .csv file")
    exit()
if not path.is_file():
    print(f'The file {args.TrainShipSegmentationCSVPath} does not exist')
    exit()


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


# Load data from csv file, extract encoded pixels to create masks for training
masks = pd.read_csv(args.TrainShipSegmentationCSVPath)
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)


def jacard_coef(y_true, y_pred):
    """
    Jacard coefficient for model metrics.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


# Parameters for model
weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Load pretrained model
model = load_model(args.ModelPath,
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef': jacard_coef})

testing_image_patches = []

image = cv2.imread(args.ImagePath)  # Read each image as BGR
SIZE_X = 768
SIZE_Y = 768
image = Image.fromarray(image)
image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
patch_size = 256
scaler = MinMaxScaler()
image = np.array(image)

# Extract patches from each image
patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, :, :]
        # Use minmaxscaler instead of just dividing by 255.
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

        # single_patch_img = (single_patch_img.astype('float32')) / 255.
        single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
        testing_image_patches.append(single_patch_img)

testing_image_patches = np.array(testing_image_patches)

buffer = np.zeros((9, 256, 256))
for l in range(9):
    prediction = (model.predict(np.expand_dims(testing_image_patches[l], 0)))
    predicted_img = np.array(prediction)[0, :, :]
    for i in range(256):
        for j in range(256):
            if predicted_img[i][j][0] < predicted_img[i][j][1]:
                buffer[l][i][j] = 1

pred_mask = np.block([[buffer[0].T, buffer[1].T, buffer[2].T],
                      [buffer[3].T, buffer[4].T, buffer[5].T],
                      [buffer[6].T, buffer[7].T, buffer[8].T]])

img = cv2.imread(args.ImagePath)
img_masks = masks.loc[masks['ImageId'] == args.ImagePath[-13:], 'EncodedPixels'].tolist()

all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask, (768, 768))


plt.figure(figsize=(15, 9))
plt.subplot(131)
plt.title("Actual Image:")
plt.imshow(image)
plt.subplot(132)
plt.title("Actual Mask:")
plt.imshow(all_masks)
plt.subplot(133)
plt.title("Predicted Mask:")
plt.imshow(pred_mask)
plt.show()
