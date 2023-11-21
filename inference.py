import argparse
from keras.models import load_model
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras import backend as K
import tensorflow as tf


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
parser.add_argument("ImagePath", help="Path to the picture (.jpg).")
parser.add_argument("ModelPath", help="Path to the model (.h5).")
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

model = load_model(args.ModelPath, custom_objects={
    'dice_coef_loss': dice_coef_loss,
    'jacard_coef': jacard_coef
})
image = np.array(Image.open(args.ImagePath))
predicted_mask = model.predict(np.expand_dims(image, 0))
predicted_mask *= 255
predicted_mask = predicted_mask.reshape((768, 768)).astype('uint8')
Image.fromarray(predicted_mask).show(title='Predicted Mask')
Image.fromarray(image).show(title='Image')
