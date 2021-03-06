# Semantic segmentation neural network on airbus detection chalange dataset
# Project goal
The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 
# Configuration instructions 
Before you can run this scipts, you need to be sure, that every needed library from requirenments.txt is installed. After that you will need to download dataset for training or some images with size of 768x768 pixels and model for testing. In this work model is somewhat special, so I recommend using model_4.h5 from model directory of this project.
# How to train model?
To train the model you need to download dataset Airbus Ship Detection Challenge from kaggle. Once you downloaded it, prepare empty folder in which all the neccesary data will be stored, and unzip everything from dataset into this folder. Once done, you may start training your deep learning neural network model. To do so, you need download file train.py from this repository, install everything specified in requirentments.txt file and you are pretty much ready to go. The first thing to do after everything is ready is to start train.py in console like this:

    C:\.\python train.py C:\..\dataset C:\...\trained_models
Where 
- C:\\.\ is path to the train.py file
- C:\\..\dataset is path to the dataset directory you created beforehand, it is a directory where train_v2, test_v2 etc. is located
- C:\\...\trained_models is path to the directory where you wish to save your trained models
# How to test model?
To test your model you need some of the images you wish to test, prefarably from Airbus Ship Detection Challenge dataset, and it is highly recomended to download this dataset or at least train_ship_segmentations_v2.csv file as nothing is going to work without it, your trained model and to install everything specified in requirentments.txt. After everything is ready download inference.py and model_4.h5 from model directory if you don't have pre trained model. Once everything is ready start inference.py in console like this:

    C:\.\python inference.py C:\..\image.jpg C:\...\model.h5 C:\....\train_ship_segmentations_v2.csv
Where
- C:\\.\ is path to the inference.py file
- C:\\..\image.jpg is path to the image using which you want to test your model
- C:\\...\model.h5 is your model that you want to test
- C:\\....\train_ship_segmentations_v2.csv is path to train_ship_segmentations_v2.csv file, it is used to see true mask of the images from train_v2 directory of the Airbus Ship Detection Challenge dataset
# Description of solution
For starters we find all images with amount of ships greater than 13, so that we can train model more quickly on them. Then we patchify those images (1 x 768x768 => 9 x 256x256), to get larger dataset and give the model more variety of situation. After that we create UNet model with 9 layers, where 5 of them are for contraction and the other 4 are for expansion. Then define function for dice score loss and Jacard coefficient for model metrics. Upgrade dice score loss function (no particular reason), compile model and start training. To get more or less good results the minimum amount of epochs should be around 30, after 50 epochs training proccess is very long (you may not see any differencies between 50 and 70 epochs of training). After training is done, you may save the model.
Inference is not very different in that aspect. Firstly we choose any image we want to use for prediction. This image will be patchified into 9 pieces, after that every piece will go through the model and be evaluated. In the end, predicted masks would be merged in one and displayed with original image and mask.
# A list of files included 
        Exploratory data analysis of the dataset.ipynb
        README.md
        inference.py
        requirements.txt
        train.py
        model\
                model_4.h5
# About model_4.h5
This model was trained on Airbus Ship Detection Challenge dataset where amount of ships on the image was 14 or 15. 
- Amount of epochs: 70
- Time per epoch: about 12 minutes
- Amount of training material: about 130 full scale images and 1200 patched ones.
- Accuracy of the model (amount of predicted pixels): 0.9949
- Loss of the model: 0.5677
- Actual accuracy of the model: 0.83835983
