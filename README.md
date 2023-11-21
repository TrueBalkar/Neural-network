# Semantic segmentation neural network on airbus detection chalange dataset
# Project goal
The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 
# Configuration instructions 
Before you can run this scipts, you need to be sure, that every needed library from requirenments.txt is installed. After that you will need to download dataset for training or some images with size of 768x768 pixels and model for testing.
# How to train model?
To train the model you need to download dataset Airbus Ship Detection Challenge from kaggle. Once you downloaded it, prepare empty folder in which all the neccesary data will be stored, and unzip everything from dataset into this folder. Once done, you may start training the model. To do so, you need download file train.py from this repository, install everything specified in requirentments.txt file and you are pretty much ready to go. The first thing to do after everything is ready is to start train.py in console like this:

    C:\.\python train.py C:\..\dataset C:\...\trained_models
Where 
- C:\\.\ is path to the train.py file
- C:\\..\dataset is path to the dataset directory you created beforehand, it is a directory where train_v2, test_v2 etc. is located
- C:\\...\trained_models is path to the directory where you wish to save your trained models
# How to test model?
To test the model you need some of the images you wish to test and to install everything specified in requirentments.txt. After everything is ready download inference.py and model_4.h5 from model directory if you don't have pre trained model. Once everything is ready start inference.py in console like this:

    C:\.\python inference.py C:\..\image.jpg C:\...\model.h5 C:\....\train_ship_segmentations_v2.csv
Where
- C:\\.\ is path to the inference.py file
- C:\\..\image.jpg is path to the image using which you want to test your model
- C:\\...\model.h5 is your model that you want to test
- C:\\....\train_ship_segmentations_v2.csv is path to train_ship_segmentations_v2.csv file, it is used to see true mask of the images from train_v2 directory of the Airbus Ship Detection Challenge dataset
# Description of solution
For starters we find all images with amount of ships greater than 0, so that we can train model using them. After that we create UNet model with 9 layers, where 4 of them are for contraction one for bottleneck and the other 4 are for expansion. Then define function for dice score loss and Jacard coefficient for model metrics. Compile model and start training. To get more or less good results the minimum amount of epochs should be around 30, after 50 epochs training proccess is very long (you may not see any differencies between 50 and 70 epochs of training). After training is done, you may save the model. In my case the model was trained only for 10 epochs as it takes a lot of time for each epoch to train.
Inference is not very different in that aspect. Firstly we choose any image we want to use for prediction. This image will be evaluated. In the end, predicted mask will be displayed with original image.
# A list of files included 
        Exploratory data analysis of the dataset.ipynb
        README.md
        inference.py
        requirements.txt
        train.py
        model\
                model.h5
# About model.h5
This model was trained on Airbus Ship Detection Challenge dataset where amount of ships on the image was more than 0.                
Metrics used: jacard coefficient, precision, recall                     
Loss functions: dice coefficient loss function                  
- Amount of epochs: 10
- Time per epoch: about 20 minutes
- Amount of training material: about 33k 768x768 images.
- loss: 0.1750
- jacard_coef: 0.7021
- precision: 0.8316
- recall: 0.7648
